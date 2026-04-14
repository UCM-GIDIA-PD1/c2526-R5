import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import scipy.sparse as sp
import gc
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import os
os.environ["WANDB_MODE"] = "offline"   
os.environ["WANDB_START_METHOD"] = "thread"  

import wandb


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.minio_client import download_df_parquet

print("Importación realizada con éxito desde:", ROOT)



#Descargar dataset
access_key = os.getenv("MINIO_ACCESS_KEY")
secret_key = os.getenv("MINIO_SECRET_KEY")

ruta_archivo = "grupo5/final/year=2025/month=01/dataset_final.parquet"
df_final = download_df_parquet(access_key, secret_key, ruta_archivo)


#Descargamos 1 mes 
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"

dates = pd.date_range(start=START_DATE, end=END_DATE).strftime("%Y-%m-%d").tolist()
dfs = []
for date in dates:
    try:
        df_gtfs = download_df_parquet(access_key, secret_key,f"grupo5/cleaned/gtfs_clean_scheduled/date={date}/gtfs_scheduled_{date}.parquet")
    except:
        print(f"Could not download data for date: {date}")
        continue
    dfs.append(df_gtfs)
df = pd.concat(dfs, ignore_index=True)


#Matriz de adyacencia
# 1. ORDENAR PARA DESCUBRIR CONEXIONES
# Aseguramos el orden temporal de los eventos dentro de cada viaje único
df = df.sort_values(by=['trip_uid', 'scheduled_seconds']).reset_index(drop=True)

# 2. CALCULAR EL TIEMPO ENTRE ESTACIONES
# Desplazamos las columnas para ver cuál es la siguiente parada en la ruta del tren
df['next_stop_id'] = df.groupby('trip_uid')['stop_id'].shift(-1)
df['next_scheduled_seconds'] = df.groupby('trip_uid')['scheduled_seconds'].shift(-1)

# Eliminamos la última parada de cada viaje (no tiene destino)
edges_df = df.dropna(subset=['next_stop_id']).copy()

# Tiempo de viaje estimado
edges_df['travel_time'] = edges_df['next_scheduled_seconds'] - edges_df['scheduled_seconds']

# Filtramos errores de los datos (tiempos negativos o de 0 segundos)
edges_df = edges_df[edges_df['travel_time'] > 0] 

# 3. CREAR EL GRAFO DE CONEXIONES Y PESOS
# Agrupamos por origen-destino para sacar la mediana del tiempo de viaje
graph_df = edges_df.groupby(['stop_id', 'next_stop_id']).agg(
    median_travel_time=('travel_time', 'median'),
    trip_count=('trip_uid', 'count')
).reset_index()

# Filtro de seguridad: quitamos "enlaces fantasma" (desvíos raros que ocurrieron < 5 veces)
graph_df = graph_df[graph_df['trip_count'] > 5]

# 4. MAPEADO DE NODOS A ÍNDICES DE LA MATRIZ
nodes = sorted(list(set(df['stop_id'].unique()) | set(df['next_stop_id'].dropna().unique())))
n_nodes = len(nodes)
node_to_idx = {stop_id: idx for idx, stop_id in enumerate(nodes)}

# 5. CREACIÓN DE LA MATRIZ PONDERADA 
A_weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
sigma = graph_df['median_travel_time'].std()

for _, row in graph_df.iterrows():
    i = node_to_idx[row['stop_id']]
    j = node_to_idx[row['next_stop_id']]
    dist = row['median_travel_time']
    
    # Asignamos peso: estaciones con menor tiempo de viaje tendrán un valor más cercano a 1.0
    peso = np.exp(- (dist ** 2) / (sigma ** 2))
    
    A_weighted[i, j] = peso
    A_weighted[j, i] = peso # Forzamos que la conexión sea bidireccional para estabilizar la GNN

# 6. NORMALIZACIÓN ESPACIAL PARA GNN: D^(-1/2) * A * D^(-1/2)
# Añadimos auto-conexiones (unos en la diagonal)
np.fill_diagonal(A_weighted, 1.0)

# Sumamos las conexiones de cada nodo (grado)
grados = np.sum(A_weighted, axis=1)

# Hacemos la inversa de la raíz cuadrada (manejando divisiones por cero de forma segura)
grados_inv_raiz = np.power(grados, -0.5, where=(grados!=0))
grados_inv_raiz[np.isinf(grados_inv_raiz)] = 0.0

matriz_diagonal = np.diag(grados_inv_raiz)

# Multiplicación matricial final
A_norm = matriz_diagonal @ A_weighted @ matriz_diagonal

# 7. CONVERSIÓN A TENSOR
A_tensor = torch.tensor(A_norm, dtype=torch.float32)

print(f"Número de nodos únicos: {n_nodes}")
print(f"Matriz de Adyacencia lista y normalizada. Forma: {A_tensor.shape}")


# Definir las variables 
variables_entrada = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
    'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay',
    'actual_headway_seconds', 'hour_sin', 'hour_cos', 'dow'
]
variables_objetivo = ['target_delay_10m', 'target_delay_20m', 'target_delay_30m', 'station_delay_10m', 'station_delay_20m',
       'station_delay_30m']

# Función para crear el tensor asegurando que cuadre con la matriz de adyacencia
def crear_tensores_stgcn(df, mapa_nodos, features, targets, freq='15min'):
    # Filtrar solo las estaciones que existen en nuestra matriz de adyacencia
    nodos_validos = list(mapa_nodos.keys())
    df = df[df['stop_id'].isin(nodos_validos)].copy()
    
    # Asegurar que el tiempo esté en el formato correcto
    df['time_bin'] = pd.to_datetime(df['merge_time']).dt.floor(freq)
    
    # Reglas de agregación por si hay varios trenes en la misma estación en esos 15 mins
    reglas_agregacion = {
        'delay_seconds': 'mean',
        'lagged_delay_1': 'mean',
        'lagged_delay_2': 'mean',
        'is_unscheduled': 'sum',
        'temp_extreme': 'max',  # Mantenemos el máximo (si hubo clima extremo, aplica a todo el bloque)
        'n_eventos_afectando': 'max',
        'route_rolling_delay': 'mean',
        'actual_headway_seconds': 'mean',
        'target_delay_10m': 'mean',
        'target_delay_20m': 'mean',
        'target_delay_30m': 'mean',
        'station_delay_10m': 'mean',
        'station_delay_20m': 'mean',
        'station_delay_30m': 'mean',
    }
    
    print("Agrupando datos por ventanas de tiempo y estación...")
    df_agrupado = df.groupby(['time_bin', 'stop_id']).agg(reglas_agregacion)
    
    # Liberar memoria del dataframe original
    del df
    gc.collect()
    
    # Crear un "Grid" (cuadrícula) perfecto: Tiempos x Nodos
    todos_los_tiempos = pd.date_range(
        start=df_agrupado.index.get_level_values('time_bin').min(),
        end=df_agrupado.index.get_level_values('time_bin').max(),
        freq=freq
    )
    
    indice_completo = pd.MultiIndex.from_product(
        [todos_los_tiempos, nodos_validos], 
        names=['time_bin', 'stop_id']
    )
    df_completo = df_agrupado.reindex(indice_completo).reset_index()
    
    del df_agrupado
    gc.collect()
    
    print("Imputando valores faltantes y reconstruyendo tensores...")
    # Rellenar retrasos inexistentes con 0
    cols_retrasos = ['delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled', 'route_rolling_delay', 'actual_headway_seconds']
    df_completo[cols_retrasos] = df_completo[cols_retrasos].fillna(0)
    
    # Rellenar clima y eventos propagando el último valor conocido de esa estación
    cols_contexto = ['temp_extreme', 'n_eventos_afectando']
    df_completo[cols_contexto] = df_completo.groupby('stop_id')[cols_contexto].ffill().bfill().fillna(0)
    
    # Recalcular variables cíclicas para que sean perfectas y no sufran distorsión por la media
    df_completo['hour_sin'] = np.sin(2 * np.pi * df_completo['time_bin'].dt.hour / 24)
    df_completo['hour_cos'] = np.cos(2 * np.pi * df_completo['time_bin'].dt.hour / 24)
    df_completo['dow'] = df_completo['time_bin'].dt.dayofweek.astype(float)
    
    # Ordenar el dataframe usando el mismo índice de la matriz de adyacencia
    df_completo['nodo_idx'] = df_completo['stop_id'].map(mapa_nodos)
    df_completo = df_completo.sort_values(['time_bin', 'nodo_idx'])
    
    # Extraer las dimensiones
    T = len(todos_los_tiempos)
    N = len(nodos_validos)
    F = len(features)
    C = len(targets)
    
    # Transformar a tensores de NumPy tridimensionales
    X_tensor = df_completo[features].values.reshape(T, N, F)
    Y_tensor = df_completo[targets].values.reshape(T, N, C)
    
    del df_completo
    gc.collect()
    
    return X_tensor, Y_tensor, todos_los_tiempos

X_full, Y_full, array_tiempos = crear_tensores_stgcn(
    df_final, 
    node_to_idx, 
    variables_entrada, 
    variables_objetivo
)

print(f"\\n Tensor de entrada X_full creado: {X_full.shape} -> (Tiempos, Nodos, Features)")
print(f"Tensor objetivo Y_full creado: {Y_full.shape} -> (Tiempos, Nodos, Targets)")

# Eliminar cualquier NaN que haya quedado en X o en Y
# Reemplazamos los NaNs por 0 
X_full = np.nan_to_num(X_full, nan=0.0)
Y_full = np.nan_to_num(Y_full, nan=0.0)

# Comprobación de seguridad para asegurarnos de que ya no hay NaNs
print(f"¿Hay NaNs en X_full?: {np.isnan(X_full).any()}")
print(f"¿Hay NaNs en Y_full?: {np.isnan(Y_full).any()}")


# Separación temporal: Usamos el 80% del tiempo para entrenar y el 20% para test
num_tiempos = X_full.shape[0]
limite_corte = int(num_tiempos * 0.8)

X_train = X_full[:limite_corte]
X_test = X_full[limite_corte:]
Y_train = Y_full[:limite_corte]
Y_test = Y_full[limite_corte:]

print(f"Secuencias de entrenamiento: {X_train.shape[0]}")
print(f"Secuencias de test: {X_test.shape[0]}")

# Escalado (Aplanamos temporalmente a 2D, escalamos y devolvemos a 3D)
T_train, N, F = X_train.shape
T_test = X_test.shape[0]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, F)).reshape(T_train, N, F)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, F)).reshape(T_test, N, F)

# Escalamos también el objetivo (Y) para que el error MSE sea numéricamente estable
C = Y_train.shape[2]
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, C)).reshape(T_train, N, C)
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, C)).reshape(T_test, N, C)

print("Datos escalados correctamente.")


class DatasetSTGCN(Dataset):
    def __init__(self, X, Y, history_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.history_len = history_len

    def __len__(self):
        # Número de ventanas posibles
        return len(self.X) - self.history_len

    def __getitem__(self, idx):
        # Cogemos 'history_len' pasos de tiempo y predecimos el instante inmediatamente posterior
        ventana_x = self.X[idx : idx + self.history_len]
        objetivo_y = self.Y[idx + self.history_len]
        return ventana_x, objetivo_y

# Configuramos la ventana: Si freq='15min', history_len=8 equivale a mirar 2 horas atrás
history_len = 8 
batch_size = 32

train_dataset = DatasetSTGCN(X_train_scaled, Y_train_scaled, history_len)
test_dataset = DatasetSTGCN(X_test_scaled, Y_test_scaled, history_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Lotes de entrenamiento (batches): {len(train_loader)}")


# Arquitectura del modelo
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (Batch, Tiempo, Nodos, Features)
        # 1. Transformación de características
        x_transformed = torch.matmul(x, self.weight)
        # 2. Propagación en el grafo usando la matriz de adyacencia (Nodos x Nodos)
        salida = torch.einsum('vw, btwd -> btvd', adj, x_transformed)
        return salida + self.bias

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STConvBlock, self).__init__()
        # Convolución Temporal (kernel_size=(3,1) afecta a 3 instantes de tiempo, 1 nodo)
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.s_conv = GraphConv(out_channels, out_channels)
        self.t_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x, adj):
        # PyTorch Conv2d requiere forma: (Batch, Canales, Tiempo, Nodos)
        x_perm = x.permute(0, 3, 1, 2)
        
        # Capa temporal 1
        x_t1 = F.relu(self.t_conv1(x_perm))
        
        # Volvemos a formato de grafo: (Batch, Tiempo, Nodos, Canales)
        x_t1_perm = x_t1.permute(0, 2, 3, 1)
        
        # Capa espacial (Grafo)
        x_s = F.relu(self.s_conv(x_t1_perm, adj))
        
        # Capa temporal 2
        x_s_perm = x_s.permute(0, 3, 1, 2)
        x_out = F.relu(self.t_conv2(x_s_perm))
        
        # Devolvemos formato original
        return x_out.permute(0, 2, 3, 1)

class STGCN_Metro(nn.Module):
    def __init__(self, num_nodes, num_features, num_targets, history_len, adj_matrix):
        super(STGCN_Metro, self).__init__()
        # Registramos la matriz de adyacencia como un tensor estático que no requiere gradientes
        self.register_buffer('adj_matrix', adj_matrix)
        
        # Bloques Spatio-Temporales
        self.block1 = STConvBlock(num_features, 32)
        self.block2 = STConvBlock(32, 64)
        
        # Capa de salida: aplanamos la ventana de tiempo (history_len) y los canales
        self.fc = nn.Linear(64 * history_len, num_targets)

    def forward(self, x):
        batch_size, tiempo, nodos, caract = x.shape
        
        # Pasamos por los bloques STGCN
        x = self.block1(x, self.adj_matrix)
        x = self.block2(x, self.adj_matrix)
        
        x = x.reshape(batch_size, nodos, -1) 
        
        # Predicción final: (Batch, Nodos, num_targets)
        out = self.fc(x)
        return out

# Instanciar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A_tensor es la matriz normalizada que creamos en los primeros pasos
modelo = STGCN_Metro(
    num_nodes=len(node_to_idx),
    num_features=len(variables_entrada),
    num_targets=len(variables_objetivo),
    history_len=history_len,
    adj_matrix=A_tensor
).to(device)

print(f"Modelo instanciado en: {device}")

import torch.nn.functional as F

#Ejecución del entrenamiento
import torch.optim as optim
import time


# 1. Configuración del entrenamiento
epocas = 50 
tasa_aprendizaje = 0.001
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

wandb.init(project="pd1-c2526-team5", name="test-stgcn-1", mode="offline")

print("Iniciando el entrenamiento de STGCN...")

for epoca in range(epocas):
    inicio_epoca = time.time()
    
    # FASE DE ENTRENAMIENTO
    modelo.train()
    loss_entrenamiento_total = 0.0
    
    for lotes_x, lotes_y in train_loader:
        # Movemos los datos a la tarjeta gráfica (o CPU)
        lotes_x = lotes_x.to(device)
        lotes_y = lotes_y.to(device)
        
        # Reseteamos los gradientes
        optimizador.zero_grad()
        
        # Propagación hacia adelante (Predicción)
        predicciones = modelo(lotes_x)
        
        # Calcular el error
        loss = criterio(predicciones, lotes_y)
        loss_entrenamiento_total += loss.item() * lotes_x.size(0)
        
        # Propagación hacia atrás y ajuste de pesos
        loss.backward()
        optimizador.step()
        
    loss_medio = loss_entrenamiento_total / len(train_dataset)
    tiempo_epoca = time.time() - inicio_epoca

    wandb.log({"loss_medio": loss_medio, "tiempo_epoca": tiempo_epoca}, step=epoca)
    
    print(f"Época [{epoca+1}/{epocas}] | Loss (MSE Escala): {loss_medio:.4f} | Tiempo: {tiempo_epoca:.1f}s")






# FASE DE EVALUACIÓN Y CÁLCULO DE MÉTRICAS REALES
print("\nExtrayendo predicciones del conjunto de Test...")
modelo.eval()

lista_predicciones = []
lista_reales = []

# Desactivamos el cálculo de gradientes para ahorrar mucha memoria y tiempo
with torch.no_grad():
    for lotes_x, lotes_y in test_loader:
        lotes_x = lotes_x.to(device)
        
        preds = modelo(lotes_x)
        
        # Nos llevamos los resultados de vuelta a la memoria RAM normal (CPU) como arrays de NumPy
        lista_predicciones.append(preds.cpu().numpy())
        lista_reales.append(lotes_y.numpy())

# Unimos todos los lotes en un solo bloque 3D: (Tiempos_Test, Nodos, Targets)
preds_test_3d = np.concatenate(lista_predicciones, axis=0)
reales_test_3d = np.concatenate(lista_reales, axis=0)

# Extraemos las dimensiones para hacer el re-moldeado (reshape)
T_test, N_test, num_objetivos = preds_test_3d.shape

# Para invertir el escalado, scikit-learn necesita que los datos sean 2D (Filas, Columnas)
preds_planas = preds_test_3d.reshape(-1, num_objetivos)
reales_planas = reales_test_3d.reshape(-1, num_objetivos)

# INVERSIÓN DEL ESCALADO: Devolvemos los datos a segundos
preds_segundos = scaler_Y.inverse_transform(preds_planas).reshape(T_test, N_test, num_objetivos)
reales_segundos = scaler_Y.inverse_transform(reales_planas).reshape(T_test, N_test, num_objetivos)

print("\n" + "="*45)
print("RESULTADOS MAE EN SEGUNDOS REALES (TEST)")
print("="*45)

# Calculamos el Error Absoluto Medio (MAE) por cada horizonte temporal (10m, 20m, 30m)
for i, objetivo in enumerate(variables_objetivo):
    # Seleccionamos solo los datos correspondientes a ese objetivo específico
    pred_horizonte = preds_segundos[:, :, i]
    real_horizonte = reales_segundos[:, :, i]
    
    # MAE = Media del valor absoluto de la diferencia
    mae_segundos = np.mean(np.abs(pred_horizonte - real_horizonte))
    
    print(f"➜ Horizonte [{objetivo}]: Error promedio de {mae_segundos:.2f} segundos")
    wandb.log({f"MAE_{objetivo}": mae_segundos})


wandb.finish()

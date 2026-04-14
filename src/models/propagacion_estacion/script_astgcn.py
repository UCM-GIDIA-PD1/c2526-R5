import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import gc
#import torch.nn.functional as F
import torch.nn.functional as F_torch

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

# Descargar dataset
access_key = os.getenv("MINIO_ACCESS_KEY")
secret_key = os.getenv("MINIO_SECRET_KEY")

ruta_archivo = "grupo5/final/year=2025/month=01/dataset_final.parquet"
df_final = download_df_parquet(access_key, secret_key, ruta_archivo)

# Descargamos 1 mes
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"

dates = pd.date_range(start=START_DATE, end=END_DATE).strftime("%Y-%m-%d").tolist()
dfs = []
for date in dates:
    try:
        df_gtfs = download_df_parquet(
            access_key,
            secret_key,
            f"grupo5/cleaned/gtfs_clean_scheduled/date={date}/gtfs_scheduled_{date}.parquet",
        )
    except:
        print(f"Could not download data for date: {date}")
        continue
    dfs.append(df_gtfs)
df = pd.concat(dfs, ignore_index=True)

# MATRIZ DE ADYACENCIA
df = df.sort_values(by=["trip_uid", "scheduled_seconds"]).reset_index(drop=True)

df["next_stop_id"] = df.groupby("trip_uid")["stop_id"].shift(-1)
df["next_scheduled_seconds"] = df.groupby("trip_uid")["scheduled_seconds"].shift(-1)

edges_df = df.dropna(subset=["next_stop_id"]).copy()
edges_df["travel_time"] = edges_df["next_scheduled_seconds"] - edges_df["scheduled_seconds"]
edges_df = edges_df[edges_df["travel_time"] > 0]

graph_df = edges_df.groupby(["stop_id", "next_stop_id"]).agg(
    median_travel_time=("travel_time", "median"),
    trip_count=("trip_uid", "count"),
).reset_index()

graph_df = graph_df[graph_df["trip_count"] > 5]

nodes = sorted(list(set(df["stop_id"].unique()) | set(df["next_stop_id"].dropna().unique())))
n_nodes = len(nodes)
node_to_idx = {stop_id: idx for idx, stop_id in enumerate(nodes)}

A_weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
sigma = graph_df["median_travel_time"].std()

if sigma == 0 or np.isnan(sigma):
    sigma = 1.0

for _, row in graph_df.iterrows():
    i = node_to_idx[row["stop_id"]]
    j = node_to_idx[row["next_stop_id"]]
    dist = row["median_travel_time"]

    peso = np.exp(- (dist ** 2) / (sigma ** 2))

    A_weighted[i, j] = peso
    A_weighted[j, i] = peso  # Forzamos conexión bidireccional

# Matriz normalizada, útil para comparaciones y depuración
A_norm = A_weighted.copy()
np.fill_diagonal(A_norm, 1.0)

grados = np.sum(A_norm, axis=1)
grados_inv_raiz = np.power(grados, -0.5, where=(grados != 0))
grados_inv_raiz[np.isinf(grados_inv_raiz)] = 0.0

matriz_diagonal = np.diag(grados_inv_raiz)
A_norm = matriz_diagonal @ A_norm @ matriz_diagonal

A_tensor = torch.tensor(A_norm, dtype=torch.float32)

print(f"Número de nodos únicos: {n_nodes}")
print(f"Matriz de adyacencia normalizada lista. Forma: {A_tensor.shape}")

# VARIABLES 
variables_entrada = [
    "delay_seconds",
    "lagged_delay_1",
    "lagged_delay_2",
    "is_unscheduled",
    "temp_extreme",
    "n_eventos_afectando",
    "route_rolling_delay",
    "actual_headway_seconds",
    "hour_sin",
    "hour_cos",
    "dow",
]

variables_objetivo = [
    "target_delay_10m",
    "target_delay_20m",
    "target_delay_30m",
    "station_delay_10m",
    "station_delay_20m",
    "station_delay_30m",
]

# TENSORES
def crear_tensores_astgcn(df, mapa_nodos, features, targets, freq="15min"):
    nodos_validos = list(mapa_nodos.keys())
    df = df[df["stop_id"].isin(nodos_validos)].copy()

    df["time_bin"] = pd.to_datetime(df["merge_time"]).dt.floor(freq)

    reglas_agregacion = {
        "delay_seconds": "mean",
        "lagged_delay_1": "mean",
        "lagged_delay_2": "mean",
        "is_unscheduled": "sum",
        "temp_extreme": "max",
        "n_eventos_afectando": "max",
        "route_rolling_delay": "mean",
        "actual_headway_seconds": "mean",
        "target_delay_10m": "mean",
        "target_delay_20m": "mean",
        "target_delay_30m": "mean",
        "station_delay_10m": "mean",
        "station_delay_20m": "mean",
        "station_delay_30m": "mean",
    }

    print("Agrupando datos por ventanas de tiempo y estación...")
    df_agrupado = df.groupby(["time_bin", "stop_id"]).agg(reglas_agregacion)

    del df
    gc.collect()

    todos_los_tiempos = pd.date_range(
        start=df_agrupado.index.get_level_values("time_bin").min(),
        end=df_agrupado.index.get_level_values("time_bin").max(),
        freq=freq,
    )

    indice_completo = pd.MultiIndex.from_product(
        [todos_los_tiempos, nodos_validos],
        names=["time_bin", "stop_id"],
    )
    df_completo = df_agrupado.reindex(indice_completo).reset_index()

    del df_agrupado
    gc.collect()

    print("Imputando valores faltantes y reconstruyendo tensores...")

    cols_retrasos = [
        "delay_seconds",
        "lagged_delay_1",
        "lagged_delay_2",
        "is_unscheduled",
        "route_rolling_delay",
        "actual_headway_seconds",
    ]
    df_completo[cols_retrasos] = df_completo[cols_retrasos].fillna(0)

    cols_contexto = ["temp_extreme", "n_eventos_afectando"]
    df_completo[cols_contexto] = df_completo.groupby("stop_id")[cols_contexto].ffill().bfill().fillna(0)

    df_completo["hour_sin"] = np.sin(2 * np.pi * df_completo["time_bin"].dt.hour / 24)
    df_completo["hour_cos"] = np.cos(2 * np.pi * df_completo["time_bin"].dt.hour / 24)
    df_completo["dow"] = df_completo["time_bin"].dt.dayofweek.astype(float)

    df_completo["nodo_idx"] = df_completo["stop_id"].map(mapa_nodos)
    df_completo = df_completo.sort_values(["time_bin", "nodo_idx"])

    T = len(todos_los_tiempos)
    N = len(nodos_validos)
    F_in = len(features)
    C_out = len(targets)

    X_tensor = df_completo[features].values.reshape(T, N, F_in)
    Y_tensor = df_completo[targets].values.reshape(T, N, C_out)

    del df_completo
    gc.collect()

    return X_tensor, Y_tensor, todos_los_tiempos


X_full, Y_full, array_tiempos = crear_tensores_astgcn(
    df_final,
    node_to_idx,
    variables_entrada,
    variables_objetivo,
)

print(f"\nTensor de entrada X_full creado: {X_full.shape} -> (Tiempos, Nodos, Features)")
print(f"Tensor objetivo Y_full creado: {Y_full.shape} -> (Tiempos, Nodos, Targets)")

X_full = np.nan_to_num(X_full, nan=0.0)
Y_full = np.nan_to_num(Y_full, nan=0.0)

print(f"¿Hay NaNs en X_full?: {np.isnan(X_full).any()}")
print(f"¿Hay NaNs en Y_full?: {np.isnan(Y_full).any()}")

# TRAIN / TEST
num_tiempos = X_full.shape[0]
limite_corte = int(num_tiempos * 0.8)

X_train = X_full[:limite_corte]
X_test = X_full[limite_corte:]
Y_train = Y_full[:limite_corte]
Y_test = Y_full[limite_corte:]

# Separar una parte del train para validación temporal
limite_val = int(X_train.shape[0] * 0.8)

X_train_final = X_train[:limite_val]
X_val = X_train[limite_val:]

Y_train_final = Y_train[:limite_val]
Y_val = Y_train[limite_val:]

print(f"Secuencias train final: {X_train_final.shape[0]}")
print(f"Secuencias validación: {X_val.shape[0]}")
print(f"Secuencias test: {X_test.shape[0]}")

print(f"Secuencias de entrenamiento: {X_train.shape[0]}")
print(f"Secuencias de test: {X_test.shape[0]}")

T_train, N, F = X_train_final.shape
T_val = X_val.shape[0]
T_test = X_test.shape[0]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_final.reshape(-1, F)).reshape(T_train, N, F)
X_val_scaled = scaler_X.transform(X_val.reshape(-1, F)).reshape(T_val, N, F)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, F)).reshape(T_test, N, F)

C = Y_train_final.shape[2]
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train_final.reshape(-1, C)).reshape(T_train, N, C)
Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, C)).reshape(T_val, N, C)
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, C)).reshape(T_test, N, C)

print("Datos escalados correctamente.")

print("STD real de cada target:")
for nombre, escala in zip(variables_objetivo, scaler_Y.scale_):
    print(f"{nombre}: {escala:.2f} s")


# DATASET
class DatasetASTGCN(Dataset):
    def __init__(self, X, Y, history_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.history_len = history_len

    def __len__(self):
        return len(self.X) - self.history_len

    def __getitem__(self, idx):
        ventana_x = self.X[idx : idx + self.history_len]
        objetivo_y = self.Y[idx + self.history_len]
        return ventana_x, objetivo_y


history_len = 12 # antes 8
batch_size = 16 # antes 32


train_dataset = DatasetASTGCN(X_train_scaled, Y_train_scaled, history_len)
val_dataset = DatasetASTGCN(X_val_scaled, Y_val_scaled, history_len)
test_dataset = DatasetASTGCN(X_test_scaled, Y_test_scaled, history_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Lotes de entrenamiento (batches): {len(train_loader)}")

# HELPERS PARA ASTGCN
def calcular_scaled_laplacian(adj_matrix):
    adj = adj_matrix.astype(np.float32).copy()
    np.fill_diagonal(adj, 0.0)

    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj

    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    laplacian_norm = d_mat_inv_sqrt @ laplacian @ d_mat_inv_sqrt

    lambda_max = np.linalg.eigvals(laplacian_norm).real.max()
    if lambda_max == 0 or np.isnan(lambda_max):
        lambda_max = 1.0

    scaled_laplacian = (2.0 / lambda_max) * laplacian_norm - np.eye(adj.shape[0], dtype=np.float32)
    return scaled_laplacian.astype(np.float32)


def calcular_polinomios_chebyshev(scaled_laplacian, K):
    N = scaled_laplacian.shape[0]
    cheb_polynomials = [np.eye(N, dtype=np.float32)]

    if K > 1:
        cheb_polynomials.append(scaled_laplacian)

    for k in range(2, K):
        cheb_polynomials.append(
            2 * scaled_laplacian @ cheb_polynomials[k - 1] - cheb_polynomials[k - 2]
        )

    return [torch.tensor(p, dtype=torch.float32) for p in cheb_polynomials]


# BLOQUES ASTGCN
class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, history_len):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj = nn.Linear(in_channels, 1, bias=False)
        self.scale = np.sqrt(max(num_nodes, 1))

    def forward(self, x):
        # x: (B, T, N, F)
        q = self.query_proj(x).squeeze(-1)           # (B, T, N)
        k = self.key_proj(x).squeeze(-1)             # (B, T, N)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # (B, T, T)
        attention = torch.softmax(scores, dim=-1)
        return attention


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, history_len):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj = nn.Linear(in_channels, 1, bias=False)
        self.scale = np.sqrt(max(history_len, 1))

    def forward(self, x):
        # x: (B, T, N, F)
        q = self.query_proj(x).squeeze(-1).transpose(1, 2)  # (B, N, T)
        k = self.key_proj(x).squeeze(-1).transpose(1, 2)    # (B, N, T)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # (B, N, N)
        attention = torch.softmax(scores, dim=-1)
        return attention


class ChebConvWithSpatialAttention(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)]
        )
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

        self.register_buffer(
            "cheb_polynomials",
            torch.stack(cheb_polynomials, dim=0),  # (K, N, N)
        )

    def forward(self, x, spatial_attention):
        # x: (B, T, N, F_in)
        B, T, N, _ = x.shape
        outputs = []

        for t in range(T):
            graph_signal = x[:, t, :, :]  # (B, N, F_in)
            output_t = torch.zeros(
                (B, N, self.out_channels),
                device=x.device,
                dtype=x.dtype,
            )

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]                 # (N, N)
                T_k_at = T_k.unsqueeze(0) * spatial_attention  # (B, N, N)
                rhs = torch.einsum("bij,bjf->bif", T_k_at, graph_signal)
                output_t = output_t + torch.matmul(rhs, self.Theta[k])

            outputs.append(output_t.unsqueeze(1))

        return F_torch.relu(torch.cat(outputs, dim=1))  # (B, T, N, out_channels)


class ASTGCNBlock(nn.Module):
    def __init__(self, in_channels, K, cheb_polynomials, num_nodes, history_len, out_channels, temporal_kernel=3):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, num_nodes, history_len)
        self.spatial_attention = SpatialAttention(in_channels, num_nodes, history_len)
        self.cheb_conv = ChebConvWithSpatialAttention(K, cheb_polynomials, in_channels, out_channels)

        self.time_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
        )

        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )

        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (B, T, N, F)
        temporal_attention = self.temporal_attention(x)  # (B, T, T)
        x_ta = torch.einsum("bts,bsnf->btnf", temporal_attention, x)

        spatial_attention = self.spatial_attention(x_ta)  # (B, N, N)
        x_gc = self.cheb_conv(x_ta, spatial_attention)    # (B, T, N, C)

        x_gc_perm = x_gc.permute(0, 3, 1, 2)              # (B, C, T, N)
        x_tc = self.time_conv(x_gc_perm)                  # (B, C, T, N)

        residual = self.residual_conv(x.permute(0, 3, 1, 2))
        x_out = F_torch.relu(x_tc + residual)                   # (B, C, T, N)

        x_out = x_out.permute(0, 2, 3, 1)                 # (B, T, N, C)
        x_out = self.layer_norm(x_out)
        return x_out


class ASTGCN_Metro(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        num_targets,
        history_len,
        cheb_polynomials,
        K=3,
        hidden_channels=32,
    ):
        super().__init__()

        self.block1 = ASTGCNBlock(
            in_channels=num_features,
            K=K,
            cheb_polynomials=cheb_polynomials,
            num_nodes=num_nodes,
            history_len=history_len,
            out_channels=hidden_channels,
        )

        self.block2 = ASTGCNBlock(
            in_channels=hidden_channels,
            K=K,
            cheb_polynomials=cheb_polynomials,
            num_nodes=num_nodes,
            history_len=history_len,
            out_channels=hidden_channels,
        )

        self.final_conv = nn.Conv2d(
            in_channels=history_len,
            out_channels=1,
            kernel_size=(1, hidden_channels),
        )


        self.dropout = nn.Dropout(p=0.2) #nuevo

        self.fc = nn.Linear(num_nodes, num_nodes * num_targets)
        self.num_nodes = num_nodes
        self.num_targets = num_targets

    def forward(self, x):
        # x: (B, T, N, F)
        x = self.block1(x)
        x = self.dropout(x) #nuevo

        x = self.block2(x)
        x = self.dropout(x) #nuevo

        # (B, T, N, C) -> usar el tiempo como canales, igual que en implementaciones ASTGCN
        x = x.permute(0, 1, 2, 3)   # (B, T, N, C)
        x = self.final_conv(x)      # (B, 1, N, 1)
        x = x.squeeze(-1).squeeze(1)  # (B, N)

        x = self.fc(x)              # (B, N * targets)
        x = x.view(-1, self.num_nodes, self.num_targets)
        return x


# INSTANCIAR MODELO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_cheb = 3
scaled_laplacian = calcular_scaled_laplacian(A_weighted)
cheb_polynomials = calcular_polinomios_chebyshev(scaled_laplacian, K_cheb)

modelo = ASTGCN_Metro(
    num_nodes=len(node_to_idx),
    num_features=len(variables_entrada),
    num_targets=len(variables_objetivo),
    history_len=history_len,
    cheb_polynomials=cheb_polynomials,
    K=K_cheb,
    hidden_channels=32,
).to(device)

print(f"Modelo ASTGCN instanciado en: {device}")

# ENTRENAMIENTO
import torch.optim as optim
import time

epocas = 50
tasa_aprendizaje = 0.0005 #antes 0.001
criterio = nn.SmoothL1Loss(beta=0.5) #antes: nn.MSELoss() 
optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

wandb.init(project="pd1-c2526-team5", name="test-astgcn-1", mode="offline")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizador, mode='min', factor=0.5, patience=5
)

mejor_val_loss = float('inf')

print("Iniciando entrenamiento...")

for epoca in range(epocas):
    inicio_epoca = time.time()

    # TRAIN
    modelo.train()
    loss_entrenamiento_total = 0.0
    mae_entrenamiento_total = 0.0
    mse_entrenamiento_total = 0.0

    for lotes_x, lotes_y in train_loader:
        lotes_x = lotes_x.to(device)
        lotes_y = lotes_y.to(device)

        optimizador.zero_grad()
        predicciones = modelo(lotes_x)

        loss = criterio(predicciones, lotes_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

        optimizador.step()

        loss_entrenamiento_total += loss.item() * lotes_x.size(0)
        mae_batch = torch.mean(torch.abs(predicciones - lotes_y)).item()
        mse_batch = torch.mean((predicciones - lotes_y) ** 2).item()

        mae_entrenamiento_total += mae_batch * lotes_x.size(0)
        mse_entrenamiento_total += mse_batch * lotes_x.size(0)

    train_loss = loss_entrenamiento_total / len(train_dataset)
    train_mae = mae_entrenamiento_total / len(train_dataset)
    train_rmse = np.sqrt(mse_entrenamiento_total / len(train_dataset))

    # VALIDATION
    modelo.eval()
    loss_validacion_total = 0.0
    mae_validacion_total = 0.0
    mse_validacion_total = 0.0

    with torch.no_grad():
        for lotes_x, lotes_y in val_loader:
            lotes_x = lotes_x.to(device)
            lotes_y = lotes_y.to(device)

            predicciones = modelo(lotes_x)

            loss = criterio(predicciones, lotes_y)

            loss_validacion_total += loss.item() * lotes_x.size(0)

            mae_batch = torch.mean(torch.abs(predicciones - lotes_y)).item()
            mse_batch = torch.mean((predicciones - lotes_y) ** 2).item()

            mae_validacion_total += mae_batch * lotes_x.size(0)
            mse_validacion_total += mse_batch * lotes_x.size(0)

    val_loss = loss_validacion_total / len(val_dataset)
    val_mae = mae_validacion_total / len(val_dataset)
    val_rmse = np.sqrt(mse_validacion_total / len(val_dataset))

    scheduler.step(val_loss)

    tiempo_epoca = time.time() - inicio_epoca
    lr_actual = optimizador.param_groups[0]["lr"]

    wandb.log({
        "train_loss": train_loss,
        "train_mae_scaled": train_mae,
        "train_rmse_scaled": train_rmse,
        "val_loss": val_loss,
        "val_mae_scaled": val_mae,
        "val_rmse_scaled": val_rmse,
        "learning_rate": lr_actual,
        "tiempo_epoca": tiempo_epoca
    }, step=epoca)

    print(
        f"Época [{epoca+1}/{epocas}] | "
        f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
        f"LR: {lr_actual:.6f} | Tiempo: {tiempo_epoca:.1f}s"
    )

    if val_loss < mejor_val_loss:
        mejor_val_loss = val_loss
        torch.save(modelo.state_dict(), "mejor_modelo_astgcn.pt")

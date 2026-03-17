import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
import os
import wandb
from lightgbm import LGBMClassifier
from wandb.integration.lightgbm import wandb_callback
from sklearn.metrics import classification_report, accuracy_score

from src.common.minio_client import download_df_parquet



def procesar(df):

    df['hora'] = df['merge_time'].dt.hour
    df['minuto'] = df['merge_time'].dt.minute
    df['dia_semana'] = df['merge_time'].dt.dayofweek # Lunes=0, Domingo=6

    df['hora_mean'] = df['merge_time_mean'].dt.hour 

    columnas_a_categoria = ['stop_id', 'route_id', 'direction']

    for col in columnas_a_categoria:
        df[col] = df[col].astype('category')

    df = df.drop(columns=['merge_time', 'merge_time_mean'])

    return df




# 1. Cargar los datos

INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'
    
ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

print('Cargando datos...')
df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
print('Todo cargado correctamente')

WANDB_PROJECT  = "pd1-c2526-team5"

wandb.init(
    project= WANDB_PROJECT, 
    name="hist-gradient-boosting-60min", 
    notes="Primer modelo por intervalos con HistGradientBoosting"
)

# Definimos los hiperparámetros en W&B para trackearlos
config = wandb.config
config.max_iter = 100
config.learning_rate = 0.1
config.random_state = 42
config.class_weight = 'balanced'


# 2. Definir los intervalos de retraso (Binning)
# Asumimos que la variable objetivo está en segundos.
# Intervalos: < 60s (Sin retraso), 60-120s (Leve), 120-300s (Moderado), > 300s (Grave)
bins = [-np.inf, -60, 60, 180, 300, 450, np.inf]
labels = [
    'Adelantado (>1 min)', 
    'Puntual (-1 a 1 min)', 
    'Retraso leve (1-3 min)', 
    'Retraso moderado (3-5 min)', 
    'Retraso grave (5-7.5 min)',
    'Retraso muy grave (>7.5 min)'
]

# Creamos la nueva columna objetivo categórica
columna_objetivo = 'target_delay_10m_max'
df['clase_retraso'] = pd.cut(df[columna_objetivo], bins=bins, labels=labels)


df_procesado = procesar(df)

columnas_a_excluir = [
    # 1. TARGETS ALTERNATIVOS (Y el tuyo propio, que pasará a ser 'y')
    'target_delay_10m_mean', 'target_delay_10m_max',
    'target_delay_20m_mean', 'target_delay_20m_max',
    'target_delay_30m_mean', 'target_delay_30m_max',
    'target_delay_45m_mean', 'target_delay_45m_max',
    'target_delay_60m_mean', 'target_delay_60m_max',
    'target_delay_end_mean', 'target_delay_end_max',

    # 2. DELTAS (Diferencia matemática exacta hacia el futuro)
    'delta_delay_10m_mean', 'delta_delay_10m_max',
    'delta_delay_20m_mean', 'delta_delay_20m_max',
    'delta_delay_30m_mean', 'delta_delay_30m_max',
    'delta_delay_45m_mean', 'delta_delay_45m_max',
    'delta_delay_60m_mean', 'delta_delay_60m_max',
    'delta_delay_end_mean', 'delta_delay_end_max',

    # 3. ESTADO FUTURO DE LAS ESTACIONES
    'station_delay_10m_mean', 'station_delay_10m_max',
    'station_delay_20m_mean', 'station_delay_20m_max',
    'station_delay_30m_mean', 'station_delay_30m_max',

    # 4. ALERTAS Y EVENTOS DEL FUTURO
    'alert_in_next_15m_max',
    'alert_in_next_30m_max',
    'seconds_to_next_alert_mean',
    'afecta_despues_max',

    # 5. RUIDO O IDENTIFICADORES INTERNOS (No predictivos)
    'match_key_nunique'
]

cols_a_excluir = columnas_a_excluir + ['merge_time', 'stop_id', 'route_id', 'clase_retraso']
# Usamos un if para evitar errores si alguna columna ya no existe
X = df.drop(columns=[c for c in cols_a_excluir if c in df.columns])
y = df['clase_retraso']

X = pd.get_dummies(X, drop_first=True)

columnas_fecha = X.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns
X = X.drop(columns=columnas_fecha)

# Manejo básico de nulos (puedes ajustar esto según tu conocimiento de los datos)
X = X.fillna(0)
y = y.dropna() 
X = X.loc[y.index] # Asegurar que X e y tengan las mismas filas

# 5. Dividir los datos en Entrenamiento y Prueba
# IMPORTANTE: shuffle=False para datos temporales (entrenamos con el pasado, probamos con el futuro)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Entrenar el Modelo
print("Entrenando el modelo...")

# ==========================================
# 2. ENTRENAR CON LOS PARÁMETROS DE W&B
# ==========================================
modelo = LGBMClassifier(
    n_estimators=config.max_iter,
    learning_rate=config.learning_rate,
    random_state=config.random_state,
    class_weight=config.class_weight
)

modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # Para ver el progreso en test
    callbacks=[wandb_callback()] # ¡Magia en vivo!
)

# ==========================================
# 3. PREDICCIONES Y LOG EN W&B
# ==========================================
print("Generando predicciones y enviando datos a W&B...")
y_pred = modelo.predict(X_test)
y_probas = modelo.predict_proba(X_test) # Necesario para las curvas ROC de W&B

# Loguear métricas simples
acc = accuracy_score(y_test, y_pred)
wandb.log({"accuracy": acc})

# Imprimir reporte en la consola local
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

# LA MAGIA DE W&B: Crear dashboard automático de Scikit-Learn
# Esto generará: Matriz de confusión, Feature Importance, Curvas ROC y Precision-Recall
wandb.sklearn.plot_classifier(
    modelo, 
    X_train, X_test, 
    y_train, y_test, 
    y_pred, y_probas, 
    labels=labels,
    model_name="HistGradientBoosting_Retrasos",
    feature_names=X.columns.tolist()
)

# Finalizar la ejecución en W&B
wandb.finish()
print("¡Entrenamiento finalizado y trackeado con éxito!")
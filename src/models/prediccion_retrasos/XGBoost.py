"""
Entrenamiento XGBOOST — Predicción de retraso en parada

Predice el tiempo de retraso absoluto (en segundos) de un tren al llegar a una estación.
    Objetivo: retraso_segundos (Variable continua)

Validación y Optimización:
    Train  → 80% de los datos históricos (aleatorio)
    Val    → 20% de los datos históricos (eval_set / early stopping)
    Aceleración: Uso de `tree_method='hist'` y `enable_categorical=True` para procesar millones de filas de forma eficiente.

Uso:
    python -m src.models.prediccion_retrasos.train_xgb_retrasos

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import os
from src.common.minio_client import download_df_parquet



OBJETIVO = 'target_delay_30m_max'

columnas_a_excluir = [
    # 1. TARGETS ALTERNATIVOS (Y el tuyo propio, que pasará a ser 'y')
    'target_delay_10m_mean', 'target_delay_10m_max',
    'target_delay_20m_mean', 'target_delay_20m_max',
    'target_delay_30m_mean', 'target_delay_30m_max', # <-- Tu objetivo probable
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




def XGBoost():
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'
    
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos...')
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
    print('Todo cargado correctamente')
    df_procesado = procesar(df)
    

    # 2. Separar las variables predictoras (X) de lo que queremos predecir (y)
    X = df_procesado.drop(columns=columnas_a_excluir + [OBJETIVO], errors='ignore') # Todo excepto el retraso
    y = df_procesado[OBJETIVO]              # Solo el retraso

    # 3. Dividir los datos en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Configurar el Modelo XGBoost
    modelo_xgb = xgb.XGBRegressor(
        n_estimators=1000,       # Número de árboles que va a construir (prueba subirlo a 500 luego)
        learning_rate=0.05,      # Qué tan rápido aprende (si subes los árboles, baja esto a 0.05 o 0.01)
        max_depth=7,            # Nivel de detalle de cada árbol (entre 3 y 10 suele ir bien)
        n_jobs=-1,              # Usa todos los núcleos de tu procesador para que vaya más rápido
        random_state=42,
        tree_method='hist',
        enable_categorical = True,
        early_stopping_rounds=20
    )

    # 5. Entrenar el Modelo
    print("Entrenando el modelo...")
    modelo_xgb.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=10)

    # 6. Predecir y Evaluar
    predicciones = modelo_xgb.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)

    print(f"Error Absoluto Medio (MAE): El modelo se equivoca en promedio por {mae:.2f} segundos.")


if __name__ == "__main__":
    XGBoost()
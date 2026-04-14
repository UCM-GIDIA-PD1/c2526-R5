"""
Entrenamiento Logistic Regression — Predicción de retraso por intervalos (Optimizado)

Predice el tiempo de retraso absoluto (en segundos) de un tren al llegar a una estación.
    Objetivo: 'clase_retraso' :
        'Adelantado (>1 min)', 
        'Puntual (-1 a 1 min)', 
        'Retraso leve (1-3 min)', 
        'Retraso moderado (3-5 min)', 
        'Retraso grave (5-7.5 min)',
        'Retraso muy grave (>7.5 min)'

Validación y Optimización:
    Train_full → 85% de los datos históricos (chronological)
    Test       → 15% de los datos históricos (chronological)
   
Uso:
    uv run python src/models/prediccion_retrasos/prediccion_por_intervalos/train/train_logistica_intervalos.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import os
import wandb

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

def main():
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'

    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos...')
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
    print('Todo cargado correctamente')

    WANDB_PROJECT  = "pd1-c2526-team5"

    wandb.init(
    project=WANDB_PROJECT, 
    group="modelos-retraso-clasificacion",
    name="logreg-60min-target_10m_max-optimizado", 
    notes="Modelo Regresión Logística entrenado con hiperparámetros optimizados para target_delay_10m_max"
    )


    config = wandb.config
    config.C = 2.764501216773756           
    config.penalty = 'l2'         
    config.solver = 'saga'        
    config.max_iter = 500         
    config.class_weight = None    
    config.l1_ratio = None        
    config.random_state = 42      

    bins = [-np.inf, -60, 60, 180, 300, 450, np.inf]
    labels = [
    'Adelantado (>1 min)', 
    'Puntual (-1 a 1 min)', 
    'Retraso leve (1-3 min)', 
    'Retraso moderado (3-5 min)', 
    'Retraso grave (5-7.5 min)',
    'Retraso muy grave (>7.5 min)'
    ]

    columna_objetivo = 'target_delay_10m_max'
    df['clase_retraso'] = pd.cut(df[columna_objetivo], bins=bins, labels=labels)

    df_procesado = procesar(df)

    columnas_a_excluir = [
    # 1. TARGETS ALTERNATIVOS 
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

    X = X.fillna(0)
    y = y.dropna() 
    X = X.loc[y.index] 

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, shuffle=False)

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    print("Escalando datos...")
    scaler = StandardScaler()
    X_train_full_scaled = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X_train_full.columns, index=X_train_full.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("Entrenando el modelo...")

    logreg_params = {
    'C': config.C,
    'penalty': config.penalty,
    'solver': config.solver,
    'max_iter': config.max_iter,
    'class_weight': config.class_weight,
    'random_state': config.random_state,
    'n_jobs': -1
    }
    if config.penalty == 'elasticnet':
        logreg_params['l1_ratio'] = config.l1_ratio

    # 1. ENTRENAR CON LOS PARÁMETROS ÓPTIMOS
    modelo = LogisticRegression(**logreg_params)

    modelo.fit(X_train_full_scaled, y_train_full)

    # 2. PREDICCIONES Y LOG EN W&B
    print("Generando predicciones y enviando datos a W&B...")
    y_pred = modelo.predict(X_test_scaled)
    y_probas = modelo.predict_proba(X_test_scaled) 

    # Calcula métricas explícitamente usando el promedio 'macro'
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    wandb.log({
    "accuracy": acc,
    "f1_macro": f1_macro,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro
    })

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))

    wandb.sklearn.plot_classifier(
    modelo, 
    X_train_full_scaled, X_test_scaled, 
    y_train_full, y_test, 
    y_pred, y_probas, 
    labels=labels,
    model_name="LogReg_Retrasos_Optimizado",
    feature_names=X.columns.tolist()
    )

    wandb.finish()
    print("¡Entrenamiento finalizado y trackeado con éxito!")

if __name__ == "__main__":
    main()
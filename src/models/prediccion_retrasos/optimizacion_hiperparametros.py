"""
Optimización de Hiperparámetros LGBM — Predicción de retraso por intervalos

Uso:
    python -m src.models.prediccion_retrasos.optimizacion_lgbm

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import pandas as pd
import numpy as np
import os
import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

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

def cargar_y_preparar_datos():
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos desde MinIO...')
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
    
    # 2. Definir intervalos
    bins = [-np.inf, -60, 60, 180, 300, 450, np.inf]
    labels = [
        'Adelantado (>1 min)', 'Puntual (-1 a 1 min)', 
        'Retraso leve (1-3 min)', 'Retraso moderado (3-5 min)', 
        'Retraso grave (5-7.5 min)', 'Retraso muy grave (>7.5 min)'
    ]

    columna_objetivo = 'target_delay_10m_max'
    df['clase_retraso'] = pd.cut(df[columna_objetivo], bins=bins, labels=labels)

    df_procesado = procesar(df)

    columnas_a_excluir = [
        'target_delay_10m_mean', 'target_delay_10m_max', 'target_delay_20m_mean', 'target_delay_20m_max',
        'target_delay_30m_mean', 'target_delay_30m_max', 'target_delay_45m_mean', 'target_delay_45m_max',
        'target_delay_60m_mean', 'target_delay_60m_max', 'target_delay_end_mean', 'target_delay_end_max',
        'delta_delay_10m_mean', 'delta_delay_10m_max', 'delta_delay_20m_mean', 'delta_delay_20m_max',
        'delta_delay_30m_mean', 'delta_delay_30m_max', 'delta_delay_45m_mean', 'delta_delay_45m_max',
        'delta_delay_60m_mean', 'delta_delay_60m_max', 'delta_delay_end_mean', 'delta_delay_end_max',
        'station_delay_10m_mean', 'station_delay_10m_max', 'station_delay_20m_mean', 'station_delay_20m_max',
        'station_delay_30m_mean', 'station_delay_30m_max', 'alert_in_next_15m_max', 'alert_in_next_30m_max',
        'seconds_to_next_alert_mean', 'afecta_despues_max', 'match_key_nunique'
    ]

    cols_a_excluir = columnas_a_excluir + ['stop_id', 'route_id', 'clase_retraso']
    X = df_procesado.drop(columns=[c for c in cols_a_excluir if c in df_procesado.columns])
    y = df_procesado['clase_retraso']

    X = pd.get_dummies(X, drop_first=True)
    columnas_fecha = X.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns
    X = X.drop(columns=columnas_fecha)

    X = X.fillna(0)
    y = y.dropna() 
    X = X.loc[y.index] 

    X = X.fillna(0)
    y = y.dropna() 
    X = X.loc[y.index] 

    # Para que Optuna sea rápido, usamos solo el último millón de registros 
    # (como están ordenados temporalmente, es el pasado más reciente)
    if len(X) > 1000000:
        X = X.tail(1000000)
        y = y.tail(1000000)

    return train_test_split(X, y, test_size=0.2, shuffle=False)

    # IMPORTANTE: shuffle=False para datos temporales
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def objective(trial, X_train, X_test, y_train, y_test):
    # 1. Definir el espacio de búsqueda (Search Space)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # 2. Entrenar el modelo
    modelo = LGBMClassifier(**param)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    # 3. Evaluar
    y_pred = modelo.predict(X_test)
    
    # Usamos f1_score macro porque nos importan todas las clases por igual (incluso los retrasos raros)
    score = f1_score(y_test, y_pred, average='macro')
    
    return score

if __name__ == "__main__":
    # 1. Preparar datos una sola vez
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos()

    # 2. Configurar Weights & Biases para rastrear Optuna
    WANDB_PROJECT  = "pd1-c2526-team5"
    wandb_kwargs = {"project": WANDB_PROJECT, "name": "optuna-lgbm-tuning"}
    wandbc = WeightsAndBiasesCallback(metric_name="f1_macro", wandb_kwargs=wandb_kwargs)

    # 3. Crear el estudio de Optuna
    # Direction="maximize" porque queremos el F1-Score más alto posible
    study = optuna.create_study(direction='maximize', study_name="lgbm_retrasos_tuning")
    
    print("Iniciando la búsqueda de hiperparámetros...")
    # Ejecutamos 30 pruebas (puedes subirlo a 50 o 100 si tienes tiempo/recursos)
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test), 
        n_trials=30, 
        callbacks=[wandbc]
    )

    # 4. Resultados finales
    print("\n--- ¡Búsqueda completada! ---")
    print(f"Mejor F1-Score: {study.best_value}")
    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    wandb.finish()
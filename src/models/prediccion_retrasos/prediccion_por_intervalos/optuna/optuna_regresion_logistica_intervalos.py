"""
Optimización de Hiperparámetros Logistic Regression — Predicción de retraso por intervalos

Uso:
    uv run python src/models/prediccion_retrasos/prediccion_por_intervalos/optuna/optuna_regresion_logistica_intervalos.py

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

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

    if len(X) > 1000000:
        X = X.tail(1000000)
        y = y.tail(1000000)
        
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, shuffle=False)

    # IMPORTANTE: Escalado de datos para Regresión Logística
    print("Escalando características con StandardScaler...")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    return X_train_scaled, X_val_scaled, y_train, y_val


def objective(trial, X_train, X_val, y_train, y_val, labels):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    
    param = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': penalty,
        'solver': 'saga', # 'saga' es la mejor opción para datasets grandes y soporta l1, l2 y elasticnet
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': 500,  
        'random_state': 42,
        'n_jobs': -1
    }

    # Si elegimos elasticnet, necesitamos definir el l1_ratio
    if penalty == 'elasticnet':
        param['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    else:
        param['l1_ratio'] = None

    # ==========================================
    # INICIAR W&B PARA ESTE INTENTO ESPECÍFICO
    # ==========================================
    run = wandb.init(
        project="pd1-c2526-team5",
        group="optuna-logreg-tuning-intervalos-group60min-obj-target10m", 
        name=f"trial_{trial.number}", 
        config=param, 
        reinit=True 
    )

    modelo = LogisticRegression(**param)
    
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_val)
    y_probas = modelo.predict_proba(X_val)
    
    f1 = f1_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    
    wandb.log({
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    })

    wandb.sklearn.plot_classifier(
        modelo, 
        X_train, X_val, 
        y_train, y_val, 
        y_pred, y_probas, 
        labels=labels,
        model_name=f"LogReg_Trial_{trial.number}",
        feature_names=X_train.columns.tolist()
    )

    run.finish()
    
    return f1

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = cargar_y_preparar_datos()

    labels = [
        'Adelantado (>1 min)', 'Puntual (-1 a 1 min)', 
        'Retraso leve (1-3 min)', 'Retraso moderado (3-5 min)', 
        'Retraso grave (5-7.5 min)', 'Retraso muy grave (>7.5 min)'
    ]

    study = optuna.create_study(direction='maximize', study_name="logreg_retrasos_tuning")
    
    print("Iniciando la búsqueda de hiperparámetros...")
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val, labels), 
        n_trials=30
    )

    print("\n--- ¡Búsqueda completada! ---")
    print(f"Mejor F1-Score general: {study.best_value}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
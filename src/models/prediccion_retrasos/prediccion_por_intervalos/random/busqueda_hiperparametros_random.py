"""
Optimización de Hiperparámetros LGBM (Random Search) — Predicción de retraso por intervalos

Uso:
    uv run python src/models/prediccion_retrasos/prediccion_por_intervalos/random/busqueda_hiperparametros_random.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import pandas as pd
import numpy as np
import os
import wandb
import scipy.stats as stats
from sklearn.model_selection import train_test_split, ParameterSampler
from lightgbm import LGBMClassifier
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
    
    # Definir intervalos
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

    return X_train, X_val, y_train, y_val

def evaluate_random_combination(trial_number, param_dict, X_train, X_val, y_train, y_val, labels):
    """
    Función que entrena el modelo con una combinación específica y loguea en W&B.
    """
    param_dict['random_state'] = 42
    param_dict['n_jobs'] = -1
    param_dict['verbose'] = -1

    run = wandb.init(
        project="pd1-c2526-team5",
        group="optuna-lgbm-tuning-intervalos-group60min-obj-target10m", 
        name=f"random_trial_{trial_number}", 
        config=param_dict, 
        reinit=True 
    )

    modelo = LGBMClassifier(**param_dict)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

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
        model_name=f"LGBM_Random_Trial_{trial_number}",
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

    param_distributions = {
        'n_estimators': stats.randint(50, 301),                  # Valores entre 50 y 300
        'learning_rate': stats.uniform(0.01, 0.19),              # Valores entre 0.01 y 0.20
        'num_leaves': stats.randint(20, 151),                    # Valores entre 20 y 150
        'max_depth': stats.randint(3, 16),                       # Valores entre 3 y 15
        'min_child_samples': stats.randint(10, 101),             # Valores entre 10 y 100
        'subsample': stats.uniform(0.6, 0.4),                    # Valores entre 0.6 y 1.0
        'colsample_bytree': stats.uniform(0.6, 0.4),             # Valores entre 0.6 y 1.0
        'class_weight': ['balanced', None]
    }

    N_INTENTOS = 30 
    
    sampler = ParameterSampler(param_distributions, n_iter=N_INTENTOS, random_state=42)
    
    print(f"\nIniciando Random Search con {N_INTENTOS} intentos aleatorios...")

    best_f1 = 0
    best_params = None

    for i, params in enumerate(sampler):
        print(f"\n--- Evaluando intento aleatorio {i+1}/{N_INTENTOS} ---")
        
        # Opcional: imprimir los parámetros actuales para tener feedback en consola
        for k, v in params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
                
        current_f1 = evaluate_random_combination(
            trial_number=i+1, 
            param_dict=params, 
            X_train=X_train, 
            X_val=X_val, 
            y_train=y_train, 
            y_val=y_val, 
            labels=labels
        )
        
        print(f"-> Resultado F1-Score (Macro): {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_params = params.copy()

    print("\n=============================================")
    print("--- ¡Búsqueda Aleatoria completada! ---")
    print(f"Mejor F1-Score general (Macro): {best_f1:.4f}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    print("=============================================")
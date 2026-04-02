"""
Entrenamiento Regresión Logística Multinomial — Predicción de retraso por intervalos

Uso:
    python -m src.models.prediccion_retrasos.prediccion_por_intervalos.train.train_logistica_intervalos

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import pandas as pd
import numpy as np
import os
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet

def cargar_y_preparar_datos():
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos...')
    # Nota: Usamos el parquet de tu pipeline, asumiendo que tiene las columnas del CSV inspeccionado
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
    print('Datos cargados correctamente.')

    # 1. Definir los intervalos de retraso (Binning) para el Target
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

    # 2. Variables a excluir (Targets, deltas y variables del futuro según inspeccion_datos.csv)
    columnas_a_excluir = [
        'target_delay_10m_mean', 'target_delay_10m_max', 'target_delay_20m_mean', 'target_delay_20m_max',
        'target_delay_30m_mean', 'target_delay_30m_max', 'target_delay_45m_mean', 'target_delay_45m_max',
        'target_delay_60m_mean', 'target_delay_60m_max', 'target_delay_end_mean', 'target_delay_end_max',
        'delta_delay_10m_mean', 'delta_delay_10m_max', 'delta_delay_20m_mean', 'delta_delay_20m_max',
        'delta_delay_30m_mean', 'delta_delay_30m_max', 'delta_delay_45m_mean', 'delta_delay_45m_max',
        'delta_delay_60m_mean', 'delta_delay_60m_max', 'delta_delay_end_mean', 'delta_delay_end_max',
        'station_delay_10m_mean', 'station_delay_10m_max', 'station_delay_20m_mean', 'station_delay_20m_max',
        'station_delay_30m_mean', 'station_delay_30m_max', 'alert_in_next_15m_max', 'alert_in_next_30m_max',
        'seconds_to_next_alert_mean', 'afecta_despues_max', 'match_key_nunique', 'merge_time', 'clase_retraso'
    ]

    # Separar X e y
    X = df.drop(columns=[c for c in columnas_a_excluir if c in df.columns])
    y = df['clase_retraso']

    # 3. Tratamiento de Variables Categóricas (One-Hot Encoding)
    # Como la Regresión Logística no maneja categorías nativas, convertimos IDs en booleanos
    columnas_a_categoria = ['stop_id', 'route_id', 'direction']
    X = pd.get_dummies(X, columns=[c for c in columnas_a_categoria if c in X.columns], drop_first=True)

    # Limpiar posibles nulos que rompan la regresión logística y sincronizar índices
    X = X.fillna(0)
    y = y.dropna()
    X = X.loc[y.index]

    columnas_fecha = X.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns
    X = X.drop(columns=columnas_fecha)

    # Para regresión logística, un tamaño inmenso puede tardar bastante, acotamos a 1M si hace falta
    if len(X) > 2000000:
        X = X.tail(2000000)
        y = y.tail(2000000)

    # IMPORTANTE: shuffle=False para mantener el orden temporal del tren/prueba
    return train_test_split(X, y, test_size=0.2, shuffle=False)

if __name__ == "__main__":
    # 1. Preparar datos
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos()

    labels_target = [
        'Adelantado (>1 min)', 'Puntual (-1 a 1 min)', 
        'Retraso leve (1-3 min)', 'Retraso moderado (3-5 min)', 
        'Retraso grave (5-7.5 min)', 'Retraso muy grave (>7.5 min)'
    ]

    # 2. Inicializar Weights & Biases
    WANDB_PROJECT = "pd1-c2526-team5"
    wandb.init(
        project=WANDB_PROJECT, 
        group="modelos-retraso-clasificacion",
        name="logistic-regression-baseline", 
        notes="Modelo Regresión Logística Multinomial con StandardScaler para target_delay_10m_max"
    )

    # ==========================================
    # 3. ESCALADO DE DATOS (Vital para Logistic Regression)
    # ==========================================
    print("Escalando características con StandardScaler...")
    scaler = StandardScaler()
    
    # Entrenamos el scaler solo con Train y aplicamos a Train y Test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertimos de nuevo a DataFrame para que W&B pueda leer los nombres de las columnas en sus plots
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # ==========================================
    # 4. ENTRENAMIENTO DEL MODELO
    # ==========================================
    print("Entrenando Regresión Logística Multinomial...")
    # max_iter=1000 es necesario porque los modelos lineales con muchos datos suelen necesitar más pasos para converger
    modelo = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' # Importante si hay pocos retrasos muy graves
    )

    modelo.fit(X_train_scaled, y_train)

    # ==========================================
    # 5. PREDICCIONES Y LOG EN W&B
    # ==========================================
    print("Generando predicciones y calculando métricas...")
    y_pred = modelo.predict(X_test_scaled)
    y_probas = modelo.predict_proba(X_test_scaled) 

    # Calcular métricas explícitamente usando el promedio 'macro' (como corregimos anteriormente)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # Loguear todas las métricas personalizadas a W&B
    wandb.log({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    })

    # Imprimir reporte en la consola local
    print("\n--- Reporte de Clasificación (Regresión Logística) ---")
    print(classification_report(y_test, y_pred))

    # Crear dashboard automático de Scikit-Learn (Matriz de confusión, ROC, PR, etc.)
    wandb.sklearn.plot_classifier(
        modelo, 
        X_train_scaled, X_test_scaled, 
        y_train, y_test, 
        y_pred, y_probas, 
        labels=labels_target,
        model_name="Logistic_Regression_Baseline",
        feature_names=X_train.columns.tolist()
    )

    # Finalizar la ejecución en W&B
    wandb.finish()
    print("¡Entrenamiento finalizado y trackeado con éxito!")
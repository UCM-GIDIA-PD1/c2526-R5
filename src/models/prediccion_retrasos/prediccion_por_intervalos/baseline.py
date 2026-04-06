"""
Baseline Heurístico — Predicción de retraso por intervalos

Heurística: "El retraso en 10 minutos será exactamente igual al retraso actual"
Esto sirve como referencia para evaluar si los modelos ML realmente aportan valor.

Uso:
    uv run python src/models/prediccion_retrasos/prediccion_por_intervalos/eval/baseline_retraso.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from src.common.minio_client import download_df_parquet

def evaluar_baseline():
    # 1. Cargar los datos
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin=60.parquet'
        
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos desde MinIO...')
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH)
    print('Datos cargados correctamente.')

    # 2. Definir los intervalos de retraso (Binning)
    bins = [-np.inf, -60, 60, 180, 300, 450, np.inf]
    labels = [
        'Adelantado (>1 min)', 
        'Puntual (-1 a 1 min)', 
        'Retraso leve (1-3 min)', 
        'Retraso moderado (3-5 min)', 
        'Retraso grave (5-7.5 min)',
        'Retraso muy grave (>7.5 min)'
    ]

    # 3. Target real (Ground Truth)
    columna_objetivo = 'target_delay_10m_max'
    df['clase_retraso_real'] = pd.cut(df[columna_objetivo], bins=bins, labels=labels)

    # 4. Target predicho (Baseline)
    columna_retraso_actual = 'delay_seconds_mean' 
    
    # (Opcional): Si no tienes la columna de retraso actual guardada, la puedes deducir matemáticamente:
    # df['retraso_actual_deducido'] = df['target_delay_10m_max'] - df['delta_delay_10m_max']
    # columna_retraso_actual = 'retraso_actual_deducido'

    df['clase_retraso_predicha'] = pd.cut(df[columna_retraso_actual], bins=bins, labels=labels)

    # Quedarnos solo con los registros que no son nulos (igual que se hace en la preparación de X e y)
    df_eval = df.dropna(subset=['clase_retraso_real', 'clase_retraso_predicha'])

    # 5. Split cronológico idéntico al de los modelos (15% final para Test)
    # Hacemos esto para evaluar exactamente sobre los mismos datos que LightGBM / LogReg
    _, df_test = train_test_split(df_eval, test_size=0.15, shuffle=False)

    y_test_real = df_test['clase_retraso_real']
    y_test_pred = df_test['clase_retraso_predicha']

    # 6. Calcular métricas
    print("\nCalculando métricas del Baseline en el conjunto de Test (último 15% de datos)...")
    
    acc = accuracy_score(y_test_real, y_test_pred)
    f1_macro = f1_score(y_test_real, y_test_pred, average='macro')
    precision_macro = precision_score(y_test_real, y_test_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_real, y_test_pred, average='macro', zero_division=0)

    # 7. Mostrar Resultados
    print(f"\n========================================================")
    print(f"   RESULTADOS BASELINE: (Predicción = Retraso Actual)   ")
    print(f"========================================================")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1-Score Macro:  {f1_macro:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro:    {recall_macro:.4f}\n")

    print("--- Reporte de Clasificación Detallado ---")
    print(classification_report(y_test_real, y_test_pred))

if __name__ == "__main__":
    evaluar_baseline()
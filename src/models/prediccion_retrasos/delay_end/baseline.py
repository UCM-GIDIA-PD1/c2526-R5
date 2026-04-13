"""
Baseline — Predicción de retraso al final del viaje (menos de 30 min restantes)

Calcula el MAE de dos baselines sobre el conjunto de validación:
  1. Predecir siempre la media del target (train)
  2. Predecir el retraso actual (delay_seconds) como retraso al final

Solo usa registros con scheduled_time_to_end < 1800s.

Uso:
    uv run python src/models/prediccion_retrasos/delay_end/baseline.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR          = 2025
TRAIN_MONTHS  = range(1, 10)
VAL_MONTHS    = range(10, 13)
TARGET        = "target_delay_end"
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"


def load_months(months: range) -> pd.DataFrame:
    """Descarga y filtra los datos de entrenamiento y validacion desde MinIO."""
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=YEAR, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            df = df[df["is_unscheduled"] == False]
            df = df.dropna(subset=[TARGET])
            df = df[df["scheduled_time_to_end"] < 1800]
            print(f"  ✓ month={month:02d}  {len(df):>10,} filas")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ month={month:02d}  no encontrado ({e})")
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(y_true, y_pred, name: str):
    """Calcula las metricas principales a partir de las predicciones y los valores reales."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f} s  ({mae/60:.3f} min)")
    print(f"  RMSE : {rmse:.2f} s")
    print(f"  R²   : {r2:.4f}")


def main():
    """Funcion principal que orquesta la carga de datos, el entrenamiento y el registro de resultados."""
    print("Cargando train (para calcular media)...")
    df_train = load_months(TRAIN_MONTHS)
    mean_target = df_train[TARGET].mean()
    print(f"  Media del target en train: {mean_target:.2f} s\n")

    print("Cargando val...")
    df_val = load_months(VAL_MONTHS)
    print(f"  Total val: {len(df_val):,} filas\n")

    y_true = df_val[TARGET]

    # Baseline 1: predecir siempre la media del train
    pred_mean = np.full(len(y_true), mean_target)
    compute_metrics(y_true, pred_mean, "Baseline 1 — siempre la media")

    # Baseline 2: predecir el retraso actual (persistencia)
    if "delay_seconds" in df_val.columns:
        mask = df_val["delay_seconds"].notna()
        compute_metrics(y_true[mask], df_val.loc[mask, "delay_seconds"].values, "Baseline 2 — retraso actual (persistencia)")
    else:
        print("⚠ No se encontró la columna 'delay_seconds' en los datos.")


if __name__ == "__main__":
    main()

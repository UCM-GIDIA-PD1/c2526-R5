"""
Procesa el DataFrame histórico (dataset_final) para generar tensores
espaciotemporales X e Y, aplica split cronológico y StandardScaler, y guarda
un diccionario .pt con todo lo necesario para entrenar y evaluar el modelo.

Carga:    artefactos/grafo.pt        (nodes, n_nodes)
Guarda:   artefactos/tensores.pt
  {
    'X_train': (T_tr, N, 14),  'Y_train': (T_tr, N, 3),
    'X_val':   (T_va, N, 14),  'Y_val':   (T_va, N, 3),
    'X_test':  (T_te, N, 14),  'Y_test':  (T_te, N, 3),
    'scaler_X': StandardScaler,
    'scaler_Y': StandardScaler,
    'times':   pd.DatetimeIndex,
    'nodes':   list[str],
  }

Uso
---
    uv run python src/models/propagacion_estacion/02_generar_tensores.py
"""
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet

# Params
RUTA_DATASET = "grupo5/final/year=2025/month=01/dataset_final.parquet"
FREQ         = "15min"
SPLIT_TRAIN  = 0.70
SPLIT_VAL    = 0.10   # el resto test

RUTA_GRAFO   = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_SALIDA  = Path(__file__).parent / "artefactos" / "tensores.pt"

FEATURE_COLS = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
    'is_unscheduled', 'temp_extreme', 'n_eventos_afectando',
    'route_rolling_delay', 'actual_headway_seconds',
    'hour_sin', 'hour_cos', 'dow',
    'afecta_previo', 'afecta_durante', 'afecta_despues',
]
TARGET_COLS = ['station_delay_10m', 'station_delay_20m', 'station_delay_30m']

# Credenciales 
access_key = os.environ["MINIO_ACCESS_KEY"]
secret_key = os.environ["MINIO_SECRET_KEY"]


def build_spatiotemporal_tensor(
    df: pd.DataFrame,
    scheduled_nodes: list[str],
    freq: str = FREQ,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Transforma un DataFrame plano de eventos en tensores (T, N, F) y (T, N, H).

    Features (14): delay_seconds, lagged_delay_1, lagged_delay_2, is_unscheduled,
                   temp_extreme, n_eventos_afectando, route_rolling_delay,
                   actual_headway_seconds, hour_sin, hour_cos, dow,
                   afecta_previo, afecta_durante, afecta_despues
    Targets  (3):  station_delay_10m, station_delay_20m, station_delay_30m
    """
    if not scheduled_nodes:
        raise ValueError("scheduled_nodes está vacío — no se puede construir el tensor.")

    df = df[df['stop_id'].isin(scheduled_nodes)].copy()
    df['time_bin'] = pd.to_datetime(df['merge_time']).dt.floor(freq)

    agg_rules = {
        'delay_seconds':          'mean',
        'lagged_delay_1':         'mean',
        'lagged_delay_2':         'mean',
        'is_unscheduled':         'sum',
        'temp_extreme':           'max',
        'n_eventos_afectando':    'max',
        'route_rolling_delay':    'mean',
        'actual_headway_seconds': 'mean',
        'afecta_previo':          'max',
        'afecta_durante':         'max',
        'afecta_despues':         'max',
        'station_delay_10m':      'mean',
        'station_delay_20m':      'mean',
        'station_delay_30m':      'mean',
    }
    grouped = df.groupby(['time_bin', 'stop_id']).agg(agg_rules)
    del df
    gc.collect()

    all_times = pd.date_range(
        start=grouped.index.get_level_values('time_bin').min(),
        end=grouped.index.get_level_values('time_bin').max(),
        freq=freq,
    )
    all_nodes  = sorted(scheduled_nodes)
    full_index = pd.MultiIndex.from_product([all_times, all_nodes], names=['time_bin', 'stop_id'])
    full_df    = grouped.reindex(full_index).reset_index()
    del grouped
    gc.collect()

    # Imputación
    zero_fill = ['delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
                 'is_unscheduled', 'route_rolling_delay', 'actual_headway_seconds']
    full_df[zero_fill] = full_df[zero_fill].fillna(0)

    ctx_cols = ['temp_extreme', 'n_eventos_afectando', 'afecta_previo', 'afecta_durante', 'afecta_despues']
    full_df[ctx_cols] = full_df.groupby('stop_id')[ctx_cols].ffill()
    full_df[ctx_cols] = full_df.groupby('stop_id')[ctx_cols].bfill()
    full_df[ctx_cols] = full_df[ctx_cols].fillna(0)

    # Features temporales derivadas del bin (sin ruido de agregación)
    full_df['hour_sin'] = np.sin(2 * np.pi * full_df['time_bin'].dt.hour / 24)
    full_df['hour_cos'] = np.cos(2 * np.pi * full_df['time_bin'].dt.hour / 24)
    full_df['dow']      = full_df['time_bin'].dt.dayofweek.astype(float)
    full_df             = full_df.sort_values(['time_bin', 'stop_id'])

    T = len(all_times)
    N = len(all_nodes)
    F = len(FEATURE_COLS)
    H = len(TARGET_COLS)

    X_tensor = full_df[FEATURE_COLS].values.reshape(T, N, F)
    Y_tensor = full_df[TARGET_COLS].values.reshape(T, N, H)
    del full_df
    gc.collect()

    return X_tensor, Y_tensor, all_times, all_nodes


def escalar(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """Ajusta StandardScaler sobre train y transforma los tres splits."""
    T_tr, N, F = X_train.shape
    H = Y_train.shape[-1]

    scaler_X = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, F))
    X_tr_s = scaler_X.transform(X_train.reshape(-1, F)).reshape(X_train.shape)
    X_va_s = scaler_X.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_te_s = scaler_X.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

    scaler_Y = StandardScaler()
    scaler_Y.fit(Y_train.reshape(-1, H))
    Y_tr_s = scaler_Y.transform(Y_train.reshape(-1, H)).reshape(Y_train.shape)
    Y_va_s = scaler_Y.transform(Y_val.reshape(-1, H)).reshape(Y_val.shape)
    Y_te_s = scaler_Y.transform(Y_test.reshape(-1, H)).reshape(Y_test.shape)

    return X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, scaler_X, scaler_Y


def main():
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("=== 02 Generar Tensores ===")

    # Carga el grafo para obtener la lista de nodos del grafo
    grafo      = torch.load(RUTA_GRAFO, weights_only=False)
    nodes_list = grafo['nodes']
    print(f"Nodos del grafo: {len(nodes_list)}")

    print("Descargando dataset_final...")
    df_final = download_df_parquet(access_key, secret_key, RUTA_DATASET)
    print(f"Filas cargadas: {len(df_final):,}")

    print("Construyendo tensor espaciotemporal...")
    X, Y, times, nodes = build_spatiotemporal_tensor(df_final, nodes_list)
    del df_final
    gc.collect()
    print(f"X: {X.shape}  Y: {Y.shape}")

    # Split cronológico
    X_np = np.nan_to_num(np.asarray(X), nan=0.0)
    Y_np = np.nan_to_num(np.asarray(Y), nan=0.0)
    del X, Y
    gc.collect()

    T         = X_np.shape[0]
    train_end = int(T * SPLIT_TRAIN)
    val_end   = train_end + int(T * SPLIT_VAL)

    X_train, X_val, X_test = X_np[:train_end], X_np[train_end:val_end], X_np[val_end:]
    Y_train, Y_val, Y_test = Y_np[:train_end], Y_np[train_end:val_end], Y_np[val_end:]
    del X_np, Y_np
    gc.collect()

    print(f"Split → train: {X_train.shape[0]} | val: {X_val.shape[0]} | test: {X_test.shape[0]}")

    print("Escalando features y targets...")
    X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, scaler_X, scaler_Y = escalar(
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    )
    del X_train, X_val, X_test, Y_train, Y_val, Y_test
    gc.collect()

    torch.save(
        {
            'X_train': X_tr_s, 'Y_train': Y_tr_s,
            'X_val':   X_va_s, 'Y_val':   Y_va_s,
            'X_test':  X_te_s, 'Y_test':  Y_te_s,
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y,
            'times':    times,
            'nodes':    nodes,
        },
        RUTA_SALIDA,
    )
    print(f"Tensores guardados en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()

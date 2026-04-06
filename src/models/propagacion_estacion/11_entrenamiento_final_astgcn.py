"""
Entrenamiento final ASTGCN sobre Train + Validación concatenados.

Descarga el dataset desde MinIO, construye el grafo, precalcula los polinomios
de Chebyshev, crea los tensores espaciotemporales, concatena Train y Val en un
único split de entrenamiento y entrena NUM_EPOCHS épocas fijas con los
hiperparámetros óptimos del HPO. Guarda los pesos del modelo y los datos de
Test para evaluación externa.
La evaluación sobre Test se realiza en 12_evaluacion_modelos.py.

Carga:  artefactos/astgcn_hpo.pt  (best_params)
        MinIO → grupo5/final/...   (dataset_final.parquet)
        MinIO → grupo5/cleaned/gtfs_clean_scheduled/...

Guarda: artefactos/astgcn_final.pth
  {
    'model_state_dict':      ...,
    'best_params':           dict,
    'scaler_Y':              StandardScaler,
    'X_test_scaled':         np.ndarray  (T_test, N, F),
    'Y_test_scaled':         np.ndarray  (T_test, N, C),
    'dow_test_raw':          np.ndarray  (T_test, N)  — day-of-week sin escalar,
    'temp_extreme_test_raw': np.ndarray  (T_test, N)  — 0/1 sin escalar,
    'history_len':           int,
    'n_nodes':               int,
    'num_features':          int,
    'num_targets':           int,
    'A_weighted':            np.ndarray  (N, N),  — matriz sin normalizar (para Laplaciano)
  }

Uso
---
    uv run python src/models/propagacion_estacion/09_entrenamiento_final_astgcn.py
"""
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_START_METHOD"] = "thread"

import wandb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.minio_client import download_df_parquet
from models.astgcn import ASTGCN_Metro, calcular_scaled_laplacian, calcular_polinomios_chebyshev

RUTA_HPO    = Path(__file__).parent / "artefactos" / "astgcn_hpo.pt"
RUTA_MODELO = Path(__file__).parent / "artefactos" / "astgcn_final.pth"

WANDB_PROJECT = "pd1-c2526-team5"
SEED          = 42
NUM_EPOCHS    = 50
FREQ          = "15min"

VARIABLES_ENTRADA = [
    "delay_seconds", "lagged_delay_1", "lagged_delay_2", "is_unscheduled",
    "temp_extreme", "n_eventos_afectando", "route_rolling_delay",
    "actual_headway_seconds", "hour_sin", "hour_cos", "dow",
]
VARIABLES_OBJETIVO = [
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
]

IDX_DOW          = VARIABLES_ENTRADA.index("dow")
IDX_TEMP_EXTREME = VARIABLES_ENTRADA.index("temp_extreme")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DatasetASTGCN(Dataset):
    def __init__(self, X, Y, history_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.history_len = history_len

    def __len__(self):
        return len(self.X) - self.history_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.history_len], self.Y[idx + self.history_len]


def descargar_datos():
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Faltan MINIO_ACCESS_KEY y/o MINIO_SECRET_KEY")
    df_final = download_df_parquet(
        access_key, secret_key,
        "grupo5/final/year=2025/month=01/dataset_final.parquet"
    )
    dates = pd.date_range(start="2025-01-01", end="2025-01-31").strftime("%Y-%m-%d").tolist()
    dfs = []
    for date in dates:
        try:
            dfs.append(download_df_parquet(
                access_key, secret_key,
                f"grupo5/cleaned/gtfs_clean_scheduled/date={date}/gtfs_scheduled_{date}.parquet"
            ))
        except Exception:
            print(f"No se pudo descargar GTFS para: {date}")
    if not dfs:
        raise RuntimeError("No se pudo descargar ningún fichero GTFS")
    return df_final, pd.concat(dfs, ignore_index=True)


def construir_grafo(df):
    df = df.sort_values(by=["trip_uid", "scheduled_seconds"]).reset_index(drop=True)
    df["next_stop_id"]           = df.groupby("trip_uid")["stop_id"].shift(-1)
    df["next_scheduled_seconds"] = df.groupby("trip_uid")["scheduled_seconds"].shift(-1)
    edges_df = df.dropna(subset=["next_stop_id"]).copy()
    edges_df["travel_time"] = edges_df["next_scheduled_seconds"] - edges_df["scheduled_seconds"]
    edges_df = edges_df[edges_df["travel_time"] > 0]
    graph_df = edges_df.groupby(["stop_id", "next_stop_id"]).agg(
        median_travel_time=("travel_time", "median"),
        trip_count=("trip_uid", "count"),
    ).reset_index()
    graph_df = graph_df[graph_df["trip_count"] > 5]
    nodes = sorted(list(set(df["stop_id"].unique()) | set(df["next_stop_id"].dropna().unique())))
    node_to_idx = {stop_id: idx for idx, stop_id in enumerate(nodes)}
    n_nodes = len(nodes)
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    sigma = graph_df["median_travel_time"].std()
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    for _, row in graph_df.iterrows():
        i = node_to_idx[row["stop_id"]]
        j = node_to_idx[row["next_stop_id"]]
        w = np.exp(-(row["median_travel_time"] ** 2) / (sigma ** 2))
        A[i, j] = w
        A[j, i] = w
    return node_to_idx, A


def crear_tensores(df, mapa_nodos, features, targets, freq=FREQ):
    nodos_validos = list(mapa_nodos.keys())
    df = df[df["stop_id"].isin(nodos_validos)].copy()
    df["time_bin"] = pd.to_datetime(df["merge_time"]).dt.floor(freq)
    reglas = {
        "delay_seconds": "mean", "lagged_delay_1": "mean", "lagged_delay_2": "mean",
        "is_unscheduled": "sum", "temp_extreme": "max", "n_eventos_afectando": "max",
        "route_rolling_delay": "mean", "actual_headway_seconds": "mean",
        "target_delay_10m": "mean", "target_delay_20m": "mean", "target_delay_30m": "mean",
        "station_delay_10m": "mean", "station_delay_20m": "mean", "station_delay_30m": "mean",
    }
    df_agr  = df.groupby(["time_bin", "stop_id"]).agg(reglas)
    tiempos = pd.date_range(
        start=df_agr.index.get_level_values("time_bin").min(),
        end=df_agr.index.get_level_values("time_bin").max(),
        freq=freq,
    )
    idx_completo = pd.MultiIndex.from_product([tiempos, nodos_validos], names=["time_bin", "stop_id"])
    df_c = df_agr.reindex(idx_completo).reset_index()
    df_c[["delay_seconds", "lagged_delay_1", "lagged_delay_2",
          "is_unscheduled", "route_rolling_delay", "actual_headway_seconds"]] = \
        df_c[["delay_seconds", "lagged_delay_1", "lagged_delay_2",
              "is_unscheduled", "route_rolling_delay", "actual_headway_seconds"]].fillna(0)
    df_c[["temp_extreme", "n_eventos_afectando"]] = (
        df_c.groupby("stop_id")[["temp_extreme", "n_eventos_afectando"]].ffill().bfill().fillna(0)
    )
    df_c["hour_sin"] = np.sin(2 * np.pi * df_c["time_bin"].dt.hour / 24)
    df_c["hour_cos"] = np.cos(2 * np.pi * df_c["time_bin"].dt.hour / 24)
    df_c["dow"]      = df_c["time_bin"].dt.dayofweek.astype(float)
    df_c["nodo_idx"] = df_c["stop_id"].map(mapa_nodos)
    df_c = df_c.sort_values(["time_bin", "nodo_idx"])
    T, N, F_in, C_out = len(tiempos), len(nodos_validos), len(features), len(targets)
    X = np.nan_to_num(df_c[features].values.reshape(T, N, F_in), nan=0.0)
    Y = np.nan_to_num(df_c[targets].values.reshape(T, N, C_out), nan=0.0)
    return X, Y


def split_y_escalar(X_full, Y_full):
    T = X_full.shape[0]
    t_test = int(T * 0.8)
    t_val  = int(t_test * 0.8)
    N, F   = X_full.shape[1], X_full.shape[2]
    C      = Y_full.shape[2]

    X_tr  = X_full[:t_val]
    X_val = X_full[t_val:t_test]
    X_te  = X_full[t_test:]
    Y_tr  = Y_full[:t_val]
    Y_val = Y_full[t_val:t_test]
    Y_te  = Y_full[t_test:]

    sc_X = StandardScaler()
    X_tr_s = sc_X.fit_transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
    X_va_s = sc_X.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_te_s = sc_X.transform(X_te.reshape(-1, F)).reshape(X_te.shape)

    sc_Y = StandardScaler()
    Y_tr_s = sc_Y.fit_transform(Y_tr.reshape(-1, C)).reshape(Y_tr.shape)
    Y_va_s = sc_Y.transform(Y_val.reshape(-1, C)).reshape(Y_val.shape)
    Y_te_s = sc_Y.transform(Y_te.reshape(-1, C)).reshape(Y_te.shape)

    return {
        'X_train': X_tr_s, 'Y_train': Y_tr_s,
        'X_val':   X_va_s, 'Y_val':   Y_va_s,
        'X_test':  X_te_s, 'Y_test':  Y_te_s,
        'scaler_Y': sc_Y,
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    hpo         = torch.load(RUTA_HPO, weights_only=False)
    best_params = hpo['best_params']

    df_final, df_gtfs  = descargar_datos()
    node_to_idx, A_raw = construir_grafo(df_gtfs)
    X_full, Y_full     = crear_tensores(df_final, node_to_idx, VARIABLES_ENTRADA, VARIABLES_OBJETIVO)

    # ── Extraer features de segmentación ANTES del escalado ──────────────────
    T_total    = X_full.shape[0]
    t_test_ini = int(T_total * 0.8)
    dow_test_raw          = X_full[t_test_ini:, :, IDX_DOW].copy()
    temp_extreme_test_raw = X_full[t_test_ini:, :, IDX_TEMP_EXTREME].copy()

    splits  = split_y_escalar(X_full, Y_full)
    N_nodes = len(node_to_idx)

    # ── Polinomios de Chebyshev ───────────────────────────────────────────────
    K_cheb = best_params['K_cheb']
    cheb_polynomials = calcular_polinomios_chebyshev(
        calcular_scaled_laplacian(A_raw), K_cheb
    )

    # ── Concatenar Train + Val ────────────────────────────────────────────────
    X_trainval = np.concatenate([splits['X_train'], splits['X_val']], axis=0)
    Y_trainval = np.concatenate([splits['Y_train'], splits['Y_val']], axis=0)
    print(f"Train+Val: {X_trainval.shape[0]} pasos  |  Test: {splits['X_test'].shape[0]} pasos")

    hl = best_params['history_len']
    bs = best_params['batch_size']

    train_loader = DataLoader(DatasetASTGCN(X_trainval, Y_trainval, hl), batch_size=bs, shuffle=True)

    model = ASTGCN_Metro(
        num_nodes=N_nodes,
        num_features=len(VARIABLES_ENTRADA),
        num_targets=len(VARIABLES_OBJETIVO),
        history_len=hl,
        cheb_polynomials=cheb_polynomials,
        K=K_cheb,
        hidden_channels=best_params['hidden_channels'],
        dropout=best_params['dropout'],
    ).to(device)

    criterion = (
        nn.MSELoss() if best_params['loss_name'] == 'mse'
        else nn.SmoothL1Loss(beta=0.5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    wandb.init(project=WANDB_PROJECT, name='astgcn-final-trainval', mode='offline', config=best_params)

    best_loss  = float('inf')
    best_epoch = 0
    best_state = None
    t0         = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        acc = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            acc += loss.item()
        train_loss = acc / len(train_loader)
        scheduler.step(train_loss)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'lr': optimizer.param_groups[0]['lr']})
        print(f"Época {epoch:02d}/{NUM_EPOCHS} | trainval={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss  = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    wandb.log({'best_trainval_loss': best_loss, 'best_epoch': best_epoch, 'train_time_sec': train_time})
    wandb.finish()

    # ── Guardar modelo + datos de test (sin evaluación sobre test) ────────────
    torch.save(
        {
            'model_state_dict':      model.state_dict(),
            'best_params':           best_params,
            'scaler_Y':              splits['scaler_Y'],
            'X_test_scaled':         splits['X_test'],
            'Y_test_scaled':         splits['Y_test'],
            'dow_test_raw':          dow_test_raw,
            'temp_extreme_test_raw': temp_extreme_test_raw,
            'history_len':           hl,
            'n_nodes':               N_nodes,
            'num_features':          len(VARIABLES_ENTRADA),
            'num_targets':           len(VARIABLES_OBJETIVO),
            'A_weighted':            A_raw,
        },
        RUTA_MODELO,
    )
    print(f"Modelo final ASTGCN guardado en: {RUTA_MODELO}")
    print("NOTA: la evaluación sobre Test se realiza en 12_evaluacion_modelos.py")


if __name__ == '__main__':
    main()

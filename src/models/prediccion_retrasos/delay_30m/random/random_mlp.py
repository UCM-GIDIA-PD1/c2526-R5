"""
Random Search — Busqueda aleatoria de hiperparametros para MLP (PyTorch)

Predice target_delay_30m usando el dataset mensual completo (todas las lineas).
Filtra scheduled_time_to_end >= 1800s.

Uso:
    uv run python src/models/prediccion_retrasos/delay_30m/random/random_mlp.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import os
import random as rnd
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# Configuracion

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR          = 2025
TRAIN_MONTHS  = range(1, 10)
VAL_MONTHS    = range(10, 13)
TARGET        = "target_delay_30m"
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

WANDB_PROJECT = "pd1-c2526-team5"
SAMPLE_FRAC   = 0.5
NUM_RUNS      = 14
SEED          = 42

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"

EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
}

# Espacio de busqueda

PARAM_SPACE = {
    "n_layers":      [2, 3, 4],
    "hidden_sizes":  [64, 128, 256, 512],
    "dropout":       (0.1, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "weight_decay":  (1e-6, 1e-3),
    "batch_size":    [1024, 2048, 4096, 8192],
}


def sample_params(rng: rnd.Random) -> dict:
    """Muestrea aleatoriamente una combinacion de hiperparametros del espacio definido."""
    n_layers = rng.choice(PARAM_SPACE["n_layers"])
    hidden_layers = [rng.choice(PARAM_SPACE["hidden_sizes"]) for _ in range(n_layers)]
    lo_dr, hi_dr = PARAM_SPACE["dropout"]
    lo_lr, hi_lr = PARAM_SPACE["learning_rate"]
    lo_wd, hi_wd = PARAM_SPACE["weight_decay"]
    return {
        "hidden_layers":  hidden_layers,
        "dropout":        rng.uniform(lo_dr, hi_dr),
        "learning_rate":  10 ** rng.uniform(np.log10(lo_lr), np.log10(hi_lr)),
        "weight_decay":   10 ** rng.uniform(np.log10(lo_wd), np.log10(hi_wd)),
        "batch_size":     rng.choice(PARAM_SPACE["batch_size"]),
    }


# Helpers

def load_data():
    """Descarga y filtra los datos de entrenamiento y validacion desde MinIO."""
    def _load(months):
        dfs = []
        for month in months:
            path = DATA_TEMPLATE.format(year=YEAR, month=month)
            try:
                df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
                df = df[df["is_unscheduled"] == False]
                df = df.dropna(subset=[TARGET])
                df = df[df["scheduled_time_to_end"] >= 1800]
                if SAMPLE_FRAC < 1.0:
                    df = df.sample(frac=SAMPLE_FRAC, random_state=SEED)
                dfs.append(df)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True)

    print("Cargando datos...")
    df_train = _load(TRAIN_MONTHS)
    df_val   = _load(VAL_MONTHS)
    print(f"  train: {len(df_train):,}  |  val: {len(df_val):,}\n")
    return df_train, df_val


def encode_categoricals(df_train, df_val):
    """Convierte las columnas categoricas a enteros usando el vocabulario del conjunto de entrenamiento."""
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variables derivadas del retraso como velocidad, aceleracion e interacciones."""
    if "lagged_delay_1" in df.columns and "delay_seconds" in df.columns:
        df["delay_velocity"] = df["delay_seconds"] - df["lagged_delay_1"]
    if "lagged_delay_1" in df.columns and "lagged_delay_2" in df.columns:
        df["delay_acceleration"] = (
            (df["delay_seconds"] - df["lagged_delay_1"])
            - (df["lagged_delay_1"] - df["lagged_delay_2"])
        )
    if "delay_seconds" in df.columns and "stops_to_end" in df.columns:
        df["delay_x_stops_remaining"] = df["delay_seconds"] * df["stops_to_end"]
    if "delay_seconds" in df.columns and "scheduled_time_to_end" in df.columns:
        df["delay_ratio"] = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1)
    return df


def add_target_encoding(df_train, df_val, col, target):
    """Aplica target encoding sobre una columna usando la media del target por grupo calculada en train."""
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val


def get_features(df):
    """Devuelve la lista de columnas que se usan como features, excluyendo el target y columnas no relevantes."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


# Modelo MLP

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        """Construye las capas de la red segun la arquitectura indicada."""
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Main

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Precargando datos (se hace una sola vez para todos los trials)...")
    df_train_global, df_val_global = load_data()
    df_train_global, df_val_global = encode_categoricals(df_train_global, df_val_global)
    df_train_global, df_val_global = add_target_encoding(df_train_global, df_val_global, STOP_ID_COL, TARGET)
    df_train_global = add_derived_features(df_train_global)
    df_val_global   = add_derived_features(df_val_global)
    feats = get_features(df_train_global)

    X_train_np = df_train_global[feats].values.astype(np.float32)
    y_train_np = df_train_global[TARGET].values.astype(np.float32)
    X_val_np   = df_val_global[feats].values.astype(np.float32)
    y_val_np   = df_val_global[TARGET].values.astype(np.float32)

    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_val_np   = np.nan_to_num(X_val_np,   nan=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled   = scaler.transform(X_val_np)

    print(f"Features ({len(feats)}): {feats}\n")

    rng = rnd.Random(SEED)
    best_mae = float("inf")
    best_params = None

    print(f"Lanzando {NUM_RUNS} trials random search MLP (target={TARGET})...\n")
    for trial in range(NUM_RUNS):
        params = sample_params(rng)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_ds = TensorDataset(torch.from_numpy(X_train_scaled.copy()), torch.from_numpy(y_train_np.copy()))
        val_ds   = TensorDataset(torch.from_numpy(X_val_scaled.copy()),   torch.from_numpy(y_val_np.copy()))
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False)

        model = MLP(
            input_dim=len(feats),
            hidden_layers=params["hidden_layers"],
            dropout=params["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        criterion = nn.L1Loss()

        best_trial_mae = float("inf")
        patience_counter = 0
        max_epochs = 100
        patience = 10

        for epoch in range(1, max_epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_losses.append(criterion(model(X_batch), y_batch).item() * len(X_batch))
            val_mae = sum(val_losses) / len(val_ds)

            if val_mae < best_trial_mae:
                best_trial_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Metricas finales
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(X_val_scaled).to(device)).cpu().numpy()
        mae  = mean_absolute_error(y_val_np, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_np, y_pred))
        r2   = r2_score(y_val_np, y_pred)

        if mae < best_mae:
            best_mae = mae
            best_params = params.copy()

        wandb.init(
            project=WANDB_PROJECT,
            name=f"random-mlp-30m-trial{trial}",
            group="prediccion-retrasos-30m",
            config={**params, "trial": trial},
            reinit=True,
        )
        wandb.log({
            "val_mae_s":  round(mae, 2),
            "val_mae_min": round(mae / 60, 2),
            "val_rmse_s": round(rmse, 2),
            "val_r2":     round(r2, 4),
        })
        wandb.finish()

        print(f"  trial {trial:02d}  mae={mae:.2f}s  best={best_mae:.2f}s")

    print("\n── Mejores hiperparametros ──────────────────────────────────────")
    print(f"  val_mae_s: {best_mae:.2f}s")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

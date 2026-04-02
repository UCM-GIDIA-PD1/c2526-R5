"""
Entrenamiento MLP (PyTorch) — Prediccion de retraso al final del viaje (Objetivo 2)

Predice target_delay_end = retraso absoluto del tren al terminar el viaje.
Solo usa registros con scheduled_time_to_end < 1800s (menos de 30 min restantes).
Usa L1 loss (MAE) para ser comparable con el LGBM.

Validacion temporal:
    Train  -> meses 01-09  (enero-septiembre 2025)
    Val    -> meses 10-12  (octubre-diciembre 2025)

Uso:
    uv run python -m src.models.prediccion_retrasos.delay_end.train.train_mlp

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
"""

import os
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

# ── Configuracion ─────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR           = 2025
TRAIN_MONTHS   = range(1, 10)
VAL_MONTHS     = range(10, 13)
TARGET         = "target_delay_end"
DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "mlp-stop-delay-end"

EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid",
    "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
}

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"

# ── Hiperparametros MLP ──────────────────────────────────────────────────────

MLP_CONFIG = {
    "hidden_layers":  [256, 128, 64],
    "dropout":        0.3,
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,
    "batch_size":     4096,
    "max_epochs":     200,
    "patience":       15,
    "seed":           42,
}

SAMPLE_FRAC = 1.0

# ── Helpers (mismos que LGBM) ─────────────────────────────────────────────────

def load_months(months: range) -> pd.DataFrame:
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=YEAR, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            total = len(df)
            df = df[df["is_unscheduled"] == False]
            df = df.dropna(subset=[TARGET])
            df = df[df["scheduled_time_to_end"] < 1800]
            if SAMPLE_FRAC < 1.0:
                df = df.sample(frac=SAMPLE_FRAC, random_state=42)
            for col in CAT_FEATURES:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            mb = df.memory_usage(deep=True).sum() / 1e6
            print(f"  month={month:02d}  {total:>10,} filas  ->  {len(df):>10,} tras filtrado  ~{mb:.0f} MB")
            dfs.append(df)
        except Exception as e:
            print(f"  month={month:02d}  no encontrado ({e})")
    return pd.concat(dfs, ignore_index=True)


def encode_categoricals(df_train, df_val):
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
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
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val


def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


def compute_metrics(y_true, y_pred, prefix="") -> dict:
    mae     = mean_absolute_error(y_true, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    r2      = r2_score(y_true, y_pred)
    mae_min = mae / 60
    return {
        f"{prefix}mae_s":   round(mae, 2),
        f"{prefix}mae_min": round(mae_min, 2),
        f"{prefix}rmse_s":  round(rmse, 2),
        f"{prefix}r2":      round(r2, 4),
    }


# ── Modelo MLP ────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
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


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(MLP_CONFIG["seed"])
    np.random.seed(MLP_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Datos ---
    print(f"Cargando datos de entrenamiento (meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS)
    print(f"  Total: {len(df_train):,} filas\n")

    print(f"Cargando datos de validacion (meses {list(VAL_MONTHS)})...")
    df_val = load_months(VAL_MONTHS)
    print(f"  Total: {len(df_val):,} filas\n")

    df_train, df_val = encode_categoricals(df_train, df_val)
    df_train, df_val = add_target_encoding(df_train, df_val, STOP_ID_COL, TARGET)
    df_train = add_derived_features(df_train)
    df_val   = add_derived_features(df_val)
    print(f"Tras filtrado + FE  --  train: {len(df_train):,}  |  val: {len(df_val):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train_np = df_train[feats].values.astype(np.float32)
    y_train_np = df_train[TARGET].values.astype(np.float32)
    X_val_np   = df_val[feats].values.astype(np.float32)
    y_val_np   = df_val[TARGET].values.astype(np.float32)

    # Reemplazar NaN por 0 (la red no tolera NaN)
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_val_np   = np.nan_to_num(X_val_np,   nan=0.0)

    # StandardScaler
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np   = scaler.transform(X_val_np)

    # Tensores y DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_np),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_np),
        torch.from_numpy(y_val_np),
    )
    train_loader = DataLoader(train_ds, batch_size=MLP_CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=MLP_CONFIG["batch_size"], shuffle=False)

    # --- Modelo ---
    model = MLP(
        input_dim=len(feats),
        hidden_layers=MLP_CONFIG["hidden_layers"],
        dropout=MLP_CONFIG["dropout"],
    ).to(device)
    print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_CONFIG["learning_rate"],
        weight_decay=MLP_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.L1Loss()

    # --- W&B ---
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-end",
        config={
            **MLP_CONFIG,
            "target":       TARGET,
            "train_months": list(TRAIN_MONTHS),
            "val_months":   list(VAL_MONTHS),
            "n_features":   len(feats),
            "train_rows":   len(df_train),
            "val_rows":     len(df_val),
            "device":       str(device),
        },
    )

    # --- Training loop ---
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, MLP_CONFIG["max_epochs"] + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * len(X_batch))

        train_mae = sum(train_losses) / len(train_ds)

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item() * len(X_batch))

        val_mae = sum(val_losses) / len(val_ds)
        scheduler.step(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_mae={train_mae:.2f}s  val_mae={val_mae:.2f}s  lr={optimizer.param_groups[0]['lr']:.1e}")

        wandb.log({"epoch": epoch, "train_mae_s": train_mae, "val_mae_s": val_mae, "lr": optimizer.param_groups[0]["lr"]})

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= MLP_CONFIG["patience"]:
                print(f"\n  Early stopping en epoch {epoch}  (best val_mae={best_val_mae:.2f}s)")
                break

    # --- Metricas finales con mejor modelo ---
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_train = model(torch.from_numpy(X_train_np).to(device)).cpu().numpy()
        y_pred_val   = model(torch.from_numpy(X_val_np).to(device)).cpu().numpy()

    metrics_train = compute_metrics(y_train_np, y_pred_train, prefix="train_")
    metrics_val   = compute_metrics(y_val_np,   y_pred_val,   prefix="val_")

    print("\nMetricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Metricas val:");   [print(f"  {k}: {v}") for k, v in metrics_val.items()]

    wandb.log({**metrics_train, **metrics_val})
    wandb.finish()
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()

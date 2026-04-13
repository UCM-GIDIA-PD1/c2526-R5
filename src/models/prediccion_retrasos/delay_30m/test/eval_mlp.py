"""
Entrenamiento MLP (PyTorch) — Prediccion de retraso a 30 min (Objetivo 1)

Predice target_delay_30m = retraso absoluto del tren en los proximos 30 min.
Usa L1 loss (MAE) para ser comparable con el LGBM.

Validacion temporal:
    Train  -> meses 01-09  (enero-septiembre 2025)
    Val    -> meses 10-12  (octubre-diciembre 2025)

Uso:
    uv run python src/models/prediccion_retrasos/delay_30m/train/train_mlp.py

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

# Configuracion

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

TRAIN_YEAR     = 2025
TRAIN_MONTHS   = range(1, 13)
TEST_YEAR      = 2026
TEST_MONTHS    = range(1, 2)
TARGET         = "target_delay_30m"
DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "mlp-delay30m-final-test"

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

# Hiperparametros MLP

MLP_CONFIG = {
    "hidden_layers":  [256, 512, 128],
    "dropout":        0.118,
    "learning_rate":  0.00238,
    "weight_decay":   4.45e-5,
    "batch_size":     8192,
    "max_epochs":     200,
    "patience":       15,
    "seed":           42,
}

SAMPLE_FRAC = 1.0

# Helpers

def load_months(months: range, year: int) -> pd.DataFrame:
    """Descarga y filtra los datos de entrenamiento y validacion desde MinIO."""
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=year, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            total = len(df)
            df = df[df["is_unscheduled"] == False]
            df = df.dropna(subset=[TARGET])
            df = df[df["scheduled_time_to_end"] >= 1800]
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


def encode_categoricals(df_train, df_test):
    """Convierte las columnas categoricas a enteros usando el vocabulario del conjunto de entrenamiento."""
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_test[col]  = df_test[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_test


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


def add_target_encoding(df_train, df_test, col, target):
    """Aplica target encoding sobre una columna usando la media del target por grupo calculada en train."""
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_test[f"{col}_target_enc"]  = df_test[col].map(means).fillna(global_mean)
    return df_train, df_test


def get_features(df):
    """Devuelve la lista de columnas que se usan como features, excluyendo el target y columnas no relevantes."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


def compute_metrics(y_true, y_pred, prefix="") -> dict:
    """Calcula las metricas principales a partir de las predicciones y los valores reales."""
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


# Entrenamiento

def main():
    """Funcion principal que orquesta la carga de datos, el entrenamiento y el registro de resultados."""
    torch.manual_seed(MLP_CONFIG["seed"])
    np.random.seed(MLP_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Datos ---
    print(f"Cargando datos de entrenamiento (año {TRAIN_YEAR}, meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS, TRAIN_YEAR)
    print(f"  Total: {len(df_train):,} filas\n")

    print(f"Cargando datos de test (año {TEST_YEAR}, meses {list(TEST_MONTHS)})...")
    df_test = load_months(TEST_MONTHS, TEST_YEAR)
    print(f"  Total: {len(df_test):,} filas\n")

    df_train, df_test = encode_categoricals(df_train, df_test)
    df_train, df_test = add_target_encoding(df_train, df_test, STOP_ID_COL, TARGET)
    df_train = add_derived_features(df_train)
    df_test  = add_derived_features(df_test)
    print(f"Tras filtrado + FE  --  train: {len(df_train):,}  |  test: {len(df_test):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train_np = df_train[feats].values.astype(np.float32)
    y_train_np = df_train[TARGET].values.astype(np.float32)
    X_test_np  = df_test[feats].values.astype(np.float32)
    y_test_np  = df_test[TARGET].values.astype(np.float32)

    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_np),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_np),
        torch.from_numpy(y_test_np),
    )
    train_loader = DataLoader(train_ds, batch_size=MLP_CONFIG["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=MLP_CONFIG["batch_size"], shuffle=False)

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
        group="prediccion-retrasos-30m",
        config={
            **MLP_CONFIG,
            "target":       TARGET,
            "train_year":   TRAIN_YEAR,
            "train_months": list(TRAIN_MONTHS),
            "test_year":    TEST_YEAR,
            "test_months":  list(TEST_MONTHS),
            "n_features":   len(feats),
            "train_rows":   len(df_train),
            "test_rows":    len(df_test),
            "device":       str(device),
        },
    )

    # --- Training loop ---
    best_test_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, MLP_CONFIG["max_epochs"] + 1):
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

        model.eval()
        test_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                test_losses.append(loss.item() * len(X_batch))

        test_mae = sum(test_losses) / len(test_ds)
        scheduler.step(test_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_mae={train_mae:.2f}s  test_mae={test_mae:.2f}s  lr={optimizer.param_groups[0]['lr']:.1e}")

        wandb.log({"epoch": epoch, "train_mae_s": train_mae, "test_mae_s": test_mae, "lr": optimizer.param_groups[0]["lr"]})

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= MLP_CONFIG["patience"]:
                print(f"\n  Early stopping en epoch {epoch}  (best test_mae={best_test_mae:.2f}s)")
                break

    # --- Metricas finales con mejor modelo ---
    model.load_state_dict(best_state)
    model.eval()

    def predict_in_batches(X_np, batch_size=4096):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                batch = torch.from_numpy(X_np[i:i+batch_size]).to(device)
                preds.append(model(batch).cpu().numpy())
        return np.concatenate(preds)

    y_pred_train = predict_in_batches(X_train_np)
    y_pred_test  = predict_in_batches(X_test_np)

    metrics_train = compute_metrics(y_train_np, y_pred_train, prefix="train_")
    metrics_test  = compute_metrics(y_test_np,  y_pred_test,  prefix="test_")

    print("\nMetricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Metricas test:");   [print(f"  {k}: {v}") for k, v in metrics_test.items()]

    wandb.log({**metrics_train, **metrics_test})
    wandb.finish()
    print("\nEvaluacion completada.")


if __name__ == "__main__":
    main()

"""
WandB Sweep — Búsqueda bayesiana de hiperparámetros para LightGBM (por línea)

Carga el dataset anual de una línea concreta desde:
    pd1/grupo5/aggregations/lines/line=XX/dataset_final.parquet

Uso:
    uv run python -m src.models.prediccion_retrasos.sweep_lgbm_line

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

LINE          = "FX"   
TARGET        = "target_delay_30m"
DATA_PATH     = f"grupo5/aggregations/lines/line={LINE}/dataset_final.parquet"
WANDB_PROJECT = "pd1-c2526-team5"
NUM_RUNS      = 20

TRAIN_MONTHS  = range(1, 10)
VAL_MONTHS    = range(10, 13)
SAMPLE_FRAC   = 1  

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]

EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_mae_s", "goal": "minimize"},
    "parameters": {
        "num_leaves":        {"values": [63, 127, 255, 511]},
        "learning_rate":     {"values": [0.01, 0.03, 0.05, 0.1]},
        "min_child_samples": {"values": [20, 50, 100, 200]},
        "feature_fraction":  {"min": 0.5, "max": 1.0},
        "bagging_fraction":  {"min": 0.5, "max": 1.0},
        "reg_alpha":         {"min": 0.0, "max": 2.0},
        "reg_lambda":        {"min": 0.0, "max": 2.0},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 100},
}

# ── Cargar datos una sola vez ──────────────────────────────────────────────────

def encode_categoricals(df_train, df_val):
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val


def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


print(f"\nCargando dataset de line={LINE}...")
df = download_df_parquet(ACCESS_KEY, SECRET_KEY, DATA_PATH)
df = df[df["is_unscheduled"] == False]
df = df.dropna(subset=[TARGET])
df["date"] = pd.to_datetime(df["date"])

df_train = df[df["date"].dt.month.isin(TRAIN_MONTHS)].copy()
df_val   = df[df["date"].dt.month.isin(VAL_MONTHS)].copy()
del df

if SAMPLE_FRAC < 1.0:
    df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=42)
    df_val   = df_val.sample(frac=SAMPLE_FRAC, random_state=42)

df_train, df_val = encode_categoricals(df_train, df_val)
feats = get_features(df_train)

X_train, y_train = df_train[feats], df_train[TARGET]
X_val,   y_val   = df_val[feats],   df_val[TARGET]

print(f"  train: {len(df_train):,}  |  val: {len(df_val):,}")
print(f"  Features ({len(feats)}): {feats}\n")

# ── Función de entrenamiento del sweep ────────────────────────────────────────

def train_sweep():
    run = wandb.init()
    cfg = run.config

    params = {
        "objective":         "regression_l1",
        "metric":            "mae",
        "learning_rate":     cfg.learning_rate,
        "num_leaves":        cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction":  cfg.feature_fraction,
        "bagging_fraction":  cfg.bagging_fraction,
        "bagging_freq":      5,
        "reg_alpha":         cfg.reg_alpha,
        "reg_lambda":        cfg.reg_lambda,
        "n_jobs":            -1,
        "verbose":           -1,
        "seed":              42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae    = mean_absolute_error(y_val, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
    r2     = r2_score(y_val, y_pred)

    wandb.log({
        "val_mae_s":      round(mae, 2),
        "val_mae_min":    round(mae / 60, 2),
        "val_rmse_s":     round(rmse, 2),
        "val_r2":         round(r2, 4),
        "best_iteration": model.best_iteration,
        "line":           LINE,
    })
    run.finish()


# ── Lanzar el sweep ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    print(f"Sweep creado: {sweep_id}")
    print(f"Lanzando {NUM_RUNS} runs para line={LINE}...\n")
    wandb.agent(sweep_id, function=train_sweep, count=NUM_RUNS)

"""
Optuna — Búsqueda bayesiana de hiperparámetros para LightGBM (menos de 30 min restantes, por línea)

Predice target_delay_end usando solo registros con scheduled_time_to_end < 1800s.
Carga el dataset anual de una línea concreta desde:
    pd1/grupo5/aggregations/lines/line=XX/dataset_final.parquet

Uso:
    uv run python -m src.models.prediccion_retrasos.delay_end.optuna.optuna_lgbm_lines

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

LINE          = "1"
TARGET        = "target_delay_end"
DATA_PATH     = f"grupo5/aggregations/lines/line={LINE}/dataset_final.parquet"
WANDB_PROJECT = "pd1-c2526-team5"
NUM_RUNS      = 50

TRAIN_MONTHS       = range(1, 10)
VAL_MONTHS         = range(10, 13)
SAMPLE_FRAC        = 1
MAX_TIME_REMAINING = 1800

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]

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

STOP_ID_COL = "stop_id"

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    if "hour" in df.columns:
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df


def add_target_encoding(df_train, df_val, col, target):
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val


def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


# ── Precargar datos ────────────────────────────────────────────────────────────

print(f"\nCargando dataset de line={LINE} (scheduled_time_to_end < {MAX_TIME_REMAINING}s)...")
print(f"  Ruta: pd1/{DATA_PATH}")
df = download_df_parquet(ACCESS_KEY, SECRET_KEY, DATA_PATH)
df = df[df["is_unscheduled"] == False]
df = df[df["scheduled_time_to_end"] < MAX_TIME_REMAINING]
df = df.dropna(subset=[TARGET])
df["date"] = pd.to_datetime(df["date"])

df_train = df[df["date"].dt.month.isin(TRAIN_MONTHS)].copy()
df_val   = df[df["date"].dt.month.isin(VAL_MONTHS)].copy()
del df

if SAMPLE_FRAC < 1.0:
    df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=42)
    df_val   = df_val.sample(frac=SAMPLE_FRAC, random_state=42)

df_train, df_val = encode_categoricals(df_train, df_val)
df_train, df_val = add_target_encoding(df_train, df_val, STOP_ID_COL, TARGET)
df_train = add_derived_features(df_train)
df_val   = add_derived_features(df_val)
feats = get_features(df_train)

X_train, y_train = df_train[feats], df_train[TARGET]
X_val,   y_val   = df_val[feats],   df_val[TARGET]

print(f"  train: {len(df_train):,}  |  val: {len(df_val):,}")
print(f"  Features ({len(feats)}): {feats}\n")

# ── Función objetivo de Optuna ─────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    objective_fn = trial.suggest_categorical("objective", ["regression_l1", "huber"])
    params = {
        "objective":         objective_fn,
        "metric":            "mae",
        "num_leaves":        trial.suggest_categorical("num_leaves", [63, 127, 255, 511]),
        "max_depth":         trial.suggest_categorical("max_depth", [-1, 8, 12, 16]),
        "learning_rate":     trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100, 200]),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      5,
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "n_jobs":            -1,
        "verbose":           -1,
        "seed":              42,
    }
    if objective_fn == "huber":
        params["alpha"] = trial.suggest_float("huber_alpha", 3.0, 30.0)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae  = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2   = r2_score(y_val, y_pred)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"optuna-end-line{LINE}-trial{trial.number}",
        group="prediccion-retrasos-end",
        config={**params, "line": LINE, "trial": trial.number},
        reinit=True,
    )
    wandb.log({
        "val_mae_s":      round(mae, 2),
        "val_mae_min":    round(mae / 60, 2),
        "val_rmse_s":     round(rmse, 2),
        "val_r2":         round(r2, 4),
        "best_iteration": model.best_iteration,
    })
    wandb.finish()

    return mae


# ── Lanzar estudio Optuna ──────────────────────────────────────────────────────

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    print(f"Lanzando {NUM_RUNS} trials Optuna para line={LINE} (target={TARGET})...\n")
    study.optimize(objective, n_trials=NUM_RUNS, show_progress_bar=True)

    print("\n── Mejores hiperparámetros ──────────────────────────────────────")
    print(f"  val_mae_s: {study.best_value:.2f}s")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

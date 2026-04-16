"""
Random Search — Búsqueda aleatoria de hiperparámetros para LightGBM (menos de 30 min restantes)

Predice target_delay_end usando solo registros con scheduled_time_to_end < 1800s.

Uso:
    uv run python src/models/prediccion_retrasos/delay_end/random/random_lgbm.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import os
import random
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# Configuracion

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR          = 2025
TRAIN_MONTHS  = range(1, 10)
VAL_MONTHS    = range(10, 13)
TARGET        = "target_delay_end"
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
    "objective":         ["regression_l1", "huber"],
    "num_leaves":        [63, 127, 255, 511],
    "max_depth":         [-1, 8, 12, 16],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [20, 50, 100, 200],
    "feature_fraction":  ("uniform", 0.5, 1.0),
    "bagging_fraction":  ("uniform", 0.5, 1.0),
    "reg_alpha":         ("uniform", 0.0, 2.0),
    "reg_lambda":        ("uniform", 0.0, 2.0),
    "min_split_gain":    ("uniform", 0.0, 1.0),
    "huber_alpha":       ("uniform", 3.0, 30.0),
}


def sample_params(rng: random.Random) -> dict:
    """Muestrea aleatoriamente una combinacion de hiperparametros del espacio definido."""
    def draw(v):
        if isinstance(v, list):
            return rng.choice(v)
        _, lo, hi = v
        return rng.uniform(lo, hi)

    objective = draw(PARAM_SPACE["objective"])
    params = {
        "objective":         objective,
        "metric":            "mae",
        "num_leaves":        draw(PARAM_SPACE["num_leaves"]),
        "max_depth":         draw(PARAM_SPACE["max_depth"]),
        "learning_rate":     draw(PARAM_SPACE["learning_rate"]),
        "min_child_samples": draw(PARAM_SPACE["min_child_samples"]),
        "feature_fraction":  draw(PARAM_SPACE["feature_fraction"]),
        "bagging_fraction":  draw(PARAM_SPACE["bagging_fraction"]),
        "bagging_freq":      5,
        "reg_alpha":         draw(PARAM_SPACE["reg_alpha"]),
        "reg_lambda":        draw(PARAM_SPACE["reg_lambda"]),
        "min_split_gain":    draw(PARAM_SPACE["min_split_gain"]),
        "feature_pre_filter": False,
        "n_jobs":            -1,
        "verbose":           -1,
        "seed":              42,
    }
    if objective == "huber":
        params["alpha"] = draw(PARAM_SPACE["huber_alpha"])
    return params

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
                df = df[df["scheduled_time_to_end"] < 1800]
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

# Main

if __name__ == "__main__":
    print("Precargando datos (se hace una sola vez para todos los trials)...")
    df_train_global, df_val_global = load_data()
    df_train_global, df_val_global = encode_categoricals(df_train_global, df_val_global)
    df_train_global, df_val_global = add_target_encoding(df_train_global, df_val_global, STOP_ID_COL, TARGET)
    df_train_global = add_derived_features(df_train_global)
    df_val_global   = add_derived_features(df_val_global)
    feats   = get_features(df_train_global)
    X_train = df_train_global[feats]
    y_train = df_train_global[TARGET]
    X_val   = df_val_global[feats]
    y_val   = df_val_global[TARGET]
    print(f"Features ({len(feats)}): {feats}\n")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    rng = random.Random(SEED)
    best_mae = float("inf")
    best_params = None

    print(f"Lanzando {NUM_RUNS} trials random search (target={TARGET}, scheduled_time_to_end < 1800s)...\n")
    for trial in range(NUM_RUNS):
        params = sample_params(rng)

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

        if mae < best_mae:
            best_mae = mae
            best_params = params.copy()

        wandb.init(
            project=WANDB_PROJECT,
            name=f"random-end-trial{trial}",
            group="prediccion-retrasos-end",
            config={**params, "trial": trial},
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

        print(f"  trial {trial:02d}  mae={mae:.2f}s  best={best_mae:.2f}s")

    print("\n── Mejores hiperparámetros ──────────────────────────────────────")
    print(f"  val_mae_s: {best_mae:.2f}s")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

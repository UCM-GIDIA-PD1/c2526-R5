"""
Entrenamiento LightGBM — Predicción de retraso al final del viaje (Objetivo 2)

Predice target_delay_end = retraso absoluto del tren al terminar el viaje.
Solo usa registros con scheduled_time_to_end < 1800s (menos de 30 min restantes).

Validación temporal:
    Train  → meses 01–09  (enero–septiembre 2025)
    Val    → meses 10–12  (octubre–diciembre 2025)

Uso:
    uv run python -m src.models.prediccion_retrasos.delay_end.train.train_lgbm

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
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


ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR           = 2025
TRAIN_MONTHS   = range(1, 10)
VAL_MONTHS     = range(10, 13)
TARGET         = "target_delay_end"
DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"
MODEL_PATH_OUT = "grupo5/models/lgbm_stop_delay_end.txt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "lgbm-stop-delay-end"


EXCLUDE_COLS = {
    # IDs y metadatos
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid",
    "is_unscheduled",
    # Todos los targets
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    # Deltas (leakage)
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    # Leakage
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    # Redundantes
    "delay_minutes", "scheduled_time", "actual_time",
}


CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"


LGBM_PARAMS = {
    "objective":         "regression_l1",
    "metric":            "mae",
    "learning_rate":     0.1,
    "num_leaves":        511,
    "max_depth":         -1,
    "min_child_samples": 200,
    "min_split_gain":    0.24014,
    "feature_fraction":  0.91404,
    "bagging_fraction":  0.6773,
    "bagging_freq":      5,
    "reg_alpha":         1.49806,
    "reg_lambda":        0.62436,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}
NUM_BOOST_ROUND = 10000
EARLY_STOPPING  = 100
SAMPLE_FRAC = 1.0


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
            print(f"  ✓ month={month:02d}  {total:>10,} filas  →  {len(df):>10,} tras filtrado  ~{mb:.0f} MB")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ month={month:02d}  no encontrado ({e})")
    return pd.concat(dfs, ignore_index=True)


def encode_categoricals(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_target_encoding(df_train: pd.DataFrame, df_val: pd.DataFrame,
                        col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val


def get_features(df: pd.DataFrame) -> list[str]:
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


def main():
    print(f"\nCargando datos de entrenamiento (meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS)
    print(f"  Total: {len(df_train):,} filas\n")

    print(f"Cargando datos de validación (meses {list(VAL_MONTHS)})...")
    df_val = load_months(VAL_MONTHS)
    print(f"  Total: {len(df_val):,} filas\n")

    df_train, df_val = encode_categoricals(df_train, df_val)
    df_train, df_val = add_target_encoding(df_train, df_val, STOP_ID_COL, TARGET)

    df_train = add_derived_features(df_train)
    df_val   = add_derived_features(df_val)
    print(f"Tras filtrado + FE  —  train: {len(df_train):,}  |  val: {len(df_val):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train, y_train = df_train[feats], df_train[TARGET]
    X_val,   y_val   = df_val[feats],   df_val[TARGET]

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-end",
        config={
            **LGBM_PARAMS,
            "target":       TARGET,
            "train_months": list(TRAIN_MONTHS),
            "val_months":   list(VAL_MONTHS),
            "n_features":   len(feats),
            "train_rows":   len(df_train),
            "val_rows":     len(df_val),
        }
    )

    print(f"Entrenando LightGBM (target={TARGET})...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    print(f"\nMejor iteración: {model.best_iteration}")

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_val   = model.predict(X_val,   num_iteration=model.best_iteration)

    metrics_train = compute_metrics(y_train, y_pred_train, prefix="train_")
    metrics_val   = compute_metrics(y_val,   y_pred_val,   prefix="val_")

    print("\nMétricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas val:");   [print(f"  {k}: {v}") for k, v in metrics_val.items()]

    wandb.log({**metrics_train, **metrics_val, "best_iteration": model.best_iteration})

    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 features:\n{importance.head(15).to_string(index=False)}")
    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(20))})

    wandb.finish()
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()

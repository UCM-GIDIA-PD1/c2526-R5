"""
Evaluación final LightGBM — Predicción de retraso al final del viaje (Objetivo 2)

Predice target_delay_end = retraso absoluto del tren al terminar el viaje.
Solo usa registros con scheduled_time_to_end < 1800s (menos de 30 min restantes).

Partición final:
    Train  → enero–diciembre 2025  (todo el año)
    Test   → enero 2026

Uso:
    uv run python -m src.models.prediccion_retrasos.delay_end.test.eval_lgbm

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

TRAIN_YEAR     = 2025
TRAIN_MONTHS   = range(1, 13)
TEST_YEAR      = 2026
TEST_MONTHS    = range(1, 2)
TARGET         = "target_delay_end"
DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"
MODEL_PATH_OUT = "grupo5/models/lgbm_stop_delay_end_final.txt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "lgbm-delay-end-final-test"


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
    "min_split_gain":    0.4288084220895492,
    "feature_fraction":  0.924842417292814,
    "bagging_fraction":  0.7040148314775926,
    "bagging_freq":      5,
    "reg_alpha":         1.5122983878652498,
    "reg_lambda":        0.9403505804168345,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING  = 100
SAMPLE_FRAC = 1.0


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


def encode_categoricals(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica target encoding sobre una columna usando la media del target por grupo calculada en train."""
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_test[f"{col}_target_enc"]  = df_test[col].map(means).fillna(global_mean)
    return df_train, df_test


def get_features(df: pd.DataFrame) -> list[str]:
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


def main():
    """Funcion principal que orquesta la carga de datos, el entrenamiento y el registro de resultados."""
    print(f"\nCargando datos de entrenamiento (año {TRAIN_YEAR}, meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS, TRAIN_YEAR)
    print(f"  Total: {len(df_train):,} filas\n")

    print(f"Cargando datos de test (año {TEST_YEAR}, meses {list(TEST_MONTHS)})...")
    df_test = load_months(TEST_MONTHS, TEST_YEAR)
    print(f"  Total: {len(df_test):,} filas\n")

    df_train, df_test = encode_categoricals(df_train, df_test)
    df_train, df_test = add_target_encoding(df_train, df_test, STOP_ID_COL, TARGET)

    df_train = add_derived_features(df_train)
    df_test  = add_derived_features(df_test)
    print(f"Tras filtrado + FE  —  train: {len(df_train):,}  |  test: {len(df_test):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train, y_train = df_train[feats], df_train[TARGET]
    X_test,  y_test  = df_test[feats],  df_test[TARGET]

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-end",
        config={
            **LGBM_PARAMS,
            "target":       TARGET,
            "train_year":   TRAIN_YEAR,
            "train_months": list(TRAIN_MONTHS),
            "test_year":    TEST_YEAR,
            "test_months":  list(TEST_MONTHS),
            "n_features":   len(feats),
            "train_rows":   len(df_train),
            "test_rows":    len(df_test),
        }
    )

    print(f"Entrenando LightGBM (target={TARGET})...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test  = lgb.Dataset(X_test,  label=y_test, reference=lgb_train)

    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_test],
        valid_names=["train", "test"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    print(f"\nMejor iteración: {model.best_iteration}")

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_test  = model.predict(X_test,  num_iteration=model.best_iteration)

    metrics_train = compute_metrics(y_train, y_pred_train, prefix="train_")
    metrics_test  = compute_metrics(y_test,  y_pred_test,  prefix="test_")

    print("\nMétricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas test:");   [print(f"  {k}: {v}") for k, v in metrics_test.items()]

    wandb.log({**metrics_train, **metrics_test, "best_iteration": model.best_iteration})

    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 features:\n{importance.head(15).to_string(index=False)}")
    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(20))})

    wandb.finish()
    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()

"""
Evaluación final LightGBM — Predicción de retraso en parada (Objetivo 1)

Predice target_delay_30m = retraso absoluto del tren en los próximos 30 min.

Partición final (Entrega 4):
    Train  → julio–diciembre 2025 + enero 2026  (sliding window, 7 meses)
    Test   → febrero 2026  (datos nuevos)
    Pesos  → exponential (elegido por menor gap train/test y mejor MAE test)

Configuración óptima hallada por window_weight_search:
    window=desde_jul25, weight=exponential, n_months=7
    test_mae=137.06s, train_mae=119.2s (gap=17.8s)

Iteraciones fijas (best_iteration del run anterior): 4260

Uso:
    uv run python src/models/prediccion_retrasos/delay_30m/test/eval_lgbm.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
"""

import gc
import json
import os
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")



ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

TRAIN_YEAR_2025   = 2025
TRAIN_MONTHS_2025 = range(7, 13)   # jul-dic 2025 (sliding window)
TRAIN_YEAR_2026   = 2026
TRAIN_MONTHS_2026 = range(1, 2)   # enero 2026
TEST_YEAR         = 2026
TEST_MONTHS       = range(2, 3)   # febrero 2026
TARGET            = "target_delay_30m"
DATA_TEMPLATE     = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "lgbm-delay30m-entrega4-feb-sliding"

# Días excluidos del test por anomalía meteorológica (nevada)
EXCLUDE_TEST_DATES = ["2026-02-23", "2026-02-24"]


EXCLUDE_COLS = {
    # IDs y metadatos
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid",
    "is_unscheduled",
    # Todos los targets (el nuestro se excluye automáticamente al ser TARGET)
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    # Deltas (todos, son targets futuros = leakage)
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    # Leakage
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    # Redundantes: delay_minutes = delay_seconds/60, scheduled/actual_time redundan con hour_sin/cos
    "delay_minutes", "scheduled_time", "actual_time",
    "station_delay_10m", "station_delay_20m", "station_delay_30m"
}


CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"


LGBM_PARAMS = {
    "objective":         "regression_l1",
    "metric":            "mae",
    "learning_rate":     0.05,
    "num_leaves":        511,
    "max_depth":         16,
    "min_child_samples": 100,
    "min_split_gain":    0.37042771510661165,
    "feature_fraction":  0.7426288737567357,
    "bagging_fraction":  0.8165370010747616,
    "bagging_freq":      5,
    "reg_alpha":         1.5346393797283635,
    "reg_lambda":        1.2926631392622208,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}
NUM_BOOST_ROUND = 4260   # best_iteration del run anterior (Entrega 3)
SAMPLE_FRAC   = 1.0
WEIGHT_SCHEME = "exponential"   # menor gap train/test (17.8s) y mejor MAE test (137.06s)


def build_weights(month_sizes: list[int], scheme: str) -> np.ndarray:
    n = len(month_sizes)
    if scheme == "linear":
        w = np.arange(1, n + 1, dtype=float)
    elif scheme == "exponential":
        lam = np.log(10) / max(n - 1, 1)
        w = np.exp(lam * np.arange(n))
    else:
        w = np.ones(n)
    w = w / w.mean()
    return np.concatenate([np.full(s, wi) for s, wi in zip(month_sizes, w)]).astype(np.float32)


def load_months(months: range, year: int) -> tuple[pd.DataFrame, list[int]]:
    """Descarga, filtra y concatena los parquets mensuales indicados.
    Filtra is_unscheduled y nulls en target MES A MES para no acumular RAM innecesaria.
    """
    dfs: list[pd.DataFrame] = []
    sizes: list[int] = []
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
            print(f"  ✓ month={month:02d}  {total:>10,} filas  →  {len(df):>10,} tras filtrado  ~{mb:.0f} MB")
            dfs.append(df)
            sizes.append(len(df))
        except Exception as e:
            print(f"  ✗ month={month:02d}  no encontrado ({e})")
    return pd.concat(dfs, ignore_index=True), sizes


def encode_categoricals(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Codifica las columnas categóricas como enteros usando el vocabulario del train.
    Los valores desconocidos en test se mapean a -1.
    Devuelve también el diccionario de vocabularios para serializar con el modelo.
    """
    vocabs: dict[str, dict] = {}
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_test[col]  = df_test[col].astype(str).map(vocab).fillna(-1).astype(int)
        vocabs[col] = vocab
    return df_train, df_test, vocabs




def add_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    """Target encoding: media del target por categoría (train→test con fallback a media global).
    Devuelve también el mapeo y la media global para serializar con el modelo.
    """
    means = df_train.groupby(col)[target].mean()
    global_mean = float(df_train[target].mean())
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_test[f"{col}_target_enc"]  = df_test[col].map(means).fillna(global_mean)
    return df_train, df_test, means.to_dict(), global_mean


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


def get_features(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas del df que son features (todo menos EXCLUDE_COLS y TARGET)."""
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


def make_progress_callback(total_rounds: int, update_every: int = 50):
    """Muestra una barra de progreso simple durante el entrenamiento de LightGBM."""
    start_time = time.time()
    bar_width = 30

    def _callback(env):
        current = env.iteration + 1
        if current == 1 or current % update_every == 0 or current == total_rounds:
            progress = min(current / total_rounds, 1.0)
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            elapsed = time.time() - start_time
            speed = current / elapsed if elapsed > 0 else 0.0
            eta = (total_rounds - current) / speed if speed > 0 else 0.0

            print(
                f"\rTrain [{bar}] {current}/{total_rounds} ({progress * 100:5.1f}%) "
                f"| elapsed {elapsed:6.1f}s | eta {eta:6.1f}s",
                end="",
                flush=True,
            )
            if current == total_rounds:
                print()

    _callback.order = 10
    _callback.before_iteration = False
    return _callback


def main():
    """Funcion principal que orquesta la carga de datos, el entrenamiento y el registro de resultados."""
    print(f"\nCargando datos de entrenamiento (año {TRAIN_YEAR_2025}, meses {list(TRAIN_MONTHS_2025)})...")
    df_train_2025, sizes_2025 = load_months(TRAIN_MONTHS_2025, TRAIN_YEAR_2025)
    print(f"  Total 2025: {len(df_train_2025):,} filas\n")

    print(f"Cargando datos de entrenamiento (año {TRAIN_YEAR_2026}, meses {list(TRAIN_MONTHS_2026)})...")
    df_train_2026, sizes_2026 = load_months(TRAIN_MONTHS_2026, TRAIN_YEAR_2026)
    print(f"  Total ene-2026: {len(df_train_2026):,} filas\n")

    df_train = pd.concat([df_train_2025, df_train_2026], ignore_index=True)
    month_sizes = sizes_2025 + sizes_2026
    del df_train_2025, df_train_2026
    gc.collect()
    print(f"  Total train combinado: {len(df_train):,} filas\n")

    print(f"Cargando datos de test (año {TEST_YEAR}, meses {list(TEST_MONTHS)})...")
    df_test, _ = load_months(TEST_MONTHS, TEST_YEAR)
    df_test = df_test[~df_test["date"].astype(str).isin(EXCLUDE_TEST_DATES)]
    print(f"  Total: {len(df_test):,} filas (excluidos {EXCLUDE_TEST_DATES})\n")

    df_train, df_test, label_vocabs = encode_categoricals(df_train, df_test)
    df_train, df_test, stop_means, global_mean = add_target_encoding(df_train, df_test, STOP_ID_COL, TARGET)

    df_train = add_derived_features(df_train)
    df_test  = add_derived_features(df_test)
    print(f"Tras filtrado + FE  —  train: {len(df_train):,}  |  test: {len(df_test):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train = df_train[feats]
    y_train = df_train[TARGET]
    X_test  = df_test[feats]
    y_test  = df_test[TARGET]
    n_train = len(df_train)
    n_test  = len(df_test)
    del df_train, df_test
    gc.collect()

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-30m",
        config={
            **LGBM_PARAMS,
            "target":          TARGET,
            "train_2025":      list(TRAIN_MONTHS_2025),
            "train_2026":      list(TRAIN_MONTHS_2026),
            "test_year":       TEST_YEAR,
            "test_months":     list(TEST_MONTHS),
            "num_boost_round": NUM_BOOST_ROUND,
            "weight_scheme":   WEIGHT_SCHEME,
            "n_features":      len(feats),
            "train_rows":      n_train,
            "test_rows":       n_test,
        }
    )

    weights = build_weights(month_sizes, WEIGHT_SCHEME)
    print(f"Construyendo lgb.Dataset (pesos={WEIGHT_SCHEME})...")
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=True)
    print(f"lgb.Dataset listo. Iniciando entrenamiento ({NUM_BOOST_ROUND} iteraciones, log cada 500)...\n")

    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            make_progress_callback(NUM_BOOST_ROUND, update_every=25),
            lgb.log_evaluation(500),
        ],
    )

    print(f"\nIteraciones completadas: {NUM_BOOST_ROUND}")

    print("Generando predicciones sobre train...")
    y_pred_train = model.predict(X_train)
    print("Generando predicciones sobre test...")
    y_pred_test  = model.predict(X_test)

    metrics_train = compute_metrics(y_train, y_pred_train, prefix="train_")
    metrics_test  = compute_metrics(y_test,  y_pred_test,  prefix="test_")

    print("\nMétricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas test:");   [print(f"  {k}: {v}") for k, v in metrics_test.items()]

    wandb.log({**metrics_train, **metrics_test, "num_boost_round": NUM_BOOST_ROUND})

    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 features:\n{importance.head(15).to_string(index=False)}")
    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(20))})

    model_filename = "lgbm_delay_30m.joblib"
    preprocessing_filename = "preprocessing_delay_30m.json"
    joblib.dump(model, model_filename)
    preprocessing = {
        "label_encoders": label_vocabs,
        "target_encoder_stop_id": stop_means,
        "target_encoder_global_mean": global_mean,
        "derived_features": ["delay_velocity", "delay_acceleration", "delay_x_stops_remaining", "delay_ratio"],
        "target": TARGET,
    }
    with open(preprocessing_filename, "w") as f:
        json.dump(preprocessing, f)

    artifact = wandb.Artifact(
        name="lgbm-delay-30m",
        type="model",
        description="LightGBM delay_30m — train 2025+ene2026, test feb2026, iter=4260, exponential weights",
    )
    artifact.add_file(model_filename)
    artifact.add_file(preprocessing_filename)
    wandb.log_artifact(artifact)
    os.remove(model_filename)
    os.remove(preprocessing_filename)
    print(f"\nModelo y preprocessing subidos como artifact wandb: {model_filename}, {preprocessing_filename}")

    wandb.finish()
    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()

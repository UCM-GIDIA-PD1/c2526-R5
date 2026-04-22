"""
Entrenamiento final de LightGBM para predicción de delta_delay.

Entrena con todo 2025 y evalúa sobre enero-febrero 2026. Sube el modelo
en formato .joblib a WandB Artifacts. No registra métricas de evaluación
en WandB, solo el artefacto del modelo.

Cambia TARGET_DELTA para entrenar el horizonte deseado:
  - "delta_delay_10m"
  - "delta_delay_20m"
  - "delta_delay_30m"

Uso:
    uv run python src/models/prediccion_retrasos/delta/entrenamiento_final_delta.py
"""

import gc
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")

TARGET_DELTA = "delta_delay_30m"   # elegir entre "delta_delay_10m", "delta_delay_20m" o "delta_delay_30m"
TARGET       = "target_mejora"

SEED         = 42

DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

TRAIN_YEAR   = 2025
TRAIN_MONTHS = range(1, 13)
TEST_YEAR    = 2026
TEST_MONTHS  = [1, 2]

WANDB_PROJECT = "pd1-c2526-team5"

EXCLUDE_COLS = {
    "date", "match_key", "merge_time", "timestamp_start",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m", "delta_delay_20m", "delta_delay_30m",
    "delta_delay_45m", "delta_delay_60m", "delta_delay_end",
    "seconds_to_next_alert", "alert_in_next_15m", "alert_in_next_30m",
}

CAT_FEATURES = [
    "route_id", "direction", "category", "tipo_referente",
    "stop_id", "is_weekend", "is_unscheduled", "temp_extreme",
    "afecta_previo", "afecta_durante", "afecta_despues",
    "is_alert_just_published", "has_alert",
]

# ── Hiperparámetros por horizonte ──────────────────────────────────────────────

LGBM_PARAMS_BY_DELTA = {
    "delta_delay_10m": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.06143017453510817,
            "num_leaves":         132,
            "max_depth":          8,
            "min_child_samples":  77,
            "feature_fraction":   0.8827511186458752,
            "bagging_fraction":   0.8579258048600183,
            "bagging_freq":       3,
            "reg_alpha":          2.1293501537723296,
            "reg_lambda":         2.220888403065774,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 2000,
    },
    "delta_delay_20m": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.07238450772957448,
            "num_leaves":         189,
            "max_depth":          8,
            "min_child_samples":  233,
            "feature_fraction":   0.964426899264662,
            "bagging_fraction":   0.9944269030513614,
            "bagging_freq":       3,
            "reg_alpha":          3.5018270288466766,
            "reg_lambda":         0.16641021767709363,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 4000,
    },
    "delta_delay_30m": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.0683508553333336073,
            "num_leaves":         168,
            "max_depth":          9,
            "min_child_samples":  160,
            "feature_fraction":   0.8725927178151603,
            "bagging_fraction":   0.6698125996673533,
            "bagging_freq":       2,
            "reg_alpha":          2.818457566956284,
            "reg_lambda":         1.772491283015965,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 3500,
    },
}

# ── Funciones auxiliares ───────────────────────────────────────────────────────

def load_months(year: int, months) -> pd.DataFrame:
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=year, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            print(f"  year={year} month={month:02d}  {len(df):>10,} filas")
            dfs.append(df)
        except Exception as e:
            print(f"  year={year} month={month:02d}  no encontrado ({e})")
    if not dfs:
        raise RuntimeError(f"No se encontraron datos para year={year}, months={list(months)}")
    return pd.concat(dfs, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delay_change"]      = df["delay_seconds"] - df["lagged_delay_1"]
    df["delay_change_prev"] = df["lagged_delay_1"] - df["lagged_delay_2"]
    df["delay_accel"]       = df["delay_change"] - df["delay_change_prev"]
    df["delay_vs_route"]    = df["delay_seconds"] - df["route_rolling_delay"]
    df["delay_vs_station"]  = df["delay_seconds"] - df["station_delay_10m"].fillna(df["delay_seconds"])
    df["station_trend"]     = df["station_delay_10m"] - df["station_delay_20m"]
    df["delay_time_ratio"]  = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1.0)
    df["has_alert"]         = (df["n_eventos_afectando"] > 0).astype(np.int8)
    df["alert_impact"]      = df["afecta_previo"] + df["afecta_durante"] + df["afecta_despues"]
    return df


def encode_categoricals(df_train: pd.DataFrame, *others: pd.DataFrame):
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        for df in others:
            if col in df.columns:
                df[col] = df[col].astype(str).map(vocab).fillna(-1).astype(int)
    return (df_train,) + others


def get_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    for col in ["merge_time", "timestamp_start"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    cfg             = LGBM_PARAMS_BY_DELTA[TARGET_DELTA]
    lgbm_params     = cfg["params"]
    num_boost_round = cfg["num_boost_round"]

    print(f"\n{'='*60}")
    print(f"  Entrenamiento final LightGBM  |  TARGET: {TARGET_DELTA}")
    print(f"{'='*60}")

    # ── Carga ─────────────────────────────────────────────────────────────────
    print(f"\nCargando datos de entrenamiento (2025, meses 1-12)...")
    df_train = load_months(TRAIN_YEAR, TRAIN_MONTHS)
    print(f"  Total 2025: {len(df_train):,} filas")

    # ── Preprocesado ──────────────────────────────────────────────────────────
    df_train = parse_dates(df_train)
    df_train = add_features(df_train)
    df_train = df_train.dropna(subset=[TARGET_DELTA])
    df_train[TARGET] = (df_train[TARGET_DELTA] < 0).astype(np.int8)

    df_train, = encode_categoricals(df_train)
    feats = get_features(df_train)

    class_ratio = df_train[TARGET].mean()
    print(f"\n  Features: {len(feats)}")
    print(f"  Clase 1 (mejora): {class_ratio*100:.1f}%")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    params = {**lgbm_params, "is_unbalance": class_ratio < 0.3 or class_ratio > 0.7}

    print(f"\n  Entrenando con num_boost_round={num_boost_round}...")
    lgb_ds      = lgb.Dataset(df_train[feats], label=df_train[TARGET])
    model_final = lgb.train(params, lgb_ds, num_boost_round=num_boost_round)
    print("  Entrenamiento completado.")
    del lgb_ds, df_train
    gc.collect()

    # ── Subir a WandB Artifacts ───────────────────────────────────────────────
    print("\n  Subiendo a WandB Artifacts...")
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"entrenamiento_final_delta_{TARGET_DELTA}",
        config={
            "target_delta":    TARGET_DELTA,
            "train_year":      TRAIN_YEAR,
            "train_months":    list(TRAIN_MONTHS),
            "num_boost_round": num_boost_round,
            "n_features":      len(feats),
            **{f"hp_{k}": v for k, v in lgbm_params.items() if k not in ("verbose", "seed")},
        },
    )

    model_name = f"lgbm_delta_{TARGET_DELTA}.joblib"
    with tempfile.TemporaryDirectory() as tmp_dir:
        ruta_modelo = Path(tmp_dir) / model_name
        joblib.dump(
            {
                "model":           model_final,
                "features":        feats,
                "target_delta":    TARGET_DELTA,
                "num_boost_round": num_boost_round,
            },
            ruta_modelo,
        )

        artifact = wandb.Artifact(
            name=f"lgbm-delta-{TARGET_DELTA}",
            type="model",
            description=f"LightGBM final entrenado sobre 2025 completo para {TARGET_DELTA}",
            metadata={
                "target_delta":    TARGET_DELTA,
                "train_year":      TRAIN_YEAR,
                "num_boost_round": num_boost_round,
                "n_features":      len(feats),
            },
        )
        artifact.add_file(str(ruta_modelo))
        wandb.log_artifact(artifact)

    run.finish()
    print(f"  Artefacto '{artifact.name}' subido a WandB.")
    print(f"\nListo.")


if __name__ == "__main__":
    main()

"""
Clasificación binaria de mejora de retraso con LightGBM — subida a WandB

Carga el dataset mensual desde MinIO, entrena un modelo binario que predice si
el retraso mejorará (delta_delay < 0) y registra todas las métricas en WandB.

Uso:
    uv run python -m src.models.prediccion_retrasos.lgbm.binary_classification_delta

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import gc
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR          = 2025
MONTHS        = range(1, 2)
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

TARGET_DELTA  = "delta_delay_10m"   # cambiar para otro horizonte: _20m, _30m, etc.
TARGET        = "target_mejora"

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
SEED          = 42

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

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.05,
    "num_leaves":        127,
    "min_child_samples": 100,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              SEED,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING  = 50

# ── Funciones auxiliares ───────────────────────────────────────────────────────

def load_months(months: range) -> pd.DataFrame:
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=YEAR, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            total = len(df)
            df = df.dropna(subset=[TARGET_DELTA])
            mb = df.memory_usage(deep=True).sum() / 1e6
            print(f"  month={month:02d}  {total:>10,} filas  ->  {len(df):>10,} tras filtrado  ~{mb:.0f} MB")
            dfs.append(df)
        except Exception as e:
            print(f"  month={month:02d}  no encontrado ({e})")
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


def encode_categoricals(df_train, df_val, df_test):
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
        df_test[col]  = df_test[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val, df_test


def get_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


def compute_metrics(y_true, y_prob, y_pred, prefix="") -> dict:
    return {
        f"{prefix}roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        f"{prefix}pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        f"{prefix}accuracy":  round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ── Entrenamiento ──────────────────────────────────────────────────────────────

def train():
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"binary_delta_{TARGET_DELTA}",
        config={
            "target_delta":    TARGET_DELTA,
            "year":            YEAR,
            "months":          list(MONTHS),
            "train_ratio":     TRAIN_RATIO,
            "val_ratio":       VAL_RATIO,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping":  EARLY_STOPPING,
            **LGBM_PARAMS,
        },
    )

    # Carga
    print(f"\nCargando datos (meses {list(MONTHS)})...")
    df = load_months(MONTHS)
    print(f"  Total: {len(df):,} filas\n")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    for col in ["merge_time", "timestamp_start"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Feature engineering y target binario
    df = add_features(df)
    df[TARGET] = (df[TARGET_DELTA] < 0).astype(np.int8)

    counts = df[TARGET].value_counts().sort_index()
    print(f"Clase 0 (no mejora): {counts[0]:,}  |  Clase 1 (mejora): {counts[1]:,}")

    # Split temporal
    sort_col = "merge_time" if df["merge_time"].notna().sum() > 0 else "date"
    df = df.sort_values(sort_col).reset_index(drop=True)
    n       = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_val   = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:i_train].copy()
    df_val   = df.iloc[i_train:i_val].copy()
    df_test  = df.iloc[i_val:].copy()
    del df
    gc.collect()

    df_train, df_val, df_test = encode_categoricals(df_train, df_val, df_test)

    feats = get_features(df_train)
    print(f"Features ({len(feats)}): {feats}\n")

    X_train, y_train = df_train[feats], df_train[TARGET]
    X_val,   y_val   = df_val[feats],   df_val[TARGET]
    X_test,  y_test  = df_test[feats],  df_test[TARGET]

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"  {name:>5s}: {len(X):>10,} filas  |  clase 1: {y.mean()*100:.1f}%")

    wandb.log({
        "n_train":              len(X_train),
        "n_val":                len(X_val),
        "n_test":               len(X_test),
        "n_features":           len(feats),
        "class1_ratio_train":   round(float(y_train.mean()), 4),
    })

    # Entrenamiento
    class_ratio = y_train.mean()
    params = {**LGBM_PARAMS, "is_unbalance": class_ratio < 0.3 or class_ratio > 0.7}
    print(f"\nClase 1 en train: {class_ratio:.3f}  ->  is_unbalance={params['is_unbalance']}")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    print(f"\nEntrenando LightGBM (target={TARGET_DELTA})...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    print(f"\nMejor iteración: {model.best_iteration}")

    # Predicciones
    y_prob_val  = model.predict(X_val,  num_iteration=model.best_iteration)
    y_prob_test = model.predict(X_test, num_iteration=model.best_iteration)

    y_pred_val  = (y_prob_val  >= 0.5).astype(int)
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    m_val  = compute_metrics(y_val,  y_prob_val,  y_pred_val,  prefix="val_")
    m_test = compute_metrics(y_test, y_prob_test, y_pred_test, prefix="test_")

    # Umbral óptimo por F1 en validación
    thresholds = np.arange(0.20, 0.80, 0.01)
    f1s = [f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx       = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    print(f"\nMejor umbral (val): {best_threshold:.2f}  ->  F1={f1s[best_idx]:.4f}")

    y_pred_test_opt = (y_prob_test >= best_threshold).astype(int)
    m_test_opt = compute_metrics(y_test, y_prob_test, y_pred_test_opt, prefix="test_opt_")

    # Métricas adicionales con umbral óptimo
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_opt).ravel()
    spec    = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    rec_val = recall_score(y_test, y_pred_test_opt, zero_division=0)
    bal_acc = (rec_val + spec) / 2 if not np.isnan(spec) else float("nan")

    # Log a WandB
    wandb.log({
        **m_val,
        **m_test,
        **m_test_opt,
        "best_iteration":          model.best_iteration,
        "best_threshold":          round(best_threshold, 2),
        "val_best_f1":             round(f1s[best_idx], 4),
        "target_delta":            TARGET_DELTA,
        "test_tp":                 int(tp),
        "test_fp":                 int(fp),
        "test_fn":                 int(fn),
        "test_tn":                 int(tn),
        "test_opt_specificity":    round(spec, 4) if not np.isnan(spec) else None,
        "test_opt_fpr":            round(fp / (fp + tn), 4) if (fp + tn) > 0 else None,
        "test_opt_fnr":            round(fn / (fn + tp), 4) if (fn + tp) > 0 else None,
        "test_opt_balanced_acc":   round(bal_acc, 4) if not np.isnan(bal_acc) else None,
    })

    # Feature importance
    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance["pct"] = (importance["importance"] / importance["importance"].sum() * 100).round(2)
    print(f"\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(30))})

    run.finish()


# ── Punto de entrada ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()

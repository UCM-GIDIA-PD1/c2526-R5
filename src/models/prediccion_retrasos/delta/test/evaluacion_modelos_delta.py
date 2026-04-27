"""
Evaluación final de los mejores modelos de clasificación binaria del delta_delay.
 
Reentrena los mejores modelos (LightGBM con Optuna y Random Forest con Optuna)
con train+val usando los mejores hiperparámetros encontrados en la fase de
optimización, y los evalúa una única vez sobre el conjunto de TEST.
 
Incluye:
  - La métrica principal (ROC-AUC)
  - Métricas complementarias (F1, Precision, Recall, Accuracy, PR-AUC,
    Specificity, Balanced Accuracy, FPR, FNR)
  - Análisis por segmentos relevantes (has_alert, is_weekend, delay_bucket)
  - Importancia de variables de entrada (feature importance)
  - Curvas PR y matrices de confusión en WandB

Uso:
    uv run python src/models/prediccion_retrasos/delta/evaluacion_modelos_delta.py
"""

import gc
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")

YEAR          = 2025
MONTHS        = range(1, 13)
SAMPLE_FRAC   = 0.1
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

TARGET_DELTA  = "delta_delay_end"   # cambiar según el horizonte evaluado
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
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
    "delay_vs_station", "station_trend",
}

CAT_FEATURES = [
    "route_id", "direction", "category", "tipo_referente",
    "stop_id", "is_weekend", "is_unscheduled", "temp_extreme",
    "afecta_previo", "afecta_durante", "afecta_despues",
    "is_alert_just_published", "has_alert",
]

# ── Mejores hiperparámetros de LightGBM encontrados por Optuna, por horizonte ──
# Los valores se seleccionan automáticamente en función de TARGET_DELTA.
# Fuente: panel de WandB -> Config parameters del mejor trial de cada run.

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
        "early_stopping":  50,
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
        "early_stopping":  75,
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
        "early_stopping":  125,
    },
    "delta_delay_45m": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.0642033097541968,
            "num_leaves":         140,
            "max_depth":          9,
            "min_child_samples":  115,
            "feature_fraction":   0.8472819498898617,
            "bagging_fraction":   0.9040462341938308,
            "bagging_freq":       7,
            "reg_alpha":          4.360431467418307,
            "reg_lambda":         2.476302332166165,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 4000,
        "early_stopping":  150,
    },
    "delta_delay_60m": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.08015448264071032,
            "num_leaves":         123,
            "max_depth":          9,
            "min_child_samples":  94,
            "feature_fraction":   0.91,
            "bagging_fraction":   0.85,
            "bagging_freq":       10,
            "reg_alpha":          4.07,
            "reg_lambda":         1.55,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 3500,
        "early_stopping":  100,
    },
    "delta_delay_end": {
        "params": {
            "objective":          "binary",
            "metric":             "auc",
            "boosting_type":      "gbdt",
            "extra_trees":        False,
            "learning_rate":      0.04900189920507535,
            "num_leaves":         135,
            "max_depth":          9,
            "min_child_samples":  112,
            "feature_fraction":   0.93,
            "bagging_fraction":   0.83,
            "bagging_freq":       7,
            "reg_alpha":          3.26,
            "reg_lambda":         0.91,
            "n_jobs":             -1,
            "verbose":            -1,
            "seed":               SEED,
            "feature_pre_filter": False,
        },
        "num_boost_round": 3000,
        "early_stopping":  150,
    },
}

# Selección automática de hiperparámetros según el horizonte activo
_lgbm_cfg          = LGBM_PARAMS_BY_DELTA[TARGET_DELTA]
BEST_LGBM_PARAMS   = _lgbm_cfg["params"]
BEST_LGBM_BOOST_ROUND = _lgbm_cfg["num_boost_round"]
BEST_LGBM_EARLY_STOP  = _lgbm_cfg["early_stopping"]

# No los configuro ya que no he tenido ningún caso en el que random forest haya sido mejor
BEST_RF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        15,
    "min_samples_leaf": 5,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "random_state":     SEED,
    "n_jobs":           -1,
}


# ── Funciones auxiliares ───────────────────────────────────────────────────────

def load_months(months: range) -> pd.DataFrame:
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=YEAR, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            total = len(df)
            df = df.dropna(subset=[TARGET_DELTA])
            if SAMPLE_FRAC < 1.0:
                df = df.sample(frac=SAMPLE_FRAC, random_state=SEED)
            mb = df.memory_usage(deep=True).sum() / 1e6
            print(f"  month={month:02d}  {total:>10,} filas  ->  {len(df):>10,} muestreadas  ~{mb:.0f} MB")
            dfs.append(df)
        except Exception as e:
            print(f"  month={month:02d}  no encontrado ({e})")
    return pd.concat(dfs, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    if "n_eventos_afectando" in df.columns:
        df["has_alert"] = (df["n_eventos_afectando"] > 0).astype(np.int8)
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec    = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    rec     = recall_score(y_true, y_pred, zero_division=0)
    bal_acc = (rec + spec) / 2 if not np.isnan(spec) else float("nan")
    return {
        f"{prefix}roc_auc":       round(roc_auc_score(y_true, y_prob), 4),
        f"{prefix}pr_auc":        round(average_precision_score(y_true, y_prob), 4),
        f"{prefix}accuracy":      round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}recall":        round(rec, 4),
        f"{prefix}f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}specificity":   round(spec, 4) if not np.isnan(spec) else None,
        f"{prefix}balanced_acc":  round(bal_acc, 4) if not np.isnan(bal_acc) else None,
        f"{prefix}fpr":           round(fp / (fp + tn), 4) if (fp + tn) > 0 else None,
        f"{prefix}fnr":           round(fn / (fn + tp), 4) if (fn + tp) > 0 else None,
        f"{prefix}tp":            int(tp),
        f"{prefix}fp":            int(fp),
        f"{prefix}fn":            int(fn),
        f"{prefix}tn":            int(tn),
    }


def best_threshold_by_f1(y_true, y_prob) -> tuple[float, float]:
    """Encuentra el umbral de clasificación óptimo maximizando el F1 en validación."""
    thresholds = np.arange(0.20, 0.80, 0.01)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate_by_segment(df_test: pd.DataFrame, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Evaluación por segmentos relevantes del dominio:
    """
    df_seg = df_test.copy()
    df_seg["y_prob"]  = y_prob
    df_seg["y_pred"]  = (y_prob >= threshold).astype(int)
    df_seg["y_true"]  = df_seg[TARGET].values

    # Segmento 1: has_alert
    rows = []
    for val in [0, 1]:
        mask = df_seg["has_alert"] == val
        sub  = df_seg[mask]
        if len(sub) < 50:
            continue
        f1  = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
        auc = roc_auc_score(sub["y_true"], sub["y_prob"]) if sub["y_true"].nunique() > 1 else float("nan")
        rows.append({
            "segmento":     "has_alert",
            "valor":        f"{'con alerta' if val else 'sin alerta'}",
            "n":            len(sub),
            "pct_mejora":   round(sub["y_true"].mean() * 100, 1),
            "roc_auc":      round(auc, 4) if not np.isnan(auc) else None,
            "f1":           round(f1, 4),
            "precision":    round(precision_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
            "recall":       round(recall_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
        })

    # Segmento 2: is_weekend
    if "is_weekend" in df_seg.columns:
        for val in [0, 1]:
            mask = df_seg["is_weekend"] == val
            sub  = df_seg[mask]
            if len(sub) < 50:
                continue
            f1  = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
            auc = roc_auc_score(sub["y_true"], sub["y_prob"]) if sub["y_true"].nunique() > 1 else float("nan")
            rows.append({
                "segmento":     "is_weekend",
                "valor":        f"{'fin de semana' if val else 'laborable'}",
                "n":            len(sub),
                "pct_mejora":   round(sub["y_true"].mean() * 100, 1),
                "roc_auc":      round(auc, 4) if not np.isnan(auc) else None,
                "f1":           round(f1, 4),
                "precision":    round(precision_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
                "recall":       round(recall_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
            })

    # Segmento 3: delay_bucket
    if "delay_seconds" in df_seg.columns:
        bins   = [-np.inf, 0, 60, 300, np.inf]
        labels = ["sin retraso", "leve (<1min)", "moderado (1-5min)", "severo (>5min)"]
        df_seg["delay_bucket"] = pd.cut(df_seg["delay_seconds"], bins=bins, labels=labels)
        for bucket in labels:
            sub = df_seg[df_seg["delay_bucket"] == bucket]
            if len(sub) < 50:
                continue
            f1  = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
            auc = roc_auc_score(sub["y_true"], sub["y_prob"]) if sub["y_true"].nunique() > 1 else float("nan")
            rows.append({
                "segmento":     "delay_bucket",
                "valor":        bucket,
                "n":            len(sub),
                "pct_mejora":   round(sub["y_true"].mean() * 100, 1),
                "roc_auc":      round(auc, 4) if not np.isnan(auc) else None,
                "f1":           round(f1, 4),
                "precision":    round(precision_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
                "recall":       round(recall_score(sub["y_true"], sub["y_pred"], zero_division=0), 4),
            })

    return pd.DataFrame(rows)


# ── Carga y preparación de datos ───────────────────────────────────────────────

def load_and_prepare():
    print(f"\nCargando datos (meses {list(MONTHS)})...")
    df = load_months(MONTHS)
    print(f"  Total: {len(df):,} filas\n")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    for col in ["merge_time", "timestamp_start"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = add_features(df)
    df[TARGET] = (df[TARGET_DELTA] < 0).astype(np.int8)

    counts = df[TARGET].value_counts().sort_index()
    print(f"Clase 0 (retraso se propaga): {counts[0]:,}  |  Clase 1 (retraso se absorbe): {counts[1]:,}")

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

    for name, X in [("train", df_train), ("val", df_val), ("test", df_test)]:
        y = X[TARGET]
        print(f"  {name:>5s}: {len(X):>10,} filas  |  clase 1 (mejora): {y.mean()*100:.1f}%")

    return df_train, df_val, df_test, feats


# ── Evaluación de modelos ──────────────────────────────────────────────────────

def evaluar_lgbm(df_train, df_val, df_test, feats):
    """
    Reentrena LightGBM con train+val usando los mejores hiperparámetros de Optuna
    y evalúa únicamente sobre el conjunto de TEST.
    """
    print("\n\n══════════════════════════════════════════════════")
    print("  MODELO 1: LightGBM (mejores parámetros Optuna)")
    print("══════════════════════════════════════════════════")

    # Reentrenar con train+val juntos (práctica estándar en evaluación final)
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    X_trainval  = df_trainval[feats]
    y_trainval  = df_trainval[TARGET]
    X_val_solo  = df_val[feats]
    y_val_solo  = df_val[TARGET]
    X_test      = df_test[feats]
    y_test      = df_test[TARGET]

    class_ratio = y_trainval.mean()
    params = {**BEST_LGBM_PARAMS, "is_unbalance": class_ratio < 0.3 or class_ratio > 0.7}

    # Primero buscamos umbral óptimo en validación (datos no vistos en la búsqueda)
    lgb_train_only = lgb.Dataset(df_train[feats], label=df_train[TARGET])
    lgb_val_only   = lgb.Dataset(X_val_solo, label=y_val_solo, reference=lgb_train_only)

    print("  Entrenando con train para buscar umbral óptimo en val...")
    model_val = lgb.train(
        params, lgb_train_only,
        num_boost_round=BEST_LGBM_BOOST_ROUND,
        valid_sets=[lgb_val_only],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(BEST_LGBM_EARLY_STOP, verbose=False)],
    )
    y_prob_val  = model_val.predict(X_val_solo, num_iteration=model_val.best_iteration)
    threshold, best_f1_val = best_threshold_by_f1(y_val_solo, y_prob_val)
    print(f"  Umbral óptimo (val F1): {threshold:.2f}  ->  F1={best_f1_val:.4f}")

    # Reentrenamos con train+val para la evaluación final en test
    print("  Reentrenando con train+val para evaluación en TEST...")
    lgb_trainval = lgb.Dataset(X_trainval, label=y_trainval)
    model_final  = lgb.train(
        params, lgb_trainval,
        num_boost_round=model_val.best_iteration,
    )

    y_prob_test = model_final.predict(X_test)
    y_pred_test = (y_prob_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_prob_test, y_pred_test, prefix="test_")
    metrics["best_threshold"] = threshold
    metrics["val_best_f1"]    = best_f1_val

    print("\n  ── Métricas en TEST ──")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # Importancia de variables
    importance = pd.DataFrame({
        "feature":    model_final.feature_name(),
        "importance": model_final.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance["pct"] = (importance["importance"] / importance["importance"].sum() * 100).round(2)
    print("\n  Top 20 features:")
    print(importance.head(20).to_string(index=False))

    # Métricas por segmento
    seg_df = evaluate_by_segment(df_test, y_prob_test, threshold)
    print("\n  ── Métricas por segmento ──")
    print(seg_df.to_string(index=False))

    return metrics, y_prob_test, y_pred_test, y_test, importance, seg_df, threshold

# Esto no serviría actualmente pero por si acaso lo dejo para un futuro
def evaluar_rf(df_train, df_val, df_test, feats):
    """
    Reentrena RandomForest con train+val usando los mejores hiperparámetros de Optuna
    y evalúa únicamente sobre el conjunto de TEST.
    """
    print("\n\n══════════════════════════════════════════════════")
    print("  MODELO 2: Random Forest (mejores parámetros Optuna)")
    print("══════════════════════════════════════════════════")

    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    X_trainval  = df_trainval[feats]
    y_trainval  = df_trainval[TARGET]
    X_val_solo  = df_val[feats]
    y_val_solo  = df_val[TARGET]
    X_test      = df_test[feats]
    y_test      = df_test[TARGET]

    # Umbral óptimo con solo train (modelo auxliar rápido en train)
    print("  Entrenando con train para buscar umbral óptimo en val...")
    model_aux = RandomForestClassifier(**BEST_RF_PARAMS)
    model_aux.fit(df_train[feats], df_train[TARGET])
    y_prob_val  = model_aux.predict_proba(X_val_solo)[:, 1]
    threshold, best_f1_val = best_threshold_by_f1(y_val_solo, y_prob_val)
    print(f"  Umbral óptimo (val F1): {threshold:.2f}  ->  F1={best_f1_val:.4f}")

    # Reentrenamos con train+val
    print("  Reentrenando con train+val para evaluación en TEST...")
    model_final = RandomForestClassifier(**BEST_RF_PARAMS)
    model_final.fit(X_trainval, y_trainval)

    y_prob_test = model_final.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_prob_test, y_pred_test, prefix="test_")
    metrics["best_threshold"] = threshold
    metrics["val_best_f1"]    = best_f1_val

    print("\n  ── Métricas en TEST ──")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # Importancia de variables
    importance = pd.DataFrame({
        "feature":    feats,
        "importance": model_final.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance["pct"] = (importance["importance"] / importance["importance"].sum() * 100).round(2)
    print("\n  Top 20 features:")
    print(importance.head(20).to_string(index=False))

    # Métricas por segmento
    seg_df = evaluate_by_segment(df_test, y_prob_test, threshold)
    print("\n  ── Métricas por segmento ──")
    print(seg_df.to_string(index=False))

    return metrics, y_prob_test, y_pred_test, y_test, importance, seg_df, threshold


# ── Subida a W&B ──────────────────────────────────────────────────────────────

def log_to_wandb(
    metrics_lgbm, y_prob_lgbm, y_pred_lgbm,
    metrics_rf,   y_prob_rf,   y_pred_rf,
    y_test,
    importance_lgbm, seg_lgbm,
    importance_rf,   seg_rf,
):
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"evaluacion_final_delta_{TARGET_DELTA}",
    )

    # Métricas escalares de ambos modelos
    wandb.log({
        **{f"lgbm_{k}": v for k, v in metrics_lgbm.items()},
        **{f"rf_{k}":   v for k, v in metrics_rf.items()},
    })

    # Tabla comparativa global
    df_comp = pd.DataFrame([
        {"modelo": "LightGBM",     **{k.replace("test_", ""): v for k, v in metrics_lgbm.items()}},
        {"modelo": "RandomForest", **{k.replace("test_", ""): v for k, v in metrics_rf.items()}},
    ])
    wandb.log({"comparativa_global": wandb.Table(dataframe=df_comp)})

    # Matrices de confusión
    wandb.log({
        "confusion_lgbm": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_lgbm.tolist(),
            class_names=["propaga (0)", "absorbe (1)"],
            title="Confusion Matrix - LightGBM",
        ),
        "confusion_rf": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_rf.tolist(),
            class_names=["propaga (0)", "absorbe (1)"],
            title="Confusion Matrix - RandomForest",
        ),
    })

    # Curvas Precision-Recall
    p_lgbm, r_lgbm, _ = precision_recall_curve(y_test, y_prob_lgbm)
    p_rf,   r_rf,   _ = precision_recall_curve(y_test, y_prob_rf)
    wandb.log({
        "pr_curve_lgbm": wandb.plot.line_series(
            xs=r_lgbm.tolist(), ys=[p_lgbm.tolist()],
            keys=["LightGBM"], title="PR Curve - LightGBM", xname="Recall",
        ),
        "pr_curve_rf": wandb.plot.line_series(
            xs=r_rf.tolist(), ys=[p_rf.tolist()],
            keys=["RandomForest"], title="PR Curve - RandomForest", xname="Recall",
        ),
    })

    # Feature importance
    wandb.log({
        "feature_importance_lgbm": wandb.Table(dataframe=importance_lgbm.head(30)),
        "feature_importance_rf":   wandb.Table(dataframe=importance_rf.head(30)),
    })

    # Métricas por segmento
    wandb.log({
        "segmentos_lgbm": wandb.Table(dataframe=seg_lgbm),
        "segmentos_rf":   wandb.Table(dataframe=seg_rf),
    })

    run.finish()
    print("\n✔ Resultados subidos a Weights & Biases.")


# ── Punto de entrada ───────────────────────────────────────────────────────────

def main():
    df_train, df_val, df_test, feats = load_and_prepare()

    metrics_lgbm, y_prob_lgbm, y_pred_lgbm, y_test, imp_lgbm, seg_lgbm, _ = evaluar_lgbm(
        df_train, df_val, df_test, feats
    )
    metrics_rf, y_prob_rf, y_pred_rf, _, imp_rf, seg_rf, _ = evaluar_rf(
        df_train, df_val, df_test, feats
    )

    print("\n\n══════════════════════════════════════════════════")
    print("  COMPARATIVA FINAL (TEST)")
    print("══════════════════════════════════════════════════")
    df_comp = pd.DataFrame([
        {"modelo": "LightGBM",     **{k.replace("test_", ""): v for k, v in metrics_lgbm.items()}},
        {"modelo": "RandomForest", **{k.replace("test_", ""): v for k, v in metrics_rf.items()}},
    ]).set_index("modelo")
    print(df_comp.to_string())

    log_to_wandb(
        metrics_lgbm, y_prob_lgbm, y_pred_lgbm,
        metrics_rf,   y_prob_rf,   y_pred_rf,
        y_test,
        imp_lgbm, seg_lgbm,
        imp_rf,   seg_rf,
    )


if __name__ == "__main__":
    main()

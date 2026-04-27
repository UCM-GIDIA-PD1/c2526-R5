"""
Optimización de hiperparámetros automatizada con Random Search para LightGBM
Registra cada prueba (trial) como una ejecución independiente en Weights & Biases.

Uso:
    uv run python src/models/prediccion_retrasos/delta/random_binary_classification_delta.py
"""

import gc
import os
import random
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

# Configuracion

ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")

YEAR          = 2025
MONTHS        = range(1, 13)
SAMPLE_FRAC   = 0.1  
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

TARGET_DELTA  = "delta_delay_end"   # cambiar para otro horizonte: _20m, _30m, _45m, _60m, etc.
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
}

CAT_FEATURES = [
    "route_id", "direction", "category", "tipo_referente",
    "stop_id", "is_weekend", "is_unscheduled", "temp_extreme",
    "afecta_previo", "afecta_durante", "afecta_despues",
    "is_alert_just_published", "has_alert",
]

# Configuración de Random Search
N_TRIALS = 20 # Número de combinaciones automáticas que probará antes de detenerse



def load_months(months: range) -> pd.DataFrame:
    """Descarga y filtra los datos de entrenamiento y validacion desde MinIO."""
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
    """Añade variables derivadas del retraso y de las alertas para enriquecer el modelo."""
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
    """Convierte las columnas categoricas a enteros usando el vocabulario del conjunto de entrenamiento."""
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
        df_test[col]  = df_test[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val, df_test


def get_features(df: pd.DataFrame) -> list:
    """Devuelve la lista de columnas que se usan como features, excluyendo el target y columnas no relevantes."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


def compute_metrics(y_true, y_prob, y_pred, prefix="") -> dict:
    """Calcula las metricas principales a partir de las predicciones y los valores reales."""
    return {
        f"{prefix}roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        f"{prefix}pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        f"{prefix}accuracy":  round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# Random Search

def load_and_prepare_data():
    """Carga y prepara todos los datos en memoria para reutilizarlos en todos los trials."""
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
    
    X_train, y_train = df_train[feats], df_train[TARGET]
    X_val,   y_val   = df_val[feats],   df_val[TARGET]
    X_test,  y_test  = df_test[feats],  df_test[TARGET]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feats


def run_random_search():
    """Ejecuta la busqueda aleatoria de hiperparametros y muestra los mejores resultados."""
    X_train, y_train, X_val, y_val, X_test, y_test, feats = load_and_prepare_data()
    class_ratio = y_train.mean()
    
    print(f"\nIniciando Búsqueda Aleatoria en {N_TRIALS} pruebas consecutivas...")
    
    best_auc = 0.0
    best_params_overall = {}
    best_trial_num = -1

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    for trial_idx in range(N_TRIALS):
        # 1. Elección Aleatoria de hiperparámetros
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "boosting_type":     "gbdt",
            "extra_trees":       random.choice([True, False]),
            "learning_rate":     10 ** random.uniform(np.log10(0.01), np.log10(0.1)),
            "num_leaves":        random.randint(50, 200),
            "max_depth":         random.randint(5, 9),
            "min_child_samples": random.randint(50, 250),
            "feature_fraction":  round(random.uniform(0.6, 1.0), 2),
            "bagging_fraction":  round(random.uniform(0.6, 1.0), 2),
            "bagging_freq":      random.randint(1, 10),
            "reg_alpha":         round(random.uniform(0.0, 5.0), 2),
            "reg_lambda":        round(random.uniform(0.0, 5.0), 2),
            "n_jobs":            -1,
            "verbose":           -1,
            "seed":              SEED,
            "is_unbalance":      class_ratio < 0.3 or class_ratio > 0.7,
            "feature_pre_filter": False,
        }
        
        num_boost_round = random.choice(range(1500, 4500, 500))
        early_stopping  = random.choice(range(50, 175, 25))
        
        # 2. Iniciar sesión de Weights and Biases
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"random_binary_delta_{TARGET_DELTA}",
            reinit=True,
            config={**params, "num_boost_round": num_boost_round, "early_stopping": early_stopping}
        )

        # 3. Entrenar el modelo
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
            ],
        )

        # 4. Calcular Métricas
        y_prob_val  = model.predict(X_val,  num_iteration=model.best_iteration)
        y_pred_val  = (y_prob_val  >= 0.5).astype(int)
        m_val  = compute_metrics(y_val,  y_prob_val,  y_pred_val,  prefix="val_")

        # Umbral óptimo por F1
        thresholds = np.arange(0.20, 0.80, 0.01)
        f1s = [f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx       = int(np.argmax(f1s))
        best_threshold = float(thresholds[best_idx])

        y_prob_test = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_test_opt = (y_prob_test >= best_threshold).astype(int)
        m_test_opt = compute_metrics(y_test, y_prob_test, y_pred_test_opt, prefix="test_opt_")

        # 5. Guardar todas las métricas en WandB
        wandb.log({
            **m_val,
            **m_test_opt,
            "best_iteration": model.best_iteration,
            "best_threshold": best_threshold,
            "val_best_f1": f1s[best_idx],
            "target_delta": TARGET_DELTA,
        })
        
        run.finish()

        # Registro del mejor modelo localmente
        val_auc = m_val["val_roc_auc"]
        if val_auc > best_auc:
            best_auc = val_auc
            best_params_overall = {**params, "num_boost_round": num_boost_round, "early_stopping": early_stopping}
            best_trial_num = trial_idx

    print("\nBÚSQUEDA ALEATORIA TERMINADA")
    print(f"La mejor prueba ha sido la Número: {best_trial_num}")
    print(f"Mejor AUC en validación: {best_auc}")
    print("Los hiperparámetros ganadores fueron:")
    for k, v in best_params_overall.items():
        print(f"    '{k}': {v},")

if __name__ == "__main__":
    run_random_search()

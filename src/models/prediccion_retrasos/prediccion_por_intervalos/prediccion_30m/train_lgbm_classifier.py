"""
Entrenamiento LightGBM — Predicción CLASIFICADA de retraso en parada (Objetivo 1)

Predice target_class = rango en el que caerá el retraso en los próximos 30 min.
Usa la función objetivo 'multiclass' para clasificación.

Validación temporal:
    Train  → meses 01–09  (enero–septiembre 2025)
    Val    → meses 10–12  (octubre–diciembre 2025)
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "TU_CLAVE")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "TU_SECRETO")

YEAR           = 2025
TRAIN_MONTHS   = range(1, 10)   
VAL_MONTHS     = range(10, 13)  

# Targets
TARGET_REG     = "target_delay_30m" # Lo mantenemos temporalmente para features
TARGET_CLASS   = "target_class"     # Nuevo target para clasificación

BINS = [-np.inf, -60, 60, 180, 300, 450, np.inf]
LABELS = [
    'Adelantado (>1 min)', 
    'Puntual (-1 a 1 min)', 
    'Retraso leve (1-3 min)', 
    'Retraso moderado (3-5 min)', 
    'Retraso grave (5-7.5 min)',
    'Retraso muy grave (>7.5 min)'
]

DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"
MODEL_PATH_OUT = "grupo5/models/lgbm_stop_delay30m_multiclass.txt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "lgbm-stop-delay30m-classification"

EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
    "delay_vs_station", "station_trend",
}

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         len(LABELS), # 6 clases
    "metric":            "multi_logloss",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         12,
    "min_child_samples": 50,
    "min_split_gain":    0.11586905952512971,
    "feature_fraction":  0.8645035840204937,
    "bagging_fraction":  0.8856351733429728,
    "bagging_freq":      5,
    "reg_alpha":         0.14808930346818072,
    "reg_lambda":        0.7169314570885452,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}

NUM_BOOST_ROUND = 20000
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
            df = df.dropna(subset=[TARGET_REG])
            df = df[df["scheduled_time_to_end"] >= 1800]
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

def create_class_target(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza el retraso en segundos a las clases definidas por el usuario."""
    # Creamos las categorías, devolviendo números enteros (0 a 5) requeridos por LightGBM
    df[TARGET_CLASS] = pd.cut(df[TARGET_REG], bins=BINS, labels=False).astype(int)
    return df

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
    return df

def add_target_encoding(df_train: pd.DataFrame, df_val: pd.DataFrame,
                        col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usamos el retraso histórico MEDIO (continuo) como feature para la predicción de clase."""
    means = df_train.groupby(col)[target].mean()
    global_mean = df_train[target].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val

def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and c not in [TARGET_REG, TARGET_CLASS]]

def compute_classification_metrics(y_true, y_pred_proba, prefix="") -> dict:
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    loss = log_loss(y_true, y_pred_proba)
    return {
        f"{prefix}accuracy": round(acc, 4),
        f"{prefix}f1_macro": round(f1_macro, 4),
        f"{prefix}log_loss": round(loss, 4),
    }

def main():
    print(f"\nCargando datos de entrenamiento (meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS)
    
    print(f"\nCargando datos de validación (meses {list(VAL_MONTHS)})...")
    df_val = load_months(VAL_MONTHS)

    df_train = create_class_target(df_train)
    df_val   = create_class_target(df_val)

    df_train, df_val = encode_categoricals(df_train, df_val)
    df_train, df_val = add_target_encoding(df_train, df_val, STOP_ID_COL, TARGET_REG)

    df_train = add_derived_features(df_train)
    df_val   = add_derived_features(df_val)
    print(f"Tras filtrado + FE  —  train: {len(df_train):,}  |  val: {len(df_val):,}\n")

    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train, y_train = df_train[feats], df_train[TARGET_CLASS]
    X_val,   y_val   = df_val[feats],   df_val[TARGET_CLASS]

    print("Distribución de clases en TRAIN:")
    print(y_train.value_counts(normalize=True).sort_index().rename(index=dict(enumerate(LABELS))))
    print("\n")

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-30m-class",
        config={
            **LGBM_PARAMS,
            "target":       TARGET_CLASS,
            "classes":      LABELS,
            "train_months": list(TRAIN_MONTHS),
            "val_months":   list(VAL_MONTHS),
            "n_features":   len(feats),
            "train_rows":   len(df_train),
            "val_rows":     len(df_val),
        }
    )

    print(f"Entrenando LightGBM Clasificador Multiclase...")
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

    y_pred_proba_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_proba_val   = model.predict(X_val,   num_iteration=model.best_iteration)

    metrics_train = compute_classification_metrics(y_train, y_pred_proba_train, prefix="train_")
    metrics_val   = compute_classification_metrics(y_val,   y_pred_proba_val,   prefix="val_")

    print("\nMétricas globales train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas globales val:");   [print(f"  {k}: {v}") for k, v in metrics_val.items()]

    wandb.log({**metrics_train, **metrics_val, "best_iteration": model.best_iteration})

    # Imprimir el reporte detallado por clases en Validación
    y_pred_class_val = np.argmax(y_pred_proba_val, axis=1)
    print("\n--- Reporte de Clasificación (Validación) ---")
    print(classification_report(y_val, y_pred_class_val, target_names=LABELS, zero_division=0))

    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 features:\n{importance.head(15).to_string(index=False)}")
    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(20))})

    # Guardar modelo
    os.makedirs(os.path.dirname(MODEL_PATH_OUT), exist_ok=True)
    model.save_model(MODEL_PATH_OUT)
    print(f"Modelo guardado en {MODEL_PATH_OUT}")

    wandb.finish()
    print("\nEntrenamiento completado.")

if __name__ == "__main__":
    main()
"""
Entrenamiento LightGBM — Predicción de retraso en parada (Objetivo 1)

Predice el retraso absoluto de un tren en una parada a distintos horizontes:
    target_delay_30m  (principal)
    target_delay_60m  (secundario)

Validación temporal:
    Train  → meses 01–09  (enero–septiembre 2025)
    Val    → meses 10–12  (octubre–diciembre 2025)

Uso:
    uv run python -m src.models.prediccion_retrasos.train_lgbm_stop_delay

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

from src.common.minio_client import download_df_parquet, upload_file

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR           = 2025
TRAIN_MONTHS   = range(1, 10)   # enero–septiembre
VAL_MONTHS     = range(10, 13)  # octubre–diciembre
TARGET         = "target_delay_30m"   # cambia a target_delay_60m para el horizonte largo
DATA_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"
MODEL_PATH_OUT = "grupo5/models/lgbm_stop_delay_30m.txt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = f"lgbm-stop-{TARGET}"

# Columnas que NO son features: IDs, todos los targets y columnas de leakage
EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "is_unscheduled",
    # Todos los targets y deltas
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    # Información futura (leakage)
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
}

# Columnas categóricas para LightGBM
CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]

# Hiperparámetros LightGBM
LGBM_PARAMS = {
    "objective":         "regression_l1", 
    "metric":            "mae",
    "learning_rate":     0.05,
    "num_leaves":        127,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}
NUM_BOOST_ROUND = 2200
EARLY_STOPPING  = 50

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_months(months: range) -> pd.DataFrame:
    """Descarga, filtra y concatena los parquets mensuales indicados.
    Filtra is_unscheduled y nulls en target MES A MES para no acumular RAM innecesaria.
    """
    dfs = []
    for month in months:
        path = DATA_TEMPLATE.format(year=YEAR, month=month)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            total = len(df)
            # Filtrar antes de acumular
            df = df[df["is_unscheduled"] == False]
            df = df.dropna(subset=[TARGET])
            # Convertir categóricas
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
    """Codifica las columnas categóricas como enteros usando el vocabulario del train.
    Los valores desconocidos en val se mapean a -1.
    Evita el problema de pd.concat convirtiendo categorías a str.
    """
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        # Construir mapeo desde train (valores únicos → entero)
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_val


def get_features(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas del df que son features (todo menos EXCLUDE_COLS)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


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

# ── Pipeline ───────────────────────────────────────────────────────────────────

def main():
    # 1. Cargar datos
    print(f"\nCargando datos de entrenamiento (meses {list(TRAIN_MONTHS)})...")
    df_train = load_months(TRAIN_MONTHS)
    print(f"  Total: {len(df_train):,} filas\n")

    print(f"Cargando datos de validación (meses {list(VAL_MONTHS)})...")
    df_val = load_months(VAL_MONTHS)
    print(f"  Total: {len(df_val):,} filas\n")

    # 2. Codificar categóricas con mapeo consistente train → val
    df_train, df_val = encode_categoricals(df_train, df_val)
    print(f"Tras filtrado  —  train: {len(df_train):,}  |  val: {len(df_val):,}\n")

    # 3. Seleccionar features automáticamente (todo lo que no esté en EXCLUDE_COLS)
    feats = get_features(df_train)
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_train, y_train = df_train[feats], df_train[TARGET]
    X_val,   y_val   = df_val[feats],   df_val[TARGET]

    # 4. Iniciar WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
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

    # 5. Entrenar LightGBM
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

    # 6. Métricas
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_val   = model.predict(X_val,   num_iteration=model.best_iteration)

    metrics_train = compute_metrics(y_train, y_pred_train, prefix="train_")
    metrics_val   = compute_metrics(y_val,   y_pred_val,   prefix="val_")

    print("\nMétricas train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas val:");   [print(f"  {k}: {v}") for k, v in metrics_val.items()]

    wandb.log({**metrics_train, **metrics_val, "best_iteration": model.best_iteration})

    # 7. Importancia de features
    importance = pd.DataFrame({
        "feature":    model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 features:\n{importance.head(15).to_string(index=False)}")
    wandb.log({"feature_importance": wandb.Table(dataframe=importance.head(20))})

    # 8. Guardar modelo en MinIO
    #tmp = "/tmp/lgbm_model.txt"
    #model.save_model(tmp)
    #upload_file(ACCESS_KEY, SECRET_KEY, MODEL_PATH_OUT, tmp)
    #print(f"\nModelo guardado en MinIO: pd1/{MODEL_PATH_OUT}")

    wandb.finish()
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()

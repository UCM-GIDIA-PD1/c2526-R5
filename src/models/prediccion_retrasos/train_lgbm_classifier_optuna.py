"""
Entrenamiento LightGBM — Predicción CLASIFICADA de retraso en parada con OPTUNA

Predice target_class = rango en el que caerá el retraso en los próximos 30 min.
Realiza una búsqueda de hiperparámetros con Optuna para minimizar el multi_logloss.
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import optuna
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "TU_CLAVE")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "TU_SECRETO")

YEAR           = 2025
TRAIN_MONTHS   = range(1, 10)   
VAL_MONTHS     = range(10, 13)  

# Targets
TARGET_REG     = "target_delay_30m"
TARGET_CLASS   = "target_class"

# Configuración de los Bins para las clases
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
MODEL_PATH_OUT = "grupo5/models/lgbm_stop_delay30m_multiclass_optuna.txt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "lgbm-stop-delay30m-optuna-best"

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

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"

# Configuración de Optuna
N_TRIALS = 30           # Cuántas combinaciones distintas probará Optuna
TUNE_BOOST_ROUND = 1000 # Menos rondas durante la búsqueda para que sea más rápido
FINAL_BOOST_ROUND = 20000 
EARLY_STOPPING  = 50
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

def add_target_encoding(df_train: pd.DataFrame, df_val: pd.DataFrame, col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    print(f"\n--- 1. CARGA DE DATOS ---")
    df_train = load_months(TRAIN_MONTHS)
    df_val = load_months(VAL_MONTHS)

    df_train = create_class_target(df_train)
    df_val   = create_class_target(df_val)

    df_train, df_val = encode_categoricals(df_train, df_val)
    df_train, df_val = add_target_encoding(df_train, df_val, STOP_ID_COL, TARGET_REG)
    df_train = add_derived_features(df_train)
    df_val   = add_derived_features(df_val)

    feats = get_features(df_train)
    X_train, y_train = df_train[feats], df_train[TARGET_CLASS]
    X_val,   y_val   = df_val[feats],   df_val[TARGET_CLASS]

    print(f"\nFeatures usadas ({len(feats)}). Iniciando creación de Datasets LightGBM...")
    
    # Creamos los datasets de LightGBM una sola vez para que Optuna los reutilice
    lgb_train = lgb.Dataset(X_train, label=y_train)
    # Durante la optimización, comparamos contra validación
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    print(f"\n--- 2. BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA ---")
    
    def objective(trial):
        # 1. Definimos el espacio de búsqueda (lo que Optuna va a variar)
        params = {
            "objective":         "multiclass",
            "num_class":         len(LABELS),
            "metric":            "multi_logloss",
            "n_jobs":            -1,
            "verbose":           -1,
            "seed":              42,
            
            # Hiperparámetros a optimizar:
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 63, 1024),
            "max_depth":         trial.suggest_int("max_depth", 5, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        # 2. Entrenamos el modelo para esta combinación
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=TUNE_BOOST_ROUND, # Usamos menos rondas para ir más rápido en la búsqueda
            valid_sets=[lgb_val],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
            ],
        )

        # 3. Optuna necesita un valor para "minimizar". Le damos el Log Loss de validación.
        return model.best_score["val"]["multi_logloss"]

    # Creamos el estudio de Optuna y le decimos que queremos minimizar el error
    study = optuna.create_study(direction="minimize", study_name="LGBM_Delay_Classification")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n✓ Búsqueda de Optuna finalizada.")
    print("Mejores hiperparámetros encontrados:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print(f"\n--- 3. ENTRENAMIENTO FINAL CON LOS MEJORES PARÁMETROS ---")
    
    # Combinamos los parámetros fijos con los mejores encontrados por Optuna
    final_params = {
        "objective": "multiclass",
        "num_class": len(LABELS),
        "metric":    "multi_logloss",
        "n_jobs":    -1,
        "verbose":   -1,
        "seed":      42,
        **best_params
    }

    # Iniciamos Weights & Biases solo para el modelo ganador
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group="prediccion-retrasos-30m-class-optuna",
        config={
            **final_params,
            "target":       TARGET_CLASS,
            "classes":      LABELS,
            "train_months": list(TRAIN_MONTHS),
            "val_months":   list(VAL_MONTHS),
            "n_features":   len(feats),
            "optuna_trials": N_TRIALS
        }
    )

    final_model = lgb.train(
        final_params,
        lgb_train,
        num_boost_round=FINAL_BOOST_ROUND, # Aquí sí le permitimos usar las 20.000 rondas
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    print(f"\nMejor iteración final: {final_model.best_iteration}")

    # Predicciones y métricas
    y_pred_proba_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
    y_pred_proba_val   = final_model.predict(X_val,   num_iteration=final_model.best_iteration)

    metrics_train = compute_classification_metrics(y_train, y_pred_proba_train, prefix="train_")
    metrics_val   = compute_classification_metrics(y_val,   y_pred_proba_val,   prefix="val_")

    print("\nMétricas globales train:"); [print(f"  {k}: {v}") for k, v in metrics_train.items()]
    print("Métricas globales val:");   [print(f"  {k}: {v}") for k, v in metrics_val.items()]

    wandb.log({**metrics_train, **metrics_val, "best_iteration": final_model.best_iteration})

    y_pred_class_val = np.argmax(y_pred_proba_val, axis=1)
    print("\n--- Reporte de Clasificación (Validación) ---")
    print(classification_report(y_val, y_pred_class_val, target_names=LABELS, zero_division=0))

    # Guardar modelo
    os.makedirs(os.path.dirname(MODEL_PATH_OUT), exist_ok=True)
    final_model.save_model(MODEL_PATH_OUT)
    print(f"Modelo guardado en {MODEL_PATH_OUT}")

    wandb.finish()
    print("\nProceso completado exitosamente.")

if __name__ == "__main__":
    main()
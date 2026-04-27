"""
Optuna — Búsqueda bayesiana de hiperparámetros para LightGBM global (todas las líneas)

Predice una clasificación multiclase del retraso usando el dataset mensual completo.

Uso:
    uv run python -m src.models.prediccion_retrasos.delay_30m.optuna.optuna_lgbm

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from optuna.integration import LightGBMPruningCallback

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR          = 2025
TRAIN_MONTHS  = range(1, 10)
VAL_MONTHS    = range(10, 13)

TARGET_CONT   = "target_delay_30m"
TARGET_CLASS  = "target_class"
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

WANDB_PROJECT = "pd1-c2526-team5"
SAMPLE_FRAC   = 0.05 
NUM_RUNS      = 30

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"

BINS = [-np.inf, -60, 60, 180, 300, 450, np.inf]
CLASS_NAMES = [
    'Adelantado (>1 min)', 
    'Puntual (-1 a 1 min)', 
    'Retraso leve (1-3 min)', 
    'Retraso moderado (3-5 min)', 
    'Retraso grave (5-7.5 min)',
    'Retraso muy grave (>7.5 min)'
]

EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
    TARGET_CONT, TARGET_CLASS,
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    def _load(months):
        dfs = []
        for month in months:
            path = DATA_TEMPLATE.format(year=YEAR, month=month)
            try:
                df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
                df = df[df["is_unscheduled"] == False]
                # Limpiar nulos del objetivo continuo original primero
                df = df.dropna(subset=[TARGET_CONT])
                df = df[df["scheduled_time_to_end"] >= 1800]
                
                # Crear variable clasificatoria con valores enteros (0-5)
                df[TARGET_CLASS] = pd.cut(df[TARGET_CONT], bins=BINS, labels=False)
                
                if SAMPLE_FRAC < 1.0:
                    df = df.sample(frac=SAMPLE_FRAC, random_state=42)
                dfs.append(df)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True)

    print("Cargando datos...")
    df_train = _load(TRAIN_MONTHS)
    df_val   = _load(VAL_MONTHS)
    print(f"  train: {len(df_train):,}  |  val: {len(df_val):,}\n")
    return df_train, df_val


def encode_categoricals(df_train, df_val):
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


def add_target_encoding(df_train, df_val, col, target_cont):
    # Utilizamos el target continuo (target_delay_30m) para el encoding
    means = df_train.groupby(col)[target_cont].mean()
    global_mean = df_train[target_cont].mean()
    df_train[f"{col}_target_enc"] = df_train[col].map(means)
    df_val[f"{col}_target_enc"]   = df_val[col].map(means).fillna(global_mean)
    return df_train, df_val


def get_features(df):
    # EXCLUDE_COLS ya excluye TARGET_CONT y TARGET_CLASS
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ── Precargar datos ────────────────────────────────────────────────────────────

print("Precargando datos (se hace una sola vez para todos los trials)...")
df_train_global, df_val_global = load_data()
df_train_global, df_val_global = encode_categoricals(df_train_global, df_val_global)

# Hacemos el target encoding basándonos en el delay continuo para no perder información
df_train_global, df_val_global = add_target_encoding(df_train_global, df_val_global, STOP_ID_COL, TARGET_CONT)

df_train_global = add_derived_features(df_train_global)
df_val_global   = add_derived_features(df_val_global)
feats   = get_features(df_train_global)

X_train = df_train_global[feats]
# Pasamos la nueva clase categórica a LightGBM
y_train = df_train_global[TARGET_CLASS]
X_val   = df_val_global[feats]
y_val   = df_val_global[TARGET_CLASS]

print(f"Features ({len(feats)}): {feats}\n")

# ── Función objetivo de Optuna ─────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective":         "multiclass",
        "num_class":         len(CLASS_NAMES),
        "metric":            "multi_logloss",
        "num_leaves":        trial.suggest_categorical("num_leaves", [31, 63, 127]),
        "max_depth":         trial.suggest_categorical("max_depth", [-1, 8, 12]),
        "learning_rate":     trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100]),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      5,
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "n_jobs":            -1,
        "verbose":           -1,
        "seed":              42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(100),
            LightGBMPruningCallback(trial, "multi_logloss", valid_name="val")
        ],
    )

    y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    logloss = log_loss(y_val, y_pred_prob)
    acc = accuracy_score(y_val, y_pred_class)
    f1_macro = f1_score(y_val, y_pred_class, average="macro")
    f1_weighted = f1_score(y_val, y_pred_class, average="weighted")

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"optuna-global-class-trial{trial.number}",
        group="prediccion-retrasos-class",
        config={**params, "trial": trial.number},
        reinit=True,
    )
    wandb.log({
        "val_logloss":     round(logloss, 4),
        "val_accuracy":    round(acc, 4),
        "val_f1_macro":    round(f1_macro, 4),
        "val_f1_weighted": round(f1_weighted, 4),
        "best_iteration":  model.best_iteration,
    })
    wandb.finish()

    return logloss


# ── Lanzar estudio Optuna ──────────────────────────────────────────────────────

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
    )
    print(f"Lanzando {NUM_RUNS} trials Optuna (target={TARGET_CLASS}, multiclase)...\n")
    study.optimize(objective, n_trials=NUM_RUNS, show_progress_bar=True)

    print("\n── Mejores hiperparámetros ──────────────────────────────────────")
    print(f"  val_logloss: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
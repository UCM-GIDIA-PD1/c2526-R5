import os
import pandas as pd
import numpy as np
import wandb
import optuna
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                              recall_score, precision_score)
from src.common.minio_client import download_df_parquet


# ── Configuración ──────────────────────────────────────────────────────────────

load_dotenv()

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

ENTITY  = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"

FEATURES = [
    # Retraso
    'delay_seconds_mean', 'delay_seconds_max',
    'lagged_delay_1_mean', 'lagged_delay_2_mean',
    'delay_1_before', 'delay_2_before', 'delay_3_before',
    'route_rolling_delay_mean', 'route_rolling_delay_max',
    # Headway
    'actual_headway_seconds_mean',
    # Trenes en la ventana
    'match_key_nunique',
    # Temporales
    'hour_sin_first', 'hour_cos_first', 'dow_first', 'is_weekend_max',
    'scheduled_time_to_end_mean', 'stops_to_end_mean',
    # Servicio
    'is_unscheduled_max', 'num_updates_sum',
    # Historial alertas
    'seconds_since_last_alert_mean',
    # Categóricas
    'direction', 'route_id',
]

TARGET = 'alert_in_next_30m_max'
N_TRIALS = 20   # número de trials de Optuna (= número de runs en W&B)
SAMPLE_FRAC = 0.05  # 5% del train para buscar hiperparámetros (rápido)


def main():

    # ── 1. Carga de datos ──────────────────────────────────────────────────────
    print("\nCargando dataset desde MinIO...")
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # ── 2. Preparación ────────────────────────────────────────────────────────
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    for col in ['route_id', 'direction']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('UNKNOWN'))

    features_disponibles = [f for f in FEATURES if f in df.columns]
    print(f"Features disponibles: {len(features_disponibles)}")

    df['seconds_since_last_alert_mean'] = df['seconds_since_last_alert_mean'].fillna(999999)
    cols_mediana = [c for c in features_disponibles if c != 'seconds_since_last_alert_mean']
    df[cols_mediana] = df[cols_mediana].fillna(df[cols_mediana].median())

    # ── 3. División temporal ───────────────────────────────────────────────────
    df = df.sort_values('merge_time').reset_index(drop=True)
    dias = sorted(df['merge_time'].dt.date.unique())
    corte_70 = dias[int(len(dias) * 0.70)]
    corte_85 = dias[int(len(dias) * 0.85)]

    train = df[df['merge_time'].dt.date <  corte_70]
    val   = df[(df['merge_time'].dt.date >= corte_70) &
               (df['merge_time'].dt.date <  corte_85)]
    test  = df[df['merge_time'].dt.date >= corte_85]

    X_train, y_train = train[features_disponibles], train[TARGET]
    X_val,   y_val   = val[features_disponibles],   val[TARGET]
    X_test,  y_test  = test[features_disponibles],  test[TARGET]

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── 4. Baseline (1 run) ────────────────────────────────────────────────────
    print("\n── Evaluando baseline ──")
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)
    y_prob_base = baseline.predict_proba(X_test)[:, 1]
    y_pred_base = baseline.predict(X_test)

    run = wandb.init(entity=ENTITY, project=PROJECT,
                     name="rf_optuna_baseline", config={"model": "baseline_estratificado"})
    wandb.log({
        "pr_auc":    average_precision_score(y_test, y_prob_base),
        "auc_roc":   roc_auc_score(y_test, y_prob_base),
        "f1":        f1_score(y_test, y_pred_base, zero_division=0),
        "recall":    recall_score(y_test, y_pred_base, zero_division=0),
        "precision": precision_score(y_test, y_pred_base, zero_division=0),
    })
    run.finish()
    print("Baseline registrado en W&B")

    # ── 5. Muestra pequeña para búsqueda ──────────────────────────────────────
    X_sample = X_train.sample(frac=SAMPLE_FRAC, random_state=42)
    y_sample = y_train.loc[X_sample.index]
    print(f"\nMuestra para búsqueda: {len(X_sample):,} filas ({SAMPLE_FRAC*100:.0f}% del train)")

    # ── 6. Optuna: cada trial = 1 run en W&B ──────────────────────────────────
    print(f"\n── Iniciando Optuna ({N_TRIALS} trials, 1 run W&B cada uno) ──")

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 400),
            'max_depth':        trial.suggest_int('max_depth', 5, 25),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
            'class_weight':     trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
            'random_state':     42,
            'n_jobs':           -1,
        }

        modelo = RandomForestClassifier(**params)
        modelo.fit(X_sample, y_sample)

        y_prob_val = modelo.predict_proba(X_val)[:, 1]
        pr_auc_val = average_precision_score(y_val, y_prob_val)
        f1_val     = f1_score(y_val, (y_prob_val >= 0.5).astype(int), zero_division=0)

        # 1 run por trial en W&B
        run = wandb.init(
            entity  = ENTITY,
            project = PROJECT,
            name    = f"rf_optuna_{trial.number+1:02d}",
            config  = params,
        )
        wandb.log({
            "pr_auc_val":  pr_auc_val,
            "f1_val":      f1_val,
            "roc_auc_val": roc_auc_score(y_val, y_prob_val),
        })
        run.finish()

        return pr_auc_val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = {**study.best_params, 'random_state': 42, 'n_jobs': -1}
    print(f"\nMejor PR-AUC en validación: {study.best_value:.4f}")
    print(f"Mejores parámetros: {best_params}")

    # ── 7. Modelo final con mejores hiperparámetros ────────────────────────────
    print("\nEntrenando modelo final (train + val completos)...")
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    modelo_final = RandomForestClassifier(**best_params)
    modelo_final.fit(X_train_final, y_train_final)

    y_prob_test = modelo_final.predict_proba(X_test)[:, 1]

    # Threshold óptimo por F1
    thresholds    = np.arange(0.05, 0.95, 0.01)
    f1_scores_thr = [f1_score(y_test, (y_prob_test >= t).astype(int), zero_division=0)
                     for t in thresholds]
    threshold_opt = thresholds[np.argmax(f1_scores_thr)]
    y_pred_test   = (y_prob_test >= threshold_opt).astype(int)

    importancias = pd.DataFrame({
        "feature":    features_disponibles,
        "importance": modelo_final.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nThreshold óptimo: {threshold_opt:.2f}")
    print(f"PR-AUC test: {average_precision_score(y_test, y_prob_test):.4f}")
    print(f"F1 test:     {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
    print("\nTop 10 features:")
    print(importancias.head(10).to_string(index=False))

    # Run final con métricas en test
    run = wandb.init(
        entity  = ENTITY,
        project = PROJECT,
        name    = "rf_optuna_FINAL",
        config  = best_params,
    )
    wandb.log({
        "pr_auc_test":    average_precision_score(y_test, y_prob_test),
        "auc_roc_test":   roc_auc_score(y_test, y_prob_test),
        "f1_test":        f1_score(y_test, y_pred_test, zero_division=0),
        "recall_test":    recall_score(y_test, y_pred_test, zero_division=0),
        "precision_test": precision_score(y_test, y_pred_test, zero_division=0),
        "threshold_opt":  float(threshold_opt),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred_test.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias),
            label="feature",
            value="importance",
            title="Feature Importance - RF Optuna",
        ),
    })
    run.finish()
    print("\nModelo final registrado en W&B")


if __name__ == "__main__":
    main()

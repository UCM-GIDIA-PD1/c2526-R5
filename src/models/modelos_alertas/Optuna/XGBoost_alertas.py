import os
import sys
import pandas as pd
import numpy as np
import wandb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.dummy import DummyClassifier
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                              recall_score, precision_score, classification_report,
                              precision_recall_curve)
from src.common.minio_client import download_df_parquet


# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH = f"grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

ENTITY = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"
NAME = "optuna_modelo_agregado_30min_XGBoost"

FEATURES = [
    "route_rolling_delay_mean", "actual_headway_seconds_mean", 
    "seconds_since_last_alert_mean", "hour_sin_first", "hour_cos_first",
    "dow_first", "is_weekend_max", "is_unscheduled_max",
    "temp_extreme_max", "stops_to_end_mean", "scheduled_time_to_end_mean",
    "num_updates_sum", "match_key_nunique", "route_id", 'delay_3_before',
]
TARGET = 'alert_in_next_15m_max'


# ── Cargar datos una sola vez ──────────────────────────────────────────────────

def filtro_comportamiento_alterado(df):
    """Elimina paradas que no tienen alerta en 15 minutos 
    y sí tienen en 30 minutos y por tanto alteran la línea"""

    mask_positivos = df[TARGET] == 1
    mask_negativos_limpios = (
        df['alert_in_next_30m_max'] == 0            
    )

    df = df[mask_positivos | mask_negativos_limpios]
    df = df.reset_index(drop=True)

    print(f"Dataset tras filtrar negativos ambiguos: {len(df):,} filas")
    print(f"  Positivos: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")
    print(f"  Negativos: {(df[TARGET]==0).sum():,} ({(df[TARGET]==0).mean()*100:.1f}%)")

    return df


def encoding_categorias(X_train, X_val, X_test):
    cols_ordinal_enc = ['route_id']   

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cols_ordinal_enc] = enc.fit_transform(X_train[cols_ordinal_enc])
    X_val[cols_ordinal_enc]   = enc.transform(X_val[cols_ordinal_enc])
    X_test[cols_ordinal_enc]  = enc.transform(X_test[cols_ordinal_enc])

    print("✓ Encoding completado")
    return X_train, X_val, X_test


def evaluar_baseline(X_train, y_train, X_test, y_test):
    """Entrena un clasificador aleatorio estratificado y devuelve sus métricas."""
    print("\nEvaluando baseline estratificado...")

    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)

    y_prob_base  = baseline.predict_proba(X_test)[:, 1]
    y_pred_base  = baseline.predict(X_test)

    metricas = {
        "baseline_pr_auc":   average_precision_score(y_test, y_prob_base),
        "baseline_roc_auc":  roc_auc_score(y_test, y_prob_base),
        "baseline_f1":       f1_score(y_test, y_pred_base, zero_division=0),
        "baseline_recall":   recall_score(y_test, y_pred_base, zero_division=0),
        "baseline_precision":precision_score(y_test, y_pred_base, zero_division=0),
    }

    print(f"  PR-AUC   baseline: {metricas['baseline_pr_auc']:.4f}  "
          f"(≈ prevalencia positivos: {y_test.mean():.4f})")
    print(f"  ROC-AUC  baseline: {metricas['baseline_roc_auc']:.4f}")
    print(f"  F1       baseline: {metricas['baseline_f1']:.4f}")

    return metricas, y_prob_base

def main():

    print(f"\nCargando dataset...")
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print("✓ Dataset cargado con exito")

    # Eliminar filas sin target
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = filtro_comportamiento_alterado(df)

    df_sorted = df.sort_values('merge_time')

    print(f"Features: {len(FEATURES)}")
    print(f"Filas:    {len(df_sorted):,}")
    print(f"\nDistribución del target:")
    print(df_sorted[TARGET].value_counts(normalize=True).round(3))


    #División de los datos en Entrenamiento-Validación-Test
    dias = df_sorted['merge_time'].dt.date.unique()
    dias_ordenados = sorted(dias)

    total_dias = len(dias_ordenados)
    corte_70   = dias_ordenados[int(total_dias * 0.70)]
    corte_85   = dias_ordenados[int(total_dias * 0.85)]

    print(f"Total días: {total_dias}")
    print(f"Primer día: {dias_ordenados[0]}")
    print(f"Último día: {dias_ordenados[-1]}")
    print(f"\nCorte train (70%): {corte_70}")
    print(f"Corte val   (85%): {corte_85}")


    train = df_sorted[df_sorted['merge_time'].dt.date <  corte_70]
    val   = df_sorted[(df_sorted['merge_time'].dt.date >= corte_70) &
            (df_sorted['merge_time'].dt.date <  corte_85)]
    test  = df_sorted[df_sorted['merge_time'].dt.date >= corte_85]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    n = len(df)
    print(f"Train: {len(train):,} ({len(train)/n*100:.0f}%)  "
        f"{train['merge_time'].min().date()} → {train['merge_time'].max().date()}")
    print(f"Val:   {len(val):,} ({len(val)/n*100:.0f}%)  "
        f"{val['merge_time'].min().date()} → {val['merge_time'].max().date()}")
    print(f"Test:  {len(test):,} ({len(test)/n*100:.0f}%)  "
        f"{test['merge_time'].min().date()} → {test['merge_time'].max().date()}")


    X_train, X_val, X_test = encoding_categorias(X_train, X_val, X_test)

    metricas_baseline, y_prob_base = evaluar_baseline(X_train, y_train, X_test, y_test)


    # ── Búsqueda de hiperparámetros óptimos ─────────────────────────────────────────

    # Ratio de desbalance para scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Ratio desbalance: {ratio:.1f}:1")

    def objective(trial):
        params = {
            'max_depth':          trial.suggest_int('max_depth', 3, 10),
            'min_child_weight':   trial.suggest_int('min_child_weight', 1, 100),
            'gamma':              trial.suggest_float('gamma', 0, 10),
            'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':       trial.suggest_int('n_estimators', 200, 800),
            'subsample':          trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel':  trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'reg_alpha':          trial.suggest_float('reg_alpha', 1e-6, 100, log=True),
            'reg_lambda':         trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
            'max_delta_step':     trial.suggest_float('max_delta_step', 0, 10),
            'scale_pos_weight':   ratio,
            'tree_method':        'hist',
            'eval_metric':        'aucpr',
            'early_stopping_rounds': 30,
            'random_state':         42,
        }
        modelo = XGBClassifier(**params)
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = modelo.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    print("\nIniciando búsqueda de hiperparámetros...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print(study.best_params)

    # ── Entrenamiento del modelo final ─────────────────────────────────────────

    best_params = study.best_params

    params_fijos = {
        'scale_pos_weight': ratio,
        'tree_method': 'hist',
        'eval_metric': 'aucpr',
        'early_stopping_rounds': 30,
        'random_state': 42
    }

    parametros = {**params_fijos, **best_params}

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=NAME,
        config=parametros,
    )

    print("\nEntrenando modelo final...")
    modelo_agregado = XGBClassifier(**parametros)
    modelo_agregado.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


    # ── Evaluación del modelo final ─────────────────────────────────────────

    y_prob = modelo_agregado.predict_proba(X_test)[:, 1]

    # Threshold óptimo por F1
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int),
                            zero_division=0) for t in thresholds]
    threshold_opt = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob >= threshold_opt).astype(int)

    print(f"\nThreshold óptimo: {threshold_opt:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Curvas PR
    precision_base, recall_base, _   = precision_recall_curve(y_test, y_prob_base)
    precision_model, recall_model, _ = precision_recall_curve(y_test, y_prob)

    #Importancia Features
    importancias = pd.DataFrame({
        "feature":    FEATURES,
        "importance": modelo_agregado.feature_importances_
    }).sort_values("importance", ascending=False)

    wandb.log({
        **metricas_baseline,

        "auc_roc":   roc_auc_score(y_test, y_prob),
        "pr_auc":    average_precision_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "threshold_opt": float(threshold_opt),

        # Curva PR - Baseline
        "curva_pr_baseline": wandb.plot.line_series(
            xs=recall_base.tolist(),
            ys=[precision_base.tolist()],
            keys=["Baseline"],
            title="Precision-Recall Curve - Baseline",
            xname="Recall"
        ),

        # Curva PR - Modelo
        "curva_pr_modelo": wandb.plot.line_series(
            xs=recall_model.tolist(),
            ys=[precision_model.tolist()],
            keys=["XGBoost"],
            title="Precision-Recall Curve - Modelo",
            xname="Recall"
        ),

        # Matriz de confusión
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred.tolist(),
            class_names=["No alerta", "Alerta"]
        ),

        # Importancia de features
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias),
            label="feature",
            value="importance",
            title="Feature Importance"
        ),
    })

    modelo_agregado.save_model("modelo_agregado_30min.json")
    artifact = wandb.Artifact("modelo_agregado_30min", type="model")
    artifact.add_file("modelo_agregado_30min.json")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    main()
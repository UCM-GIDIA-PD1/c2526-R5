"""
Entrena un XGBoost para detección temprana de alertas MTA
trabajando a nivel de línea (route_id + direction + ventana 30min).
Primero se carga el dataset raw de 21 M filas parada x 30 mins. Después se agrega
a nivel de línea, lo que supone una reducción de filas a ~800k aprox.
Posteriormente, se lleva a cabo un encoding ordinal de route_id y direction y un
split temporal 70/15/15 por fechas.
A continuación, se realiza la búsqueda de hiperparámetros con Optuna (30 trials),
optimizando PR-AUC en validación con early stopping.
Finalmente, se entrena el modelo final con los mejores hiperparámetros y se
registran en W&B las métricas, curvas PR, matriz de confusión e importancia
de features.
 
Target:  alert_in_next_30m  (1 si hay alerta MTA en los próximos 30 min)
Métrica: PR-AUC  (clases desbalanceadas)
 
Uso:
  python -m src.models.modelos_alertas.Optuna.XGBoost_alertas
 
"""

import os
import pandas as pd
import numpy as np
import wandb
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                              precision_recall_curve)
from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, TARGET, evaluar_baseline, evaluar_test, filtro_comportamiento_alterado,
    encoding_categorias, FEATURES_CON,
)


# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH = f"grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

ENTITY = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"
NAME = "optuna_modelo_agregado_30min_XGBoost"
TRIALS = 30



def main():

    print(f"\nCargando dataset...")
    df_raw = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print("✓ Dataset cargado con exito")

    df_raw = filtro_comportamiento_alterado(df_raw)

    # ── Re-agregación por línea ────────────────────────────────────────────────
    print("\nRe-agregando a nivel de línea...")
    df_linea = agregar_por_linea(df_raw)
    df_linea = agregar_features_rolling_retraso(df_linea)

    # ── Features ──────────────────────────────────────────────────────────────
    feats = [f for f in FEATURES_CON if f in df_linea.columns]

    train, val, test = split_temporal(df_linea)

    X_train, y_train = train[feats], train[TARGET]
    X_val,   y_val   = val[feats],   val[TARGET]
    X_test,  y_test  = test[feats],  test[TARGET]


    X_train, X_val, X_test = encoding_categorias(X_train, X_val, X_test)

    metricas_baseline, y_prob_base = evaluar_baseline(X_train, y_train, X_test, y_test)


    # ── Búsqueda de hiperparámetros óptimos ─────────────────────────────────────────

    # Ratio de desbalance para scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Ratio desbalance: {ratio:.1f}:1")

    print(f"\n-- Optuna {TRIALS} trials --")

    def objective(trial):
        params = {
            'max_depth':          trial.suggest_int('max_depth', 3, 10),
            'min_child_weight':   trial.suggest_int('min_child_weight', 1, 100),
            'gamma':              trial.suggest_float('gamma', 0, 10),
            'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':       trial.suggest_int('n_estimators', 200, 1000),
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
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
    print(study.best_params)

    # ── Entrenamiento del modelo final ─────────────────────────────────────────

    parametros = {
        'scale_pos_weight':      ratio,
        'tree_method':           'hist',
        'eval_metric':           'aucpr',
        'early_stopping_rounds': 30,
        'random_state':          42,
        **study.best_params,
    }

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

    metricas, y_prob, y_pred = evaluar_test(modelo_agregado, X_test, y_test)

    print(f"PR-AUC test: {metricas['pr_auc_test']:.4f}")
    print(f"F1 test:     {metricas['f1_test']:.4f}")
    print(f"Threshold:   {metricas['threshold_opt']:.2f}")

    print(classification_report(y_test, y_pred, zero_division=0))

    # Curvas PR
    precision_base, recall_base, _   = precision_recall_curve(y_test, y_prob_base)
    precision_model, recall_model, _ = precision_recall_curve(y_test, y_prob)

    #Importancia Features
    importancias = pd.DataFrame({
        "feature":    feats,
        "importance": modelo_agregado.feature_importances_
    }).sort_values("importance", ascending=False)

    wandb.log({
        **metricas_baseline,
        **metricas,
        
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
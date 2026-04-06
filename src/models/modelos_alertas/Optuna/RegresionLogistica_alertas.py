"""
Entrena una Regresion Logistica para deteccion temprana de alertas MTA
trabajando a nivel de linea (route_id + direction + ventana 30min).

-Carga el dataset raw desde MinIO.
-Agrega a nivel de linea via agregar_por_linea().
-Añade features de tendencia rolling via agregar_features_rolling_retraso().
-Split temporal 70/15/15 por fechas.
-Busqueda de hiperparametros con Optuna (30 trials) sobre X_train.
     route_id y direction se codifican con OneHotEncoder (mejor para modelos lineales).
     Numericas escaladas con StandardScaler. Todo dentro de sklearn Pipeline.
-Modelo final reentrenado en train+val con los mejores params.

Target:  alert_in_next_30m  (1 si hay alerta MTA en los proximos 30 min)
Metrica: PR-AUC  (clases desbalanceadas, ~18% positivos)

Uso:
  python -m src.models.modelos_alertas.Optuna.RegresionLogistica_alertas
"""

import os
import gc
import numpy as np
import pandas as pd
import wandb
import optuna
from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score, precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, TARGET, FEATURES_CON,
)

load_dotenv()

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH       = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"
ENTITY     = "pd1-c2526-team5"
PROJECT    = "pd1-c2526-team5"
NAME       = "logreg_linea_optuna"

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Categoricas con OneHotEncoder (mejor que ordinal para modelos lineales)
CAT_FEATURES = ['route_id', 'direction']



def build_pipeline(numeric_features, C=1.0, class_weight=None):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,    numeric_features),
        ("cat", categorical_transformer, CAT_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=C, class_weight=class_weight,
            max_iter=300, random_state=42, solver="lbfgs",
        )),
    ])


def main():
    print("Cargando dataset...")
    df_raw = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print(f"Dataset raw: {df_raw.shape[0]:,} filas")

    df = agregar_por_linea(df_raw)
    del df_raw
    gc.collect()
    df = agregar_features_rolling_retraso(df)

    train, val, test = split_temporal(df)

    FEATURES = [f for f in FEATURES_CON if f in df.columns]
    NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]

    X_train, y_train = train[FEATURES].fillna(0), train[TARGET]
    X_val,   y_val   = val[FEATURES].fillna(0),   val[TARGET]
    X_test,  y_test  = test[FEATURES].fillna(0),  test[TARGET]

    print(f"Features: {len(FEATURES)} ({len(NUM_FEATURES)} numericas + {len(CAT_FEATURES)} categoricas)")

    print("\n-- Baseline --")
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)
    y_prob_base = baseline.predict_proba(X_test)[:, 1]
    y_pred_base = baseline.predict(X_test)

    run_base = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="logreg_linea_optuna_baseline",
        config={"model": "baseline_estratificado", "nivel": "linea"},
        tags=["linea", "logreg", "optuna", "baseline"],
    )
    wandb.log({
        "pr_auc":    average_precision_score(y_test, y_prob_base),
        "auc_roc":   roc_auc_score(y_test, y_prob_base),
        "f1":        f1_score(y_test, y_pred_base, zero_division=0),
        "recall":    recall_score(y_test, y_pred_base, zero_division=0),
        "precision": precision_score(y_test, y_pred_base, zero_division=0),
    })
    run_base.finish()

    print(f"\n-- Optuna 30 trials --")

    def objective(trial):
        C            = trial.suggest_float("C", 1e-3, 10.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        pipe = build_pipeline(NUM_FEATURES, C=C, class_weight=class_weight)
        pipe.fit(X_train, y_train)
        pr_auc_val = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])

        print(f"  Trial {trial.number+1:02d}/30 | C={C:.4f} | class_weight={class_weight} | PR-AUC val: {pr_auc_val:.4f}")
        return pr_auc_val

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_C            = study.best_params["C"]
    best_class_weight = study.best_params["class_weight"]
    print(f"\nMejor PR-AUC val: {study.best_value:.4f}")
    print(f"Mejores params: C={best_C:.4f}, class_weight={best_class_weight}")

    print("\n-- Modelo final (train+val) --")
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])

    pipeline_final = build_pipeline(NUM_FEATURES, C=best_C, class_weight=best_class_weight)
    pipeline_final.fit(X_tv, y_tv)

    # threshold optimo por F1 en val
    y_val_prob  = pipeline_final.predict_proba(X_val)[:, 1]
    thresholds  = np.arange(0.05, 0.95, 0.01)
    f1s         = [f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    thr         = float(thresholds[np.argmax(f1s)])

    y_prob = pipeline_final.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= thr).astype(int)

    metricas = {
        "pr_auc_test":    float(average_precision_score(y_test, y_prob)),
        "auc_roc_test":   float(roc_auc_score(y_test, y_prob)),
        "f1_test":        float(f1_score(y_test, y_pred, zero_division=0)),
        "recall_test":    float(recall_score(y_test, y_pred, zero_division=0)),
        "precision_test": float(precision_score(y_test, y_pred, zero_division=0)),
        "threshold_opt":  thr,
    }

    print(f"PR-AUC test: {metricas['pr_auc_test']:.4f}")
    print(f"F1 test:     {metricas['f1_test']:.4f}")
    print(f"Threshold:   {metricas['threshold_opt']:.2f}")

    # feature importance por coeficientes
    preprocessor    = pipeline_final.named_steps["preprocessor"]
    feature_names   = list(NUM_FEATURES) + list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CAT_FEATURES))
    coef_abs        = np.abs(pipeline_final.named_steps["model"].coef_[0])
    importancias_df = pd.DataFrame({"feature": feature_names, "importance": coef_abs}).sort_values("importance", ascending=False)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_table = wandb.Table(
        data=[[r, p] for r, p in zip(rec.tolist(), prec.tolist())],
        columns=["recall", "precision"],
    )

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="logreg_linea_optuna_FINAL",
        config={"C": best_C, "class_weight": best_class_weight,
                "threshold_opt": thr, "nivel": "linea"},
        tags=["linea", "logreg", "optuna", "final"],
    )
    wandb.log({
        **metricas,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias_df.head(30)),
            label="feature", value="importance",
            title="Top 30 coeficientes - LogReg Optuna",
        ),
        "pr_curve": wandb.plot.line(
            pr_table, "recall", "precision",
            title="Precision-Recall Curve - LogReg Optuna",
        ),
    })
    run.finish()
    print("Logueado en W&B: logreg_linea_optuna_FINAL")


if __name__ == "__main__":
    main()

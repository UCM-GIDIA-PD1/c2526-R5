"""
Entrena una Regresion Logistica para deteccion temprana de alertas MTA
trabajando a nivel de linea (route_id + direction + ventana 30min).

Flujo:
-Carga el dataset raw desde MinIO.
-Agrega a nivel de linea via agregar_por_linea().
-Añade features de tendencia rolling via agregar_features_rolling_retraso().
-Split temporal 70/15/15 por fechas.
-Busqueda de hiperparametros con ParameterSampler (20 combinaciones) sobre X_train.
     route_id y direction se codifican con OneHotEncoder (mejor para modelos lineales).
     Numericas escaladas con StandardScaler. Todo dentro de sklearn Pipeline.
-Modelo final reentrenado en train+val con los mejores params.

Target:  alert_in_next_15m  (1 si hay alerta MTA en los proximos 15 min)
Metrica: PR-AUC  (clases desbalanceadas, ~18% positivos)

Uso:
  uv run python  src/models/modelos_alertas/Random/RegresionLogistica_alertas_random.py

"""

import os
import gc
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from scipy.stats import loguniform

# Componentes de sklearn para preprocessing, modelos y evaluación
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score, precision_recall_curve,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Funciones propias del proyecto
from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, TARGET, FEATURES_CON,
)

# Carga variables de entorno (credenciales, etc.)
load_dotenv()

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

# Path del dataset en MinIO
PATH       = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

# Configuración de W&B
ENTITY     = "pd1-c2526-team5"
PROJECT    = "pd1-c2526-team5"
NAME       = "logreg_linea_random"

# Número de combinaciones para Random Search
N_ITER = 20

# Variables categóricas (se codifican con OneHot)
CAT_FEATURES = ['route_id', 'direction']


# --------------------------------------------------------------------------
# Construcción del pipeline completo (preprocesado + modelo)
# --------------------------------------------------------------------------
def build_pipeline(numeric_features, C=1.0, class_weight=None):
    """
    Crea un pipeline de sklearn con:
    - Imputación + escalado para variables numéricas
    - Imputación + OneHot para categóricas
    - Regresión logística como modelo final
    """

    # Pipeline para variables numéricas
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # rellena NaNs con la mediana
        ("scaler",  StandardScaler()),                  # estandariza (media 0, varianza 1)
    ])

    # Pipeline para variables categóricas
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # rellena NaNs con valor más frecuente
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),   # codificación one-hot
    ])

    # Combina ambos tipos de features
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     numeric_features),
        ("cat", categorical_transformer, CAT_FEATURES),
    ])

    # Pipeline final: preprocesado + modelo
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=C,                          # regularización (hiperparámetro clave)
            class_weight=class_weight,    # balanceo de clases
            max_iter=300,
            random_state=42,
            solver="lbfgs",
        )),
    ])


def main():

    # ----------------------------------------------------------------------
    # 1. Carga y preparación de datos
    # ----------------------------------------------------------------------
    print("Cargando dataset...")
    df_raw = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print(f"Dataset raw: {df_raw.shape[0]:,} filas")

    # Agregación a nivel de línea (route_id + direction)
    df = agregar_por_linea(df_raw)

    # Liberar memoria
    del df_raw
    gc.collect()

    # Añadir features rolling (tendencias temporales)
    df = agregar_features_rolling_retraso(df)

    # Split temporal (train/val/test)
    train, val, test = split_temporal(df)

    # Selección de features disponibles
    FEATURES     = [f for f in FEATURES_CON if f in df.columns]
    NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]

    # Separación X / y
    X_train, y_train = train[FEATURES].fillna(0), train[TARGET]
    X_val,   y_val   = val[FEATURES].fillna(0),   val[TARGET]
    X_test,  y_test  = test[FEATURES].fillna(0),  test[TARGET]

    print(f"Features: {len(FEATURES)} ({len(NUM_FEATURES)} numericas + {len(CAT_FEATURES)} categoricas)")

    # ----------------------------------------------------------------------
    # 2. Baseline (modelo dummy)
    # ----------------------------------------------------------------------
    print("\n-- Baseline --")

    # Modelo trivial (predicción aleatoria estratificada)
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)

    y_prob_base = baseline.predict_proba(X_test)[:, 1]
    y_pred_base = baseline.predict(X_test)

    # Log en W&B
    run_base = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="logreg_linea_random_baseline",
        config={"model": "baseline_estratificado", "nivel": "linea"},
        tags=["linea", "logreg", "random_search", "baseline"],
    )

    wandb.log({
        "pr_auc":    average_precision_score(y_test, y_prob_base),
        "auc_roc":   roc_auc_score(y_test, y_prob_base),
        "f1":        f1_score(y_test, y_pred_base, zero_division=0),
        "recall":    recall_score(y_test, y_pred_base, zero_division=0),
        "precision": precision_score(y_test, y_pred_base, zero_division=0),
    })
    run_base.finish()

    # ----------------------------------------------------------------------
    # 3. Random Search de hiperparámetros
    # ----------------------------------------------------------------------
    param_dist = {
        "C":            loguniform(1e-3, 10.0),  # fuerza de regularización
        "class_weight": [None, "balanced"],      # balanceo de clases
    }

    combinaciones  = list(ParameterSampler(param_dist, n_iter=N_ITER, random_state=42))

    print(f"\n-- Random Search {N_ITER} combinaciones --")

    mejor_pr_auc   = 0
    mejores_params = None

    # Itera sobre combinaciones
    for i, params in enumerate(combinaciones):

        pipe = build_pipeline(NUM_FEATURES, C=params["C"], class_weight=params["class_weight"])
        pipe.fit(X_train, y_train)

        # Evaluación en validation
        pr_auc_val = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])

        print(f"  [{i+1:02d}/{N_ITER}] C={params['C']:.4f} | class_weight={params['class_weight']} | PR-AUC val: {pr_auc_val:.4f}")

        # Guardar mejor modelo
        if pr_auc_val > mejor_pr_auc:
            mejor_pr_auc   = pr_auc_val
            mejores_params = params

    print(f"\nMejor PR-AUC val: {mejor_pr_auc:.4f}")
    print(f"Mejores params: {mejores_params}")

    # ----------------------------------------------------------------------
    # 4. Entrenamiento final (train + val)
    # ----------------------------------------------------------------------
    print("\n-- Modelo final (train+val) --")

    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])

    pipeline_final = build_pipeline(
        NUM_FEATURES,
        C=mejores_params["C"],
        class_weight=mejores_params["class_weight"]
    )
    pipeline_final.fit(X_tv, y_tv)

    # ----------------------------------------------------------------------
    # 5. Optimización del threshold (maximizar F1)
    # ----------------------------------------------------------------------
    y_val_prob = pipeline_final.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s        = [f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0) for t in thresholds]

    # Threshold óptimo
    thr = float(thresholds[np.argmax(f1s)])

    # ----------------------------------------------------------------------
    # 6. Evaluación en test
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # 7. Importancia de variables (coeficientes del modelo)
    # ----------------------------------------------------------------------
    preprocessor    = pipeline_final.named_steps["preprocessor"]

    # Nombres de features tras OneHot
    feature_names   = list(NUM_FEATURES) + list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CAT_FEATURES)
    )

    # Importancia = valor absoluto de coeficientes
    coef_abs        = np.abs(pipeline_final.named_steps["model"].coef_[0])

    importancias_df = pd.DataFrame({
        "feature": feature_names,
        "importance": coef_abs
    }).sort_values("importance", ascending=False)

    # ----------------------------------------------------------------------
    # 8. Curva Precision-Recall
    # ----------------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    pr_table = wandb.Table(
        data=[[r, p] for r, p in zip(rec.tolist(), prec.tolist())],
        columns=["recall", "precision"],
    )

    # ----------------------------------------------------------------------
    # 9. Logging final en W&B
    # ----------------------------------------------------------------------
    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="logreg_linea_random_FINAL",
        config={**mejores_params, "threshold_opt": thr, "nivel": "linea"},
        tags=["linea", "logreg", "random_search", "final"],
    )

    wandb.log({
        **metricas,

        # Matriz de confusión
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred.tolist(),
            class_names=["No alerta", "Alerta"],
        ),

        # Importancia de variables
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias_df.head(30)),
            label="feature", value="importance",
            title="Top 30 coeficientes - LogReg Random",
        ),

        # Curva PR
        "pr_curve": wandb.plot.line(
            pr_table, "recall", "precision",
            title="Precision-Recall Curve - LogReg Random",
        ),
    })

    run.finish()
    print("Logueado en W&B: logreg_linea_random_FINAL")


if __name__ == "__main__":
    main()
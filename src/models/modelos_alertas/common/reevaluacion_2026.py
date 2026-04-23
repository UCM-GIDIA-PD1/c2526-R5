"""
Rutas MinIO:
  TRAIN : grupo5/aggregations/DataFrameGroupedByMin=30.parquet      (2025 agregado)
  TEST  : grupo5/aggregations/DataFrameGroupedByMin=30-2026.parquet (2026 agregado)

Variables de entorno requeridas:
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  WANDB_API_KEY

Uso:
  uv run python src/models/modelos_alertas/common/reevaluacion_2026.py
"""

import gc
import os

import joblib
from dotenv import load_dotenv

load_dotenv()
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    FEATURES_CON,
    TARGET,
    agregar_features_rolling_retraso,
    agregar_por_linea,
    evaluar_baseline,
    evaluar_test,
    filtro_comportamiento_alterado,
)

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

PATH_TRAIN = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"
PATH_TEST  = "grupo5/aggregations/DataFrameGroupedByMin=30-2026.parquet"

ENTITY = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"
NAME = "reevaluacion_alertas_2026"

# Métricas de referencia obtenidas en la fase anterior (test 2025)
# Se usan en 1.3 para comparar si el modelo se degrada sobre 2026
PR_AUC_FASE3 = None  


def encoding_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OrdinalEncoder para route_id y direction.
    Se ajusta sobre train; líneas desconocidas en test reciben -1.
    """
    cols_cat = ["route_id", "direction"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cols_cat] = enc.fit_transform(X_train[cols_cat])
    X_test[cols_cat]  = enc.transform(X_test[cols_cat])
    print("✓ Encoding completado")
    return X_train, X_test


def cargar_y_preparar(access_key, secret_key, path, label):
    """Descarga, filtra y agrega a nivel de línea un dataset de nivel parada."""
    print(f"\nCargando {label}...")
    df = download_df_parquet(access_key, secret_key, path)
    print(f"  ✓ {len(df):,} filas")
    df = filtro_comportamiento_alterado(df)
    df = agregar_por_linea(df)
    df = agregar_features_rolling_retraso(df)
    return df


def main():
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    #cargamos lod datos
    print("=" * 60)
    print("  [1/3] CARGA Y PREPARACIÓN DE DATOS")
    print("=" * 60)

    df_train = cargar_y_preparar(ACCESS_KEY, SECRET_KEY, PATH_TRAIN, "TRAIN – 2025")
    df_test  = cargar_y_preparar(ACCESS_KEY, SECRET_KEY, PATH_TEST,  "TEST  – 2026")
    gc.collect()

    feats = [f for f in FEATURES_CON if f in df_train.columns]

    X_train = df_train[feats].copy()
    y_train = df_train[TARGET]
    X_test  = df_test[feats].copy()
    y_test  = df_test[TARGET]

    X_train, X_test = encoding_train_test(X_train, X_test)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    prev_train = y_train.mean() * 100
    prev_test  = y_test.mean() * 100
    print(f"\n  Ratio desbalance (train 2025): {ratio:.1f}:1")
    print(f"  Prevalencia positivos – train 2025: {prev_train:.1f}%")
    print(f"  Prevalencia positivos – test  2026: {prev_test:.1f}%")

    print("\n" + "=" * 60)
    print("  [2/3] REENTRENAMIENTO CON TODO 2025 (apartado 1.1)")
    print("=" * 60)

    best_xgb_params = {
        "colsample_bylevel": 0.920703389007254,
        "colsample_bytree":  0.863489949535421,
        "gamma":             7.980950294359764,
        "learning_rate":     0.04638574237732011,
        "max_delta_step":    5.630240841414502,
        "max_depth":         9,
        "min_child_weight":  39,
        "n_estimators":      912,
        "reg_alpha":         0.0030248836472922427,
        "reg_lambda":        0.007687070109538379,
        "subsample":         0.6763208314934956,
    }

    modelo_xgb = XGBClassifier(
        **best_xgb_params,
        scale_pos_weight=ratio,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
    )
    print("Entrenando XGBoost sobre todos los datos de 2025...")
    modelo_xgb.fit(X_train, y_train, verbose=False)
    print("✓ Modelo entrenado")

    # Guardar modelo como artifact en W&B (v1 de modelo_xgb_alertas)
    joblib.dump(modelo_xgb, "modelo_xgb_produccion.joblib")
    artifact = wandb.Artifact(
        name="modelo_xgb_alertas",
        type="model",
        description=(
            "XGBoost reentrenado con todos los datos 2025 (train+val+test). "
            "Modelo final para producción – detección de alertas MTA."
        ),
    )
    artifact.add_file("modelo_xgb_produccion.joblib")
    run.log_artifact(artifact)
    print("✓ Artifact subido a W&B: modelo_xgb_alertas (v1)")

    print("\n" + "=" * 60)
    print("  [3/3] EVALUACIÓN SOBRE NUEVOS DATOS 2026 (apartado 1.3)")
    print("=" * 60)

    metricas_base, _ = evaluar_baseline(X_train, y_train, X_test, y_test)
    metricas_xgb, y_prob_xgb, y_pred_xgb = evaluar_test(modelo_xgb, X_test, y_test)

    print(f"\n  PR-AUC  (test 2026): {metricas_xgb['pr_auc_test']:.4f}")
    print(f"  ROC-AUC (test 2026): {metricas_xgb['auc_roc_test']:.4f}")
    print(f"  F1      (test 2026): {metricas_xgb['f1_test']:.4f}")
    print(f"  Recall  (test 2026): {metricas_xgb['recall_test']:.4f}")
    print(f"  Precision(test 2026): {metricas_xgb['precision_test']:.4f}")
    if PR_AUC_FASE3:
        delta = metricas_xgb["pr_auc_test"] - PR_AUC_FASE3
        print(f"\n  Δ PR-AUC vs fase anterior: {delta:+.4f}")

    # Feature importance
    imp_xgb = pd.DataFrame({
        "feature":    feats,
        "importance": modelo_xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  Top-5 variables más importantes:")
    print(imp_xgb.head(5).to_string(index=False))

    # Análisis por línea
    resultados_lineas = []
    for linea in df_test["route_id"].unique():
        mask = df_test["route_id"] == linea
        if mask.sum() == 0:
            continue
        y_seg   = y_test[mask]
        pred_seg = (
            modelo_xgb.predict_proba(X_test[mask])[:, 1] >= metricas_xgb["threshold_opt"]
        ).astype(int)
        resultados_lineas.append({
            "Linea":          linea,
            "Muestras":       int(mask.sum()),
            "Alertas_Reales": int(y_seg.sum()),
            "Prevalencia_%":  round(y_seg.mean() * 100, 1),
            "F1_XGBoost":     round(f1_score(y_seg, pred_seg, zero_division=0), 4),
        })

    df_lineas = pd.DataFrame(resultados_lineas).sort_values("F1_XGBoost", ascending=False)

    # Log en wandb
    p, r, _ = precision_recall_curve(y_test, y_prob_xgb)

    df_comp = pd.DataFrame([
        {"modelo": "Baseline", **metricas_base},
        {"modelo": "XGBoost",  **metricas_xgb},
    ]).set_index("modelo")

    wandb.log({
        "comparativa":             wandb.Table(dataframe=df_comp.reset_index()),
        "segmentacion_por_lineas": wandb.Table(dataframe=df_lineas),
        "feature_importance_xgb":  wandb.Table(dataframe=imp_xgb),

        "curva_pr_xgb": wandb.plot.line_series(
            xs=r.tolist(), ys=[p.tolist()],
            keys=["XGBoost"],
            title="PR Curve – XGBoost (evaluación 2026)",
            xname="Recall",
        ),
        "confusion_xgb": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred_xgb.tolist(),
            class_names=["No alerta", "Alerta"],
        ),

        # Métricas planas para comparar entre runs
        **{f"xgb_{k}": v for k, v in metricas_xgb.items()},

        # Distribución del target en train vs test
        "prevalencia_positivos_train_%": round(prev_train, 2),
        "prevalencia_positivos_test_%":  round(prev_test, 2),
    })

    run.finish()
    print("\n✓ Reevaluación completada. Resultados subidos a W&B.")
    print("  Modelo de producción guardado en: modelo_xgb_produccion.joblib")


if __name__ == "__main__":
    main()

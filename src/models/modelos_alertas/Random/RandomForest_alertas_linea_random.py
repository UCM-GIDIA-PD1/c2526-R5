"""
Entrena un Random Forest para deteccion temprana de alertas MTA
trabajando a nivel de linea (route_id + direction + ventana 30min).
Primero se carga el dataset raw de 21 m filas parada x 30 mins. Después agregamosa nivel de línea, esto supone una reducción de filas a 800k aprox.
Posteriormente, se lleva a cabo un encoding cardinal de route_id y direction con Label Encoder y un split temporal 70/15/15 por fechas
A continuación, va la búsqueda de hiperparámetros con ParameterSampler(implementación de Random Search en scikit-learn) .
Finalmente, se registra el modelo final tanto con el feature ('seconds_since_last_alert') como sin él. Lo hacemos así para demostrara el gran impacto
que esta tiene en nuestros modelos.

Cada trial = 1 run en W&B. Los modelos finales loguean ademas
PR curve, feature importance y confusion matrix.

Target:  alert_in_next_15m  (1 si hay alerta MTA en los proximos 15 min)
Metrica: PR-AUC  (clases desbalanceadas, ~18% positivos)


Uso:
  uv run python src/models/modelos_alertas/Random/RandomForest_alertas_linea_random.py

"""

import os
import gc
import pandas as pd
import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                              recall_score, precision_score, precision_recall_curve)
from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, TARGET, evaluar_test,
    FEATURES_CON, FEATURES_SIN,
)


load_dotenv()

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

ENTITY  = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"

N_ITER      = 30
SAMPLE_FRAC = 0.30


def log_final_wandb(run_name, config, metricas, y_test, y_prob, y_pred,
                    features, importances, tags):
    """Loguea modelo final con todas las visualizaciones en W&B."""
    importancias_df = pd.DataFrame({
        "feature":    features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_table = wandb.Table(
        data=[[r, p] for r, p in zip(rec.tolist(), prec.tolist())],
        columns=["recall", "precision"]
    )

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name=run_name, config=config, tags=tags,
    )
    wandb.log({
        **metricas,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias_df),
            label="feature", value="importance",
            title=f"Feature Importance - {run_name}",
        ),
        "pr_curve": wandb.plot.line(
            pr_table, "recall", "precision",
            title=f"Precision-Recall Curve - {run_name}",
        ),
    })
    run.finish()


def main():

   
    print("\nCargando dataset desde MinIO...")
    df_raw = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print(f"Dataset raw: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")

   
    print("\nAgregando por linea...")
    df_linea = agregar_por_linea(df_raw)
    del df_raw
    gc.collect()
    df_linea = agregar_features_rolling_retraso(df_linea)
    print(f"Dataset linea: {df_linea.shape[0]:,} filas x {df_linea.shape[1]} columnas")
    print(f"Positivos: {df_linea[TARGET].mean()*100:.1f}%")

    # encoding necesario para random forest
    for col in ['route_id', 'direction']:
        le = LabelEncoder()
        df_linea[col] = le.fit_transform(df_linea[col].astype(str))

    feats_con = [f for f in FEATURES_CON if f in df_linea.columns]
    feats_sin = [f for f in FEATURES_SIN if f in df_linea.columns]

    print(f"Features con seg_alerta: {len(feats_con)}")
    print(f"Features sin seg_alerta: {len(feats_sin)}")

    
    train, val, test = split_temporal(df_linea)

    y_train = train[TARGET]
    y_val   = val[TARGET]
    y_test  = test[TARGET]

    train_search = train.sample(frac=SAMPLE_FRAC, random_state=42)
    y_search     = train_search[TARGET]

    print(f"Muestra busqueda ({SAMPLE_FRAC*100:.0f}%): {len(train_search):,}")

    print("\n-- Baseline --")
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(train[feats_con].fillna(0), y_train)
    y_prob_base = baseline.predict_proba(test[feats_con].fillna(0))[:, 1]
    y_pred_base = baseline.predict(test[feats_con].fillna(0))

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="rf_linea_random_15m_baseline",
        config={"model": "baseline_estratificado", "nivel": "linea"},
        tags=["linea", "random_forest", "random_search", "baseline"],
    )
    wandb.log({
        "pr_auc":    average_precision_score(y_test, y_prob_base),
        "auc_roc":   roc_auc_score(y_test, y_prob_base),
        "f1":        f1_score(y_test, y_pred_base, zero_division=0),
        "recall":    recall_score(y_test, y_pred_base, zero_division=0),
        "precision": precision_score(y_test, y_pred_base, zero_division=0),
    })
    run.finish()
    print("Baseline logueado en W&B")

    param_dist = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [5, 10, 15, 20, None],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features':     ['sqrt', 'log2', 0.3],
        'class_weight':     ['balanced', 'balanced_subsample'],
    }
    combinaciones = list(ParameterSampler(param_dist, n_iter=N_ITER, random_state=42))

    print(f"\n-- Random Search {N_ITER} combinaciones (con seg_alerta) --")

    X_search = train_search[feats_con].fillna(0)
    X_val_c  = val[feats_con].fillna(0)

    mejor_pr_auc  = 0
    mejores_params = None

    for i, params in enumerate(combinaciones):
        params_completos = {**params, 'random_state': 42, 'n_jobs': -1}

        modelo = RandomForestClassifier(**params_completos)
        modelo.fit(X_search, y_search)

        y_prob_val = modelo.predict_proba(X_val_c)[:, 1]
        pr_auc_val = average_precision_score(y_val, y_prob_val)
        f1_val     = f1_score(y_val, (y_prob_val >= 0.5).astype(int), zero_division=0)

        print(f"  [{i+1:02d}/{N_ITER}] PR-AUC val: {pr_auc_val:.4f} | F1 val: {f1_val:.4f}")

        if pr_auc_val > mejor_pr_auc:
            mejor_pr_auc   = pr_auc_val
            mejores_params = params_completos

    print(f"\nMejor PR-AUC val: {mejor_pr_auc:.4f}")
    print(f"Mejores params: {mejores_params}")

    # 8. Modelo final CON seg_alerta
    print("\n-- Modelo final CON seg_alerta (train + val) --")
    X_train_final = pd.concat([train[feats_con], val[feats_con]]).fillna(0)
    y_train_final = pd.concat([y_train, y_val])

    modelo_con = RandomForestClassifier(**mejores_params)
    modelo_con.fit(X_train_final, y_train_final)

    metricas_con, y_prob_con, y_pred_con = evaluar_test(modelo_con, test[feats_con].fillna(0), y_test)

    print(f"PR-AUC test: {metricas_con['pr_auc_test']:.4f}")
    print(f"F1 test:     {metricas_con['f1_test']:.4f}")
    print(f"Threshold:   {metricas_con['threshold_opt']:.2f}")

    log_final_wandb(
        run_name="rf_linea_random_15m_FINAL_con_seg",
        config={**mejores_params, "nivel": "linea", "con_seg_alerta": True},
        metricas=metricas_con,
        y_test=y_test, y_prob=y_prob_con, y_pred=y_pred_con,
        features=feats_con, importances=modelo_con.feature_importances_,
        tags=["linea", "random_forest", "random_search", "final", "con_seg_alerta"],
    )
    print("Logueado en W&B: rf_linea_random_FINAL_con_seg")

    # 9. Modelo final SIN seg_alerta (mismos hiperparametros)
    print("\n-- Modelo final SIN seg_alerta (mismos hiperparametros) --")
    X_train_sin = pd.concat([train[feats_sin], val[feats_sin]]).fillna(0)
    X_test_sin  = test[feats_sin].fillna(0)

    modelo_sin = RandomForestClassifier(**mejores_params)
    modelo_sin.fit(X_train_sin, y_train_final)

    metricas_sin, y_prob_sin, y_pred_sin = evaluar_test(modelo_sin, X_test_sin, y_test)

    print(f"PR-AUC test: {metricas_sin['pr_auc_test']:.4f}")
    print(f"F1 test:     {metricas_sin['f1_test']:.4f}")
    print(f"Threshold:   {metricas_sin['threshold_opt']:.2f}")

    log_final_wandb(
        run_name="rf_linea_random_15m_FINAL_sin_seg",
        config={**mejores_params, "nivel": "linea", "con_seg_alerta": False},
        metricas=metricas_sin,
        y_test=y_test, y_prob=y_prob_sin, y_pred=y_pred_sin,
        features=feats_sin, importances=modelo_sin.feature_importances_,
        tags=["linea", "random_forest", "random_search", "final", "sin_seg_alerta"],
    )
    print("Logueado en W&B: rf_linea_random_FINAL_sin_seg")

    # 10. Resumen
    print(f"\n{'='*60}")
    print("RESUMEN - RF por linea (Random Search)")
    print(f"{'='*60}")
    resumen = pd.DataFrame({
        "con seg_alerta": metricas_con,
        "sin seg_alerta": metricas_sin,
    })
    print(resumen.to_string(float_format='{:.4f}'.format))
    print(f"{'='*60}")
    print(f"Impacto de seg_alerta en PR-AUC: "
          f"{metricas_con['pr_auc_test'] - metricas_sin['pr_auc_test']:+.4f}")


if __name__ == "__main__":
    main()

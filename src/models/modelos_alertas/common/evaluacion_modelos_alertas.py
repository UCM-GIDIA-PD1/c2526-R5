import os
import pandas as pd
import numpy as np
import wandb
import optuna

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, encoding_categorias,
    evaluar_baseline, evaluar_test,
    TARGET, FEATURES_CON, filtro_comportamiento_alterado
)

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH       = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"
ENTITY     = "pd1-c2526-team5"
PROJECT    = "pd1-c2526-team5"

NAME = "evaluacion_modelos_alertas_FINAL"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():

    # ── Carga y pipeline compartido ────────────────────────────────────────────
    print(f"\nCargando dataset...")
    df_raw = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print("✓ Dataset cargado con exito")

    df_raw = filtro_comportamiento_alterado(df_raw)
    
    # ── Re-agregación por línea ────────────────────────────────────────────────
    print("\nRe-agregando a nivel de línea...")
    df_linea = agregar_por_linea(df_raw)
    df_linea = agregar_features_rolling_retraso(df_linea)

    feats = [f for f in FEATURES_CON if f in df_linea.columns]


    # ── Split y encoding ───────────────────────────────────────────────────────
    train, val, test = split_temporal(df_linea)

    X_train, y_train = train[feats], train[TARGET]
    X_val,   y_val   = val[feats],   val[TARGET]
    X_test,  y_test  = test[feats],  test[TARGET]

    X_train, X_val, X_test = encoding_categorias(X_train, X_val, X_test)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Ratio desbalance: {ratio:.1f}:1")


    # ── Escalado adicional para regresión logística ────────────────────────────    
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_test_lr  = X_test.copy()

    X_train_lr = X_train_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_lr  = X_test_lr.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_train_lr_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_lr),
        columns=X_train_lr.columns,
        index=X_train_lr.index,
    )
    X_test_lr_scaled = pd.DataFrame(
        scaler.transform(X_test_lr),
        columns=X_test_lr.columns,
        index=X_test_lr.index,
    )

    # ── Baseline ───────────────────────────────────────────────────────────────
    metricas_base, y_prob_base = evaluar_baseline(X_train, y_train, X_test, y_test)

    print("Entrenamiento de los modelos empleando los parámetros óptimos...")
    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("ENtrenando XGBoost...")
    best_xgb = {
        "colsample_bylevel": 0.920703389007254,
        "colsample_bytree":0.863489949535421,
        "gamma":7.980950294359764,
        "learning_rate":0.04638574237732011,
        "max_delta_step":5.630240841414502,
        "max_depth":9,
        "min_child_weight":39,
        "n_estimators":912,
        "reg_alpha":0.0030248836472922427,
        "reg_lambda":0.007687070109538379,
        "subsample":0.6763208314934956
    }
    modelo_xgb = XGBClassifier(
        **best_xgb,
        scale_pos_weight=ratio, tree_method='hist',
        eval_metric='aucpr', early_stopping_rounds=30, random_state=42,
    )
    modelo_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metricas_xgb, y_prob_xgb, y_pred_xgb = evaluar_test(modelo_xgb, X_test, y_test)


    # ── RandomForest ───────────────────────────────────────────────────────────
    print("Entrenando Random Forest...")
    best_rf = {
        "n_estimators":384,
        "max_depth":12,
        "min_samples_leaf":5,
        "max_features":0.3,
        "class_weight":'balanced',
    }
    modelo_rf = RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train_val, y_train_val)
    metricas_rf, y_prob_rf, y_pred_rf = evaluar_test(modelo_rf, X_test, y_test)


    # ── Regresión Logística ────────────────────────────────────────────────────
    print("Entrenando Regresión Logística...")
    best_lr   = {
        "C": 9.816798250253951,
        "class_weight" : None
    }
    modelo_lr = LogisticRegression(
        C=best_lr["C"], class_weight=best_lr["class_weight"],
        max_iter=300, random_state=42, solver="lbfgs",
    )
    modelo_lr.fit(X_train_lr_scaled, y_train_val)
    metricas_lr, y_prob_lr, y_pred_lr = evaluar_test(modelo_lr, X_test_lr_scaled, y_test)


    # ── Tabla comparativa ──────────────────────────────────────────────────────
    print("\n── Comparativa final ─────────────────────────────────────────────")
    df_comp = pd.DataFrame([
        {"modelo": "Baseline",     **metricas_base},
        {"modelo": "XGBoost",      **metricas_xgb},
        {"modelo": "RandomForest", **metricas_rf},
        {"modelo": "LogisticRegression", **metricas_lr},
    ]).set_index("modelo")
    print(df_comp.to_string())



    # ── Cálculo de Feature Importance ──────────────────────────────────────────
    print("\nCalculando importancia de variables...")

    imp_xgb = pd.DataFrame({
        "feature": feats,
        "importance": modelo_xgb.feature_importances_
    }).sort_values(by="importance", ascending=False)

    imp_rf = pd.DataFrame({
        "feature": feats,
        "importance": modelo_rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importancias_lr = np.abs(modelo_lr.coef_[0])
    imp_lr = pd.DataFrame({
        "feature": feats,
        "importance": importancias_lr / importancias_lr.sum() # Normalizamos
    }).sort_values(by="importance", ascending=False)


    # ── Análisis por líneas para todos los modelos ─────────────────
    print("\nAnalizando rendimiento por líneas para todos los modelos...")
    
    lineas_test = test['route_id'].unique()
    resultados_segmentos = []

    for linea in lineas_test:
        mask = (test['route_id'] == linea)
        
        if mask.sum() > 0:

            X_seg = X_test[mask]
            y_seg = y_test[mask]
            
            X_seg_lr = X_test_lr_scaled[mask]

            pred_xgb = (modelo_xgb.predict_proba(X_seg)[:, 1] >= metricas_xgb["threshold_opt"]).astype(int)
            pred_rf  = (modelo_rf.predict_proba(X_seg)[:, 1] >= metricas_rf["threshold_opt"]).astype(int)
            pred_lr  = (modelo_lr.predict_proba(X_seg_lr)[:, 1] >= metricas_lr["threshold_opt"]).astype(int)

            resultados_segmentos.append({
                "Linea": linea,
                "Muestras": mask.sum(),
                "Alertas_Reales": y_seg.sum(),
                "F1_XGBoost": round(f1_score(y_seg, pred_xgb, zero_division=0), 4),
                "F1_RandomForest": round(f1_score(y_seg, pred_rf, zero_division=0), 4),
                "F1_LogisticReg": round(f1_score(y_seg, pred_lr, zero_division=0), 4)
            })

    df_segmentos = pd.DataFrame(resultados_segmentos)


    # ── W&B ────────────────────────────────────────────────────────────────────
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    p_xgb, r_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    p_rf,  r_rf,  _ = precision_recall_curve(y_test, y_prob_rf)
    p_lr,  r_lr,  _ = precision_recall_curve(y_test, y_prob_lr)

    wandb.log({

        "comparativa": wandb.Table(dataframe=df_comp.reset_index()),
        "Segmentacion por Lineas": wandb.Table(dataframe=df_segmentos),

        "Feature Importance XGBoost": wandb.Table(dataframe=imp_xgb),
        "Feature Importance RandomForest": wandb.Table(dataframe=imp_rf),
        "Feature Importance LogisticReg": wandb.Table(dataframe=imp_lr),

        "curva_pr_xgb": wandb.plot.line_series(
            xs=r_xgb.tolist(),
            ys=[p_xgb.tolist()],
            keys=["XGBoost"],
            title="PR Curve - XGBoost",
            xname="Recall",
        ),
        "curva_pr_rf": wandb.plot.line_series(
            xs=r_rf.tolist(),
            ys=[p_rf.tolist()],
            keys=["RandomForest"],
            title="PR Curve - RandomForest",
            xname="Recall",
        ),
        "curva_pr_lr": wandb.plot.line_series(
            xs=r_lr.tolist(),
            ys=[p_lr.tolist()],
            keys=["LogisticRegression"],
            title="PR Curve - LogisticRegression",
            xname="Recall",
        ),
        "Matriz Confusión XGBoost": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_xgb.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "Matriz Confusión Random Forest": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_rf.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "Matriz Confusión Regresión Logística": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_lr.tolist(),
            class_names=["No alerta", "Alerta"],
        ),

        **{f"xgb_{k}": v for k, v in metricas_xgb.items()},
        **{f"rf_{k}":  v for k, v in metricas_rf.items()},
        **{f"lr_{k}":  v for k, v in metricas_lr.items()},
    })
    run.finish()


if __name__ == "__main__":
    main()
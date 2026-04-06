import os
import pandas as pd
import numpy as np
import wandb
import optuna

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet
from src.models.modelos_alertas.common.pipeline_linea import (
    agregar_por_linea, agregar_features_rolling_retraso,
    split_temporal, get_features, encoding_categorias,
    evaluar_baseline, evaluar_test,
    TARGET, TARGET_RAW, filtro_comportamiento_alterado
)

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
PATH       = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"
ENTITY     = "pd1-c2526-team5"
PROJECT    = "pd1-c2526-team5"

NAME = "evaluacion_modelos_alertas"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Búsqueda de hiperparámetros ────────────────────────────────────────────────

def buscar_xgboost(X_train, y_train, X_val, y_val, ratio, n_trials=30):
    def objective(trial):
        params = {
            'max_depth':             trial.suggest_int('max_depth', 3, 10),
            'min_child_weight':      trial.suggest_int('min_child_weight', 1, 100),
            'gamma':                 trial.suggest_float('gamma', 0, 10),
            'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':          trial.suggest_int('n_estimators', 200, 1000),
            'subsample':             trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree':      trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel':     trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'reg_alpha':             trial.suggest_float('reg_alpha', 1e-6, 100, log=True),
            'reg_lambda':            trial.suggest_float('reg_lambda', 1e-6, 100, log=True),
            'max_delta_step':        trial.suggest_float('max_delta_step', 0, 10),
            'scale_pos_weight':      ratio,
            'tree_method':           'hist',
            'eval_metric':           'aucpr',
            'early_stopping_rounds': 30,
            'random_state':          42,
        }
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Mejor PR-AUC val XGBoost: {study.best_value:.4f}")
    return study.best_params

def buscar_rf(X_train, y_train, X_val, y_val, n_trials=20):
    X_sample = X_train.sample(frac=0.05, random_state=42)
    y_sample  = y_train.loc[X_sample.index]

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
        m = RandomForestClassifier(**params)
        m.fit(X_sample, y_sample)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Mejor PR-AUC val RandomForest: {study.best_value:.4f}")
    return study.best_params


def buscar_logreg(X_train_lr, y_train, X_val_lr, y_val, n_trials=30):
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        m = LogisticRegression(
            C=C, class_weight=class_weight,
            max_iter=300, random_state=42, solver="lbfgs",
        )
        m.fit(X_train_lr, y_train)
        return average_precision_score(y_val, m.predict_proba(X_val_lr)[:, 1])
 
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Mejor PR-AUC val LogReg: {study.best_value:.4f}")
    return study.best_params


def main():

    # ── Carga y pipeline compartido ────────────────────────────────────────────
    print("Cargando dataset...")
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    df = df.dropna(subset=[TARGET_RAW])
    df[TARGET_RAW] = df[TARGET_RAW].astype(int)
    
    df = agregar_por_linea(df)
    df = agregar_features_rolling_retraso(df)

    FEATURES = get_features(df)
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())


    # ── Split y encoding ───────────────────────────────────────────────────────
    train, val, test = split_temporal(df)

    X_train, y_train = train[FEATURES].copy(), train[TARGET]
    X_val,   y_val   = val[FEATURES].copy(),   val[TARGET]
    X_test,  y_test  = test[FEATURES].copy(),  test[TARGET]

    X_train, X_val, X_test = encoding_categorias(X_train, X_val, X_test)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Ratio desbalance: {ratio:.1f}:1")


    # ── Escalado adicional para regresión logística ────────────────────────────
    scaler = StandardScaler()
    X_train_lr = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_lr = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_lr = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )


    # ── Baseline ───────────────────────────────────────────────────────────────
    metricas_base, y_prob_base = evaluar_baseline(X_train, y_train, X_test, y_test)


    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("\n── XGBoost ──")
    best_xgb = buscar_xgboost(X_train, y_train, X_val, y_val, ratio)
    modelo_xgb = XGBClassifier(
        **best_xgb,
        scale_pos_weight=ratio, tree_method='hist',
        eval_metric='aucpr', early_stopping_rounds=30, random_state=42,
    )
    modelo_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metricas_xgb, y_prob_xgb, y_pred_xgb = evaluar_test(modelo_xgb, X_test, y_test)


    # ── RandomForest ───────────────────────────────────────────────────────────
    print("\n── RandomForest ──")
    best_rf = buscar_rf(X_train, y_train, X_val, y_val)
    modelo_rf = RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train, y_train)
    metricas_rf, y_prob_rf, y_pred_rf = evaluar_test(modelo_rf, X_test, y_test)


    # ── Regresión Logística ────────────────────────────────────────────────────
    print("\n── Regresión Logística ──")
    best_lr   = buscar_logreg(X_train_lr, y_train, X_val_lr, y_val)
    modelo_lr = LogisticRegression(
        C=best_lr["C"], class_weight=best_lr["class_weight"],
        max_iter=300, random_state=42, solver="lbfgs",
    )
    modelo_lr.fit(X_train_lr, y_train)
    metricas_lr, y_prob_lr, y_pred_lr = evaluar_test(modelo_lr, X_test_lr, y_test)


    # ── Tabla comparativa ──────────────────────────────────────────────────────
    print("\n── Comparativa final ─────────────────────────────────────────────")
    df_comp = pd.DataFrame([
        {"modelo": "Baseline",     **metricas_base},
        {"modelo": "XGBoost",      **metricas_xgb},
        {"modelo": "RandomForest", **metricas_rf},
        {"modelo": "LogisticRegression", **metricas_lr},
    ]).set_index("modelo")
    print(df_comp.to_string())


    # ── W&B ────────────────────────────────────────────────────────────────────
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    p_xgb, r_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    p_rf,  r_rf,  _ = precision_recall_curve(y_test, y_prob_rf)
    p_lr,  r_lr,  _ = precision_recall_curve(y_test, y_prob_lr)

    wandb.log({
        "comparativa": wandb.Table(dataframe=df_comp.reset_index()),
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
        "confusion_xgb": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_xgb.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "confusion_rf": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(), preds=y_pred_rf.tolist(),
            class_names=["No alerta", "Alerta"],
        ),
        "confusion_lr": wandb.plot.confusion_matrix(
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
import os
import gc
import json
import joblib
import numpy as np
import pandas as pd
import wandb

from scipy.stats import loguniform

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.minio_client import download_df_parquet


# ── Configuración ──────────────────────────────────────────────────────────────

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

PATH = "grupo5/aggregations/DataFrameGroupedByMin=30.parquet"

ENTITY = "pd1-c2526-team5"
PROJECT = "pd1-c2526-team5"
NAME = "randomized_modelo_agregado_30min_logreg"

TARGET = "alert_in_next_15m_max"
TARGET_RAW = TARGET

COLS_RAW = [
    "route_id", "direction", "merge_time",
    "delay_seconds_mean",
    "lagged_delay_1_mean", "lagged_delay_2_mean", "delay_3_before",
    "actual_headway_seconds_mean", "is_unscheduled_max",
    "num_updates_sum", "match_key_nunique",
    "hour_sin_first", "hour_cos_first", "dow_first", "is_weekend_max",
    "seconds_since_last_alert_mean",
    TARGET,
]


# ── Preprocesado de datos ──────────────────────────────────────────────────────

def filtro_comportamiento_alterado(df: pd.DataFrame) -> pd.DataFrame:
    if "alert_in_next_30m_max" not in df.columns:
        print("No existe 'alert_in_next_30m_max'; se omite el filtro de negativos ambiguos.")
        return df.reset_index(drop=True)

    mask_positivos = df[TARGET] == 1
    mask_negativos_limpios = df["alert_in_next_30m_max"] == 0

    df = df[mask_positivos | mask_negativos_limpios].reset_index(drop=True)

    print(f"Dataset tras filtrar negativos ambiguos: {len(df):,} filas")
    print(f"  Positivos: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")
    print(f"  Negativos: {(df[TARGET] == 0).sum():,} ({(df[TARGET] == 0).mean()*100:.1f}%)")
    return df


def agregar_por_linea(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols_base = [
        "route_id", "direction", "merge_time",
        "delay_seconds_mean",
        "actual_headway_seconds_mean",
        "is_unscheduled_max",
        "num_updates_sum",
        "match_key_nunique",
        "hour_sin_first",
        "hour_cos_first",
        "dow_first",
        "is_weekend_max",
        "seconds_since_last_alert_mean",
        TARGET_RAW,
    ]

    cat_cols = [c for c in df_raw.columns if c.startswith("category_")]
    print(f"Categorías de alerta encontradas: {cat_cols}")

    if all(c in df_raw.columns for c in ["lagged_delay_1_mean", "lagged_delay_2_mean", "delay_3_before"]):
        lag_cols = ["lagged_delay_1_mean", "lagged_delay_2_mean", "delay_3_before"]
        lag_rename = {
            "lagged_delay_1_mean_mean": "lag1_mean_linea",
            "lagged_delay_2_mean_mean": "lag2_mean_linea",
            "delay_3_before_mean": "delay_3_before_mean",
        }
    elif all(c in df_raw.columns for c in ["delay_1_before", "delay_2_before", "delay_3_before"]):
        lag_cols = ["delay_1_before", "delay_2_before", "delay_3_before"]
        lag_rename = {
            "delay_1_before_mean": "lag1_mean_linea",
            "delay_2_before_mean": "lag2_mean_linea",
            "delay_3_before_mean": "delay_3_before_mean",
        }
    else:
        lag_cols = []
        lag_rename = {}

    cols_usar = [c for c in cols_base + lag_cols + cat_cols if c in df_raw.columns]

    df = df_raw[cols_usar].copy()
    df = df[df[TARGET_RAW].notna()].copy()

    df["merge_time"] = pd.to_datetime(df["merge_time"])
    df[TARGET_RAW] = df[TARGET_RAW].astype(int)
    df["parada_retrasada"] = (df["delay_seconds_mean"] > 60).astype(int)

    agg_dict = {
        "delay_seconds_mean": ["mean", "max", "std"],
        "parada_retrasada": ["sum", "count"],
        "actual_headway_seconds_mean": ["mean", "std"],
        "is_unscheduled_max": "max",
        "num_updates_sum": "sum",
        "match_key_nunique": "sum",
        "hour_sin_first": "first",
        "hour_cos_first": "first",
        "dow_first": "first",
        "is_weekend_max": "max",
        "seconds_since_last_alert_mean": "min",
        TARGET_RAW: "max",
    }

    for c in lag_cols:
        agg_dict[c] = "mean"

    for c in cat_cols:
        agg_dict[c] = "max"

    print("Agregando por línea...")
    df_linea = df.groupby(
        ["route_id", "direction", pd.Grouper(key="merge_time", freq="30min")],
        observed=True
    ).agg(agg_dict).reset_index()

    df_linea.columns = [
        "_".join(filter(None, col)) if isinstance(col, tuple) else col
        for col in df_linea.columns
    ]

    rename_dict = {
        "delay_seconds_mean_mean": "delay_mean_linea",
        "delay_seconds_mean_max": "delay_max_linea",
        "delay_seconds_mean_std": "delay_std_linea",
        "parada_retrasada_sum": "paradas_retrasadas",
        "parada_retrasada_count": "total_paradas",
        "actual_headway_seconds_mean_mean": "headway_mean_linea",
        "actual_headway_seconds_mean_std": "headway_std_linea",
        "is_unscheduled_max_max": "is_unscheduled",
        "num_updates_sum_sum": "num_updates",
        "match_key_nunique_sum": "match_key_nunique",
        "hour_sin_first_first": "hour_sin",
        "hour_cos_first_first": "hour_cos",
        "dow_first_first": "dow",
        "is_weekend_max_max": "is_weekend",
        "seconds_since_last_alert_mean_min": "seg_desde_ultima_alerta_linea",
        f"{TARGET_RAW}_max": TARGET,
        **lag_rename,
    }

    df_linea = df_linea.rename(columns=rename_dict)
    df_linea = df_linea.loc[:, ~df_linea.columns.duplicated()].copy()

    for c in ["lag1_mean_linea", "lag2_mean_linea", "delay_3_before_mean"]:
        if c not in df_linea.columns:
            df_linea[c] = 0.0

    fill_zero_cols = [
        "delay_mean_linea", "delay_max_linea", "delay_std_linea",
        "lag1_mean_linea", "lag2_mean_linea", "delay_3_before_mean",
        "headway_mean_linea", "headway_std_linea",
        "is_unscheduled", "num_updates", "match_key_nunique",
        "hour_sin", "hour_cos", "dow", "is_weekend",
        "seg_desde_ultima_alerta_linea",
    ]

    for col in fill_zero_cols:
        if col in df_linea.columns:
            df_linea[col] = pd.to_numeric(df_linea[col], errors="coerce").fillna(0)

    df_linea["paradas_retrasadas"] = df_linea["paradas_retrasadas"].fillna(0)
    df_linea["total_paradas"] = df_linea["total_paradas"].fillna(0)

    df_linea["pct_paradas_retrasadas"] = (
        df_linea["paradas_retrasadas"] / df_linea["total_paradas"].clip(lower=1)
    )
    df_linea["delay_acceleration_linea"] = (
        df_linea["delay_mean_linea"] - df_linea["lag1_mean_linea"]
    )
    df_linea["headway_cv"] = (
        df_linea["headway_std_linea"] / df_linea["headway_mean_linea"].clip(lower=1)
    )
    df_linea["colapso_linea"] = (
        (df_linea["pct_paradas_retrasadas"] > 0.5).astype(int)
    )
    df_linea["delay_x_aceleracion"] = (
        df_linea["delay_mean_linea"] * df_linea["delay_acceleration_linea"].clip(lower=0)
    )

    df_linea["seg_desde_ultima_alerta_linea"] = (
        df_linea["seg_desde_ultima_alerta_linea"].fillna(999999)
    )

    df_linea = df_linea.dropna(subset=[TARGET]).copy()
    df_linea[TARGET] = df_linea[TARGET].astype(int)

    del df
    gc.collect()

    print(f"Dataset línea: {df_linea.shape[0]:,} filas x {df_linea.shape[1]} columnas")
    print(f"Positivos: {df_linea[TARGET].mean()*100:.1f}%")

    return df_linea, cat_cols


def agregar_features_rolling_retraso(df_linea: pd.DataFrame) -> pd.DataFrame:
    df_linea = df_linea.sort_values(["route_id", "direction", "merge_time"]).copy()
    grp = df_linea.groupby(["route_id", "direction"])

    df_linea["delay_rolling4_mean"] = (
        grp["delay_mean_linea"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        .fillna(0)
    )

    df_linea["delay_rolling4_max"] = (
        grp["delay_max_linea"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
        .fillna(0)
    )

    df_linea["headway_rolling4_std"] = (
        grp["headway_mean_linea"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
        .fillna(0)
    )

    return df_linea


def get_features(cat_cols: list[str], df: pd.DataFrame) -> list[str]:
    features = [
        "headway_cv", "colapso_linea", "delay_x_aceleracion",
        "delay_mean_linea", "delay_max_linea", "delay_std_linea",
        "paradas_retrasadas", "pct_paradas_retrasadas",
        "lag1_mean_linea", "lag2_mean_linea", "delay_3_before_mean",
        "delay_acceleration_linea", "delay_rolling4_mean", "delay_rolling4_max", "headway_rolling4_std",
        "headway_mean_linea", "headway_std_linea",
        "is_unscheduled", "num_updates", "match_key_nunique",
        "hour_sin", "hour_cos", "dow", "is_weekend",
        "route_id", "direction",
        "seg_desde_ultima_alerta_linea",
    ] + cat_cols
    return [f for f in features if f in df.columns]


def preparar_dataset_modelo(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    df_model = df[features + [TARGET, "merge_time"]].copy()

    df_model = df_model.dropna(subset=[TARGET]).copy()
    df_model[TARGET] = pd.to_numeric(df_model[TARGET], errors="coerce")
    df_model = df_model.dropna(subset=[TARGET]).copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    categorical_features = ["direction", "route_id"] + [c for c in features if c.startswith("category_")]
    categorical_features = [c for c in categorical_features if c in df_model.columns]

    binary_features = [c for c in ["is_unscheduled", "is_weekend", "colapso_linea"] if c in df_model.columns]
    numeric_features = [c for c in features if c not in categorical_features]

    df_model[numeric_features] = df_model[numeric_features].replace([np.inf, -np.inf], np.nan)

    for col in binary_features:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0).astype(int)

    for col in categorical_features:
        df_model[col] = df_model[col].astype("string")

    df_model = df_model.sort_values("merge_time").reset_index(drop=True)

    fill_zero_cols = [
        "delay_mean_linea", "delay_max_linea", "delay_std_linea",
        "paradas_retrasadas", "total_paradas", "pct_paradas_retrasadas",
        "lag1_mean_linea", "lag2_mean_linea", "delay_3_before_mean",
        "delay_acceleration_linea", "delay_x_aceleracion",
        "headway_mean_linea", "headway_std_linea", "headway_cv",
        "delay_rolling4_mean", "delay_rolling4_max", "headway_rolling4_std",
        "is_unscheduled", "num_updates", "match_key_nunique",
        "hour_sin", "hour_cos", "dow", "is_weekend",
        "seg_desde_ultima_alerta_linea", "colapso_linea",
    ]

    for col in fill_zero_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

    if {"paradas_retrasadas", "total_paradas"}.issubset(df_model.columns):
        df_model["pct_paradas_retrasadas"] = (
            df_model["paradas_retrasadas"] / df_model["total_paradas"].clip(lower=1)
        )

    if {"delay_mean_linea", "lag1_mean_linea"}.issubset(df_model.columns):
        df_model["delay_acceleration_linea"] = (
            df_model["delay_mean_linea"] - df_model["lag1_mean_linea"]
        )

    if {"delay_mean_linea", "delay_acceleration_linea"}.issubset(df_model.columns):
        df_model["delay_x_aceleracion"] = (
            df_model["delay_mean_linea"] * df_model["delay_acceleration_linea"].clip(lower=0)
        )

    if {"headway_std_linea", "headway_mean_linea"}.issubset(df_model.columns):
        df_model["headway_cv"] = (
            df_model["headway_std_linea"] / df_model["headway_mean_linea"].clip(lower=1)
        )

    print("Shape tras limpieza:", df_model.shape)
    print(df_model[features + [TARGET]].isna().sum().sort_values(ascending=False).head(20))

    return df_model, categorical_features, numeric_features


def build_pipeline(categorical_features: list[str], numeric_features: list[str], C: float = 1.0, class_weight=None) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=300,
            random_state=42,
            solver="lbfgs",
        ))
    ])
    return pipeline


def evaluar_baseline(y_train, y_test):
    print("\nEvaluando baseline estratificado...")
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(np.zeros((len(y_train), 1)), y_train)

    X_dummy_test = np.zeros((len(y_test), 1))
    y_prob_base = baseline.predict_proba(X_dummy_test)[:, 1]
    y_pred_base = baseline.predict(X_dummy_test)

    metricas = {
        "baseline_pr_auc": average_precision_score(y_test, y_prob_base),
        "baseline_roc_auc": roc_auc_score(y_test, y_prob_base),
        "baseline_f1": f1_score(y_test, y_pred_base, zero_division=0),
        "baseline_recall": recall_score(y_test, y_pred_base, zero_division=0),
        "baseline_precision": precision_score(y_test, y_pred_base, zero_division=0),
    }

    print(f"  PR-AUC   baseline: {metricas['baseline_pr_auc']:.4f}")
    print(f"  ROC-AUC  baseline: {metricas['baseline_roc_auc']:.4f}")
    print(f"  F1       baseline: {metricas['baseline_f1']:.4f}")

    return metricas, y_prob_base


def main():
    print("\nCargando dataset...")
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, PATH)
    print("✓ Dataset cargado con éxito")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    df = filtro_comportamiento_alterado(df)

    print("\nRe-agregando a nivel de línea...")
    df_linea, cat_cols = agregar_por_linea(df)
    df_linea = agregar_features_rolling_retraso(df_linea)

    FEATURES = get_features(cat_cols, df_linea)
    df_model, categorical_features, numeric_features = preparar_dataset_modelo(df_linea, FEATURES)

    df_sorted = df_model.sort_values("merge_time").copy()

    print(f"\nFeatures: {len(FEATURES)}")
    print(f"Filas:    {len(df_sorted):,}")
    print("\nDistribución del target:")
    print(df_sorted[TARGET].value_counts(normalize=True).round(3))

    dias = df_sorted["merge_time"].dt.date.unique()
    dias_ordenados = sorted(dias)

    total_dias = len(dias_ordenados)
    corte_70 = dias_ordenados[int(total_dias * 0.70)]
    corte_85 = dias_ordenados[int(total_dias * 0.85)]

    print(f"Total días: {total_dias}")
    print(f"Primer día: {dias_ordenados[0]}")
    print(f"Último día: {dias_ordenados[-1]}")
    print(f"\nCorte train (70%): {corte_70}")
    print(f"Corte val   (85%): {corte_85}")

    train = df_sorted[df_sorted["merge_time"].dt.date < corte_70]
    val = df_sorted[
        (df_sorted["merge_time"].dt.date >= corte_70) &
        (df_sorted["merge_time"].dt.date < corte_85)
    ]
    test = df_sorted[df_sorted["merge_time"].dt.date >= corte_85]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    n = len(df_sorted)
    print(f"Train: {len(train):,} ({len(train)/n*100:.0f}%)  {train['merge_time'].min().date()} → {train['merge_time'].max().date()}")
    print(f"Val:   {len(val):,} ({len(val)/n*100:.0f}%)  {val['merge_time'].min().date()} → {val['merge_time'].max().date()}")
    print(f"Test:  {len(test):,} ({len(test)/n*100:.0f}%)  {test['merge_time'].min().date()} → {test['merge_time'].max().date()}")

    metricas_baseline, y_prob_base = evaluar_baseline(y_train, y_test)

    # ── RandomizedSearchCV ────────────────────────────────────────────────────
    # búsqueda solo sobre train; threshold se ajusta después con validación
    tscv = TimeSeriesSplit(n_splits=3)

    pipeline_search = build_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        C=1.0,
        class_weight=None,
    )

    param_distributions = {
        "model__C": loguniform(1e-3, 10.0),
        "model__class_weight": [None, "balanced"],
    }

    print("\nIniciando búsqueda de hiperparámetros con RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=pipeline_search,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    random_search.fit(X_train, y_train)

    print("Mejores hiperparámetros:")
    print(random_search.best_params_)
    print(f"Mejor score CV: {random_search.best_score_:.4f}")

    pipeline_final = random_search.best_estimator_

    # threshold óptimo en validación
    y_val_prob = pipeline_final.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores_val = [
        f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    threshold_opt = float(thresholds[np.argmax(f1_scores_val)])

    # evaluación final en test
    y_prob = pipeline_final.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold_opt).astype(int)

    print(f"\nThreshold óptimo: {threshold_opt:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    precision_base, recall_base, _ = precision_recall_curve(y_test, y_prob_base)
    precision_model, recall_model, _ = precision_recall_curve(y_test, y_prob)

    preprocessor = pipeline_final.named_steps["preprocessor"]
    model = pipeline_final.named_steps["model"]

    feature_names_out = preprocessor.get_feature_names_out()
    coef_abs = np.abs(model.coef_[0])

    importancias = pd.DataFrame({
        "feature": feature_names_out,
        "importance": coef_abs,
    }).sort_values("importance", ascending=False)

    parametros = {
        "model_type": "logistic_regression",
        "search_type": "RandomizedSearchCV",
        "target": TARGET,
        "features": FEATURES,
        "best_C": float(random_search.best_params_["model__C"]),
        "best_class_weight": random_search.best_params_["model__class_weight"],
        "threshold_opt": threshold_opt,
        "n_iter": 20,
        "cv_n_splits": 3,
        "cv_best_score": float(random_search.best_score_),
        "scoring": "average_precision",
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=NAME,
        config=parametros,
    )

    wandb.log({
        **metricas_baseline,
        "auc_roc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "threshold_opt": threshold_opt,
        "cv_best_score": float(random_search.best_score_),

        "curva_pr_baseline": wandb.plot.line_series(
            xs=recall_base.tolist(),
            ys=[precision_base.tolist()],
            keys=["Baseline"],
            title="Precision-Recall Curve - Baseline",
            xname="Recall",
        ),

        "curva_pr_modelo": wandb.plot.line_series(
            xs=recall_model.tolist(),
            ys=[precision_model.tolist()],
            keys=["LogisticRegression"],
            title="Precision-Recall Curve - Modelo",
            xname="Recall",
        ),

        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred.tolist(),
            class_names=["No alerta", "Alerta"],
        ),

        "feature_importance": wandb.plot.bar(
            wandb.Table(dataframe=importancias.head(30)),
            label="feature",
            value="importance",
            title="Top 30 | Importancia por |coef|",
        ),
    })

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/modelo_logreg_randomized.joblib"
    params_path = "artifacts/modelo_logreg_randomized_params.json"

    joblib.dump(pipeline_final, model_path)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(parametros, f, ensure_ascii=False, indent=2)

    artifact_model = wandb.Artifact("modelo_logreg_randomized", type="model")
    artifact_model.add_file(model_path)
    artifact_model.add_file(params_path)
    run.log_artifact(artifact_model)

    run.finish()


if __name__ == "__main__":
    main()
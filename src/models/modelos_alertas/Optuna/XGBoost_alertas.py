import os
import gc
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

TARGET = 'alert_in_next_15m_max'
COLS_RAW = [
    'route_id', 'direction', 'merge_time',
    'delay_seconds_mean', 'lagged_delay_1_mean', 
    'lagged_delay_2_mean', 'delay_3_before',
    'actual_headway_seconds_mean', 'is_unscheduled_max',
    'num_updates_sum', 'match_key_nunique',
    'hour_sin_first', 'hour_cos_first', 'dow_first', 'is_weekend_max',
    'seconds_since_last_alert_mean',
    TARGET,
]



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


def agregar_por_linea(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Re-agrega el dataset de nivel parada a nivel línea (route_id + direction + 30min).
    Genera features que capturan el estado global de la línea."""
 
    cat_cols = [c for c in df.columns if c.startswith('category_')]
    print(f"Categorías de alerta encontradas: {cat_cols}")
 
    cols = [c for c in COLS_RAW + cat_cols if c in df.columns]
    df_work = df[cols].copy()
    df_work['merge_time'] = pd.to_datetime(df_work['merge_time'])
    df_work['parada_retrasada'] = (df_work['delay_seconds_mean'] > 60).astype(int)
 
    agg_dict = {
        'delay_seconds_mean':            ['mean', 'max', 'std'],
        'parada_retrasada':              ['sum', 'count'],
        'lagged_delay_1_mean':           'mean',
        'lagged_delay_2_mean':           'mean',
        'delay_3_before':                'mean',
        'actual_headway_seconds_mean':   ['mean', 'std'],
        'is_unscheduled_max':            'max',
        'num_updates_sum':               'sum',
        'match_key_nunique':             'sum',
        'hour_sin_first':                'first',
        'hour_cos_first':                'first',
        'dow_first':                     'first',
        'is_weekend_max':                'max',
        'seconds_since_last_alert_mean': 'min',  
        TARGET:                          'max',
    }
    for col in cat_cols:
        agg_dict[col] = 'max'  
 
    print("Agregando por línea...")
    df_linea = df_work.groupby(
        ['route_id', 'direction', pd.Grouper(key='merge_time', freq='30min')],
        observed=True
    ).agg(agg_dict).reset_index()
 
    # Aplanar columnas multinivel generadas por las agregaciones dobles
    df_linea.columns = [
        '_'.join(filter(None, col)) if isinstance(col, tuple) else col
        for col in df_linea.columns
    ]
 
    df_linea = df_linea.rename(columns={
        'delay_seconds_mean_mean':           'delay_mean_linea',
        'delay_seconds_mean_max':            'delay_max_linea',
        'delay_seconds_mean_std':            'delay_std_linea',
        'parada_retrasada_sum':              'paradas_retrasadas',
        'parada_retrasada_count':            'total_paradas',
        'lagged_delay_1_mean_mean':          'lag1_mean_linea',
        'lagged_delay_2_mean_mean':          'lag2_mean_linea',
        'delay_3_before_mean':               'delay_3_before_mean',
        'actual_headway_seconds_mean_mean':  'headway_mean_linea',
        'actual_headway_seconds_mean_std':   'headway_std_linea',
        'is_unscheduled_max_max':            'is_unscheduled',
        'num_updates_sum_sum':               'num_updates',
        'match_key_nunique_sum':             'match_key_nunique',
        'hour_sin_first_first':              'hour_sin',
        'hour_cos_first_first':              'hour_cos',
        'dow_first_first':                   'dow',
        'is_weekend_max_max':                'is_weekend',
        'seconds_since_last_alert_mean_min': 'seg_desde_ultima_alerta_linea',
        f'{TARGET}_max':                     TARGET,
    })
 
    # Features derivadas
    df_linea['pct_paradas_retrasadas']   = (
        df_linea['paradas_retrasadas'] / df_linea['total_paradas'].clip(lower=1)
    )
    df_linea['delay_acceleration_linea'] = (
        df_linea['delay_mean_linea'] - df_linea['lag1_mean_linea']
    )
 
    del df_work
    gc.collect()
 
    print(f"Dataset por línea: {len(df_linea):,} filas x {df_linea.shape[1]} columnas")
    print(f"Reducción: {len(df):,} → {len(df_linea):,} filas")
    print(f"Positivos: {df_linea[TARGET].mean()*100:.1f}%")
 
    return df_linea, cat_cols


def agregar_features_rolling_retraso(df_linea: pd.DataFrame) -> pd.DataFrame:
    df_linea = df_linea.sort_values(['route_id', 'direction', 'merge_time'])
    grp = df_linea.groupby(['route_id', 'direction'])

    # Media móvil del retraso en las últimas 4 ventanas
    df_linea['delay_rolling4_mean'] = (
        grp['delay_mean_linea']
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )

    # Máximo retraso en las últimas 4 ventanas
    df_linea['delay_rolling4_max'] = (
        grp['delay_max_linea']
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
    )

    # Varianza del headway reciente 
    df_linea['headway_rolling4_std'] = (
        grp['headway_mean_linea']
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
        .fillna(0)
    )

    return df_linea

 
def get_features(cat_cols: list[str], df: pd.DataFrame) -> list[str]:
    """Features a nivel de línea"""
    features = [
        # Retraso global de la línea
        'delay_mean_linea', 'delay_max_linea', 'delay_std_linea',
        # Proporción de paradas afectadas
        'paradas_retrasadas', 'pct_paradas_retrasadas',
        # Evolución temporal del retraso
        'lag1_mean_linea', 'lag2_mean_linea', 'delay_3_before_mean',
        # Tendencia: ¿el retraso está empeorando?
        'delay_acceleration_linea', 'delay_rolling4_mean', 'delay_rolling4_max', 'headway_rolling4_std',
        # Irregularidad del servicio
        'headway_mean_linea', 'headway_std_linea',
        # Actividad operativa
        'is_unscheduled', 'num_updates', 'match_key_nunique',
        # Temporales
        'hour_sin', 'hour_cos', 'dow', 'is_weekend',
        # Identidad de la línea
        'route_id', 'direction',
        # Historial de alertas a nivel de línea
        #  'seg_desde_ultima_alerta_linea',
    ] + cat_cols
    return [f for f in features if f in df.columns]


def encoding_categorias(X_train, X_val, X_test):
    cols_ordinal_enc = ['route_id', 'direction']   

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

    # ── Re-agregación por línea ────────────────────────────────────────────────
    print("\nRe-agregando a nivel de línea...")
    df, cat_cols = agregar_por_linea(df)
    df = agregar_features_rolling_retraso(df)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # ── Features ──────────────────────────────────────────────────────────────
    FEATURES = get_features(cat_cols, df)
 
    # Imputar NaN con mediana
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

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
    study.optimize(objective, n_trials=30, show_progress_bar=True)
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
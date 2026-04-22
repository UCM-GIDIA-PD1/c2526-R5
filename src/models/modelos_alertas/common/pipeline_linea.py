"""
pipeline_linea.py
-----------------
Funciones compartidas para el Problema 3: Deteccion temprana de alertas MTA.

Contexto:
    El dataset original tiene ~21M filas a nivel de parada x ventana de 30min.
    Las alertas MTA se emiten a nivel de linea, no de parada, por lo que
    agregamos a nivel de linea (~755K filas) antes de modelar.

Features:
    FEATURES_SIN -> 20 features operacionales sin historial de alertas
    FEATURES_CON -> igual + seg_desde_ultima_alerta_linea (21 features)

Las features elegidas son:
-'delay_mean_linea': retraso media de todas las paradas de la línea
-'delay_max_linea': retraso máximo en cualquier parada
-'delay_Std_linea': dispersion del retraso de la linea
-delay_acceleration_linea: diferencia entre retraso actual y el de hace 30 mins
-paradas_retrasadas : numero paradas con mas de 60 segs de retraso
-pct_paradas_retrasadas : porcentaje de paradas retrasadas
-delay_1_before_mean: retraso medio de la línea hace 30 mins.
-delay_2_before_mean: retraso medio de la línea hace 60 mins.
-delay_3_before_mean: retraso medio de la línea hace 90 mins.
-headway_mean_linea: separacion media entre trenes.
-headway_std_linea  : irregularidad en separacion de trenes.
-is_unscheduled  :si hay trenes no programados activos (servicio de emergencia)
- match_key_nunique : numero de trenes distintos activos en la línea en esa ventan
-hour_sin, hour_cos :hora del dia codificada de forma ciclica (seno y coseno) para que las 23h y las 0h sean cercanas para el modelo
-dow: dia de la semana
-route_id : linea de metro codificada
FEATURES CREADAS NUEVAS:
-headway_cv : irregularidad relativa del intervalo entre trenes, calculada como desviación típica dividida entre la media del headway
-colapso_linea: variable binaria que vale 1 si más del 50% de las paradas de la
línea correspondiente tienen más de 60 segundo de retraso
- 'delay_x_aceleracion': interacción entre el retraso medio actual y la aceleracion positiva
del retraso.
-'delay_rolling4_mean': media móvil del retraso medio de la línea en las últimas
4 ventanas de 30 minutos.
-'delay_rolling4_max': máximo retraso registrado en cualquier parada de la línea en las últimas 4 
ventanas de 30 minutos.
-'headway_rolling4_std': mide si la irregularidad del servicio es sostenida en el tiempo.





-direction
-seg_desde_ultima_alerta_linea

Split: temporal 70/15/15 por fechas. Nunca aleatorio en series temporales.
Metrica principal: PR-AUC (clases desbalanceadas, ~18% positivos).
Target: alert_in_next_30m (1 si hay alerta en los proximos 30 min).

Uso:
    from src.models.modelos_alertas.pipeline_linea import (
        agregar_por_linea, split_temporal,
        FEATURES_CON, FEATURES_SIN, TARGET, evaluar_test,
    )
"""

import gc
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score,
)

TARGET     = 'alert_in_next_15m'
TARGET_RAW = 'alert_in_next_15m_max'

FEATURES_SIN = [
    'delay_mean_linea', 'delay_max_linea', 'delay_std_linea', 'delay_acceleration_linea',
    'paradas_retrasadas', 'pct_paradas_retrasadas',
    'delay_1_before_mean', 'delay_2_before_mean', 'delay_3_before_mean',
    'delay_rolling4_mean', 'delay_rolling4_max',   # tendencia ultimas 2h
    'headway_mean_linea', 'headway_std_linea', 'headway_rolling4_std',
    'is_unscheduled', 'num_updates', 'match_key_nunique',
    'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    'route_id', 'direction',
]

FEATURES_CON = FEATURES_SIN + ['seg_desde_ultima_alerta_linea']


def filtro_comportamiento_alterado(df):
    """Elimina paradas que no tienen alerta en 15 minutos 
    y sí tienen en 30 minutos y por tanto alteran la línea"""

    mask_positivos = df[TARGET_RAW] == 1
    mask_negativos_limpios = (
        df['alert_in_next_30m_max'] == 0            
    )

    df = df[mask_positivos | mask_negativos_limpios]
    df = df.reset_index(drop=True)

    print(f"Dataset tras filtrar negativos ambiguos: {len(df):,} filas")
    print(f"  Positivos: {df[TARGET_RAW].sum():,} ({df[TARGET_RAW].mean()*100:.1f}%)")
    print(f"  Negativos: {(df[TARGET_RAW]==0).sum():,} ({(df[TARGET_RAW]==0).mean()*100:.1f}%)")

    return df


def agregar_por_linea(df_raw):
    COLS = [
        'route_id', 'direction', 'merge_time',
        'delay_seconds_mean',
        'delay_1_before', 'delay_2_before', 'delay_3_before',
        'actual_headway_seconds_mean', 'is_unscheduled_max',
        'num_updates_sum', 'match_key_nunique',
        'hour_sin_first', 'hour_cos_first', 'dow_first', 'is_weekend_max',
        'seconds_since_last_alert_mean', TARGET_RAW,
    ]
    COLS = [c for c in COLS if c in df_raw.columns]

    df = df_raw[COLS].copy()
    df = df[df[TARGET_RAW].notna()]
    df['merge_time'] = pd.to_datetime(df['merge_time'])
    df[TARGET_RAW] = df[TARGET_RAW].astype(int)
    df['parada_retrasada'] = (df['delay_seconds_mean'] > 60).astype(int)

    agg_dict = {
        'delay_seconds_mean':            ['mean', 'max', 'std'],
        'parada_retrasada':              ['sum', 'count'],
        'delay_1_before':                'mean',
        'delay_2_before':                'mean',
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
        TARGET_RAW:                      'max',
    }

    print("Agregando por linea...")
    df_linea = df.groupby(
        ['route_id', 'direction', pd.Grouper(key='merge_time', freq='30min')],
        observed=True
    ).agg(agg_dict).reset_index()

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
        'delay_1_before_mean':               'delay_1_before_mean',
        'delay_2_before_mean':               'delay_2_before_mean',
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
        f'{TARGET_RAW}_max':                 TARGET,
    })


    del df
    gc.collect()

    # Features derivadas
    df_linea['pct_paradas_retrasadas']   = df_linea['paradas_retrasadas'] / df_linea['total_paradas'].clip(lower=1)
    df_linea['delay_acceleration_linea'] = df_linea['delay_mean_linea'] - df_linea['delay_1_before_mean']
    df_linea['headway_cv'] = (
        df_linea['headway_std_linea'] / df_linea['headway_mean_linea'].clip(lower=1)
    )
    df_linea['colapso_linea'] = (
        (df_linea['pct_paradas_retrasadas'] > 0.5).astype(int)
    )
    df_linea['delay_x_aceleracion'] = (
        df_linea['delay_mean_linea'] * df_linea['delay_acceleration_linea'].clip(lower=0)
    )

    df_linea['seg_desde_ultima_alerta_linea'] = df_linea['seg_desde_ultima_alerta_linea'].fillna(999999)
    df_linea = df_linea.dropna(subset=[TARGET])
    df_linea[TARGET] = df_linea[TARGET].astype(int)

    print(f"Dataset linea: {df_linea.shape[0]:,} filas x {df_linea.shape[1]} columnas")
    print(f"Positivos: {df_linea[TARGET].mean()*100:.1f}%")

    return df_linea


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
        'headway_cv', 'colapso_linea', 'delay_x_aceleracion',
        # Retraso global de la línea
        'delay_mean_linea', 'delay_max_linea', 'delay_std_linea',
        # Proporción de paradas afectadas
        'paradas_retrasadas', 'pct_paradas_retrasadas',
        # Evolución temporal del retraso
        'delay_1_before_mean', 'delay_2_before_mean', 'delay_3_before_mean',
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
        'seg_desde_ultima_alerta_linea',
    ]
    return [f for f in features if f in df.columns]


def encoding_categorias(X_train, X_val, X_test):
    cols_ordinal_enc = ['route_id', 'direction']

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cols_ordinal_enc] = enc.fit_transform(X_train[cols_ordinal_enc])
    X_val[cols_ordinal_enc]   = enc.transform(X_val[cols_ordinal_enc])
    X_test[cols_ordinal_enc]  = enc.transform(X_test[cols_ordinal_enc])

    print("✓ Encoding completado")
    return X_train, X_val, X_test, enc


def split_temporal(df_linea, train_frac=0.70, val_frac=0.15):
    df_linea = df_linea.sort_values('merge_time').reset_index(drop=True)
    dias = sorted(df_linea['merge_time'].dt.date.unique())

    corte_train = dias[int(len(dias) * train_frac)]
    corte_val   = dias[int(len(dias) * (train_frac + val_frac))]

    train = df_linea[df_linea['merge_time'].dt.date <  corte_train].copy()
    val   = df_linea[(df_linea['merge_time'].dt.date >= corte_train) &
                     (df_linea['merge_time'].dt.date <  corte_val)].copy()
    test  = df_linea[df_linea['merge_time'].dt.date >= corte_val].copy()

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Positivos -> train: {train[TARGET].mean()*100:.1f}% | val: {val[TARGET].mean()*100:.1f}% | test: {test[TARGET].mean()*100:.1f}%")

    return train, val, test


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


def evaluar_test(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    thr = thresholds[np.argmax(f1s)]
    y_pred = (y_prob >= thr).astype(int)

    return {
        "pr_auc_test":    float(average_precision_score(y, y_prob)),
        "auc_roc_test":   float(roc_auc_score(y, y_prob)),
        "f1_test":        float(f1_score(y, y_pred, zero_division=0)),
        "recall_test":    float(recall_score(y, y_pred, zero_division=0)),
        "precision_test": float(precision_score(y, y_pred, zero_division=0)),
        "threshold_opt":  float(thr),
    }, y_prob, y_pred

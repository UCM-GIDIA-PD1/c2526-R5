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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score,
)

TARGET     = 'alert_in_next_30m'
TARGET_RAW = 'alert_in_next_30m_max'

FEATURES_SIN = [
    'delay_mean_linea', 'delay_max_linea', 'delay_std_linea', 'delay_acceleration_linea',
    'paradas_retrasadas', 'pct_paradas_retrasadas',
    'delay_1_before_mean', 'delay_2_before_mean', 'delay_3_before_mean',
    'headway_mean_linea', 'headway_std_linea',
    'is_unscheduled', 'num_updates', 'match_key_nunique',
    'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    'route_id', 'direction',
]

FEATURES_CON = FEATURES_SIN + ['seg_desde_ultima_alerta_linea']


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

    df_linea['pct_paradas_retrasadas']   = df_linea['paradas_retrasadas'] / df_linea['total_paradas'].clip(lower=1)
    df_linea['delay_acceleration_linea'] = df_linea['delay_mean_linea'] - df_linea['delay_1_before_mean']

    for col in ['route_id', 'direction']:
        le = LabelEncoder()
        df_linea[col] = le.fit_transform(df_linea[col].astype(str).fillna('UNKNOWN'))

    df_linea['seg_desde_ultima_alerta_linea'] = df_linea['seg_desde_ultima_alerta_linea'].fillna(999999)
    df_linea = df_linea.dropna(subset=[TARGET])
    df_linea[TARGET] = df_linea[TARGET].astype(int)

    print(f"Dataset linea: {df_linea.shape[0]:,} filas x {df_linea.shape[1]} columnas")
    print(f"Positivos: {df_linea[TARGET].mean()*100:.1f}%")

    return df_linea


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

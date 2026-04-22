"""
Este script se encarga de agregar el dataset en tiempo real generado por generate_realtime_dataset.py
en ventanas temporales de X minutos (por defecto 30), agrupando por parada,
línea y dirección.

Este script es el equivalente en tiempo real de time_aggregations.py,
adaptado para trabajar con el momento actual en lugar de
datos históricos mensuales almacenados en MinIO.

Uso:
    from src.ETL.pipelines.generate_realtime_dataset import build_realtime_dataset
    from src.ETL.pipelines.aggregate_realtime_dataset import aggregate_realtime

    df_rt = build_realtime_dataset()
    df_agregado = aggregate_realtime(df_rt, tiempo='30')
"""

import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def agrupar_realtime(df: pd.DataFrame, tiempo: str = '30') -> pd.DataFrame:
    """
    Agrega el DataFrame de tiempo real en ventanas de `tiempo` minutos,
    agrupando por parada, línea y dirección.

    Los params son:
    -df : pd.DataFrame
        DataFrame generado por build_realtime_dataset() a nivel de parada.
    -tiempo : str
        Tamaño de la ventana de tiempo en minutos (por defecto '30').

    
      Retorna un DataFrame agregado con una fila por (parada, línea, dirección, ventana).
    """
    if df.empty:
        print("[WARN] El DataFrame de entrada está vacío.")
        return pd.DataFrame()

    df = df.copy()
    df['merge_time'] = pd.to_datetime(df['merge_time'], errors='coerce')
    if df['merge_time'].dt.tz is not None:
        df['merge_time'] = df['merge_time'].dt.tz_convert('America/New_York').dt.tz_localize(None)

    limite_segundos = 43200
    columnas_tiempo = [
        'delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
        'route_rolling_delay',
    ]
    for col in columnas_tiempo:
        if col in df.columns:
            df.loc[(df[col] > limite_segundos) | (df[col] < -limite_segundos), col] = np.nan

    # One-Hot Encoding de variables categóricas
    cols_ohe = [c for c in ['category', 'tipo_referente'] if c in df.columns]
    if cols_ohe:
        df = pd.get_dummies(df, columns=cols_ohe, dummy_na=False)

    cat_columns = [col for col in df.columns if col.startswith(('category_', 'tipo_referente_'))]

    # Diccionario de agregación (mismo esquema que time_aggregations.py)
    agg_dict = {}

    for col in ['is_unscheduled', 'is_weekend', 'temp_extreme',
                'afecta_previo', 'afecta_durante', 'afecta_despues',
                'is_alert_just_published', 'alert_in_next_15m', 'alert_in_next_30m']:
        if col in df.columns:
            agg_dict[col] = 'max'

    for col in ['delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'route_rolling_delay']:
        if col in df.columns:
            agg_dict[col] = ['mean', 'max']

    if 'match_key' in df.columns:
        agg_dict['match_key'] = 'nunique'
    if 'actual_headway_seconds' in df.columns:
        agg_dict['actual_headway_seconds'] = 'mean'
    if 'stops_to_end' in df.columns:
        agg_dict['stops_to_end'] = 'mean'
    if 'scheduled_time_to_end' in df.columns:
        agg_dict['scheduled_time_to_end'] = 'mean'
    if 'seconds_since_last_alert' in df.columns:
        agg_dict['seconds_since_last_alert'] = 'mean'
    if 'num_updates' in df.columns:
        agg_dict['num_updates'] = 'sum'
    if 'n_eventos_afectando' in df.columns:
        agg_dict['n_eventos_afectando'] = 'max'

    for col in ['hour_sin', 'hour_cos', 'dow']:
        if col in df.columns:
            agg_dict[col] = 'first'

    for col in cat_columns:
        agg_dict[col] = 'sum'

    frecuencia = tiempo + 'min'
    print(f"Iniciando agregación en ventanas de {tiempo} min...")

    df_grouped = df.groupby([
        'stop_id',
        'route_id',
        'direction',
        pd.Grouper(key='merge_time', freq=frecuencia)
    ]).agg(agg_dict).reset_index()

    nuevas_columnas = []
    for col in df_grouped.columns.values:
        if isinstance(col, tuple):
            nuevas_columnas.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        else:
            nuevas_columnas.append(col)
    df_grouped.columns = nuevas_columnas

    # Lags de retraso previo: en tiempo real no hay ventanas anteriores disponibles,
    # ya que solo contamos con una snapshot del momento actual.
    # Se dejan como NaN para que el modelo los trate como datos ausentes.
    df_grouped['delay_1_before'] = np.nan
    df_grouped['delay_2_before'] = np.nan
    df_grouped['delay_3_before'] = np.nan


    minutos_ventana = int(tiempo)
    ahora_ny = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)

    # Borde inferior de la ventana actual
    ventana_actual = ahora_ny.replace(second=0, microsecond=0)
    ventana_actual = ventana_actual.replace(
        minute=(ventana_actual.minute // minutos_ventana) * minutos_ventana
    )

    # ¿Estamos en la primera o segunda mitad de la ventana?
    minutos_transcurridos = (ahora_ny - ventana_actual).total_seconds() / 60

    if minutos_transcurridos >= minutos_ventana / 2:
        # Segunda mitad: asignamos la ventana en curso
        ventana_asignada = ventana_actual
    else:
        # Primera mitad: asignamos la anterior (ya cerrada)
        ventana_asignada = ventana_actual - timedelta(minutes=minutos_ventana)

    antes = len(df_grouped)
    df_grouped = df_grouped[df_grouped['merge_time'] == ventana_asignada].reset_index(drop=True)
    print(f"  Ventana asignada: {ventana_asignada} "
        f"({len(df_grouped)} filas de {antes} totales)")


    print(f"  Ventanas generadas: {len(df_grouped):,} | "
          f"Líneas: {df_grouped['route_id'].nunique()} | "
          f"Paradas únicas: {df_grouped['stop_id'].nunique()}")

    return df_grouped


def aggregate_realtime(df_rt: pd.DataFrame, tiempo: str = '30') -> pd.DataFrame:
    """
    Punto de entrada principal. Toma el DataFrame en tiempo real a nivel de
    parada y devuelve el DataFrame agregado en ventanas temporales, listo
    para pasar a agregar_por_linea().

    Los params son:
    -df_rt : pd.DataFrame
        Salida de build_realtime_dataset().
    -tiempo : str
        Tamaño de la ventana en minutos (por defecto '30').

      Retorna un DataFrame agregado por (parada, línea, dirección, ventana de tiempo),
        con el mismo esquema que time_aggregations.py.
    """
    print(f"\n{'='*55}")
    print("  AGGREGATE REALTIME DATASET")
    print(f"{'='*55}\n")

    if df_rt.empty:
        raise ValueError("El DataFrame de tiempo real está vacío. Revisa generate_realtime_dataset.py.")

    print(f"  Filas de entrada (nivel parada): {len(df_rt):,}")
    print(f"  Líneas detectadas: {df_rt['route_id'].nunique()}")

    df_agregado = agrupar_realtime(df_rt, tiempo=tiempo)

    gc.collect()

    print(f"\n[OK] Agregación completada: {len(df_agregado):,} filas")

    return df_agregado


if __name__ == "__main__":
    from src.ETL.pipelines.generate_realtime_dataset import build_realtime_dataset

    df_rt = build_realtime_dataset()
    df_agregado = aggregate_realtime(df_rt, tiempo='30')
    print(df_agregado.head(10))

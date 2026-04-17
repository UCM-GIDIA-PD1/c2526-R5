"""
Agregación del DataFrame de retrasos del metro de Nueva York en ventanas de tiempo.

Toma el DataFrame final producido por realtime_data.py y lo agrega en ventanas
de X minutos (por defecto 30), agrupando por parada, línea y dirección.

Parámetros
    ----------
    df : DataFrame final de realtime_data.py con columnas:
         viaje_id, linea_id, parada_id, hora_llegada, direccion,
         dow, is_weekend, delay, hour_sin, hour_cos
    tiempo : tamaño de la ventana en minutos (por defecto 30)

    Output
    ------
    DataFrame agregado con una fila por (parada, línea, dirección, ventana),
    con estadísticas de delay y variables auxiliares.
"""

import math
import pandas as pd
import numpy as np


def agregar_por_ventana(df: pd.DataFrame, tiempo: int = 30) -> pd.DataFrame:
    """
    Agrega el DataFrame de retrasos en ventanas de `tiempo` minutos,
    agrupando por parada, línea y dirección.
    """

    df = df.copy()

    # hora_llegada es string HH:MM:SS → convertir a datetime para poder usar Grouper
    # Usamos la fecha de hoy como base (solo nos importa la componente horaria para agrupar)

    df['delay'] = pd.to_numeric(df['delay'], errors='coerce')

    frecuencia = f'{tiempo}min'

    agg_dict = {
        'delay':       ['mean', 'max', 'min', 'std', 'count'],
        'hour_sin':    'first',
        'hour_cos':    'first',
        'dow':         'first',
        'is_weekend':  'first',
        'viaje_id':    'nunique',
    }

    df_agrupado = df.groupby([
        'parada_id',
        'linea_id',
        'direccion',
        pd.Grouper(key='hora_llegada', freq=frecuencia)
    ]).agg(agg_dict).reset_index()

    # Aplanar columnas multi-nivel generadas por las agregaciones múltiples de delay
    nuevas_columnas = []
    for col in df_agrupado.columns:
        if isinstance(col, tuple):
            nuevas_columnas.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        else:
            nuevas_columnas.append(col)
    df_agrupado.columns = nuevas_columnas

    # Renombrar para claridad
    df_agrupado = df_agrupado.rename(columns={
        'hora_llegada': 'ventana_inicio',
        'delay_mean':      'delay_medio',
        'delay_max':       'delay_maximo',
        'delay_min':       'delay_minimo',
        'delay_std':       'delay_std',
        'delay_count':     'num_trenes',
        'viaje_id_nunique':'viajes_unicos',
        'hour_sin_first':  'hour_sin',
        'hour_cos_first':  'hour_cos',
        'dow_first':       'dow',
        'is_weekend_first':'is_weekend',
    })

    # Eliminar ventanas sin ningún tren
    df_agrupado = df_agrupado[df_agrupado['num_trenes'] > 0].reset_index(drop=True)

    print(f"  Agregación a {tiempo} min: {len(df_agrupado)} ventanas | "
          f"paradas únicas: {df_agrupado['parada_id'].nunique()} | "
          f"líneas: {sorted(df_agrupado['linea_id'].unique())}")

    return df_agrupado  
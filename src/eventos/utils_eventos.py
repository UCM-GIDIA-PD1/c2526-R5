"""
utils.py — Funciones compartidas para todos los scripts de ingesta de eventos.

Incluye:
  - Re-exportación de funciones MinIO desde src.common.minio_client
  - Descarga y cálculo de paradas de metro afectadas (Haversine)
  - Fusión de estaciones duplicadas
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.common.minio_client import (         
    download_df_parquet,
    upload_file,
    download_file,
    upload_json,
    download_json,
    DEFAULT_ENDPOINT,
    DEFAULT_BUCKET,
)

#  Constantes propias

_METRO_CSV_URL = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"



#  Paradas de metro
def cargar_paradas_df():
    """
    Descarga el CSV de paradas del metro de NY y lo devuelve como DataFrame
    con columnas: nombre, lineas, lon, lat.
    Devuelve None si la descarga falla.
    """
    try:
        df = pd.read_csv(_METRO_CSV_URL)
        df = df[["Stop Name", "Daytime Routes", "GTFS Longitude", "GTFS Latitude"]].copy()
        df = df.rename(columns={
            "Stop Name":      "nombre",
            "Daytime Routes": "lineas",
            "GTFS Longitude": "lon",
            "GTFS Latitude":  "lat",
        })
        return df
    except Exception as exc:
        print(f"[utils] Error descargando paradas de metro: {exc}")
        return None


def obtener_paradas_afectadas(coords, df_paradas, max_metros=500):
    """
    Devuelve las paradas de metro a menos de max_metros metros de coords.

    Parámetros
    ----------
    coords      : (longitud, latitud) del evento.
    df_paradas  : DataFrame con columnas lon, lat, nombre, lineas.
    max_metros  : radio de búsqueda en metros.

    Devuelve
    -------
    Lista de tuplas [(nombre_parada, lineas)].
    """
    if not coords or None in coords or df_paradas is None or df_paradas.empty:
        return []

    lon_ev, lat_ev = coords

    # Haversine formula
    lat1 = np.radians(lat_ev)
    lon1 = np.radians(lon_ev)
    lat2 = np.radians(df_paradas["lat"].to_numpy())
    lon2 = np.radians(df_paradas["lon"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distancias = 2 * np.arcsin(np.sqrt(a)) * 6371000  

    cercanas = df_paradas[distancias <= max_metros]
    return [(row["nombre"], row["lineas"]) for _, row in cercanas.iterrows()]


def fusionar_lista_estaciones(lista_tuplas):
    """
    Agrupa las tuplas (nombre, lineas) por nombre de estación,
    unificando las líneas en un único string ordenado.
    """
    if not isinstance(lista_tuplas, list) or not lista_tuplas:
        return lista_tuplas

    fusionadas = defaultdict(set)
    for nombre, lineas in lista_tuplas:
        fusionadas[nombre].update(str(lineas).split())

    return [
        (nombre, " ".join(sorted(lineas_set)))
        for nombre, lineas_set in fusionadas.items()
    ]
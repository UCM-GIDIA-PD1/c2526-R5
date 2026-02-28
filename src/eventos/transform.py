import io
import zipfile

import numpy as np
import pandas as pd
import os
import sys
import requests

from src.common.minio_client import (
    download_df_parquet,
    upload_df_parquet,
)

from datetime import date, timedelta


IDS = ["eventos", "eventos_deporte", "eventos_concierto"]

_MTA_GTFS_ZIP_URL = "http://web.mta.info/developers/data/nyct/subway/google_transit.zip"
_METRO_CSV_URL    = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"


def _descargar_stops_gtfs() -> pd.DataFrame:
    """
    Descarga el ZIP del GTFS estático del MTA en memoria y extrae stops.txt
    como DataFrame con stop_id, stop_name, stop_lat, stop_lon.
    """
    resp = requests.get(_MTA_GTFS_ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open("stops.txt") as f:
            df = pd.read_csv(f)
    return df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()


def _construir_tabla_correspondencias_stop_id() -> dict:
    """
    Construye un diccionario  parada_nombre → [stop_id, ...]  combinando:
      1) Correspondencia exacta por nombre normalizado entre data.ny.gov y stops.txt
      2) Correspondencia por proximidad geográfica (< 100 m) para las paradas
         que no se hayan juntado por nomnre
    """
    df_paradas_gtfs = _descargar_stops_gtfs()
    df_paradas_nyc  = pd.read_csv(_METRO_CSV_URL)

    df_paradas_gtfs["nombre_normalizado"] = df_paradas_gtfs["stop_name"].str.lower().str.strip()
    df_paradas_nyc["nombre_normalizado"]  = df_paradas_nyc["Stop Name"].str.lower().str.strip()

    correspondencias = df_paradas_nyc[
        ["Stop Name", "GTFS Latitude", "GTFS Longitude", "nombre_normalizado"]
    ].merge(
        df_paradas_gtfs[["stop_id", "stop_lat", "stop_lon", "nombre_normalizado"]],
        on="nombre_normalizado",
        how="left",
    )

    sin_correspondencia = correspondencias[correspondencias["stop_id"].isna()]["Stop Name"].unique()
    if len(sin_correspondencia) > 0:
        latitudes_gtfs  = np.radians(df_paradas_gtfs["stop_lat"].values)
        longitudes_gtfs = np.radians(df_paradas_gtfs["stop_lon"].values)
        correspondencias_por_coordenada = []
        for nombre in sin_correspondencia:
            fila = df_paradas_nyc[df_paradas_nyc["Stop Name"] == nombre].iloc[0]
            lat1 = np.radians(fila["GTFS Latitude"])
            lon1 = np.radians(fila["GTFS Longitude"])
            delta_lat = latitudes_gtfs - lat1
            delta_lon = longitudes_gtfs - lon1
            factor_haversine = (
                np.sin(delta_lat / 2) ** 2
                + np.cos(lat1) * np.cos(latitudes_gtfs) * np.sin(delta_lon / 2) ** 2
            )
            distancia_m = 2 * np.arcsin(np.sqrt(factor_haversine)) * 6_371_000
            for id_parada in df_paradas_gtfs.loc[distancia_m <= 100, "stop_id"]:
                correspondencias_por_coordenada.append({"Stop Name": nombre, "stop_id": id_parada})
        if correspondencias_por_coordenada:
            correspondencias = pd.concat(
                [correspondencias, pd.DataFrame(correspondencias_por_coordenada)],
                ignore_index=True,
            )

    tabla_correspondencias = (
        correspondencias[correspondencias["stop_id"].notna()]
        .groupby("Stop Name")["stop_id"]
        .apply(list)
        .to_dict()
    )
    print(f"  [paradas] tabla de correspondencias construida: {len(tabla_correspondencias)} paradas")
    return tabla_correspondencias


def iterate_dates(start, end):
    """Itera fechas (start y end inclusive)"""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_cleaned_object(day):
    return f"grupo5/cleaned/eventos_nyc/date={day}/eventos_{day}.parquet"


def build_processed_object(day):
    return f"grupo5/processed/eventos_nyc/date={day}/eventos_{day}.parquet"

def _normalizar_paradas(value):
    """
    Convierte 'paradas_afectadas' a lista de tuplas [(nombre, lineas), ...]
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, np.ndarray)):
        out = []
        for x in value:
            if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 2:
                out.append((str(x[0]), str(x[1])))
        return out
    return []


def transform_gtfs_processed_range_to_cleaned(start, end, access_key, secret_key):
    print("  [paradas] Descargando stops.txt del MTA en memoria...")
    tabla_correspondencias = _construir_tabla_correspondencias_stop_id()

    for d in iterate_dates(start, end):
        day = d.strftime("%Y-%m-%d")

        in_obj = build_processed_object(day)
        try:
            df = download_df_parquet(access_key, secret_key, in_obj)
            print(f"  encontrado: {in_obj}")
        except Exception:
            print(f"  No encontrado: {in_obj}, saltando...")
            continue

        if df is None or df.empty:
            print(f"  Sin datos para {day}, saltando...")
            continue

        # 1) Arreglao del score de eventos deportivos a 1.0
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1.0)
        else:
            df["score"] = 1.0

        # 2) Añadida la fecha final a todos los eventos
        if "fecha_final" not in df.columns and "fecha_inicio" in df.columns:
            df["fecha_final"] = df["fecha_inicio"]
        else:
            df["fecha_final"] = df["fecha_final"].fillna(df["fecha_inicio"])

        # 3) normalizar paradas_afectadas a lista
        df["paradas_afectadas"] = df["paradas_afectadas"].apply(_normalizar_paradas)

        # 4) dropear filas sin paradas afectadas
        df = df[df["paradas_afectadas"].map(len) > 0].copy()
        if df.empty:
            print(f"  {day}: todas las filas sin paradas afectadas, saltando subida...")
            continue

        # 5) Dejamos 1 fila por parada afectada
        df = df.explode("paradas_afectadas", ignore_index=True) 

        # 6) Separamos en dos columnas: parada_nombre / parada_lineas
        df["parada_nombre"] = df["paradas_afectadas"].apply(lambda x: x[0] if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 2 else None)
        df["parada_lineas"] = df["paradas_afectadas"].apply(lambda x: x[1] if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 2 else None)
        df = df.drop(columns=["paradas_afectadas"])

        #7) Establecemos su stop_id según su nombre
        df["stop_id"] = df["parada_nombre"].map(lambda n: tabla_correspondencias.get(n, [None]))
        df = df.explode("stop_id", ignore_index=True)

        df.drop_duplicates(subset=['nombre_evento', 'stop_id'], inplace=True)
        df= df[df['stop_id'].str.endswith(('N', 'S'))]

        out_obj = build_cleaned_object(day)
        upload_df_parquet(access_key, secret_key, out_obj, df)
        print(f"Subido: {out_obj} ({len(df)} filas)")


def run_transform(start, end):
    """Función usada por runner externo para ejecutar la transformacion.

    Convierte string dates a objetos date, obtiene credenciales de MinIO
    de las variables de entorno y delega a transform_gtfs_processed_range_to_cleaned
    """
    from datetime import datetime

    access_key = os.getenv("MINIO_ACCESS_KEY")
    if access_key is None:
        raise AssertionError("MINIO_ACCESS_KEY no definida")

    secret_key = os.getenv("MINIO_SECRET_KEY")
    if secret_key is None:
        raise AssertionError("MINIO_SECRET_KEY no definida")

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    transform_gtfs_processed_range_to_cleaned(start_date, end_date, access_key, secret_key)

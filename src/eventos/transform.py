import numpy as np
import pandas as pd
import os
import sys


from src.common.minio_client import (
    download_df_parquet,
    upload_df_parquet,
)

from datetime import date, timedelta


IDS = ["eventos", "eventos_deporte", "eventos_concierto"]


def iterate_dates(start, end):
    """Itera fechas (start y end inclusive)"""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_cleaned_object(day):
    return f"grupo5/cleaned/eventos_nyc/dia={day}/eventos_{day}.parquet"


def build_processed_object(day):
    return f"grupo5/processed/eventos_nyc/dia={day}/eventos_{day}.parquet"

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

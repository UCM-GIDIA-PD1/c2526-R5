import numpy as np
import pandas as pd
import os




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


#def transform_gtfs_processed_range_to_cleaned(start, end, access_key, secret_key):


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

    #def transform_gtfs_processed_range_to_cleaned(start, end, access_key, secret_key)
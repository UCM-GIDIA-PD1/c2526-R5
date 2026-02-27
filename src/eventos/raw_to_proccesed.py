import numpy as np
import pandas as pd
import os
import sys

ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)


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


def build_raw_object(day, id):
    return f"grupo5/raw/eventos_nyc/dia={day}/{id}_{day}.parquet"


def build_processed_object(day):
    return f"grupo5/processed/eventos_nyc/dia={day}/eventos_{day}.parquet"


def transform_events_raw_range_to_proccesed(start, end, access_key, secret_key):
    for d in iterate_dates(start, end):
        day = d.strftime("%Y-%m-%d")
        dfs = []
        for id in IDS:
            in_obj = build_raw_object(day, id)
            try:
                dfs.append(download_df_parquet(access_key, secret_key, in_obj))
                print(f"  encontrado: {in_obj}")
            except Exception:
                print(f"  No encontrado: {in_obj}, saltando...")

        if not dfs:
            print(f"  Sin datos para {day}, saltando...")
            continue

        df_processed = pd.concat(dfs, ignore_index=True)
        out_obj = build_processed_object(day)
        upload_df_parquet(access_key, secret_key, out_obj, df_processed)
        print(f"Subido: {out_obj} ({len(df_processed)} filas)")


def run_transform(start, end):
    """Función usada por runner externo para ejecutar la transformacion.

    Convierte string dates a objetos date, obtiene credenciales de MinIO
    de las variables de entorno y delega a transform_events_raw_range_to_proccesed
    """
    start_date = "2025-01-01"
    end_date = "2025-12-01"

    from datetime import datetime

    access_key = os.getenv("MINIO_ACCESS_KEY")
    if access_key is None:
        raise AssertionError("MINIO_ACCESS_KEY no definida")

    secret_key = os.getenv("MINIO_SECRET_KEY")
    if secret_key is None:
        raise AssertionError("MINIO_SECRET_KEY no definida")

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    transform_events_raw_range_to_proccesed(start_date, end_date, access_key, secret_key)

if __name__ == "__main__":
    
    FECHA_INICIO = "2025-01-01"
    FECHA_FIN = "2025-12-31"

try:
    run_transform(FECHA_INICIO, FECHA_FIN)
    print("\n¡Transformación finalizada correctamente!")
except AssertionError as ae:
    print(f"\nError de configuración: {ae}")
except Exception as e:
        print(f"\nError inesperado: {e}")
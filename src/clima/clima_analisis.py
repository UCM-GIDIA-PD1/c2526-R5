INPUT_CLIMATE_PATH = "grupo5/cleaned/clima_clean/date={day}/clima_{day}.parquet"
INPUT_GTFS_PATH = "grupo5/cleaned/gtfs_clean_scheduled/date={day}/gtfs_schedulled_{day}.parquet"
INPUT_GTFS_PATH = "grupo5/cleaned/gtfs_clean_unscheduled/date={day}/gtfs_unschedulled_{day}.parquet"
OUTPUT_CLIMATE_PATH ="grupo5/analytics/clima_clean/date={day}/clima_{day}.parquet"

import os
from datetime import date, timedelta, datetime

from src.common.minio_client import download_df_parquet, upload_df_parquet, upload_json

def analisis(df_weather):


def ejecucion(fechaini, fechafin):
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    start_dt = datetime.strptime(fechaini, "%Y-%m-%d").date()
    end_dt = datetime.strptime(fechafin, "%Y-%m-%d").date()

    curr = start_dt
    while curr <= end_dt:
        day = curr.strftime("%Y-%m-%d")
        try:
            # Descarga
            df_weather = download_df_parquet(access_key, secret_key, INPUT_CLIMATE_PATH.format(day=day))
            
            # Analisis
            df_clean, report = analisis(df_weather)
            
            # Carga de datos 
            #upload_df_parquet(access_key, secret_key, OUTPUT_DATA_PATH.format(day=day), df_clean)
            
            print(f"{day}: Procesado correctamente.")
        except Exception as e:
            print(f"{day}: Error en transformación -> {str(e)}")
            
        curr += timedelta(days=1)

if __name__ == "__main__":
    ejecucion()
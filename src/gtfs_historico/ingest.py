'''
Ejecutar usando -m
'''
import os
from datetime import date, timedelta
from minio import Minio
from src.gtfs_historico.historical_gtfs_builder import process_mta_date
from src.common.minio_client import upload_file, DEFAULT_BUCKET 

# Configuracion MinIO
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
assert ACCESS_KEY is not None, "La variable de entorno MINIO_ACCESS_KEY no está definida."
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
assert SECRET_KEY is not None, "La variable de entorno MINIO_SECRET_KEY no está definida."

BUCKET_NAME = DEFAULT_BUCKET

# Función generadora de fechas
def daterange(start_date, end_date):
    """Generador para iterar día a día entre dos fechas"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def process_and_store_gtfs_range(start: str, end: str) -> None:
    """Procesa y sube GTFS para un rango de fechas.

    Args:
        start: fecha inicio 'YYYY-MM-DD'
        end: fecha fin 'YYYY-MM-DD'
    """
    from datetime import datetime

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    print(f"Iniciando pipeline para fechas desde {start_date} hasta {end_date}")

    for single_date in daterange(start_date, end_date):
        target_date_str = single_date.strftime("%Y-%m-%d")
        print(f"Procesando día: {target_date_str}")

        try:
            # Procesa los datos static y realtime del día y devuelve ruta del Parquet final
            local_parquet_path = process_mta_date(target_date_str)

            # Subir el parquet a MinIO
            minio_destination_path = f"grupo5/processed/gtfs_with_delays/date={target_date_str}/mta_delays_{target_date_str}.parquet"

            print(f"Subiendo archivo a MinIO: {minio_destination_path} en el bucket {BUCKET_NAME}...")

            # Subida usando helper
            upload_file(
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                object_name=minio_destination_path,
                file_path=local_parquet_path,
                bucket=BUCKET_NAME,
            )

            print(f" Archivo subido correctamente a MinIO.")

        except Exception as e:
            import traceback
            print(f"Error procesando la fecha {target_date_str}: {e}")
            traceback.print_exc()

        finally:
            # Limpiar el parquet temporal del disco local (para no saturar disco)
            if local_parquet_path and os.path.exists(local_parquet_path):
                try:
                    os.remove(local_parquet_path)
                    print(f"Archivo temporal local '{local_parquet_path}' eliminado.")
                except Exception:
                    pass

if __name__ == "__main__":
    # Definir el rango de fechas a procesar
    start = date(2025, 12, 1)
    end = date(2025, 12, 31)
    
    process_and_store_gtfs_range(start, end)
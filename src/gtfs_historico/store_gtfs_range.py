import os
from datetime import date, timedelta
from minio import Minio
from historical_gtfs_builder import process_mta_date

# Configuracion MinIO
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
assert ACCESS_KEY is not None, "La variable de entorno MINIO_ACCESS_KEY no está definida."
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
assert SECRET_KEY is not None, "La variable de entorno MINIO_SECRET_KEY no está definida."

# Crear un cliente de MinIO apuntando a la UCM
client = Minio(
    endpoint="minio.fdi.ucm.es",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
)

BUCKET_NAME = "pd1" 

# Función generadora de fechas
def daterange(start_date, end_date):
    """Generador para iterar día a día entre dos fechas"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

if __name__ == "__main__":
    # Definir el rango de fechas a procesar
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    
    print(f"Iniciando pipeline para fechas desde {start} hasta {end}")
    
    for single_date in daterange(start, end):
        target_date_str = single_date.strftime("%Y-%m-%d")
        print(f"Procesando días: {target_date_str}")
        
        try:
            # Procesa los datos static y realtime del día y devuelve ruta del CSV final
            local_csv_path = process_mta_date(target_date_str)
            
            # Subir el CSV a MinIO
            minio_destination_path = f"grupo5/processed/gtfs_with_delays/date={target_date_str}/mta_delays_{target_date_str}.csv"
            
            print(f"Subiendo archivo a MinIO: {minio_destination_path} en el bucket {BUCKET_NAME}...")
            
            client.fput_object(
                bucket_name=BUCKET_NAME,
                object_name=minio_destination_path,
                file_path=local_csv_path,
            )
            
            print(f" Archivo subido correctamente a MinIO.")
            
            # Limpiar el CSV temporal del disco local (para no saturar disco)
            if os.path.exists(local_csv_path):
                os.remove(local_csv_path)
                print(f"Archivo temporal local '{local_csv_path}' eliminado.")
                
        except Exception as e:
            import traceback
            print(f"Error procesando la fecha {target_date_str}: {e}")
            traceback.print_exc()
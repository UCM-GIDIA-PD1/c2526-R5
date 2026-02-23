'''
Ejemplo de uso del API de MinIO para subir y descargar archivos.
'''
from minio import Minio
from dotenv import load_dotenv
from pathlib import Path
import os

# Cargar .env desde la misma carpeta del script
load_dotenv(Path(__file__).resolve().parent / ".env")

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

assert ACCESS_KEY is not None, "MINIO_ACCESS_KEY no definida"
assert SECRET_KEY is not None, "MINIO_SECRET_KEY no definida"

client = Minio(
    "minio.fdi.ucm.es",
    #access_key=ACCESS_KEY,
    #secret_key=SECRET_KEY,
    access_key="e8T60glxkUKjHHEhcNwR",
    secret_key="VeV7cbs96fDRyh3c0aw6lFKZxENmPvhoZeooNXhb",
)

print("Conectando...")
print([b.name for b in client.list_buckets()])

# Subir el archivo f3.txt a pd1/comun/f3.txt
bucket = "pd1"
source_file = "mta_dataset.csv"
destination_file = "grupo5/processed/official_alerts/mta_dataset_ultimos_30_mins.csv"
client.fput_object(
  bucket_name=bucket,
  object_name=destination_file,
  file_path=source_file,
)

print(f"Archivo '{source_file}' subido a MinIO como '{destination_file}' en el bucket '{bucket}'.")

'''
# Desgarcar el archivo pd1/comun/f3.txt de MinIO como f4.txt
source_file = "comun/pruebaSubir2.txt"
destination_file = "traidoDeVuelta2.txt"
client.fget_object(
  bucket_name=bucket,
  object_name=source_file,
  file_path=destination_file,
)

print(f"Archivo '{source_file}' bajado del bucket '{bucket}' de MinIO como '{destination_file}'.")
'''

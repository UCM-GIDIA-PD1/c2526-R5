'''
Script para borrar archivos de MinIO

Definir ruta a borrar en CARPETA_A_BORRAR
'''
import os
from minio import Minio
from minio.deleteobjects import DeleteObject

# CONFIGURACIÓN MINIO
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

client = Minio(
    endpoint="minio.fdi.ucm.es",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    # secure=False  # Descomenta esto si en tu script principal lo tuviste que poner
)

BUCKET_NAME = "pd1"  # Cambia por tu bucket real si es otro

# Define aquí la ruta exacta de la "carpeta" que quieres vaciar.
# IMPORTANTE: Asegúrate de que termina en "/" para no borrar archivos que empiecen igual por accidente.
CARPETA_A_BORRAR = "grupo5/cleaned/eventos_nyc/dia="

if __name__ == "__main__":
    print(f"Buscando archivos en '{CARPETA_A_BORRAR}' dentro del bucket '{BUCKET_NAME}'...")

    # 1. Listar todos los objetos que tengan ese prefijo (recursive=True busca en subcarpetas)
    objetos = client.list_objects(BUCKET_NAME, prefix=CARPETA_A_BORRAR, recursive=True, include_user_meta=True)

    # 2. Preparar la lista de objetos a borrar
    elementos_a_borrar = [DeleteObject(obj.object_name) for obj in objetos]

    # 3. Ejecutar el borrado masivo
    if elementos_a_borrar:
        print(f"Se han encontrado {len(elementos_a_borrar)} archivos. Procediendo a borrarlos...")
        
        # remove_objects devuelve un generador con los errores (si los hay)
        errores = client.remove_objects(BUCKET_NAME, elementos_a_borrar)
        
        hubo_errores = False
        for error in errores:
            print(f"Error al borrar: {error}")
            hubo_errores = True
            
        if not hubo_errores:
            print("Carpeta borrada con éxito.")
    else:
        print("La carpeta está vacía o no existe. No se ha borrado nada.")
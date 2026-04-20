"""
Orquesta la ejecución del pipeline RT y sube la ventana agregada más reciente
a MinIO, manteniendo solo las N ventanas más recientes (ventana deslizante).

Flujo:
    1. Genera el dataset RT a nivel de parada (build_realtime_dataset).
    2. Agrega en ventanas temporales de 30 min (aggregate_realtime).
    3. Sube la ventana a MinIO con nombre ISO (orden cronológico).
    4. Lista las ventanas en MinIO y borra las más antiguas si hay más de N.

Uso:
    uv run python src/ETL/pipelines/upload_realtime_window.py

Variables de entorno requeridas:
    MINIO_ACCESS_KEY, MINIO_SECRET_KEY
"""

import os

from dotenv import load_dotenv

from src.ETL.pipelines.generate_realtime_dataset import build_realtime_dataset
from src.ETL.pipelines.aggregate_realtime_dataset import aggregate_realtime
from src.common.minio_client import (
    upload_df_parquet,
    list_objects,
    delete_object,
)

load_dotenv()

# Configuración
MINIO_PREFIX = "grupo5/realtime_windows/"
NUM_VENTANAS_A_MANTENER = 4
TIEMPO_VENTANA = "30"  


def main():
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY y MINIO_SECRET_KEY no están definidas.")

    # 1) Generar dataset RT
    print("\n[1/3] Generando dataset RT...")
    df_rt = build_realtime_dataset()

    # 2) Agregar en ventanas
    print("\n[2/3] Agregando en ventanas temporales...")
    df_agregado = aggregate_realtime(df_rt, tiempo=TIEMPO_VENTANA)

    if df_agregado.empty:
        print("[WARN] Agregado vacío. No se sube nada a MinIO.")
        return

    # 3) Subir a MinIO con nombre basado en la ventana del DataFrame
    ventana = df_agregado['merge_time'].iloc[0]
    nombre = ventana.strftime("%Y-%m-%d_%H-%M")
    object_name = f"{MINIO_PREFIX}ventana_{nombre}.parquet"

    print(f"\n[3/3] Subiendo a MinIO: {object_name}")
    upload_df_parquet(access_key, secret_key, object_name, df_agregado)
    print(f"  [OK] Subida: {len(df_agregado)} filas")

    # 4) Mantener solo las N ventanas más recientes (ventana deslizante)
    print(f"\n[CLEANUP] Conservando solo las {NUM_VENTANAS_A_MANTENER} ventanas más recientes...")
    objetos = list_objects(access_key, secret_key, prefix=MINIO_PREFIX)

    # Filtrar solo parquets de ventana (por si hay otros archivos en la carpeta)
    objetos = [o for o in objetos if o.endswith(".parquet") and "ventana_" in o]

    # Ordenar cronológicamente (el nombre es ISO, orden alfabético = cronológico)
    objetos.sort()

    # Si hay más de N, borrar los más antiguos
    if len(objetos) > NUM_VENTANAS_A_MANTENER:
        a_borrar = objetos[:-NUM_VENTANAS_A_MANTENER]
        for obj in a_borrar:
            delete_object(access_key, secret_key, obj)
            print(f"  [DEL] {obj}")

    # Mostrar las ventanas conservadas
    objetos_finales = sorted([
        o for o in list_objects(access_key, secret_key, prefix=MINIO_PREFIX)
        if o.endswith(".parquet")
    ])
    print(f"\n  Ventanas en MinIO ({len(objetos_finales)}):")
    for o in objetos_finales:
        print(f"    - {o}")


if __name__ == "__main__":
    main()
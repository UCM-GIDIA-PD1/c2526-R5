"""
Módulo de ingesta histórica de alertas oficiales MTA.
Se integra con el orquestador 'run_extraccion' y permite:
- Descargar alertas históricas desde data.ny.gov
- Extraer los datos en un rango de fechas indicadas
- Subir los datos a la carpeta raw de MINIO en forma JSON

La función principal 'ingest_alertas' es el punto de entrada
llamado por el orquestador del pipeline.
"""

import requests
import os
import json
from datetime import datetime
from typing import List, Dict
from src.common.minio_client import upload_json
BASE_URL = "https://data.ny.gov/resource/7kct-peq7.json"
MINIO_BASE_PATH = "grupo5/raw/official_alerts"


def fetch_data(start_date: str, end_date: str, limit: int = 50000):
    """ 
    Descarga datos históricos del dataset de alertas oficiales
    utilizando paginación.
    limit : Número máximo de registros por petición
    where: devuelve los registos cuya columna date esté entre esas fechas
    offset : es lo que permite la paginación. 
    En cada iteración descarga un bloque de 50.000, los añade a all_results, incremneta
    el offset y repite hasta que no queden datos
    """
    all_results = []
    offset = 0

    while True:
        params = {
            "$where": f"date between '{start_date}' and '{end_date}'",
            "$limit": limit,
            "$offset": offset
        }

        print(f"[alertas] Descargando offset={offset}")

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_results.extend(data)
        offset += limit

    return all_results




def ingest_alertas(start: str, end: str) -> None:
    """
    Función principal de ingesta llamada por el orquestador.
    Realiza lo siguiente:
    -Valida las credenciales de minio.
    - Descarga los datos históricos en el rango indicado.
    - Construye la ruta de almacenamiento
    - Sube el JSON directamente a MINIO
    """
    print(f"[alertas] START start={start} end={end}")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            "Las variables de entorno MINIO_ACCESS_KEY y MINIO_SECRET_KEY deben estar definidas."
        )
    data = fetch_data(start, end)
    object_name = (
        f"{MINIO_BASE_PATH}/"
        f"range={start}_to_{end}/"
        f"alertas_oficiales_2025.json"
    )
    upload_json(
        access_key=access_key,
        secret_key=secret_key,
        object_name=object_name,
        data=data
    )
    print(f"[alertas] Subido a Minio://pd1/{object_name}")

    print("[alertas] DONE")
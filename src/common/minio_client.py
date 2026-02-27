'''
Script simplificado para subir y descargar datos desde MinIO.

Ejecutar siempre usando -m (ejecuta archivo como módulo dentro del paquete)
Ejemplo: uv run python -m src.nombre_archivo

Permite trabajar con:
- Archivos locales
- DataFrames (se guardan como Parquet)
- Objetos JSON

Valores por defecto:
- endpoint = "minio.fdi.ucm.es"
- bucket = "pd1"

El usuario solo debe proporcionar:
access_key, secret_key, object_name (ruta en MinIO) y el dato correspondiente.

Casos de uso:

1) Subir/Descargar archivo local:
   upload_file(access_key, secret_key, object_name, file_path, 
                    endpoint, bucket)

   download_file(access_key, secret_key, object_name, file_path, 
                    endpoint, bucket)

2) Subir DataFrame como Parquet:
   upload_df_parquet(access_key, secret_key, object_name, df, 
                        endpoint, bucket)

3) Descargar Parquet como DataFrame:
   df = download_df_parquet(access_key, secret_key, object_name, 
                                endpoint, bucket)

4) Subir / descargar JSON:
   upload_json(access_key, secret_key, object_name, data, 
                    endpoint, bucket)

   data = download_json(access_key, secret_key, object_name, 
                        endpoint, bucket)

Nota: para Parquet necesitas tener instalado 'pyarrow'
'''

import io
import json
from typing import Any

import pandas as pd
from minio import Minio

DEFAULT_ENDPOINT = "minio.fdi.ucm.es"
DEFAULT_BUCKET = "pd1"

def _client(access_key: str, secret_key: str, endpoint: str = DEFAULT_ENDPOINT) -> Minio:
    """Crear un cliente de MinIO"""
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)


# Archivos (upload/download)

def upload_file(
    access_key: str,
    secret_key: str,
    object_name: str,
    file_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """Subir un archivo local a MinIO con ruta object_name"""
    c = _client(access_key, secret_key, endpoint)
    c.fput_object(bucket, object_name, file_path)


def download_file(
    access_key: str,
    secret_key: str,
    object_name: str,
    file_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """Descargar object_name de MinIO a ruta local file_path"""
    c = _client(access_key, secret_key, endpoint)
    c.fget_object(bucket, object_name, file_path)


# DataFrames como Parquet (upload/download)

def upload_df_parquet(
    access_key: str,
    secret_key: str,
    object_name: str,
    df: pd.DataFrame,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """Subir un pandas Dataframe como objeto parquet"""
    c = _client(access_key, secret_key, endpoint)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    c.put_object(bucket, object_name, buf, length=buf.getbuffer().nbytes)


def download_df_parquet(
    access_key: str,
    secret_key: str,
    object_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> pd.DataFrame:
    """Descargar un archivo parquet como pandas Dataframe"""
    c = _client(access_key, secret_key, endpoint)
    resp = c.get_object(bucket, object_name)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return pd.read_parquet(io.BytesIO(data))


# JSON (upload/download)

def upload_json(
    access_key: str,
    secret_key: str,
    object_name: str,
    data: Any,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """Subir un objeto de Python como JSON"""
    c = _client(access_key, secret_key, endpoint)
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    buf = io.BytesIO(raw)
    c.put_object(bucket, object_name, buf, length=len(raw))


def download_json(
    access_key: str,
    secret_key: str,
    object_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> Any:
    """Descargar un objeto JSON desde MinIO como objeto de Python"""
    c = _client(access_key, secret_key, endpoint)
    resp = c.get_object(bucket, object_name)
    try:
        raw = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return json.loads(raw.decode("utf-8"))

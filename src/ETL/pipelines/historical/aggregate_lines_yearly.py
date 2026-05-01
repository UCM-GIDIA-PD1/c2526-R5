"""
Script para juntar todos los meses de cada línea en un único parquet anual.

Lee los parquets mensuales por línea de MinIO:
    pd1/grupo5/final/year=2025/month=MM/lines/line=XX/dataset_final.parquet

Y sube un parquet anual por línea a:
    pd1/grupo5/aggregations/lines/line=XX/dataset_final.parquet

Estrategia de memoria:
    Descubre las líneas disponibles en month=01, luego procesa una línea a la vez
    (lee sus 12 meses, concatena, sube). En RAM nunca hay más de una línea completa.

Uso:
    uv run python src/ETL/pipelines/historical/aggregate_lines_yearly.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
"""

import os

import pandas as pd
from minio import Minio

from src.common.minio_client import download_df_parquet, upload_df_parquet, DEFAULT_ENDPOINT, DEFAULT_BUCKET



ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR   = 2025
MONTHS = range(1, 13)

INPUT_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/lines/line={line}/dataset_final.parquet"
OUTPUT_TEMPLATE = "grupo5/aggregations/lines/line={line}/dataset_final.parquet"



def discover_lines(access_key: str, secret_key: str, year: int) -> list[str]:
    """Lista las líneas disponibles buscando en todos los meses."""
    client = Minio(DEFAULT_ENDPOINT, access_key=access_key, secret_key=secret_key, secure=True)
    lines = set()
    for month in MONTHS:
        prefix = f"grupo5/final/year={year}/month={month:02d}/lines/"
        try:
            objects = client.list_objects(DEFAULT_BUCKET, prefix=prefix)
            for obj in objects:
                # obj.object_name → "grupo5/.../lines/line=A/dataset_final.parquet"
                parts = obj.object_name.split("/")
                for part in parts:
                    if part.startswith("line="):
                        lines.add(part.replace("line=", ""))
        except Exception:
            pass
    return sorted(lines)

print(f"Descubriendo líneas disponibles en year={YEAR}...")
lines = discover_lines(ACCESS_KEY, SECRET_KEY, YEAR)

if not lines:
    raise RuntimeError("No se encontraron líneas. Revisa que el split por línea se haya ejecutado primero.")

print(f"Líneas encontradas ({len(lines)}): {lines}\n")



errores       = []
lineas_ok     = []

for line in lines:
    dfs = []
    meses_disponibles = []

    for month in MONTHS:
        path = INPUT_TEMPLATE.format(year=YEAR, month=month, line=line)
        try:
            df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
            dfs.append(df)
            meses_disponibles.append(month)
        except Exception:
            pass

    if not dfs:
        print(f"  ✗ line={line:<4}  sin datos en ningún mes, omitida")
        continue

    df_yearly = pd.concat(dfs, ignore_index=True)
    del dfs

    path_out = OUTPUT_TEMPLATE.format(line=line)
    try:
        upload_df_parquet(ACCESS_KEY, SECRET_KEY, path_out, df_yearly)
        print(f" line={line:<4}  {len(df_yearly):>10,} filas  meses={meses_disponibles}  →  {path_out}")
        lineas_ok.append(line)
    except Exception as e:
        errores.append(line)
        print(f"line={line:<4}  ERROR al subir: {e}")

    del df_yearly



print("\n" + "─" * 60)
print(f"Resumen:")
print(f"  Líneas subidas   : {len(lineas_ok)}/{len(lines)}  {lineas_ok}")
if errores:
    print(f"  Líneas con error : {errores}")
print(f"  Destino          : pd1/grupo5/aggregations/lines/line=XX/dataset_final.parquet")
print("─" * 60)

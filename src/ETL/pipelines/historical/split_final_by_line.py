"""
Script para particionar el dataset final por mes y línea de metro.

Lee los archivos mensuales de MinIO:
    pd1/grupo5/final/year=2025/month=MM/dataset_final.parquet

Y sube un parquet por línea para cada mes:
    pd1/grupo5/final/year=2025/month=MM/line=XX/dataset_final.parquet

Estrategia de memoria:
    Procesa un mes a la vez → parte por route_id → sube cada línea → pasa al siguiente mes.
    En RAM nunca hay más de un mes cargado.

Uso:
    uv run python src/ETL/pipelines/historical/split_final_by_line.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
"""

import os

from src.common.minio_client import download_df_parquet, upload_df_parquet

# Configuración

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

YEAR        = 2025
MONTHS      = range(1, 13)
LINE_COLUMN = "route_id"

INPUT_TEMPLATE  = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"
OUTPUT_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/lines/line={line}/dataset_final.parquet"

#Procesar mes a mes

meses_ok        = []
meses_faltantes = []
total_subidos   = 0
errores         = []

print(f"Particionando year={YEAR} por línea...\n")

for month in MONTHS:
    path_in = INPUT_TEMPLATE.format(year=YEAR, month=month)

    try:
        df_month = download_df_parquet(ACCESS_KEY, SECRET_KEY, path_in)
        meses_ok.append(month)
        print(f"month={month:02d}  {len(df_month):>10,} filas  →  particionando...")
    except Exception as e:
        meses_faltantes.append(month)
        print(f"month={month:02d}  ✗ no encontrado ({e})")
        continue

    if LINE_COLUMN not in df_month.columns:
        raise ValueError(
            f"Columna '{LINE_COLUMN}' no encontrada. "
            f"Columnas disponibles: {list(df_month.columns)}"
        )

    # Partir por línea y subir cada chunk directamente a MinIO
    for line, df_line in df_month.groupby(LINE_COLUMN, sort=False):
        path_out = OUTPUT_TEMPLATE.format(year=YEAR, month=month, line=line)
        try:
            upload_df_parquet(ACCESS_KEY, SECRET_KEY, path_out, df_line)
            print(f"  ✓ line={line:<4}  {len(df_line):>8,} filas  →  {path_out}")
            total_subidos += 1
        except Exception as e:
            errores.append((month, line))
            print(f"  ✗ line={line:<4}  ERROR: {e}")

    del df_month  

# Resumen final 

print("\n" + "─" * 60)
print(f"Resumen:")
print(f"  Meses procesados : {len(meses_ok)}/12  {meses_ok}")
if meses_faltantes:
    print(f"  Meses faltantes  : {meses_faltantes}")
print(f"  Parquets subidos : {total_subidos}")
if errores:
    print(f"  Errores          : {errores}")
print(f"  Destino          : grupo5/final/year={YEAR}/month=MM/lines/line=XX/dataset_final.parquet")
print("─" * 60)

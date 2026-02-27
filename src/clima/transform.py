"""
Clima histórico - Transformación (processed -> cleaned)

Columnas de entrada:
  "Date",
    "Temperature",
    "Rain",
    "Precipitation",
    "Wind Speed",
    "Snow",
    "Cloud Cover",

Lee de (MinIO):
  Cambiar: grupo5/processed/Clima/Clima_Historico/YYYY-MM-DD/Clima_Historico_YYYY-MM-DD.parquet

Escribe a (MinIO):
    grupo5/cleaned/clima_clean/date=YYYY-MM-DD/clima_YYYY-MM-DD.parquet
    grupo5/cleaned/clima_clean/date=YYYY-MM-DD/quality_report_YYYY-MM-DD.json

Creación de nuevas variables:
- Sensación térmica ('apparent_temp')
- Lluvia acumulada en las últimas 3 horas  ('precip_3h_accum')
- Condiciones extremas ('is_heavy_rain', 'is_high_wind', 'is_freezing')
- Hora ('hour')
- Hora punta o no ('is_rush_hour')
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from scipy import stats 

from src.common.minio_client import download_df_parquet, upload_df_parquet, upload_json


REQUIRED_COLS = [
    "Date",
    "Temperature",
    "Rain",
    "Precipitation",
    "Wind Speed",
    "Snow",
    "Cloud Cover",
]


INPUT_BASE_PATH = "grupo5/processed/Clima/Clima_Historico/{day}/Clima_Historico_{day}.parquet"
OUTPUT_DATA_PATH = "grupo5/cleaned/clima_clean/date={day}/clima_{day}.parquet"
OUTPUT_JSON_PATH = "grupo5/cleaned/clima_clean/date={day}/quality_report_{day}.json"


def calculate_apparent_temp(t, ws):
    """Calcula sensación térmica (Wind Chill simplificado)."""
    # Aproximación para climas como NYC
    return 13.12 + 0.6215*t - 11.37*(ws**0.16) + 0.3965*t*(ws**0.16)

def generate_quality_report(df_before, df_after):
    """Genera el informe de calidad en formato JSON."""
    return {
        "execution_at": datetime.now().isoformat(),
        "stats": {
            "rows_raw": len(df_before),
            "rows_clean": len(df_after),
            "removed_rows": len(df_before) - len(df_after),
            "nulls_in_temp": int(df_before["Temperature"].isna().sum())
        },
        "data_ranges": {
            "min_temp": float(df_after["Temperature"].min()) if not df_after.empty else 0,
            "max_temp": float(df_after["Temperature"].max()) if not df_after.empty else 0,
            "total_precip_day": float(df_after["Precipitation"].sum()) if not df_after.empty else 0
        }
    }

def transform_weather_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Lógica de transformación:
    - Limpieza de tipos y nulos usando 'Date'.
    - Filtrado de outliers estadísticos.
    - Features de impacto para el Metro de NYC.
    """
    df_raw = df.copy()

    # 1. Asegurar Tipos (Columna correcta: 'Date')
    df['Date'] = pd.to_datetime(df['Date'])
    numeric_cols = ["Temperature", "Precipitation", "Wind Speed", "Snow"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Limpieza básica
    # Eliminamos duplicados y filas donde la temperatura o fecha sean nulas
    df = df.dropna(subset=['Date', 'Temperature']).drop_duplicates()

    # 3. Filtrado de Outliers (Z-Score)
    # Si hay suficientes datos, eliminamos errores de sensor (>3 desviaciones estándar)
    if len(df) > 5:
        z = np.abs(stats.zscore(df['Temperature']))
        df = df[z < 3]

    # 4. Feature Engineering (Contexto Operativo Metro)
    df = df.sort_values('Date')
    
    # Sensación térmica: importante para predecir demanda en estaciones exteriores
    df['apparent_temp'] = df.apply(lambda x: calculate_apparent_temp(x['Temperature'], x['Wind Speed']), axis=1)
    
    # Acumulado de lluvia (Rolling 3h): El metro se inunda por acumulación, no solo por intensidad instantánea
    df['precip_3h_accum'] = df['Precipitation'].rolling(window=3, min_periods=1).sum()
    
    # Flags de riesgo para el Metro NYC
    df['is_freezing'] = (df['Temperature'] <= 0).astype(int)  # Hielo en tercer raíl
    df['is_high_wind'] = (df['Wind Speed'] > 50).astype(int)  # Peligro en puentes (N, Q, B, D)
    
    # Variables de tiempo para facilitar JOINS con el dataset de la API de metro
    df['hour'] = df['Date'].dt.hour
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    report = generate_quality_report(df_raw, df)
    return df, report

def run_pipeline(start_str: str, end_str: str):
    """Automatización del rango de fechas y subida a MinIO."""
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    curr = start_dt
    while curr <= end_dt:
        day = curr.strftime("%Y-%m-%d")
        try:
            # Descarga
            df_weather = download_df_parquet(access_key, secret_key, INPUT_BASE_PATH.format(day=day))
            
            # Transformación
            df_clean, report = transform_weather_data(df_weather)
            
            # Carga de datos y reporte JSON
            upload_df_parquet(access_key, secret_key, OUTPUT_DATA_PATH.format(day=day), df_clean)
            upload_json(access_key, secret_key, OUTPUT_JSON_PATH.format(day=day), report)
            
            print(f"{day}: Procesado correctamente.")
        except Exception as e:
            print(f"{day}: Error en transformación -> {str(e)}")
            
        curr += timedelta(days=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    run_pipeline(args.start, args.end)
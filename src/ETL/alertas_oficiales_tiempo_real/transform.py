"""
Transformación de alertas oficiales MTA. Coordinado con el orquestador

1. Lee RAW desde MinIO.
2. Aplica transformación básica.
3. Sube partición diaria a processed.
4. Aplica limpieza final.
5. Sube partición diaria a cleaned.
"""

import os
import pandas as pd
from src.ETL.common.minio_client import download_json, upload_df_parquet

BUCKET = "pd1"

RAW_BASE = "grupo5/raw/official_alerts"
PROCESSED_BASE = "grupo5/processed/official_alerts"
CLEANED_BASE = "grupo5/cleaned/official_alerts"


def upload_processed_day(access_key: str, secret_key: str, df_day: pd.DataFrame, day_date):
    """
    Sube el parquet diario a la capa PROCESSED.
    """
    processed_prefix = f"{PROCESSED_BASE}/date={day_date}"

    if df_day.empty:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{processed_prefix}/_empty.parquet",
            pd.DataFrame()
        )
        print(f"[processed] Carpeta creada sin datos: {day_date}")
    else:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{processed_prefix}/alerts.parquet",
            df_day
        )
        #print(f"[processed] Subido día {day_date}")


def agrupar_alertas(df):
    """Varias líneas del datframe tienen el mismo código de evento_id eso significa que son la misma alerta pero 
    actualizada , el objetivo de esta función es si varias lineas contienen las mismas líneas afectadas, el mismo header y el 
     mismo status_label , se agrupa en una sola con timestamp inicial y final y el número de actualizaciones que ha sufrido"""
    df_grouped = (
        df.groupby(
            ["event_id", "status_label", "affected", "header"],
            as_index=False
        )
        .agg(
            timestamp_inicial=("date", "min"),
            timestamp_final=("date", "max"),
            agency=("agency", "first"),
            description=("description", "last"),
            num_updates=("date", "count")
        )
    )
    df_grouped["num_updates"] = df_grouped["num_updates"] - 1
    df_grouped["num_updates"] = df_grouped["num_updates"].clip(lower=0)
    df_grouped = df_grouped.sort_values("timestamp_inicial")
    return df_grouped

def map_category(status):
    """Esta función se encarga clasificar status label ya que algunas lineas contienen mas de un valor, pero solo
    nos interesa uno para el posterior análisis, dando prioridad a las supensiones y delays"""
    if pd.isna(status):
        return "Other"
    status = status.lower()
    if "suspended"in status or "part-suspended" in status:
        return "Suspension"
    elif "severe-delays" in status:
        return "Severe Delay"
    elif "delay" in status:
        return "Delay"
    elif "reroute" in status or "express-to-local" in status or "stops-skipped" in status or "boarding-change" in status:
        return "Service Change"
    elif "cancellations" in status:
        return "Cancellation"
    else:
        return "Other"
    
def upload_cleaned_day(access_key: str, secret_key: str, df_day: pd.DataFrame, day_date):
    """
    Aplica limpieza final y sube el parquet diario a la capa CLEANED. El objetivo es coordinarlo 
    lo máximo posible con el dataframe de realtime
    """
    cleaned_prefix = f"{CLEANED_BASE}/date={day_date}"

    # Cogemos solo alertas metro
    df_day_clean = df_day[df_day["agency"] == "NYCT Subway"].copy()
    
    
    if df_day_clean.empty:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{cleaned_prefix}/_empty.parquet",
            pd.DataFrame()
        )
        print(f"[cleaned] Carpeta creada sin datos: {day_date}")
    else:
        df_day_clean = agrupar_alertas(df_day_clean)
        df_day_clean["category"] = df_day_clean["status_label"].apply(map_category)
        df_day_clean.columns = df_day_clean.columns.str.strip()
        df_day_clean = df_day_clean.drop(
        columns=["agency", "status_label"],
        errors="ignore"
        )
        #renombre de columnas para asemejarse al dataframe realtime
        df_day_clean = df_day_clean.rename(columns={
            "timestamp_inicial": "timestamp_start",
            "timestamp_final": "timestamp_end",
            "affected": "lines",
            "header": "text_snippet"
        })
        #reordenar columnas
        df_day_clean = df_day_clean[
            [
                "event_id",
                "timestamp_start",
                "timestamp_end",
                "category",
                "lines",
                "text_snippet",
                "description"
            ]
        ]
        #en la columna lines, sustituimos la barra (|) por comas
        df_day_clean["lines"] = (
            df_day_clean["lines"]
                .str.replace(r"\s*\|\s*", ", ", regex=True)
        )
        upload_df_parquet(
            access_key,
            secret_key,
            f"{cleaned_prefix}/alerts.parquet",
            df_day_clean
        )
        #print(f"[cleaned] Subido día {day_date}")



def run_transform(start: str, end: str) :
    """
    Función llamada por el orquestador run_transform.
    """

    print(f"[alertas_transform] START start={start} end={end}")

    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not access_key or not secret_key:
        raise ValueError(
            "MINIO_ACCESS_KEY y MINIO_SECRET_KEY deben estar definidas."
        )
    
    raw_file = (
        f"{RAW_BASE}/"
        f"range={start}_to_{end}/"
        f"alertas_oficiales_2025.json"
    )

    print("[alertas_transform] Descargando RAW...")

    data = download_json(
        access_key=access_key,
        secret_key=secret_key,
        object_name=raw_file
    )

    df = pd.DataFrame(data)

    print(f"[alertas_transform] Registros RAW: {len(df)}")


    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates()

    print(f"[alertas_transform] Registros tras transform: {len(df)}")

    all_days = pd.date_range(start, end, freq="D")

    for day in all_days:
        day_date = day.date()

        df_day = df[df["date"].dt.date == day_date]

        upload_processed_day(access_key, secret_key, df_day, day_date)
        upload_cleaned_day(access_key, secret_key, df_day, day_date)

    print("[alertas_transform] DONE")
"""
GTFS histórico - Transformación (processed -> cleaned)

Columnas de entrada:
  match_key, route_id, stop_id, is_unscheduled,
  scheduled_seconds, actual_seconds, delay_seconds, delay_minutes

Lee de (MinIO):
  grupo5/processed/gtfs_with_delays/date=YYYY-MM-DD/mta_delays_YYYY-MM-DD.parquet

Escribe a (MinIO):
  Scheduled:
    grupo5/cleaned/gtfs_clean_scheduled/date=YYYY-MM-DD/gtfs_scheduled_YYYY-MM-DD.parquet
    grupo5/cleaned/gtfs_clean_scheduled/date=YYYY-MM-DD/quality_report_YYYY-MM-DD.json

  Unscheduled:
    grupo5/cleaned/gtfs_clean_unscheduled/date=YYYY-MM-DD/gtfs_unscheduled_YYYY-MM-DD.parquet
    grupo5/cleaned/gtfs_clean_unscheduled/date=YYYY-MM-DD/quality_report_YYYY-MM-DD.json
"""

import os
import math
from datetime import date, timedelta
from typing import Dict, Any, List
import pandas as pd

from src.common.minio_client import download_df_parquet, upload_df_parquet, upload_json


REQUIRED_COLS = [
    "match_key",
    "route_id",
    "stop_id",
    "is_unscheduled",
    "scheduled_seconds",
    "actual_seconds",
    "delay_seconds",
    "delay_minutes",
]


def iterate_dates(start: date, end: date):
    """Itera fechas (start y end inclusive)"""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def build_processed_object(day: str) -> str:
    return f"grupo5/processed/gtfs_with_delays/date={day}/mta_delays_{day}.parquet"


def build_cleaned_scheduled_object(day: str) -> str:
    return f"grupo5/cleaned/gtfs_clean_scheduled/date={day}/gtfs_scheduled_{day}.parquet"


def build_cleaned_unscheduled_object(day: str) -> str:
    return f"grupo5/cleaned/gtfs_clean_unscheduled/date={day}/gtfs_unscheduled_{day}.parquet"


def build_quality_scheduled_object(day: str) -> str:
    return f"grupo5/cleaned/gtfs_clean_scheduled/date={day}/quality_report_{day}.json"


def build_quality_unscheduled_object(day: str) -> str:
    return f"grupo5/cleaned/gtfs_clean_unscheduled/date={day}/quality_report_{day}.json"


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validar que el dataframe tiene las columnas requeridas
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en processed: {missing}")


def add_derivated_features(df: pd.DataFrame, service_date: str) -> pd.DataFrame:
    """
    Genera features derivados sin agregaciones:
    - service_date
    - hour (aprox desde scheduled_seconds si existe; si no desde actual_seconds)
    - dow, is_weekend
    - hour_sin/cos
    - scheduled_time y actual_time en formato HH:MM:SS (para cruzar con clima/eventos)
    """
    out = df.copy()
    out["service_date"] = service_date

    sec_base = out["scheduled_seconds"].where(~out["scheduled_seconds"].isna(), out["actual_seconds"])
    out["hour"] = ((sec_base // 3600) % 24).astype("Int64")

    hour_float = out["hour"].astype("float")
    out["hour_sin"] = hour_float.apply(lambda h: math.sin(2 * math.pi * h / 24) if pd.notna(h) else None)
    out["hour_cos"] = hour_float.apply(lambda h: math.cos(2 * math.pi * h / 24) if pd.notna(h) else None)

    dt = pd.to_datetime(out["service_date"], format="%Y-%m-%d", errors="coerce")
    out["dow"] = dt.dt.dayofweek.astype("Int64")
    out["is_weekend"] = out["dow"].isin([5, 6]).astype("Int64")

    # Añadir columnas de tiempo formateado (HH:MM:SS) para scheduled y actual, si existen

    if "scheduled_seconds" in out.columns:
        # Añadimos fillna(0) para que Numpy no lanze error al convertir NaNs a timedelta, luego corregimos a None
        td_sched = pd.to_timedelta(out["scheduled_seconds"].fillna(0), unit='s')
        # Formatear rellenando con ceros a la izquierda (ej. 08:05:09)
        out["scheduled_time"] = (
            td_sched.dt.components.hours.astype(str).str.zfill(2) + ":" +
            td_sched.dt.components.minutes.astype(str).str.zfill(2) + ":" +
            td_sched.dt.components.seconds.astype(str).str.zfill(2)
        )
        # Los NaNs se convertirán en "nan:nan:nan", los limpiamos:
        out.loc[out["scheduled_seconds"].isna(), "scheduled_time"] = None

    if "actual_seconds" in out.columns:
        td_act = pd.to_timedelta(out["actual_seconds"].fillna(0), unit='s')
        out["actual_time"] = (
            td_act.dt.components.hours.astype(str).str.zfill(2) + ":" +
            td_act.dt.components.minutes.astype(str).str.zfill(2) + ":" +
            td_act.dt.components.seconds.astype(str).str.zfill(2)
        )
        out.loc[out["actual_seconds"].isna(), "actual_time"] = None

    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forzar datatypes
    """
    out = df.copy()
    # strings
    for c in ["match_key", "route_id", "stop_id"]:
        out[c] = out[c].astype("string")

    # trip_uid (opcional)
    if "trip_uid" in out.columns:
        out["trip_uid"] = out["trip_uid"].astype("string")

    # booleans
    out["is_unscheduled"] = out["is_unscheduled"].astype("bool")

    # numeric
    for c in ["scheduled_seconds", "actual_seconds", "delay_seconds", "delay_minutes"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicación más robusta que drop_duplicates() global.
    - match_key + stop_id suele identificar un stop-event
    - añadimos actual_seconds para diferenciar casos raros
    """
    subset = ["match_key", "stop_id", "actual_seconds"]
    subset = [c for c in subset if c in df.columns]
    return df.drop_duplicates(subset=subset)


def filter_delay_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro suave de outliers: delays fuera de +/- 2.5h suelen ser ruido (pero ajustable)
    """
    return df[(df["delay_seconds"].isna()) | (df["delay_seconds"].between(-9000, 9000))]


def quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame, name: str) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "dataset": name,
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "dropped_rows": int(len(df_before) - len(df_after)),
        "nulls_after": {c: int(df_after[c].isna().sum()) for c in df_after.columns},
    }
    s = df_after["delay_seconds"].dropna()
    rep["delay_seconds_stats"] = {
        "min": None if s.empty else float(s.min()),
        "max": None if s.empty else float(s.max()),
        "mean": None if s.empty else float(s.mean()),
        "p50": None if s.empty else float(s.quantile(0.5)),
        "p95": None if s.empty else float(s.quantile(0.95)),
    }
    return rep


# Transformación por dia

def transform_processed_day_to_cleaned(
    df_processed: pd.DataFrame,
    service_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Devuelve dict con dos DataFrames:
      - scheduled
      - unscheduled
    """
    validate_schema(df_processed)
    df = coerce_types(df_processed)

    # Limpieza común
    df = df.dropna(subset=["match_key", "stop_id"])  # mínimo para identificar viaje/parada
    df = deduplicate(df)
    df = filter_delay_outliers(df)
    df = add_derivated_features(df, service_date)

    # Split
    scheduled = df[df["is_unscheduled"] == False].copy()
    unscheduled = df[df["is_unscheduled"] == True].copy()

    # Scheduled: debe tener referencia teórica para modelar delay vs horario
    scheduled = scheduled.dropna(subset=["route_id", "scheduled_seconds"])

    # Unscheduled: permitimos route_id/scheduled_seconds nulos (es normal)
    # pero sí exigimos actual_seconds (si no, no aporta nada)
    unscheduled = unscheduled.dropna(subset=["actual_seconds"])

    return {"scheduled": scheduled, "unscheduled": unscheduled}


def transform_gtfs_processed_range_to_cleaned(
    start: date,
    end: date,
    access_key: str,
    secret_key: str,
) -> None:
    for d in iterate_dates(start, end):
        day = d.strftime("%Y-%m-%d")

        in_obj = build_processed_object(day)
        df_before = download_df_parquet(access_key, secret_key, in_obj)

        outputs = transform_processed_day_to_cleaned(df_before, service_date=day)
        df_sched = outputs["scheduled"]
        df_uns = outputs["unscheduled"]

        # write scheduled
        upload_df_parquet(access_key, secret_key, build_cleaned_scheduled_object(day), df_sched)
        upload_json(access_key, secret_key, build_quality_scheduled_object(day), quality_report(df_before, df_sched, "scheduled"))

        # write unscheduled
        upload_df_parquet(access_key, secret_key, build_cleaned_unscheduled_object(day), df_uns)
        upload_json(access_key, secret_key, build_quality_unscheduled_object(day), quality_report(df_before, df_uns, "unscheduled"))

        print(
            f"[gtfs_historico.transform] OK {day} "
            f"scheduled={len(df_sched)} unscheduled={len(df_uns)}"
        )


def run_transform(start: str, end: str) -> None:
    """Función usada por runner externo para ejecutar la transformacion.

    Convierte string dates a objetos ``date``, obtiene credenciales de MinIO 
    de las variables de entorno y delega a transform_gtfs_processed_range_to_cleaned
    """
    from datetime import datetime

    access_key = os.getenv("MINIO_ACCESS_KEY")
    if access_key is None:
        raise AssertionError("MINIO_ACCESS_KEY no definida")

    secret_key = os.getenv("MINIO_SECRET_KEY")
    if secret_key is None:
        raise AssertionError("MINIO_SECRET_KEY no definida")

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    transform_gtfs_processed_range_to_cleaned(
        start=start_date,
        end=end_date,
        access_key=access_key,
        secret_key=secret_key,
    )


if __name__ == "__main__":
    start = date(2025, 12, 1)
    end = date(2025, 12, 31)

    # delegar a función principal de transformación
    run_transform(start, end)
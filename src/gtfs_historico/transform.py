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


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade características temporales y de viaje al DataFrame:
    - **lagged_delay_1 / lagged_delay_2**: retraso del stop anterior y del antepenúltimo dentro del mismo viaje
    - **actual_headway_seconds**: tiempo transcurrido desde el tren previo en la misma parada
    - **route_rolling_delay**: promedio móvil a corto plazo de los retrasos en la misma ruta (no incluye el tren actual)
    - **period_of_day**: franja horaria categórica (morning_peak, midday, evening_peak, off_peak)
    - **is_peak**: indicador booleano de si la observación cae en hora punta
    - **trip_progress**: proporción de la duración planificada del viaje que ya ha transcurrido
    - **rolling_mean_delay_trip**: promedio móvil de retrasos a lo largo de un mismo viaje
    - **headway_ratio**: ratio entre el headway actual y el anterior en la misma parada

    Devuelve el DataFrame ordenado por su índice original.
    """
    out = df.copy()
    out["direction"] = out["stop_id"].str[-1]

    # lagged delays: retrasos previos del mismo viaje
    if "delay_seconds" in out.columns and "match_key" in out.columns:
        out = out.sort_values(by=["match_key", "actual_seconds"], na_position="last")
        out["lagged_delay_1"] = out.groupby("match_key", group_keys=False)["delay_seconds"].shift(1)
        out["lagged_delay_2"] = out.groupby("match_key", group_keys=False)["delay_seconds"].shift(2)

    # actual headway: tiempo desde el tren anterior en la misma parada
    if "actual_seconds" in out.columns and "stop_id" in out.columns:
        out = out.sort_values(by=["stop_id", "actual_seconds"], na_position="last")
        out["actual_headway_seconds"] = out.groupby("stop_id", group_keys=False)["actual_seconds"].diff()
        # headway ratio respecto al headway previo
        prev_headway = out.groupby("stop_id")["actual_headway_seconds"].shift(1).replace(0, pd.NA)
        out["headway_ratio"] = out["actual_headway_seconds"] / prev_headway

    # rolling delay por ruta
    if "delay_seconds" in out.columns and "route_id" in out.columns:
        out = out.sort_values(by=["route_id", "direction", "actual_seconds"], na_position="last")
        out["route_rolling_delay"] = out.groupby(["route_id", "direction"])["delay_seconds"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )

    # franja horaria / hora punta
    if "hour" in out.columns:
        def period_of_day(hour):
            if pd.isna(hour): return None
            if 6 <= hour < 10: return "morning_peak"
            if 10 <= hour < 16: return "midday"
            if 16 <= hour < 20: return "evening_peak"
            return "off_peak"
        out["period_of_day"] = out["hour"].apply(lambda h: period_of_day(h) if pd.notna(h) else None)
        out["is_peak"] = out["period_of_day"].isin(["morning_peak", "evening_peak"]).astype("Int64")

    # progreso del viaje y rolling interno
    if "trip_uid" in out.columns and "scheduled_seconds" in out.columns:
        min_sched = out.groupby("trip_uid")["scheduled_seconds"].transform("min")
        max_sched = out.groupby("trip_uid")["scheduled_seconds"].transform("max")
        # Evitar división por cero si el viaje solo tiene 1 parada programada o duracion 0
        trip_duration = (max_sched - min_sched).replace(0, pd.NA)
        out["trip_progress"] = (out["scheduled_seconds"] - min_sched) / trip_duration
        
        # Rolling mean delay del viaje intacto usando transform()
        out = out.sort_values(by=["trip_uid", "scheduled_seconds"], na_position="last")
        out["rolling_mean_delay_trip"] = out.groupby("trip_uid")["delay_seconds"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    # Borrar la columna temporal direction y restaurar el orden original
    if "direction" in out.columns:
        out = out.drop(columns=["direction"])
    out = out.sort_index()
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
    df = add_time_series_features(df)

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
    start = date(2025, 1, 1)
    end = date(2025, 1, 1)

    # delegar a función principal de transformación
    #run_transform(start, end)
    
    
    # Para pruebas, descomentar esto para guardar resultados en CSV:
    from datetime import datetime
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    for d in iterate_dates(start, end):
        day = d.strftime("%Y-%m-%d")
        in_obj = build_processed_object(day)
        df_processed = download_df_parquet(access_key, secret_key, in_obj)
        output = transform_processed_day_to_cleaned(df_processed, service_date=day)
        df_sched = output["scheduled"]
        df_uns = output["unscheduled"]
        df_sched.to_csv(f"tmp/gtfs_scheduled_{day}.csv", index=False)
        df_uns.to_csv(f"tmp/gtfs_unscheduled_{day}.csv", index=False)
        print(f"Saved test csvs for {day}")
    
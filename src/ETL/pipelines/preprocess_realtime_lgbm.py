"""
Genera el dataset de inferencia en tiempo real e indexa por trip_id (match_key).

Los datos diarios estáticos (clima, eventos, stop_times) se leen desde Google Drive
(carpeta MTA_Daily_Data/), actualizada por upload_daily_data.py una vez al día.
Las únicas llamadas en tiempo real son:
  - GTFS-RT  : endpoint de la línea concreta (extraída del trip_id)
  - Alertas  : Gmail MTA

Uso desde otro módulo:
    from src.ETL.pipelines.preprocess_realtime_lgbm import build_index, get_trip_features

    index    = build_index()
    features = get_trip_features(index, "033150_2..N08R")

Uso standalone (un único trip):
    uv run python src/ETL/pipelines/preprocess_realtime_lgbm.py <trip_id>
"""

import gc
import sys
import logging

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.ETL.pipelines.generate_realtime_dataset import (
    load_realtime_gtfs,
    load_realtime_alerts,
    _prepare_alert_route,
    merge_gtfs_weather_rt,
    merge_gtfs_events_rt,
    merge_gtfs_alerts_rt,
    apply_final_column_policy,
    reduce_mem_usage,
    normalize_route_id,
)
from src.ETL.tiempo_real_metro.realtime_data import (
    FUENTES,
    extraccion_linea,
    conversion_hora_NYC,
    dia_segun_fecha_y_formato,
    direccion_tren,
    union_dataframes,
)
from app.data.drive import download_daily_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Columnas que no son features (igual que en eval_lgbm.py).
# stop_id se conserva: necesario para el target encoding del modelo.
DROP_COLS = {
    "date", "merge_time", "timestamp_start", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
}


# ── Features derivadas ───────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mismas features derivadas que add_derived_features() en los scripts de evaluación."""
    if "lagged_delay_1" in df.columns and "delay_seconds" in df.columns:
        df["delay_velocity"] = df["delay_seconds"] - df["lagged_delay_1"]
    if "lagged_delay_1" in df.columns and "lagged_delay_2" in df.columns:
        df["delay_acceleration"] = (
            (df["delay_seconds"] - df["lagged_delay_1"])
            - (df["lagged_delay_1"] - df["lagged_delay_2"])
        )
    if "delay_seconds" in df.columns and "stops_to_end" in df.columns:
        df["delay_x_stops_remaining"] = df["delay_seconds"] * df["stops_to_end"]
    if "delay_seconds" in df.columns and "scheduled_time_to_end" in df.columns:
        df["delay_ratio"] = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1)
    return df


# ── Fuentes diarias desde Drive ──────────────────────────────────────────────

def _load_weather_drive() -> pd.DataFrame:
    log.info("  [CLIMA] Leyendo desde Drive...")
    return download_daily_file("clima_hoy.parquet", subfolder="clima")


def _load_events_drive() -> pd.DataFrame:
    log.info("  [EVENTOS] Leyendo desde Drive...")
    try:
        df = download_daily_file("eventos_hoy.parquet", subfolder="eventos")
        log.info("  [EVENTOS] %d filas.", len(df))
        return df
    except Exception as e:
        log.warning("  [EVENTOS] No disponible en Drive: %s", e)
        return pd.DataFrame()


def _load_stop_times_drive() -> pd.DataFrame:
    log.info("  [STOP TIMES] Leyendo desde Drive...")
    return download_daily_file("stop_times.parquet", subfolder="gtfs_supplemented")


# ── GTFS-RT para una línea concreta ─────────────────────────────────────────

def _route_id_from_trip(trip_id: str) -> str:
    """Extrae route_id del trip_id: '033150_2..N08R' → '2'."""
    return trip_id.split("_")[1].split(".")[0]


def _load_gtfs_rt_line(route_id: str) -> pd.DataFrame:
    """Llama solo al endpoint GTFS-RT de la línea dada."""
    url = None
    for info in FUENTES.values():
        if route_id.upper() in [l.upper() for l in info["lineas"]]:
            url = info["url"]
            break
    if url is None:
        raise ValueError(f"route_id '{route_id}' no encontrado en FUENTES")

    log.info("  [GTFS RT] Llamando endpoint para línea %s...", route_id)
    datos = extraccion_linea(url, route_id)
    df = pd.DataFrame(datos)
    if df.empty:
        raise ValueError(f"Sin datos RT para la línea {route_id}")

    df = conversion_hora_NYC(df)
    df = dia_segun_fecha_y_formato(df)
    df = direccion_tren(df)
    df = df.dropna(subset=["hora_llegada", "viaje_id", "parada_id", "linea_id"])
    df["segundos_reales"] = (
        df["hora_llegada"].dt.hour * 3600
        + df["hora_llegada"].dt.minute * 60
        + df["hora_llegada"].dt.second
    )
    log.info("  [GTFS RT] %d filas para línea %s.", len(df), route_id)
    return df


def _gtfs_rt_to_features(df_real: pd.DataFrame, df_previsto: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del feed RT de una línea y los horarios previstos (ya cargados),
    calcula el delay y adapta las columnas al esquema del pipeline.
    Equivalente a load_realtime_gtfs() pero para una sola línea.
    """
    log.info("  [GTFS] Calculando retrasos...")
    df = union_dataframes(df_real, df_previsto)
    if df.empty:
        raise ValueError("DataFrame GTFS vacío tras unión.")

    rename_map = {}
    if "linea_id"  in df.columns: rename_map["linea_id"]  = "route_id"
    if "parada_id" in df.columns: rename_map["parada_id"] = "stop_id"
    if "delay"     in df.columns: rename_map["delay"]     = "delay_seconds"
    if "direccion" in df.columns: rename_map["direccion"] = "direction"
    df = df.rename(columns=rename_map)

    if "hora_llegada" in df.columns:
        df["merge_time"] = pd.to_datetime(df["hora_llegada"], errors="coerce")
    else:
        df["merge_time"] = pd.Timestamp.now(tz="America/New_York")

    df["hour"]           = df["merge_time"].dt.hour
    df["date"]           = df["merge_time"].dt.date
    df["service_date"]   = df["merge_time"].dt.strftime("%Y-%m-%d")
    df["actual_seconds"] = (
        df["merge_time"].dt.hour * 3600
        + df["merge_time"].dt.minute * 60
        + df["merge_time"].dt.second
    )

    if "viaje_id" in df.columns:
        df["match_key"] = df["viaje_id"].astype(str)
    if "route_id" in df.columns:
        df["route_id"] = normalize_route_id(df["route_id"])
    if "stop_id" in df.columns:
        df["stop_id"] = df["stop_id"].astype("string")

    return df


# ── Merges comunes ────────────────────────────────────────────────────────────

def _merge_all(df_gtfs: pd.DataFrame) -> pd.DataFrame:
    """Aplica merges de clima, eventos y alertas sobre el DataFrame GTFS."""
    try:
        weather = _load_weather_drive()
    except Exception as e:
        log.warning("Clima no disponible: %s", e)
        weather = pd.DataFrame()

    merged = merge_gtfs_weather_rt(df_gtfs, weather)
    del df_gtfs, weather
    gc.collect()

    events = _load_events_drive()
    merged = merge_gtfs_events_rt(merged, events)
    del events
    gc.collect()

    try:
        alerts_raw = load_realtime_alerts()
        alerts = _prepare_alert_route(alerts_raw) if not alerts_raw.empty else pd.DataFrame()
    except Exception as e:
        log.warning("Alertas no disponibles: %s", e)
        alerts = pd.DataFrame()

    merged = merge_gtfs_alerts_rt(merged, alerts)
    del alerts
    gc.collect()

    merged = reduce_mem_usage(merged)
    return apply_final_column_policy(merged)


# ── API pública ───────────────────────────────────────────────────────────────

def build_index() -> dict[str, dict]:
    """
    Construye el índice de todos los trips activos indexado por match_key.
      - Stop times / Clima / Eventos : Drive (MTA_Daily_Data/)
      - GTFS-RT                      : todos los endpoints
      - Alertas                      : Gmail
    """
    log.info("=== BUILD INDEX ===")

    df_previsto = _load_stop_times_drive()
    df_gtfs = load_realtime_gtfs(df_previsto=df_previsto)

    df = _merge_all(df_gtfs)
    df = _add_derived_features(df)
    df = df.dropna(subset=["match_key"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Por trip, conservar la parada más cercana al final del viaje con stops_to_end > 0
    if "stops_to_end" in df.columns:
        df = (
            df[df["stops_to_end"] > 0]
            .sort_values("stops_to_end")
            .drop_duplicates(subset=["match_key"], keep="first")
        )

    index = df.set_index("match_key").to_dict(orient="index")
    log.info("Índice construido: %d trips.", len(index))
    return index


def get_single_trip_features(trip_id: str) -> dict | None:
    """
    Features para un único trip_id sin construir el índice completo.
      - Stop times / Clima / Eventos : Drive (MTA_Daily_Data/)
      - GTFS-RT                      : solo el endpoint de la línea del trip
      - Alertas                      : Gmail
    """
    log.info("=== SINGLE TRIP: %s ===", trip_id)

    route_id    = _route_id_from_trip(trip_id)
    df_previsto = _load_stop_times_drive()

    df_real = _load_gtfs_rt_line(route_id)
    if df_real[df_real["viaje_id"] == trip_id].empty:
        log.warning("trip_id '%s' no encontrado en el feed RT.", trip_id)
        return None

    # Pasar todo el feed de la línea para que stops_to_end y lags se calculen correctamente
    df_gtfs = _gtfs_rt_to_features(df_real, df_previsto)

    # Filtrar al trip concreto después de calcular features
    df = df_gtfs[df_gtfs["match_key"] == trip_id].copy()
    if df.empty:
        return None

    df = _merge_all(df)
    df = _add_derived_features(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df_mid = df[df["stops_to_end"] > 0]
    if df_mid.empty:
        return None
    return df_mid.iloc[-1].to_dict()


def get_trip_features(index: dict[str, dict], trip_id: str) -> dict | None:
    """Devuelve el dict de features para el trip_id dado, o None si no existe."""
    return index.get(trip_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/ETL/pipelines/preprocess_realtime_lgbm.py <trip_id>")
        sys.exit(1)

    trip_id  = sys.argv[1]
    features = get_single_trip_features(trip_id)

    if features is None:
        print(f"trip_id '{trip_id}' no encontrado.")
        sys.exit(1)

    for k, v in features.items():
        print(f"{k:35s} {v}")

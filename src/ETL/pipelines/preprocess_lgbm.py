"""
Genera el dataset de inferencia en tiempo real con exactamente las mismas
columnas que el dataset de entrenamiento (incluyendo features derivadas).

Uso desde otro módulo:
    from predict_realtime_lgbm import build_index, get_trip_features

    index    = build_index()
    features = get_trip_features(index, "033150_2..N08R")

Uso standalone:
    uv run python predict_realtime_lgbm.py 033150_2..N08R
"""

import sys
import logging

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.ETL.pipelines.generate_realtime_dataset import build_realtime_dataset



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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las mismas features derivadas que en eval_lgbm.py (add_derived_features)."""
    if "delay_seconds" in df.columns and "lagged_delay_1" in df.columns:
        df["delay_velocity"] = df["delay_seconds"] - df["lagged_delay_1"]

    if "delay_seconds" in df.columns and "lagged_delay_1" in df.columns and "lagged_delay_2" in df.columns:
        df["delay_acceleration"] = (
            (df["delay_seconds"] - df["lagged_delay_1"])
            - (df["lagged_delay_1"] - df["lagged_delay_2"])
        )

    if "delay_seconds" in df.columns and "stops_to_end" in df.columns:
        df["delay_x_stops_remaining"] = df["delay_seconds"] * df["stops_to_end"]

    if "delay_seconds" in df.columns and "scheduled_time_to_end" in df.columns:
        df["delay_ratio"] = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1)

    return df


def build_index() -> dict[str, dict]:
    """
    Genera el dataset RT, añade features derivadas y devuelve
    un dict match_key → {features} 
    El dict de cada trip tiene las mismas columnas que el dataset de entrenamiento.
    """
    df = build_realtime_dataset()
    df = _add_derived_features(df)
    df = df.dropna(subset=["match_key"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df.set_index("match_key").to_dict(orient="index")


def get_trip_features(index: dict[str, dict], trip_id: str) -> dict | None:
    """Devuelve el dict de features para el trip_id dado, o None si no existe."""
    return index.get(trip_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_realtime_lgbm.py <trip_id>")
        sys.exit(1)

    trip_id  = sys.argv[1]
    index    = build_index()
    features = get_trip_features(index, trip_id)

    if features is None:
        print(f"trip_id '{trip_id}' no encontrado.")
        sys.exit(1)

    for k, v in features.items():
        print(f"{k:35s} {v}")

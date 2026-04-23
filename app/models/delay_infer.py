"""LightGBM delay/end inference: per-stop absolute delay prediction."""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_delay_features
from app.models.registry import LGBMDelayEntry
from app.schemas import DelayPrediction, DelayResponse

logger = logging.getLogger(__name__)


def _apply_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Apply label encoding, target encoding and derived features from preprocessing JSON."""
    df = df.copy()

    # Label encoding for categorical columns
    for col, mapping in prep.get("label_encoders", {}).items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Target encoding for stop_id
    stop_enc = prep.get("target_encoder_stop_id", {})
    global_mean = prep.get("target_encoder_global_mean", 0.0)
    if stop_enc and "stop_id" in df.columns:
        df["stop_id_encoded"] = (
            df["stop_id"].astype(str).map(stop_enc).fillna(global_mean)
        )

    # Derived features
    for feat in prep.get("derived_features", []):
        if feat == "delay_velocity" and "delay_seconds_mean" in df.columns and "lagged_delay_1_mean" in df.columns:
            df["delay_velocity"] = df["delay_seconds_mean"] - df["lagged_delay_1_mean"]
        elif feat == "delay_acceleration" and "delay_seconds_mean" in df.columns:
            d1 = df.get("lagged_delay_1_mean", 0)
            d2 = df.get("lagged_delay_2_mean", 0)
            df["delay_acceleration"] = (df["delay_seconds_mean"] - d1) - (d1 - d2)
        elif feat == "delay_x_stops_remaining" and "delay_seconds_mean" in df.columns and "stops_to_end_mean" in df.columns:
            df["delay_x_stops_remaining"] = df["delay_seconds_mean"] * df["stops_to_end_mean"]
        elif feat == "delay_ratio" and "delay_seconds_mean" in df.columns and "scheduled_time_to_end_mean" in df.columns:
            df["delay_ratio"] = df["delay_seconds_mean"] / (df["scheduled_time_to_end_mean"] + 1)

    return df


def run_delays(
    entry: LGBMDelayEntry,
    windows: list,
    route_id_filter: Optional[str] = None,
    stop_id_filter: Optional[str] = None,
    min_delay_seconds: float = 0.0,
) -> DelayResponse:
    """Run LightGBM delay inference and return a DelayResponse."""
    df = windows_to_delay_features(windows)

    if route_id_filter and "route_id" in df.columns:
        df = df[df["route_id"].astype(str) == route_id_filter]
    if stop_id_filter and "stop_id" in df.columns:
        base = df["stop_id"].astype(str).str.rstrip("NS")
        df = df[(df["stop_id"].astype(str) == stop_id_filter) | (base == stop_id_filter)]

    if df.empty:
        return DelayResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            target=entry.preprocessing.get("target", "unknown"),
            n_stops=0,
            predictions=[],
        )

    # Keep metadata columns before encoding
    stop_ids = df["stop_id"].astype(str).tolist() if "stop_id" in df.columns else []
    route_ids = df["route_id"].astype(str).tolist() if "route_id" in df.columns else []
    directions = df["direction"].astype(str).tolist() if "direction" in df.columns else []

    df = _apply_preprocessing(df, entry.preprocessing)

    # Align to model's expected features
    model = entry.model
    try:
        feature_names = model.feature_name()
    except Exception:
        feature_names = []

    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_names]
    else:
        X = df.select_dtypes(include=[np.number])

    preds = model.predict(X)

    predictions: list[DelayPrediction] = []
    for i, pred in enumerate(preds):
        if pred < min_delay_seconds:
            continue
        predictions.append(DelayPrediction(
            stop_id=stop_ids[i] if i < len(stop_ids) else "?",
            route_id=route_ids[i] if i < len(route_ids) else "?",
            direction=directions[i] if i < len(directions) else "?",
            delay_seconds=float(np.clip(pred, 0, None)),
            delay_minutes=float(np.clip(pred, 0, None)) / 60.0,
        ))

    return DelayResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        target=entry.preprocessing.get("target", "unknown"),
        n_stops=len(predictions),
        predictions=predictions,
    )

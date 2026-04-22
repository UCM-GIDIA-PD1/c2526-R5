"""LightGBM delta inference: binary classification of delay improvement."""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_delay_features
from app.models.registry import DeltaEntry
from app.schemas import DeltaPrediction, DeltaResponse

logger = logging.getLogger(__name__)


def _apply_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Apply categorical vocab encoding from preprocessing JSON."""
    df = df.copy()
    vocabs: dict = prep.get("vocabs", {})
    for col, vocab in vocabs.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df


def run_delta(
    entry: DeltaEntry,
    windows: list,
    horizon: str,
    threshold: Optional[float] = None,
    route_id_filter: Optional[str] = None,
    stop_id_filter: Optional[str] = None,
) -> DeltaResponse:
    """Run LightGBM delta inference and return a DeltaResponse."""
    prep = entry.preprocessing
    thr = threshold if threshold is not None else float(prep.get("best_threshold", 0.5))

    df = windows_to_delay_features(windows)

    if route_id_filter and "route_id" in df.columns:
        df = df[df["route_id"].astype(str) == route_id_filter]
    if stop_id_filter and "stop_id" in df.columns:
        df = df[df["stop_id"].astype(str) == stop_id_filter]

    if df.empty:
        return DeltaResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            horizon=horizon,
            threshold_used=thr,
            n_stops=0,
            predictions=[],
        )

    stop_ids = df["stop_id"].astype(str).tolist() if "stop_id" in df.columns else []
    route_ids = df["route_id"].astype(str).tolist() if "route_id" in df.columns else []
    directions = df["direction"].astype(str).tolist() if "direction" in df.columns else []

    df = _apply_preprocessing(df, prep)

    model = entry.model
    feature_list: list[str] = prep.get("features", [])
    try:
        feature_list = model.feature_name() or feature_list
    except Exception:
        pass

    if feature_list:
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_list]
    else:
        X = df.select_dtypes(include=[np.number])

    X = X.fillna(0)
    probs = model.predict(X)

    predictions: list[DeltaPrediction] = []
    for i, prob in enumerate(probs):
        predictions.append(DeltaPrediction(
            stop_id=stop_ids[i] if i < len(stop_ids) else "?",
            route_id=route_ids[i] if i < len(route_ids) else "?",
            direction=directions[i] if i < len(directions) else "?",
            mejora_prob=float(prob),
            mejora_predicted=bool(prob >= thr),
        ))

    return DeltaResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        threshold_used=thr,
        n_stops=len(predictions),
        predictions=predictions,
    )

"""XGBoost alert inference: per-line alert probability."""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_alertas_features
from app.models.registry import AlertEntry
from app.schemas import AlertPrediction, AlertResponse

logger = logging.getLogger(__name__)


def run_alerts(
    entry: AlertEntry,
    windows: list,
    threshold: Optional[float] = None,
    route_id_filter: Optional[str] = None,
    min_prob: float = 0.0,
) -> AlertResponse:
    """Run XGBoost alert inference and return an AlertResponse."""
    thr = threshold if threshold is not None else entry.threshold

    df_linea = windows_to_alertas_features(windows)

    if df_linea.empty:
        return AlertResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            threshold_used=thr,
            n_lines=0,
            predictions=[],
        )

    if route_id_filter:
        df_linea = df_linea[df_linea["route_id"].astype(str) == route_id_filter]

    model = entry.model
    feat_attr = getattr(model, "feature_names_in_", None)
    known_features = list(feat_attr) if feat_attr is not None else []
    if not known_features:
        try:
            known_features = model.get_booster().feature_names or []
        except Exception:
            known_features = []

    df_feat = df_linea.copy()
    # OrdinalEncoder used during training → encode categoricals as integer codes
    for col in ("route_id", "direction"):
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("category").cat.codes

    if known_features:
        for col in known_features:
            if col not in df_feat.columns:
                df_feat[col] = 0
        X = df_feat[known_features]
    else:
        X = df_feat.select_dtypes(include=[np.number])

    X = X.fillna(0)
    probs = model.predict_proba(X)[:, 1]

    predictions: list[AlertPrediction] = []
    for i, prob in enumerate(probs):
        if prob < min_prob:
            continue
        route_id = str(df_linea["route_id"].iloc[i]) if "route_id" in df_linea.columns else "?"
        direction = str(df_linea["direction"].iloc[i]) if "direction" in df_linea.columns else "?"
        pct = float(df_linea["pct_paradas_retrasadas"].iloc[i]) if "pct_paradas_retrasadas" in df_linea.columns else None
        delay_mean = float(df_linea["delay_mean_linea"].iloc[i]) if "delay_mean_linea" in df_linea.columns else None

        predictions.append(AlertPrediction(
            route_id=route_id,
            direction=direction,
            alert_probability=float(prob),
            alert_predicted=bool(prob >= thr),
            pct_stops_delayed=pct,
            delay_mean_seconds=delay_mean,
        ))

    return AlertResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        threshold_used=thr,
        n_lines=len(predictions),
        predictions=predictions,
    )

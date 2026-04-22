"""DCRNN inference: spatiotemporal delay propagation per station."""
import logging
from datetime import datetime, timezone
from typing import Optional

import torch

from app.data.transforms import windows_to_dcrnn_tensor
from app.models.registry import DCRNNEntry
from app.schemas import PropagationPrediction, PropagationResponse

logger = logging.getLogger(__name__)


def run_propagation(
    entry: DCRNNEntry,
    windows: list,
    stations_meta: Optional[dict] = None,
    stop_id_filter: Optional[str] = None,
    route_id_filter: Optional[str] = None,
) -> PropagationResponse:
    """Run DCRNN inference and return a PropagationResponse.

    Args:
        entry:          Loaded DCRNNEntry from the registry.
        windows:        List of 15-min DataFrames (oldest → newest).
        stations_meta:  Optional dict {stop_id: {lat, lon}} for coordinates.
        stop_id_filter: If set, return only this station.
        route_id_filter: If set, return only stations served by this route.
    """
    X = windows_to_dcrnn_tensor(
        windows=windows,
        nodes=entry.nodes,
        feature_set=entry.feature_set,
        scaler_X=entry.scaler_X,
        history_len=entry.history_len,
    )  # (1, history_len, N, n_features)

    with torch.no_grad():
        y_hat = entry.model(X, entry.edge_index, entry.edge_weight)
    # y_hat: (1, 1, N, 3) → (N, 3)
    y_scaled = y_hat.squeeze(0).squeeze(0).cpu().numpy()

    # Inverse-transform predictions to seconds
    import numpy as np
    N, H = y_scaled.shape
    y_sec = entry.scaler_Y.inverse_transform(y_scaled.reshape(-1, H)).reshape(N, H)

    nodes_sorted = sorted(entry.nodes)
    predictions: list[PropagationPrediction] = []

    for i, stop_id in enumerate(nodes_sorted):
        if stop_id_filter and stop_id != stop_id_filter:
            continue
        if route_id_filter and stations_meta:
            meta = stations_meta.get(stop_id, {})
            routes = str(meta.get("routes", ""))
            if route_id_filter not in routes:
                continue

        meta = (stations_meta or {}).get(stop_id, {})
        predictions.append(
            PropagationPrediction(
                stop_id=stop_id,
                lat=meta.get("lat"),
                lon=meta.get("lon"),
                delay_10m=float(np.clip(y_sec[i, 0], 0, None)),
                delay_20m=float(np.clip(y_sec[i, 1], 0, None)),
                delay_30m=float(np.clip(y_sec[i, 2], 0, None)),
            )
        )

    return PropagationResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        n_stations=len(predictions),
        predictions=predictions,
    )

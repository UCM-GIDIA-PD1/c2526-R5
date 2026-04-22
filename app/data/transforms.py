"""
Transforms raw 15-min Drive windows (time_aggregations.py format) into
the input tensors / DataFrames expected by each model.

Column naming convention from time_aggregations.py:
  - GroupBy keys  : stop_id, route_id, direction, merge_time  (no suffix)
  - Single agg    : col_<agg>   e.g. is_unscheduled_max, hour_sin_first
  - Multi agg     : col_mean, col_max
"""
import gc
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ── DCRNN ─────────────────────────────────────────────────────────────────────

ALL_FEATURE_COLS = [
    "delay_seconds",
    "lagged_delay_1",
    "lagged_delay_2",
    "is_unscheduled",
    "temp_extreme",
    "n_eventos_afectando",
    "route_rolling_delay",
    "actual_headway_seconds",
    "hour_sin",
    "hour_cos",
    "dow",
    "afecta_previo",
    "afecta_durante",
    "afecta_despues",
]

# Map from DCRNN logical name → aggregated column name in Drive parquets
_DCRNN_COL_MAP: dict[str, str] = {
    "delay_seconds":          "delay_seconds_mean",
    "lagged_delay_1":         "lagged_delay_1_mean",
    "lagged_delay_2":         "lagged_delay_2_mean",
    "is_unscheduled":         "is_unscheduled_max",
    "temp_extreme":           "temp_extreme_max",
    "n_eventos_afectando":    "n_eventos_afectando_max",
    "route_rolling_delay":    "route_rolling_delay_mean",
    "actual_headway_seconds": "actual_headway_seconds_mean",
    "hour_sin":               "hour_sin_first",
    "hour_cos":               "hour_cos_first",
    "dow":                    "dow_first",
    "afecta_previo":          "afecta_previo_max",
    "afecta_durante":         "afecta_durante_max",
    "afecta_despues":         "afecta_despues_max",
}


def windows_to_dcrnn_tensor(
    windows: list[pd.DataFrame],
    nodes: list[str],
    feature_set: list[int],
    scaler_X,
    history_len: int,
) -> torch.Tensor:
    """Build a (1, history_len, N, n_features) tensor from the Drive windows.

    Args:
        windows:     List of DataFrames, oldest → newest, one per 15-min bin.
        nodes:       Ordered list of stop_id strings matching the DCRNN graph.
        feature_set: Indices into ALL_FEATURE_COLS selected during HPO.
        scaler_X:    StandardScaler fitted on training data (all 14 features).
        history_len: Number of past time steps the model expects.

    Returns:
        FloatTensor of shape (1, history_len, N, len(feature_set)).
    """
    N = len(nodes)
    F_all = len(ALL_FEATURE_COLS)
    node_set = set(nodes)

    df = pd.concat(windows, ignore_index=True)
    df["merge_time"] = pd.to_datetime(df["merge_time"])

    # Rename aggregated columns to DCRNN logical names
    rename = {v: k for k, v in _DCRNN_COL_MAP.items() if v in df.columns}
    df = df.rename(columns=rename)

    # Re-derive temporal features from the bin time (more reliable than _first)
    df["time_bin"] = df["merge_time"].dt.floor("15min")
    df["hour_sin"] = np.sin(2 * np.pi * df["time_bin"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_bin"].dt.hour / 24)
    df["dow"] = df["time_bin"].dt.dayofweek.astype(float)

    # Filter to known graph nodes and aggregate to (time_bin, stop_id)
    df = df[df["stop_id"].isin(node_set)]
    agg_rules = {f: "mean" for f in ALL_FEATURE_COLS if f in df.columns}
    df_st = df.groupby(["time_bin", "stop_id"]).agg(agg_rules).reset_index()
    del df
    gc.collect()

    # Build full (T × N) grid with zero-fill for missing entries
    time_bins = sorted(df_st["time_bin"].unique())
    T = len(time_bins)
    nodes_sorted = sorted(nodes)
    full_idx = pd.MultiIndex.from_product(
        [time_bins, nodes_sorted], names=["time_bin", "stop_id"]
    )
    df_full = (
        df_st.set_index(["time_bin", "stop_id"])
        .reindex(full_idx)
        .reset_index()
        .fillna(0)
        .sort_values(["time_bin", "stop_id"])
    )
    del df_st
    gc.collect()

    # Build (T, N, F_all) array
    X = np.zeros((T, N, F_all), dtype=np.float32)
    for fi, feat in enumerate(ALL_FEATURE_COLS):
        if feat in df_full.columns:
            X[:, :, fi] = df_full[feat].values.reshape(T, N)
    del df_full
    gc.collect()

    # Scale using training scaler
    X_scaled = scaler_X.transform(X.reshape(-1, F_all)).reshape(T, N, F_all).astype(np.float32)

    # Select HPO-chosen feature subset
    X_sel = X_scaled[:, :, feature_set]  # (T, N, n_sel)

    # Pad or trim to history_len
    n_sel = len(feature_set)
    if T < history_len:
        pad = np.zeros((history_len - T, N, n_sel), dtype=np.float32)
        X_sel = np.concatenate([pad, X_sel], axis=0)
    else:
        X_sel = X_sel[-history_len:]

    return torch.from_numpy(X_sel).unsqueeze(0)  # (1, history_len, N, n_sel)


# ── XGBoost delay ─────────────────────────────────────────────────────────────

_DELAY_EXCLUDE = {
    "target_delay_10m_mean", "target_delay_10m_max",
    "target_delay_20m_mean", "target_delay_20m_max",
    "target_delay_30m_mean", "target_delay_30m_max",
    "target_delay_45m_mean", "target_delay_45m_max",
    "target_delay_60m_mean", "target_delay_60m_max",
    "target_delay_end_mean", "target_delay_end_max",
    "delta_delay_10m_mean", "delta_delay_10m_max",
    "delta_delay_20m_mean", "delta_delay_20m_max",
    "delta_delay_30m_mean", "delta_delay_30m_max",
    "delta_delay_45m_mean", "delta_delay_45m_max",
    "delta_delay_60m_mean", "delta_delay_60m_max",
    "delta_delay_end_mean", "delta_delay_end_max",
    "station_delay_10m_mean", "station_delay_10m_max",
    "station_delay_20m_mean", "station_delay_20m_max",
    "station_delay_30m_mean", "station_delay_30m_max",
    "alert_in_next_15m_max", "alert_in_next_30m_max",
    "seconds_to_next_alert_mean", "afecta_despues_max",
    "match_key_nunique",
}


def windows_to_delay_features(windows: list[pd.DataFrame]) -> pd.DataFrame:
    """Prepare the most-recent window for XGBoost delay inference.

    The delay model was trained on time_aggregations.py output (60-min bins).
    We use the latest 15-min window as a proxy; column names already match
    since both use the same aggregation format.

    Returns a DataFrame ready for model.predict(), with the same feature names
    the model was trained on (the model's feature_names_in_ is used to select).
    """
    df = windows[-1].copy()
    df["merge_time"] = pd.to_datetime(df["merge_time"])

    # Temporal features added by procesar() in the training script
    df["hora"] = df["merge_time"].dt.hour
    df["minuto"] = df["merge_time"].dt.minute
    df["dia_semana"] = df["merge_time"].dt.dayofweek
    df["hora_mean"] = (
        pd.to_datetime(df["merge_time_mean"]).dt.hour
        if "merge_time_mean" in df.columns
        else df["hora"]
    )

    # Cast categorical columns
    for col in ("stop_id", "route_id", "direction"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    drop = [c for c in ("merge_time", "merge_time_mean") if c in df.columns]
    df = df.drop(columns=drop)

    # Remove training-time-only columns
    drop_exc = [c for c in _DELAY_EXCLUDE if c in df.columns]
    df = df.drop(columns=drop_exc)

    return df


# ── XGBoost alertas ───────────────────────────────────────────────────────────


def windows_to_alertas_features(windows: list[pd.DataFrame]) -> pd.DataFrame:
    """Transform 8 × 15-min windows into per-(route_id, direction) alert features.

    Strategy:
    1. Concat all windows and compute per-stop lag columns (30/60/90 min back).
    2. Add a dummy target column (inference: not available).
    3. Delegate to pipeline_linea.agregar_por_linea() which groups to 30-min × line.
    4. Compute rolling-4 features via agregar_features_rolling_retraso().
    5. Return the most-recent row per (route_id, direction).
    """
    from src.models.modelos_alertas.common.pipeline_linea import (
        agregar_por_linea,
        agregar_features_rolling_retraso,
    )

    df = pd.concat(windows, ignore_index=True)
    df["merge_time"] = pd.to_datetime(df["merge_time"])
    df = df.sort_values(["stop_id", "route_id", "direction", "merge_time"]).reset_index(drop=True)

    # Per-stop lags in seconds. The alert model was trained on 30-min aggregated
    # data where lag(1) = 30 min. With 15-min bins that means shift(2).
    grp = df.groupby(["stop_id", "route_id", "direction"])
    df["delay_1_before"] = grp["delay_seconds_mean"].shift(2).fillna(0)
    df["delay_2_before"] = grp["delay_seconds_mean"].shift(4).fillna(0)
    df["delay_3_before"] = grp["delay_seconds_mean"].shift(6).fillna(0)

    # seconds_since_last_alert: use column if present, else fill unknown
    if "seconds_since_last_alert_mean" not in df.columns:
        df["seconds_since_last_alert_mean"] = 999_999.0

    # Dummy target so agregar_por_linea doesn't drop rows
    df["alert_in_next_15m_max"] = 0

    # Run pipeline: groups to 30-min × line internally
    df_linea = agregar_por_linea(df)
    df_linea = agregar_features_rolling_retraso(df_linea)

    # Keep only the most-recent 30-min bin per line
    df_linea = df_linea.sort_values("merge_time")
    df_latest = df_linea.groupby(["route_id", "direction"]).last().reset_index()

    return df_latest

"""
Per-train feature extraction from GTFS-RT trip_update entities.

Given a trip_id + route_id + current stop_id, downloads the relevant GTFS-RT
feed, locates the trip_update, and constructs a single-row DataFrame whose
column names match the Drive-window aggregated format.  This lets us reuse
the existing run_delays() / run_delta() inference functions directly.
"""
import logging
import math
from datetime import datetime, timezone

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)


class FeedUnavailable(Exception):
    """Raised when the GTFS-RT feed request fails (timeout, HTTP error, etc.)."""


class TripNotFound(Exception):
    """Raised when the trip_id is not present in the feed."""

# One feed URL per route (same endpoints used by vehicles.py)
_FEED_FOR_ROUTE: dict[str, str] = {
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
       for r in ("A", "C", "E")},
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm"
       for r in ("B", "D", "F", "M")},
    "G": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz"
       for r in ("J", "Z")},
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw"
       for r in ("N", "Q", "R", "W")},
    "L": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs"
       for r in ("1", "2", "3", "4", "5", "6", "7", "S", "GS", "FS", "H")},
    "SIR": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
}


def _su_delay(su) -> float | None:
    """Extract delay in seconds from a StopTimeUpdate protobuf, or None."""
    try:
        if su.HasField("arrival") and su.arrival.delay != 0:
            return float(su.arrival.delay)
    except Exception:
        pass
    try:
        if su.HasField("departure") and su.departure.delay != 0:
            return float(su.departure.delay)
    except Exception:
        pass
    return None


def _normalize_route(rid: str) -> str:
    return rid.strip().split("-")[0].split("_")[0]


def fetch_train_features(
    trip_id: str,
    route_id: str,
    stop_id: str,
    windows: list,
) -> pd.DataFrame | None:
    """
    Builds a single-row DataFrame in Drive-window aggregated format for the
    given trip.  The returned DataFrame is compatible with the existing
    windows_to_delay_features() → run_delays() / run_delta() pipeline:
    pass it as ``fake_windows = [returned_df]``.

    Raises:
        FeedUnavailable  – network / HTTP error fetching the GTFS-RT feed.
        TripNotFound     – feed responded OK but trip_id is not present.
    """
    route_norm = _normalize_route(route_id)
    feed_url = _FEED_FOR_ROUTE.get(route_norm)
    if feed_url is None:
        raise FeedUnavailable(f"No GTFS-RT feed URL for route '{route_norm}'")

    # ── 1. Download the relevant GTFS-RT feed ────────────────────────────────
    try:
        resp = requests.get(feed_url, timeout=10)
        resp.raise_for_status()
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(resp.content)
    except Exception as exc:
        raise FeedUnavailable(f"Feed for route {route_norm} unavailable: {exc}") from exc

    # ── 2. Locate the trip_update entity for this trip_id ────────────────────
    trip_update = None
    for entity in msg.entity:
        if entity.HasField("trip_update") and entity.trip_update.trip.trip_id == trip_id:
            trip_update = entity.trip_update
            break

    if trip_update is None:
        raise TripNotFound(f"trip_id '{trip_id}' not found in feed for route {route_norm}")

    stops = list(trip_update.stop_time_update)
    if not stops:
        raise TripNotFound(f"trip_id '{trip_id}' has no stop_time_updates")

    # ── 3. Find current stop position inside the update list ─────────────────
    current_idx = next(
        (i for i, s in enumerate(stops) if s.stop_id == stop_id),
        len(stops) - 1,   # fall back to last known stop
    )
    current_su = stops[current_idx]
    future_stops = stops[current_idx + 1:]

    # ── 4. Derive delay features from trip_update ─────────────────────────────
    delay_now   = _su_delay(current_su) or 0.0
    prev1_delay = _su_delay(stops[current_idx - 1]) if current_idx >= 1 else 0.0
    prev2_delay = _su_delay(stops[current_idx - 2]) if current_idx >= 2 else 0.0

    stops_to_end = len(future_stops)

    # scheduled_time_to_end = (last_scheduled_arrival) − (current_scheduled_arrival)
    # scheduled = actual_time − delay
    scheduled_time_to_end = 0.0
    try:
        last_su = future_stops[-1] if future_stops else current_su
        curr_t = current_su.arrival.time if current_su.HasField("arrival") else 0
        last_t = last_su.arrival.time    if last_su.HasField("arrival")    else 0
        last_d = _su_delay(last_su) or delay_now
        if curr_t > 0 and last_t > 0:
            scheduled_time_to_end = float((last_t - last_d) - (curr_t - delay_now))
    except Exception:
        pass

    # ── 5. Temporal & meta features ───────────────────────────────────────────
    direction    = "N" if stop_id.endswith("N") else "S"
    is_unscheduled = int(trip_update.trip.schedule_relationship != 0)

    now   = datetime.now(timezone.utc)
    hour  = now.hour
    dow   = float(now.weekday())

    # ── 6. Build row in Drive-window aggregated column format ─────────────────
    row: dict = {
        # Groupby keys (no suffix)
        "stop_id":    stop_id,
        "route_id":   route_norm,
        "direction":  direction,
        "merge_time": now,          # consumed by windows_to_delay_features()

        # Core delay (aggregated-style names the model expects)
        "delay_seconds_mean":  float(max(0.0, delay_now)),
        "delay_seconds_max":   float(max(0.0, delay_now)),
        "lagged_delay_1_mean": float(max(0.0, prev1_delay or 0.0)),
        "lagged_delay_1_max":  float(max(0.0, prev1_delay or 0.0)),
        "lagged_delay_2_mean": float(max(0.0, prev2_delay or 0.0)),
        "lagged_delay_2_max":  float(max(0.0, prev2_delay or 0.0)),

        # Trip progress
        "stops_to_end_mean":          float(stops_to_end),
        "scheduled_time_to_end_mean": float(max(0.0, scheduled_time_to_end)),

        # Temporal
        "hour_sin_first": math.sin(2 * math.pi * hour / 24),
        "hour_cos_first": math.cos(2 * math.pi * hour / 24),
        "dow_first":      dow,
        "is_weekend_max": float(1 if now.weekday() >= 5 else 0),

        # Categorical flags
        "is_unscheduled_max": float(is_unscheduled),

        # Route-level defaults (overridden below from Drive windows)
        "route_rolling_delay_mean":     float(max(0.0, delay_now)),
        "actual_headway_seconds_mean":  0.0,
        "n_eventos_afectando_max":      0.0,
        "temp_extreme_max":             0.0,
        "afecta_previo_max":            0.0,
        "afecta_durante_max":           0.0,
        "afecta_despues_max":           0.0,
        "seconds_since_last_alert_mean": 999_999.0,
    }

    # ── 7. Enrich route-level context from the latest Drive window ────────────
    if windows:
        try:
            df_w = windows[-1]
            mask = df_w["route_id"].astype(str) == route_norm
            if "direction" in df_w.columns:
                mask &= df_w["direction"].astype(str) == direction
            sub = df_w[mask]
            if not sub.empty:
                for col in (
                    "route_rolling_delay_mean",
                    "actual_headway_seconds_mean",
                    "n_eventos_afectando_max",
                    "temp_extreme_max",
                    "afecta_previo_max",
                    "afecta_durante_max",
                    "afecta_despues_max",
                    "seconds_since_last_alert_mean",
                ):
                    if col in sub.columns:
                        val = sub[col].mean()
                        import math as _m
                        if not _m.isnan(val):
                            row[col] = float(val)
        except Exception as exc:
            logger.debug("Could not enrich from Drive windows: %s", exc)

    return pd.DataFrame([row])

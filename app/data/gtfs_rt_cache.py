"""Snapshot compartido de feeds GTFS-RT MTA.

Garantiza que /api/vehicles (entity.vehicle) y /predict/train (entity.trip_update)
operen sobre el MISMO snapshot mientras esté fresco, eliminando el desfase
temporal que producía 404 al pedir trenes recién aparecidos en el mapa.
"""
import logging
import threading
import time

import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30

_lock = threading.Lock()
_cache: dict[str, tuple[float, "gtfs_realtime_pb2.FeedMessage"]] = {}


def get_feed_message(
    url: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    timeout: int = 10,
) -> "gtfs_realtime_pb2.FeedMessage | None":
    """Devuelve el FeedMessage parseado para `url`, descargando si no hay snapshot fresco.

    Thread-safe. Si la descarga falla y existe un snapshot previo, se devuelve
    el previo (mejor que None) y se loguea el fallo. Si nunca se descargó, devuelve None.
    """
    now = time.time()
    with _lock:
        cached = _cache.get(url)
        if cached and (now - cached[0]) < ttl_seconds:
            return cached[1]

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(resp.content)
    except Exception as exc:
        logger.warning("Feed %s no disponible: %s", url, exc)
        with _lock:
            cached = _cache.get(url)
        return cached[1] if cached else None

    with _lock:
        _cache[url] = (now, msg)
    return msg

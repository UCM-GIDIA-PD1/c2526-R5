"""
Extrae la posición actual de todos los trenes del metro de Nueva York
a partir del feed GTFS-RT de la MTA.

La API no publica coordenadas GPS reales (latitude/longitude = 0.0).
En su lugar, cada entidad vehicle incluye el stop_id de la parada en la
que está o hacia la que se dirige. Las coordenadas se obtienen cruzando
ese stop_id con stops.txt del GTFS suplementado.

OUTPUT:
    DataFrame con una fila por tren en circulación:
        viaje_id, linea_id, parada_actual_id, parada_actual_nombre,
        lat, lon, estado, timestamp
"""

import io
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2


FUENTES = {
    "ACES":     "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
    "BDFMS":    "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
    "G":        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
    "JZ":       "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz",
    "NQRW":     "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
    "L":        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
    "1234567S": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
    "SIR":      "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
}

# current_status codes del protobuf
ESTADO = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}

GTFS_SUPPLEMENTED_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"


def _descargar_stops() -> pd.DataFrame:
    """Descarga stops.txt del GTFS suplementado y devuelve stop_id → lat/lon/nombre."""
    with urllib.request.urlopen(GTFS_SUPPLEMENTED_URL) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    with z.open("stops.txt") as f:
        stops = pd.read_csv(f, usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops = stops.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
    return stops.set_index("stop_id")


def _extraer_vehicles(url: str, reintentos: int = 3) -> list[dict]:
    """Extrae entidades vehicle de un feed GTFS-RT."""
    for intento in range(reintentos):
        try:
            resp = requests.get(url, timeout=10)
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(resp.content)

            rows = []
            for entity in feed.entity:
                if not entity.HasField("vehicle"):
                    continue
                v = entity.vehicle
                if not v.trip.trip_id:
                    continue
                rows.append({
                    "viaje_id":        v.trip.trip_id,
                    "linea_id":        v.trip.route_id,
                    "parada_actual_id": v.stop_id,
                    "stop_sequence":   v.current_stop_sequence,
                    "estado":          ESTADO.get(v.current_status, str(v.current_status)),
                    "timestamp":       datetime.fromtimestamp(v.timestamp, tz=timezone.utc)
                                       if v.timestamp else datetime.now(tz=timezone.utc),
                })
            return rows

        except Exception as e:
            if intento == reintentos - 1:
                print(f"  [ERROR] {url}: {e}")
                return []
            time.sleep(2 ** intento)
    return []


def obtener_posiciones() -> pd.DataFrame:
    """
    Devuelve un DataFrame con la posición actual de todos los trenes en circulación.

    Columnas:
        viaje_id          — trip_id del tren
        linea_id          — route_id (línea)
        parada_actual_id  — stop_id donde está o al que se dirige
        parada_actual_nombre — nombre legible de la parada
        lat, lon          — coordenadas de esa parada
        stop_sequence     — número de secuencia de la parada actual
        estado            — STOPPED_AT | IN_TRANSIT_TO | INCOMING_AT
        timestamp         — momento de la última actualización del vehículo
    """
    print("Descargando stops.txt del GTFS suplementado...")
    stops = _descargar_stops()
    print(f"  {len(stops)} paradas cargadas")

    print("Extrayendo posiciones de vehículos...")
    filas = []
    for nombre, url in FUENTES.items():
        vehicles = _extraer_vehicles(url)
        print(f"  {nombre:12} → {len(vehicles)} vehículos")
        filas.extend(vehicles)

    if not filas:
        print("[WARN] No se obtuvieron posiciones de vehículos.")
        return pd.DataFrame()

    df = pd.DataFrame(filas)
    df["parada_actual_id"] = df["parada_actual_id"].astype(str)

    # Cruzar con coordenadas
    df = df.join(stops, on="parada_actual_id", how="left")
    df = df.rename(columns={"stop_name": "parada_actual_nombre"})

    # Filtrar trenes sin parada conocida (no deberían existir, pero por robustez)
    sin_coords = df["lat"].isna().sum()
    if sin_coords:
        print(f"  [WARN] {sin_coords} vehículos sin coordenadas (stop_id desconocido)")

    df = df[[
        "viaje_id", "linea_id",
        "parada_actual_id", "parada_actual_nombre",
        "lat", "lon",
        "stop_sequence", "estado", "timestamp",
    ]].reset_index(drop=True)

    print(f"\nTotal: {len(df)} trenes | {df['linea_id'].nunique()} líneas")
    return df


if __name__ == "__main__":
    df = obtener_posiciones()
    print()
    print(df.to_string())

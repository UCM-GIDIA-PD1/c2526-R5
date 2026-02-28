"""
eventos_nyc.py — Ingesta de eventos públicos de NYC desde NYC Open Data.

Fuente  : data.cityofnewyork.us  (dataset bkfu-528j)
Destino : MinIO  grupo5/raw/eventos_nyc/date=YYYY-MM-DD/eventos_YYYY-MM-DD.parquet
"""

import os

import numpy as np
import pandas as pd
import requests
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from .utils_eventos import (
    cargar_paradas_df,
    fusionar_lista_estaciones,
    obtener_paradas_afectadas,
)
from src.common.minio_client import upload_df_parquet, DEFAULT_BUCKET

#  Constantes


URL_EVENTOS   = "https://data.cityofnewyork.us/resource/bkfu-528j.json"
RADIO_METRO_M = 500

RIESGO_MAP = {
    "Parade":                       10,
    "Athletic Race / Tour":         10,
    "Street Event":                  8,
    "Stationary Demonstration":      7,
    "Street Festival":               7,
    "Special Event":                 7,
    "Single Block Festival":         6,
    "Bike the Block":                6,
    "BID Multi-Block":               6,
    "Plaza Event":                   6,
    "Plaza Partner Event":           6,
    "Block Party":                   5,
    "Theater Load in and Load Outs": 5,
    "Open Culture":                  4,
    "Religious Event":               3,
    "Press Conference":              3,
    "Health Fair":                   3,
    "Rigging Permit":                3,
    "Farmers Market":                2,
    "Sidewalk Sale":                 2,
    "Shooting Permit":               2,
    "Filming/Photography":           2,
    "Open Street Partner Event":     2,
    "Production Event":              1,
    "Sport - Adult":                 1,
    "Sport - Youth":                 1,
    "Miscellaneous":                 1,
    "Stickball":                     1,
    "Clean-Up":                      1,
}

RIESGO_MINIMO = 8

SCORE_MAP = {8: 0.8, 9: 0.9, 10: 1.0}


# ─────────────────────────────────────────────────────────────────
#  Extracción desde NYC Open Data
# ─────────────────────────────────────────────────────────────────

def _fmt_inicio(fecha):
    return f"{fecha}T00:00:00.000"

def _fmt_fin(fecha):
    return f"{fecha}T23:59:59.000"


def descargar_eventos(start_date, end_date, token):
    """Descarga los eventos del rango [start_date, end_date] desde NYC Open Data."""
    limit  = 100000
    offset = 0
    chunks = []

    while True:
        params = {
            "$where":  f"start_date_time >= '{_fmt_inicio(start_date)}' AND start_date_time <= '{_fmt_fin(end_date)}'",
            "$limit":  limit,
            "$offset": offset,
        }
        r = requests.get(URL_EVENTOS, params=params, headers={"X-App-Token": token}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}\n{r.text[:2000]}")

        data = r.json()
        if not data:
            break

        chunks.append(pd.DataFrame(data))
        offset += limit
        print(f"  Descargadas ~{offset} filas...")

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()



#  Geocodificación


def _extraer_intersecciones(localizacion, barrio):
    intersecciones = []
    for segmento in localizacion.split(","):
        segmento = segmento.strip()
        if " between " in segmento:
            partes = segmento.split(" between ")
            calle_principal = partes[0].strip()
            for cruce in partes[1].split(" and "):
                cruce = cruce.strip()
                if cruce:
                    intersecciones.append(f"{calle_principal} & {cruce}, {barrio}, New York")
    return intersecciones or [f"{localizacion}, {barrio}, New York"]


def _extraer_coord(localizacion, barrio, geocode_fn):
    """Devuelve (longitud, latitud) para la ubicación del evento. (0, 0) si falla."""
    if pd.isna(localizacion):
        return 0.0, 0.0

    if ":" in localizacion:
        res = geocode_fn(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        return (res.longitude, res.latitude) if res else (0.0, 0.0)

    coords = []
    for interseccion in _extraer_intersecciones(localizacion, barrio):
        try:
            res = geocode_fn(interseccion)
            if res:
                coords.append((res.latitude, res.longitude))
        except Exception:
            continue

    if coords:
        return float(np.mean([c[1] for c in coords])), float(np.mean([c[0] for c in coords]))

    return 0.0, 0.0



#  Extracción pública

def extraer_eventos_nyc(start_date, end_date, df_paradas=None):
    """
    Descarga, filtra y enriquece los eventos de NYC Open Data.
    Devuelve un DataFrame listo para subir.
    """
    token = os.environ.get("NYC_OPEN_DATA_TOKEN")
    if not token:
        raise ValueError("Falta la variable de entorno NYC_OPEN_DATA_TOKEN")

    print(f"[eventos_nyc] Descargando eventos {start_date} - {end_date}...")
    df = descargar_eventos(start_date, end_date, token)

    if df.empty:
        return df

    df["borough"] = df["event_borough"]
    df = df[["event_name", "event_type", "start_date_time", "end_date_time",
             "event_location", "borough", "community_board"]].copy()
    df["start_date_time"] = pd.to_datetime(df["start_date_time"])
    df["end_date_time"]   = pd.to_datetime(df["end_date_time"])
    df = df.dropna(subset=["event_type"])

    df["nivel_riesgo_tipo"] = df["event_type"].map(RIESGO_MAP)
    df = df[df["nivel_riesgo_tipo"] >= RIESGO_MINIMO]
    df = df.drop_duplicates(subset=["event_name", "start_date_time", "borough", "event_location"])
    df["score"] = df["nivel_riesgo_tipo"].map(SCORE_MAP)

    print(f"[eventos_nyc] Geocodificando {len(df)} eventos...")
    geolocator = Nominatim(user_agent="pd1_eventos_nyc", timeout=10)
    geocode_fn = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1.2,
        max_retries=5,
        error_wait_seconds=2,
        swallow_exceptions=True,
        return_value_on_exception=None,
    )

    total = len(df)
    paso  = max(1, total // 10)
    lons, lats = [], []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        lon, lat = _extraer_coord(row["event_location"], row["borough"], geocode_fn)
        lons.append(lon)
        lats.append(lat)
        if i % paso == 0 or i == total:
            pct = min(100, (int(round(i * 100 / total)) // 10) * 10)
            print(f"  Coordenadas: {pct}% ({i}/{total})")

    df["lon"] = pd.to_numeric(lons, errors="coerce")
    df["lat"] = pd.to_numeric(lats, errors="coerce")
    df.loc[(df["lon"] == 0) & (df["lat"] == 0), ["lon", "lat"]] = np.nan

    if df_paradas is not None:
        print(f"[eventos_nyc] Calculando paradas afectadas (radio {RADIO_METRO_M} m)...")
        df["paradas_afectadas"] = df.apply(
            lambda r: fusionar_lista_estaciones(
                obtener_paradas_afectadas((float(r["lon"]), float(r["lat"])), df_paradas, max_metros=RADIO_METRO_M)
            ) if pd.notna(r["lon"]) and pd.notna(r["lat"]) else [],
            axis=1,
        )
    else:
        df["paradas_afectadas"] = [[] for _ in range(len(df))]

    df["hora_inicio"]          = df["start_date_time"].dt.strftime("%H:%M")
    df["hora_salida_estimada"] = df["end_date_time"].dt.strftime("%H:%M")
    df["fecha_inicio"]         = df["start_date_time"].dt.strftime("%Y-%m-%d")
    df["fecha_final"]          = df["end_date_time"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"event_name": "nombre_evento"})

    return df[["nombre_evento", "fecha_inicio", "hora_inicio", "fecha_final",
               "hora_salida_estimada", "score", "paradas_afectadas"]].reset_index(drop=True)



#  Ingesta completa

def ingest_eventos_nyc(start_date, end_date):
    """Punto de entrada para el orquestador."""
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    print("[eventos_nyc] Cargando paradas de metro...")
    df_paradas = cargar_paradas_df()

    df = extraer_eventos_nyc(start_date, end_date, df_paradas=df_paradas)

    if df.empty:
        print("[eventos_nyc] Sin eventos en el rango. Nada que subir.")
        return

    print("[eventos_nyc] Subiendo parquets a MinIO...")
    subidos = 0
    for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
        df_dia = df_dia.reset_index(drop=True)
        obj = f"grupo5/raw/eventos_nyc/date={fecha}/eventos_{fecha}.parquet"
        try:
            upload_df_parquet(access_key, secret_key, obj, df_dia)
            print(f"  Subido: {DEFAULT_BUCKET}/{obj} ({len(df_dia)} filas)")
            subidos += 1
        except Exception as exc:
            print(f"  Error subiendo {obj}: {exc}")

    print(f"[eventos_nyc] Terminado. {subidos} archivos subidos.")
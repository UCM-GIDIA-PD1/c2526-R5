"""
conciertos.py — Ingesta de conciertos en NYC desde Setlist.fm.

Fuente  : api.setlist.fm
Destino : MinIO  grupo5/raw/eventos_nyc/date=YYYY-MM-DD/eventos_concierto_YYYY-MM-DD.parquet
"""

import os
import random
import time

import pandas as pd
import requests

from .utils_eventos import (
    cargar_paradas_df,
    fusionar_lista_estaciones,
    obtener_paradas_afectadas,
)
from src.common.minio_client import upload_df_parquet, DEFAULT_BUCKET

# ─────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────

BASE_URL            = "https://api.setlist.fm/rest/1.0"
SEARCH_SETLISTS_URL = f"{BASE_URL}/search/setlists"
MAX_PAGINAS_API     = 500
RADIO_METRO_M       = 500

ARTISTAS_NY = {
    "Taylor Swift", "Dua Lipa", "Gracie Abrams", "Tate McRae", "Benson Boone",
    "Chappell Roan", "Mary J. Blige", "Sabrina Carpenter", "Katy Perry",
    "Deftones", "Ghost", "Avril Lavigne", "Pierce The Veil",
    "Shinedown", "Eric Clapton", "Nick Cave & The Bad Seeds",
    "The Black Keys", "Pulp", "Bloc Party", "Alabama Shakes",
    "Oasis", "Cage The Elephant",
    "Tyler, The Creator", "Kali Uchis", "Muni Long", "Eladio Carrión",
    "Rod Wave", "Kendrick Lamar", "SZA", "Chris Brown",
    "Chris Stapleton", "Hardy", "Lainey Wilson", "Tyler Childers",
    "Phish", "King Gizzard & the Lizard Wizard",
    "Christian Nodal", "Los Tigres Del Norte", "Alejandro Fernández", "Shakira",
    "Vybz Kartel", "Bounty Killer", "Capleton", "Shenseea",
    "Andrea Bocelli", "Hans Zimmer", "Bruno Mars", "Ariana Grande",
    "Beyoncé", "Billie Eilish", "Post Malone", "Olivia Rodrigo",
    "Bruce Springsteen", "J. Cole", "Bad Bunny", "Karol G",
    "Linkin Park", "Metallica", "Stray Kids", "The Weeknd",
    "Justin Timberlake", "Adele", "Ed Sheeran", "Lady Gaga",
}

MAPEO_HORARIOS = {
    "Madison Square Garden":     "19:30",
    "Barclays Center":           "19:30",
    "UBS Arena":                 "19:30",
    "Forest Hills Stadium":      "18:00",
    "Kings Theatre":             "20:00",
    "Brooklyn Paramount":        "20:00",
    "Amazura Concert Hall":      "21:00",
    "Great Lawn (Central Park)": "16:00",
    "Flushing Meadows Park":     "13:00",
    "Under the K Bridge":        "18:00",
    "Lincoln Center":            "19:30",
}



#  Helpers Setlist.fm


def build_headers():
    api_key = os.environ.get("SETLIST_API_KEY")
    if not api_key:
        raise ValueError("Falta la variable de entorno SETLIST_API_KEY")
    return {"x-api-key": api_key, "Accept": "application/json"}


def request_with_retry(session, url, params=None, timeout=30, max_retries=8, base_sleep=2.0):
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else base_sleep * (2 ** (attempt - 1)) + random.uniform(1, 2)
                print(f"  [Límite API] Esperando {wait:.2f}s...")
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                print(f"  [Error {r.status_code}] Reintentando en {wait:.2f}s...")
                time.sleep(wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

            return r

        except requests.exceptions.RequestException as exc:
            wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(1, 3)
            print(f"  [Red] {type(exc).__name__}. Reintentando en {wait:.2f}s...")
            time.sleep(wait)

    raise RuntimeError(f"No se pudo completar la petición tras {max_retries} reintentos.")


def convertir_fecha(fecha_str):
    try:
        return pd.to_datetime(fecha_str, format="%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        return fecha_str



#  Extracción de setlists

def _fetch_setlists_nyc(year, pagina_inicio=1):
    """Descarga todos los setlists de NYC para el año dado desde Setlist.fm."""
    all_items = []
    page  = pagina_inicio
    total = None
    params_base = {
        "cityName":    "New York",
        "stateCode":   "NY",
        "countryCode": "US",
        "year":        year,
    }

    with requests.Session() as session:
        session.headers.update(build_headers())
        while True:
            r = request_with_retry(session, SEARCH_SETLISTS_URL, params={**params_base, "p": page})
            payload = r.json()
            batch   = payload.get("setlist", [])
            all_items.extend(batch)

            if total is None:
                total = int(payload.get("total", 0))

            print(f"  Página {page} → acumulado {len(all_items)}/{total}")

            if not batch or len(all_items) >= total or page >= MAX_PAGINAS_API:
                break

            page += 1
            time.sleep(3 + random.uniform(1, 2))

    return all_items


def setlists_to_df(setlists, df_paradas):
    '''Convertir en df'''
    rows = []
    for s in setlists:
        venue  = s.get("venue", {}) or {}
        city   = venue.get("city", {}) or {}
        coords = city.get("coords", {}) or {}
        artist = s.get("artist", {}) or {}

        lon, lat = coords.get("long"), coords.get("lat")
        paradas = []
        if lon and lat:
            raw = obtener_paradas_afectadas([lon, lat], df_paradas, max_metros=RADIO_METRO_M)
            paradas = fusionar_lista_estaciones(raw)

        rows.append({
            "fecha_inicio":      convertir_fecha(s.get("eventDate")),
            "nombre_evento":     artist.get("name"),
            "venue_name":        venue.get("name"),
            "lat":               lat,
            "lng":               lon,
            "paradas_afectadas": paradas,
        })
    return pd.DataFrame(rows)



#  Extracción pública

def extraer_conciertos(start_date, end_date, df_paradas=None):
    """
    Extrae conciertos de NYC en el rango [start_date, end_date].
    Devuelve un DataFrame listo para subir.
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    years = range(start.year, end.year + 1)

    all_setlists = []
    for year in years:
        print(f"[conciertos] Descargando setlists de NYC para {year}...")
        all_setlists.extend(_fetch_setlists_nyc(year))

    df_completo = setlists_to_df(all_setlists, df_paradas)

    df = df_completo[df_completo["nombre_evento"].isin(ARTISTAS_NY)].copy()

    df["hora_inicio"] = df["venue_name"].map(MAPEO_HORARIOS).fillna("20:00")
    df["hora_salida_estimada"] = df["hora_inicio"].apply(
        lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=3)).strftime("%H:%M")
    )
    df["score"] = 1.0 #alta influencia
    df["nombre_evento"] = "Concierto: " + df["nombre_evento"]
    df["tipo"] = "Concierto"

    df = df[(df["fecha_inicio"] >= start_date) & (df["fecha_inicio"] <= end_date)]
    df = df.drop(columns=["lat", "lng", "venue_name"]).reset_index(drop=True)
    return df.sort_values(["fecha_inicio", "hora_inicio"]).reset_index(drop=True)



#  Ingesta completa

def ingest_conciertos(start_date, end_date):
    """Punto de entrada para el orquestador."""
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    print("[conciertos] Cargando paradas de metro...")
    df_paradas = cargar_paradas_df()

    df = extraer_conciertos(start_date, end_date, df_paradas=df_paradas)

    if df.empty:
        print("[conciertos] Sin conciertos en el rango. Nada que subir.")
        return

    print("[conciertos] Subiendo parquets a MinIO...")
    subidos = 0
    for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
        df_dia = df_dia.reset_index(drop=True)
        obj = f"grupo5/raw/eventos_nyc/date={fecha}/eventos_concierto_{fecha}.parquet"
        try:
            upload_df_parquet(access_key, secret_key, obj, df_dia)
            print(f"  Subido: {DEFAULT_BUCKET}/{obj} ({len(df_dia)} filas)")
            subidos += 1
        except Exception as exc:
            print(f"  Error subiendo {obj}: {exc}")

    print(f"[conciertos] Terminado. {subidos} archivos subidos.")
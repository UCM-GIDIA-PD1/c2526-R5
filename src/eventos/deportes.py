"""
deportes.py — Ingesta de eventos deportivos de equipos NYC desde ESPN.

Fuente  : site.api.espn.com
Destino : MinIO  grupo5/raw/eventos_nyc/date=YYYY-MM-DD/eventos_deporte_YYYY-MM-DD.parquet
"""



import os
import sys

import requests
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from .utils_eventos import (
    cargar_paradas_df,
    fusionar_lista_estaciones,
    obtener_paradas_afectadas,
)
from src.common.minio_client import upload_df_parquet, DEFAULT_BUCKET

import pandas as pd


#  Constantes


BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

NYC_TEAMS = {
    "basketball/nba": ["knicks", "nets"],
    "baseball/mlb":   ["yankees", "mets"],
    "hockey/nhl":     ["rangers", "islanders", "devils"],
    "football/nfl":   ["giants", "jets"],
    "soccer/usa.1":   ["new-york-city-fc", "new-york-red-bulls"],
}

DURACIONES_HORAS = {
    "nba":   2.5,
    "mlb":   3.0,
    "nhl":   2.5,
    "nfl":   3.5,
    "usa.1": 2.0,
}

VENUES_NYC = {
    "Madison Square Garden": [-73.9934, 40.7505],
    "UBS Arena":             [-73.7229, 40.7226],
    "Prudential Center":     [-74.1713, 40.7334],
    "Yankee Stadium":        [-73.9262, 40.8296],
    "Citi Field":            [-73.8456, 40.7571],
    "MetLife Stadium":       [-74.0744, 40.8135],
    "Red Bull Arena":        [-74.1502, 40.7369],
}

CIUDADES_NYC = {"New York", "Elmont", "Newark", "East Rutherford", "Harrison"}

RADIO_METRO_M = 500



#  Helpers ESPN

def _extraer_scoreboard(session, sport, fecha_gte, fecha_lte):
    '''Request a la API, para sacar los datos'''
    url = f"{BASE_URL}/{sport}/scoreboard"
    r = session.get(url, params={"dates": f"{fecha_gte}-{fecha_lte}"}, timeout=(10, 60))
    r.raise_for_status()
    return r.json()


def _es_partido_casa_nyc(evento, equipos_nyc):
    '''Comprueba si el partido es en casa de algún local'''
    for comp in evento.get("competitions", [{}])[:1]:
        for competidor in comp.get("competitors", []):
            if competidor.get("homeAway") == "home":
                equipo = competidor.get("team", {})
                slug   = equipo.get("slug", "").lower()
                nombre = equipo.get("displayName", "").lower()
                if any(nyc in slug or nyc in nombre for nyc in equipos_nyc):
                    return True
    return False


def _es_venue_nyc(competicion):
    '''Comprueba si el partido es en algún venue de NYC'''
    venue  = competicion.get("venue", {})
    ciudad = venue.get("address", {}).get("city", "")
    nombre = venue.get("fullName", "")
    return (ciudad in CIUDADES_NYC) or (nombre in VENUES_NYC)


def _geocodificar_venue(nombre_venue, geocode_fn):
    '''Retorna coordenadas [long,lat] de los venues'''
    if nombre_venue in VENUES_NYC:
        lon, lat = VENUES_NYC[nombre_venue]
        return lat, lon
    try:
        res = geocode_fn(f"{nombre_venue}, New York")
        if res:
            return res.latitude, res.longitude
    except Exception:
        pass
    return None, None



#  Extracción mensual / anual

def extraer_deportes(start_date, end_date):
    """
    Extrae eventos deportivos de equipos NYC entre start_date y end_date
    (formato YYYY-MM-DD). Devuelve un DataFrame unificado.
    """
    
    fecha_gte = start_date.replace("-", "")
    fecha_lte = end_date.replace("-", "")

    session    = requests.Session()
    geolocator = Nominatim(user_agent="espn_nyc_geocoder")
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

    filas = []
    for sport, equipos in NYC_TEAMS.items():
        try:
            data = _extraer_scoreboard(session, sport, fecha_gte, fecha_lte)
        except requests.exceptions.RequestException:
            continue

        liga = sport.split("/")[1]
        for ev in data.get("events", []):
            comp_ppal = ev.get("competitions", [{}])[0]
            if _es_partido_casa_nyc(ev, equipos) and _es_venue_nyc(comp_ppal):
                nombre_venue = comp_ppal.get("venue", {}).get("fullName", "")
                lat, lon = _geocodificar_venue(nombre_venue, geocode_fn)
                filas.append({
                    "nombre_evento": ev.get("name"),
                    "fecha_cruda":   ev.get("date"),
                    "duracion":      DURACIONES_HORAS.get(liga, 2.5),
                    "coordinates":   [lon, lat] if lon and lat else [],
                })

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    dt_ny = pd.to_datetime(df["fecha_cruda"]).dt.tz_convert("America/New_York")
    df["fecha_inicio"]         = dt_ny.dt.strftime("%Y-%m-%d")
    df["hora_inicio"]          = dt_ny.dt.strftime("%H:%M")
    df["hora_salida_estimada"] = (dt_ny + pd.to_timedelta(df["duracion"] + 0.2, unit="h")).dt.strftime("%H:%M")
    df = df.drop(columns=["fecha_cruda", "duracion"])
    return df.sort_values(["fecha_inicio", "hora_inicio"]).reset_index(drop=True)




#  Ingesta completa


def ingest_deportes(start_date, end_date):
    """Punto de entrada para el orquestador."""
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    print("[deportes] Cargando paradas de metro...")
    df_paradas = cargar_paradas_df()

    print(f"[deportes] Extrayendo eventos {start_date} - {end_date}...")
    df = extraer_deportes(start_date, end_date)

    if df.empty:
        print("[deportes] Sin eventos en el rango. Nada que subir.")
        return

    if df_paradas is not None:
        df["paradas_afectadas"] = df["coordinates"].apply(
            lambda c: fusionar_lista_estaciones(
                obtener_paradas_afectadas(c, df_paradas, max_metros=RADIO_METRO_M)
            ) if c else []
        )

    df = df.drop(columns=["coordinates"], errors="ignore")
    df["score"] = 1.0 #eventos de alta influencia a priori

    print("[deportes] Subiendo parquets a MinIO...")
    subidos = 0
    for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
        df_dia = df_dia.reset_index(drop=True)
        obj = f"grupo5/raw/eventos_nyc/dia={fecha}/eventos_deporte_{fecha}.parquet"
        try:
            upload_df_parquet(access_key, secret_key, obj, df_dia)
            print(f"  Subido: {DEFAULT_BUCKET}/{obj} ({len(df_dia)} filas)")
            subidos += 1
        except Exception as exc:
            print(f"  Error subiendo {obj}: {exc}")

    print(f"[deportes] Terminado. {subidos} archivos subidos.")


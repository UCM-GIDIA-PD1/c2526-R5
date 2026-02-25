import io
import os
import time
import calendar
import requests
import pandas as pd
from collections import defaultdict
from pymongo import MongoClient
from minio import Minio
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

DEFAULT_ENDPOINT = "minio.fdi.ucm.es"
DEFAULT_BUCKET   = "pd1"

NYC_TEAMS = {
    'basketball/nba': ['knicks', 'nets'],
    'baseball/mlb':   ['yankees', 'mets'],
    'hockey/nhl':     ['rangers', 'islanders', 'devils'],
    'football/nfl':   ['giants', 'jets'],
    'soccer/usa.1':   ['new-york-city-fc', 'new-york-red-bulls'],
}

DURACIONES_HORAS = {
    'nba':   2.5,
    'mlb':   3.0,
    'nhl':   2.5,
    'nfl':   3.5,
    'usa.1': 2.0,
}

VENUES_NYC = {
    'Madison Square Garden':  [-73.9934, 40.7505],
    'UBS Arena':              [-73.7229, 40.7226],
    'Prudential Center':      [-74.1713, 40.7334],
    'Yankee Stadium':         [-73.9262, 40.8296],
    'Citi Field':             [-73.8456, 40.7571],
    'MetLife Stadium':        [-74.0744, 40.8135],
    'Red Bull Arena':         [-74.1502, 40.7369],
    'Yankee Stadium II':      [-73.9262, 40.8296],
}

CIUDADES_NYC = {'New York', 'Elmont', 'Newark', 'East Rutherford', 'Harrison'}


# ─────────────────────────────────────────────
#  MinIO
# ─────────────────────────────────────────────
def minio_client(access_key, secret_key, endpoint=DEFAULT_ENDPOINT):
    return Minio(endpoint, access_key=access_key, secret_key=secret_key)


def upload_df_parquet(access_key, secret_key, object_name, df,
                      endpoint=DEFAULT_ENDPOINT, bucket=DEFAULT_BUCKET):
    c = minio_client(access_key, secret_key, endpoint)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    c.put_object(bucket, object_name, buf, length=buf.getbuffer().nbytes)


# ─────────────────────────────────────────────
#  MongoDB
# ─────────────────────────────────────────────
def conectar_mongo():
    url_servidor = 'mongodb://127.0.0.1:27017/'
    client = MongoClient(url_servidor)
    try:
        client.server_info()
        db = client["PD1"]
        return db
    except Exception:
        return None


def cursor_paradas_afectadas(coordinates, db):
    return db.subway.find({
        "ubicacion": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": coordinates},
                "$maxDistance": 700
            }
        }
    })


def extraccion_paradas(cursor):
    return [(doc["nombre"], doc["lineas"]) for doc in cursor]


def fusionar_lista_estaciones(lista_tuplas):
    if not isinstance(lista_tuplas, list):
        return lista_tuplas

    estaciones_fusionadas = defaultdict(set)
    for nombre, lineas in lista_tuplas:
        estaciones_fusionadas[nombre].update(lineas.split())

    return [(nombre, " ".join(sorted(lineas_set))) for nombre, lineas_set in estaciones_fusionadas.items()]


# ─────────────────────────────────────────────
#  ESPN
# ─────────────────────────────────────────────
def extraer_scoreboard(session, sport, fecha_gte, fecha_lte):
    url = f"{BASE_URL}/{sport}/scoreboard"
    parametros = {"dates": f"{fecha_gte}-{fecha_lte}"}
    respuesta = session.get(url, params=parametros, timeout=(10, 60))
    respuesta.raise_for_status()
    return respuesta.json()


def es_partido_en_casa_nyc(evento, equipos_nyc):
    competiciones = evento.get("competitions", [])
    if competiciones:
        for competidor in competiciones[0].get("competitors", []):
            if competidor.get("homeAway") == "home":
                equipo = competidor.get("team", {})
                slug  = equipo.get("slug", "").lower()
                nombre = equipo.get("displayName", "").lower()
                if any(nyc in slug or nyc in nombre for nyc in equipos_nyc):
                    return True
    return False


def es_venue_nyc(competicion):
    if not competicion:
        return False
    venue = competicion.get("venue", {})
    if not venue:
        return False
    ciudad      = venue.get("address", {}).get("city", "")
    nombre_venue = venue.get("fullName", "")
    return (ciudad in CIUDADES_NYC) or (nombre_venue in VENUES_NYC)


def geocodificar_venue(nombre_venue, funcion_geocode):
    if nombre_venue in VENUES_NYC:
        longitud, latitud = VENUES_NYC[nombre_venue]
        return latitud, longitud
    try:
        resultado = funcion_geocode(f"{nombre_venue}, New York")
        if resultado:
            return resultado.latitude, resultado.longitude
    except Exception:
        pass
    return None, None


def extraer_fila(evento, sport, funcion_geocode):
    competiciones = evento.get("competitions", [])
    competicion   = competiciones[0] if competiciones else {}
    nombre_venue  = competicion.get("venue", {}).get("fullName", "")
    liga          = sport.split("/")[1]
    latitud, longitud = geocodificar_venue(nombre_venue, funcion_geocode)

    return {
        "nombre_evento": evento.get("name"),
        "fecha_cruda":   evento.get("date"),
        "duracion":      DURACIONES_HORAS.get(liga, 2.5),
        "coordinates":   [longitud, latitud] if longitud and latitud else [],
    }


def extraccion_nyc(year, month):
    dia_final = calendar.monthrange(year, month)[1]
    fecha_gte = f"{year}{month:02d}01"
    fecha_lte = f"{year}{month:02d}{dia_final}"

    session        = requests.Session()
    geolocator     = Nominatim(user_agent="espn_nyc_geocoder")
    funcion_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

    filas_datos = []

    for sport, equipos in NYC_TEAMS.items():
        try:
            data   = extraer_scoreboard(session, sport, fecha_gte, fecha_lte)
            eventos = data.get("events", [])
            for ev in eventos:
                comp_ppal = ev.get("competitions", [{}])[0]
                if es_partido_en_casa_nyc(ev, equipos) and es_venue_nyc(comp_ppal):
                    filas_datos.append(extraer_fila(ev, sport, funcion_geocode))
        except requests.exceptions.RequestException:
            pass
        except Exception:
            pass

    df = pd.DataFrame(filas_datos)

    if not df.empty:
        dt_ny = pd.to_datetime(df['fecha_cruda']).dt.tz_convert('America/New_York')
        df['fecha_inicio']         = dt_ny.dt.strftime('%Y-%m-%d')
        df['hora_inicio']          = dt_ny.dt.strftime('%H:%M')
        dt_salida                  = dt_ny + pd.to_timedelta(df['duracion'], unit='h')
        df['hora_salida_estimada'] = dt_salida.dt.strftime('%H:%M')
        df = df.drop(columns=['fecha_cruda', 'duracion'])
        df = df.sort_values(by=["fecha_inicio", "hora_inicio"]).reset_index(drop=True)

    return df


def extraccion_nyc_anual(year):
    dataframes_mensuales = []

    for month in range(1, 13):
        df_mes = extraccion_nyc(year, month)
        if not df_mes.empty:
            dataframes_mensuales.append(df_mes)
        time.sleep(2)

    if dataframes_mensuales:
        df_anual = pd.concat(dataframes_mensuales, ignore_index=True)
        return df_anual.sort_values(by=["fecha_inicio", "hora_inicio"]).reset_index(drop=True)
    return pd.DataFrame()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ANIO_A_EXTRAER = 2025

    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    assert MINIO_ACCESS_KEY is not None, "Falta MINIO_ACCESS_KEY"
    assert MINIO_SECRET_KEY is not None, "Falta MINIO_SECRET_KEY"

    db = conectar_mongo()
    df = extraccion_nyc_anual(ANIO_A_EXTRAER)

    if df.empty:
        print("No hay eventos para el año indicado.")
        exit()

    if db is not None:
        df["paradas_afectadas"] = df["coordinates"].apply(
            lambda cor: extraccion_paradas(cursor_paradas_afectadas(cor, db)) if cor else []
        )
        df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)

    df = df.drop(columns=["coordinates"])

    # Subir a MinIO un parquet por día
    subidos = 0
    for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
        if df_dia.empty:
            continue

        df_dia = df_dia.reset_index(drop=True)
        object_name = f"grupo5/raw/eventos_nyc/dia={fecha}/eventos_deporte_{fecha}.parquet"

        try:
            upload_df_parquet(MINIO_ACCESS_KEY, MINIO_SECRET_KEY, object_name, df_dia)
            print(f"Subido: {DEFAULT_BUCKET}/{object_name} (filas: {len(df_dia)})")
            subidos += 1
        except Exception as e:
            print(f"Error subiendo {object_name}: {e}")

    print(f"\nTerminado. {subidos} archivos subidos a MinIO.")
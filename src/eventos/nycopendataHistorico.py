import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import json
from pymongo import MongoClient
import numpy as np
import io
from minio import Minio
from typing import Any

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


urlbase = "https://data.cityofnewyork.us/resource/"

DEFAULT_ENDPOINT = "minio.fdi.ucm.es"
DEFAULT_BUCKET = "pd1"

def desde_fecha(fecha_str):
       return f'{fecha_str}T00:00:00.000'

def hasta_fecha(fecha_str):
   return f'{fecha_str}T23:59:59.000'

def extraccion_actual(ini, fin, token):
    url_eventos = f"{urlbase}bkfu-528j.json"

    limit = 10000

    header = {
        "X-App-Token": token
    }

    chunks = []
    offset = 0

    while True:
        param = {
            "$where": f"start_date_time >= '{ini}' AND start_date_time <= '{fin}'",
            "$limit": limit,
            "$offset": offset
        }

        response = requests.get(url_eventos, params=param, headers=header, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"Error en la extracción. HTTP {response.status_code}\n{response.text[:2000]}")

        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Respuesta no es JSON válido (posible corte por tamaño). "
                               f"Primeros 2000 chars:\n{response.text[:2000]}") from e

        if not data:
            break

        chunks.append(pd.DataFrame(data))
        offset += limit
        print(f"Descargadas ~{offset} filas...")

        break #para comprobar

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return df

TOKEN = os.getenv("NYC_OPEN_DATA_TOKEN")
assert TOKEN is not None, "Falta la variable de entorno NYCOPENDATA_TOKEN"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
assert MINIO_ACCESS_KEY is not None, "Falta MINIO_ACCESS_KEY"
assert MINIO_SECRET_KEY is not None, "Falta MINIO_SECRET_KEY"

def _client(access_key: str, secret_key: str, endpoint: str = DEFAULT_ENDPOINT) -> Minio:
    """Crear un cliente de MinIO"""
    return Minio(endpoint, access_key=access_key, secret_key=secret_key)

def upload_df_parquet(
    access_key: str,
    secret_key: str,
    object_name: str,
    df: pd.DataFrame,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """Subir un pandas Dataframe como objeto parquet"""
    c = _client(access_key, secret_key, endpoint)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    c.put_object(bucket, object_name, buf, length=buf.getbuffer().nbytes)



inicio_2025 = desde_fecha('2025-01-01')
final_2025 = hasta_fecha('2025-12-31')
#print("Iniciando el proceso de extracción")
df = extraccion_actual(inicio_2025, final_2025, TOKEN)
#print(df.shape)
#print(df.columns)

if df.empty:
    print("No hay eventos en ese rango de fechas")
    exit()

df["borough"] = df["event_borough"]

df = df[['event_name', 'event_type', 'start_date_time', 'end_date_time',
         'event_location', 'borough', 'community_board']]

df['start_date_time'] = pd.to_datetime(df['start_date_time'])
df['end_date_time'] = pd.to_datetime(df['end_date_time'])

df['duration_hours'] = (df['end_date_time'] - df['start_date_time']).dt.total_seconds() / 3600

df = df.dropna(subset=['event_type'])

riesgo_map = {
        'Parade': 10,
        'Athletic Race / Tour': 10,
        'Street Event': 8,
        'Stationary Demonstration': 7,
        'Street Festival': 7,
        'Special Event': 7,
        'Single Block Festival': 6,
        'Bike the Block': 6,
        'BID Multi-Block': 6,
        'Plaza Event': 6,
        'Plaza Partner Event': 6,
        'Block Party': 5,
        'Theater Load in and Load Outs': 5,
        'Open Culture': 4,
        'Religious Event': 3,
        'Press Conference': 3,
        'Health Fair': 3,
        'Rigging Permit': 3,
        'Farmers Market': 2,
        'Sidewalk Sale': 2,
        'Shooting Permit': 2,
        'Filming/Photography': 2,
        'Open Street Partner Event': 2,
        'Production Event': 1,  
        'Sport - Adult': 1,
        'Sport - Youth': 1,
        'Miscellaneous': 1,
        'Stickball': 1,
        'Clean-Up': 1
}

df['nivel_riesgo_tipo'] = df['event_type'].map(riesgo_map)
df = df[df["nivel_riesgo_tipo"] >= 8]
df = df.drop_duplicates(subset=["event_name", "start_date_time", "borough", "event_location"])
df["score"] = df["nivel_riesgo_tipo"].map({8: 0.8, 9: 0.9, 10: 1.0})
#print(df)
#print("Proceso finalizado")
#print("Calculando coordenadas")

def extraer_intersecciones(localizacion, barrio):
    """
    Extrae las intersecciones de las calles del evento
    """
    intersecciones = []
    segmentos = localizacion.split(",")

    for segmento in segmentos:
        segmento = segmento.strip()
        if " between " in segmento:
            partes = segmento.split(" between ")
            calle_principal = partes[0].strip()

            cruces = partes[1].split(" and ")
            for cruce in cruces:
                cruce = cruce.strip()
                if cruce:
                    intersecciones.append(f"{calle_principal} & {cruce}, {barrio}, New York")

    return intersecciones if intersecciones else [localizacion + f", {barrio}, New York"]


def extraer_coord(localizacion, barrio, geocode):
    """
    Devuelve las coordenadas del centro de las ubicaciones (calles que cruzan), o la coordenada del parque
    Devuelve longitud-latitud
    """
    if pd.isna(localizacion):
        return 0, 0

    if ":" in localizacion:
        resultado = geocode(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        if resultado:
            return resultado.longitude, resultado.latitude
        return 0, 0

    intersections = extraer_intersecciones(localizacion, barrio)

    coords = []
    for intersection in intersections:
        try:
            resultado = geocode(intersection)
            if resultado:
                coords.append((resultado.latitude, resultado.longitude))
        except:
            continue

    if coords:
        lat = np.mean([c[0] for c in coords])
        lon = np.mean([c[1] for c in coords])
        return lon, lat

    return 0, 0



geolocator = Nominatim(user_agent="pd1_eventos_nyc")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)



total = len(df)
paso = max(1, total // 10)

lons = []
lats = []
for i, (_, r) in enumerate(df.iterrows(), start=1):
    lon, lat = extraer_coord(r["event_location"], r["borough"], geocode)
    lons.append(lon)
    lats.append(lat)

    if i % paso == 0 or i == total:
        pct = int(round(i * 100 / total))
        pct = min(100, (pct // 10) * 10)
        print(f"Coordenadas: {pct}% ({i}/{total})")

df["lon"] = lons
df["lat"] = lats


df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df.loc[(df["lon"] == 0) & (df["lat"] == 0), ["lon", "lat"]] = np.nan


#print("Calculando paradas afectadas")

url_servidor = 'mongodb://127.0.0.1:27017/'
client = MongoClient(url_servidor)

# código para ver si se ha conectado bien
try:
    s = client.server_info()  # si hay error tendremos una excepción
    print("Conectado a MongoDB, versión", s["version"])
    db = client["PD1"]
except:
    print("Error de conexión ¿está arrancado el servidor?")


def cursor_paradas_afectedas(coordinates):  # coordinates de esta forma [longitud, latitud]
    cursor = db.subway.find(
        {
            "ubicacion":
                {"$near":
                    {
                        "$geometry": {"type": "Point", "coordinates": coordinates},
                        "$maxDistance": 500
                    }
                }
        }
    )
    return cursor


def extraccion_paradas(cursor):
    afectadas = []
    for doc in cursor:
        afectadas.append((doc["nombre"], doc["lineas"]))
    return afectadas



def paradas_afectadas_evento(lon, lat):
    if pd.isna(lon) or pd.isna(lat):
        return []
    cursor = cursor_paradas_afectedas([float(lon), float(lat)])
    return extraccion_paradas(cursor)

df["paradas_afectadas"] = df.apply(
    lambda r: paradas_afectadas_evento(r["lon"], r["lat"]),
    axis=1
)

#print(df[["event_name", "event_location", "borough", "lon", "lat", "paradas_afectadas"]].head(10))
df["hora_inicio"] = df["start_date_time"].dt.strftime("%H:%M")
df["hora_salida_estimada"] = df["end_date_time"].dt.strftime("%H:%M")
df["fecha_inicio"] = df["start_date_time"].dt.strftime("%Y-%m-%d")
df["fecha_final"] = df["end_date_time"].dt.strftime("%Y-%m-%d")

df = df.rename(columns={"event_name": "nombre_evento"})
df = df.reset_index(drop=True)
df = df[["nombre_evento", "fecha_inicio", "hora_inicio", "fecha_final", "hora_salida_estimada", "score", "paradas_afectadas"]]
print(df.head(10))
print(len(df))

output_dir = "eventos_por_dia"
os.makedirs(output_dir, exist_ok=True)

df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"]).dt.strftime("%Y-%m-%d")

print("Guardando dataframes por día en local...")

for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
    df_dia = df_dia.reset_index(drop=True)

    filename = f"{output_dir}/eventos_{fecha}.parquet"

    df_dia.to_parquet(filename, index=False)

    print(f"Guardado: {filename} (filas: {len(df_dia)})")

print("Terminado.")

'''
df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"]).dt.strftime("%Y-%m-%d")

for fecha, df_dia in df.groupby("fecha_inicio", sort=True):
    df_dia = df_dia.reset_index(drop=True)

    object_name = f"eventos_nyc/dia={fecha}/eventos_{fecha}.parquet"
    upload_df_parquet(MINIO_ACCESS_KEY, MINIO_SECRET_KEY, object_name, df_dia)

    print(f"Subido: {DEFAULT_BUCKET}/{object_name} (filas: {len(df_dia)})")

print("Terminado.")
'''
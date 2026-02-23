import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import json
import numpy as np
from pymongo import MongoClient
from collections import defaultdict
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def fusionar_lista_estaciones(lista_tuplas):
    '''fusiona lineas con el mismo nombre'''
    if not isinstance(lista_tuplas, list):
        return lista_tuplas
        
    estaciones_fusionadas = defaultdict(set)
    
    for nombre, lineas in lista_tuplas:
        estaciones_fusionadas[nombre].update(lineas.split())
        
    resultado = []
    for nombre, lineas_set in estaciones_fusionadas.items():
        lineas_ordenadas = " ".join(sorted(lineas_set))
        resultado.append((nombre, lineas_ordenadas))
        
    return resultado

def extraccion_actual(fecha, CLIENT_ID, manana):
    url="https://api.seatgeek.com/2/events"
    params = {
        "client_id": CLIENT_ID,
        "venue.city": "New York",   
        "sort": "score.desc",        
        "per_page": 100,            
        "datetime_local.gte": fecha,
        "datetime_local.lte": manana,
    }
    
    response = requests.get(url, params=params)
    assert response.status_code == 200, "Error en la extracción de eventos"
    return response.json()

def calcular_salida(fila, tiempos_salida):
    tipo_evento = fila['tipo']
    horas_duracion = tiempos_salida.get(tipo_evento, 2.5) 
    hora_inicio = pd.to_datetime(fila['hora_inicio'])
    hora_fin = hora_inicio + timedelta(hours=horas_duracion)
    
    return hora_fin.strftime('%H:%M')

def conectar_mongo():
    url_servidor = 'mongodb://127.0.0.1:27017/'
    client = MongoClient(url_servidor)
    try:
        s = client.server_info() 
        print("Conectado a MongoDB, versión", s["version"])
        db = client["PD1"]
        return db
    except Exception as e:
        print(f"Error de conexión: {e}")
        print("¿Está arrancado el servidor de Mongo?")
        return None

def cursor_paradas_afectedas(coordinates, db): 
    cursor = db.subway.find(
       {
         "ubicacion":
           { "$near" :
              {
                "$geometry": { "type": "Point",  "coordinates": coordinates },
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

def api_seatgeek(db):
    fecha_hoy_obj = datetime.now() 
    manana_obj = fecha_hoy_obj + timedelta(days = 1)
    fecha_hoy_str = fecha_hoy_obj.strftime('%Y-%m-%d')
    manana_str = manana_obj.strftime('%Y-%m-%d')
    
    API_KEY = os.getenv('CLIENT_ID_SEATGEEK')
    assert API_KEY is not None, "Falta la variable de entorno CLIENT_ID_SEATGEEK"
    
    data = extraccion_actual(fecha_hoy_str, API_KEY, manana_str)
    eventos_limpios = []

    for e in data['events']:
        info = {
            'nombre_evento': e.get('title'),
            'tipo': e.get('type'),
            'hora_inicio': e.get('datetime_local'),
            'lugar': e['venue'].get('name'),
            'direccion': e['venue'].get('address', 'Dirección no disponible'),
            'latitud': e['venue']['location'].get('lat'),
            'longitud': e['venue']['location'].get('lon'),
            'capacidad': e['venue'].get('capacity'),
            'popularidad_score': e.get('score'), 
            'venue_score': e['venue'].get('score') 
        }
        eventos_limpios.append(info)
        
    df = pd.DataFrame(eventos_limpios)
    df['capacidad'] = df['capacidad'].replace(0, np.nan)
    
    df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
    df['hora_inicio_str'] = df['hora_inicio'].dt.strftime('%H:%M') 
    
    tiempos_salida = {
        'nba': 2.5, 'nfl': 3.5, 'mlb': 3.0, 'nhl': 2.5, 'mls': 2.0, 
        'ncaa_basketball': 2.5, 'ncaa_football': 3.5, 'sports': 2.5,
        'tennis': 4.0, 'wwe': 3.0, 'boxing': 3.5, 'mma': 3.5,
        'concert': 3.0, 'music_festival': 8.0, 'classical': 2.5, 'opera': 3.0,
        'theater': 2.5, 'broadway_tickets_national': 2.5, 'comedy': 2.0, 
        'family': 2.0, 'ballet': 2.5, 'cirque_du_soleil': 2.0
    }

    df['hora_salida_estimada'] = df.apply(lambda fila: calcular_salida(fila, tiempos_salida), axis=1)
    df['hora_inicio'] = df['hora_inicio_str'] 
    df = df.drop(columns=['hora_inicio_str'])
    
    df["coordinates"] = df.apply(lambda fila: [fila['longitud'], fila['latitud']], axis=1)
    df = df.drop(['longitud', 'latitud', 'lugar', 'direccion'], axis=1)

    coords_invalidas = df["coordinates"].apply(lambda c: c == [0, 0] or None in c)
    n_invalidas = coords_invalidas.sum()
    if n_invalidas > 0:
        df = df[~coords_invalidas].copy()

    df["paradas_afectadas"] = df["coordinates"].apply(
        lambda cor: extraccion_paradas(cursor_paradas_afectedas(cor, db))
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)
    df = df.drop(columns=["coordinates", "tipo"], axis=1)
    
    return df

def desde_fecha(fecha_str):
    return f'{fecha_str}T00:00:00.000'

def hasta_fecha(fecha_str):
    return f'{fecha_str}T23:59:59.000'

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
    Devuelve las coordenadas del centro de las ubicaciones (calles que cruzan), o la coordenada del parque.
    Devuelve longitud-latitud.
    """
    if pd.isna(localizacion):
        return None, None
    
    if ":" in localizacion:
        resultado = geocode(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        if resultado:
            return resultado.longitude, resultado.latitude
        return None, None

    intersections = extraer_intersecciones(localizacion, barrio)
    
    coords = []
    for intersection in intersections:
        try:
            resultado = geocode(intersection)
            if resultado:
                coords.append((resultado.latitude, resultado.longitude))
        except Exception:
            continue
    
    if coords:
        lat = np.mean([c[0] for c in coords])
        lon = np.mean([c[1] for c in coords])
        return lon, lat
    
    return None, None

def api_nycopendata(db):
    urlbase = "https://data.cityofnewyork.us/resource/"
    url_eventos = f"{urlbase}tvpp-9vvx.json"

    load_dotenv()
    token = os.getenv('NYC_OPEN_DATA_TOKEN')
    assert token is not None, "Falta la variable de entorno NYC_OPEN_DATA_TOKEN"

    fecha_hoy_str = datetime.now().strftime('%Y-%m-%d')
    fecha_actual = desde_fecha(fecha_hoy_str)
    fecha_fin = hasta_fecha(fecha_hoy_str)
    
    param = {
        "$where": f"start_date_time >= '{fecha_actual}' AND start_date_time <= '{fecha_fin}'",    
    }

    header = {"X-App-Token": token}
    eventos = requests.get(url=url_eventos, params=param, headers=header)
    
    if eventos.status_code != 200:
        print(f"Error Parques: {eventos.text}")
    assert eventos.status_code == 200, "Error en la extracción de eventos"

    df = pd.DataFrame(eventos.json())
    
    if df.empty:
        return df

    df['start_date_time'] = pd.to_datetime(df['start_date_time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
    df["start_date_time"] = df["start_date_time"].dt.strftime('%H:%M')
    df['end_date_time'] = pd.to_datetime(df['end_date_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f')
    df["end_date_time"] = df["end_date_time"].dt.strftime('%H:%M')

    df = df.drop(["event_id", "event_agency", "street_closure_type", 'community_board',
                  'police_precinct', 'cemsid', 'event_street_side'], axis=1)
    
    riesgo_map = {
        'Parade': 10,
        'Athletic Race / Tour': 10,
        'Street Event': 8,
        'Stationary Demonstration': 7,
        'Street Festival': 7,
        'Special Event': 6,
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
    tipos_nuevos = df[df['nivel_riesgo_tipo'].isna()]['event_type'].unique()
    if len(tipos_nuevos) > 0:
        df['nivel_riesgo_tipo'] = df['nivel_riesgo_tipo'].fillna(1)

    df = df.sort_values(by="nivel_riesgo_tipo", ascending=False)
    df = df[df.nivel_riesgo_tipo > 6]

    geolocator = Nominatim(user_agent="nyc_events_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)
    
    df["coordenadas"] = df.apply(
        lambda row: list(extraer_coord(row["event_location"], row["event_borough"], geocode)), axis=1
    )

    coords_invalidas = df["coordenadas"].apply(lambda c: None in c)
    n_invalidas = coords_invalidas.sum()
    if n_invalidas > 0:   
        df = df[~coords_invalidas].copy()

    df["paradas_afectadas"] = df["coordenadas"].apply(
        lambda cor: extraccion_paradas(cursor_paradas_afectedas(cor, db))
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)

    df = df.drop(columns=["coordenadas", "event_location", "event_type","event_borough"], axis=1)

    mapeo_columnas = {
        'event_name': 'nombre_evento',
        'start_date_time': 'hora_inicio',
        'end_date_time': 'hora_salida_estimada',
    }
    df = df.rename(columns=mapeo_columnas)

    return df

def fusionar_dataframes(df_seat_geek, df_nyc):
        df_seat_geek['score'] = (df_seat_geek['popularidad_score'] + df_seat_geek['venue_score']) / 2
        df_seat_geek = df_seat_geek.drop(columns=['popularidad_score', 'venue_score', 'capacidad'])

        df_nyc['score'] = df_nyc['nivel_riesgo_tipo'] / 10
        df_nyc = df_nyc.drop(columns=['nivel_riesgo_tipo'])

    

        cols_comunes = ['nombre_evento', 'hora_inicio', 'hora_salida_estimada', 'score', 'paradas_afectadas']
        df_final = pd.concat([
            df_seat_geek[cols_comunes],
            df_nyc[cols_comunes]
            ], ignore_index=True)

        df_final = df_final.sort_values('score', ascending=False).reset_index(drop=True)

        return df_final

if __name__ == "__main__":

    db = conectar_mongo()
    if db is not None:
        df_seat_geek = None
        df_nyc = None

        try:
            print("\nExtrayendo eventos de SeatGeek...")
            df_seat_geek = api_seatgeek(db)
            print(f"  {len(df_seat_geek)} eventos extraídos de SeatGeek")
        except Exception as e:
            print(f"  Error en SeatGeek: {e}")

        try:
            print("\nExtrayendo eventos de NYC Open Data...")
            df_nyc = api_nycopendata(db)
            print(f"  {len(df_nyc)} eventos extraídos de NYC Open Data")
        except Exception as e:
            print(f"  Error en NYC Open Data: {e}")

        df_final = fusionar_dataframes(df_seat_geek, df_nyc)
        print(df_final)
"""
eventos_actual_final.py — Extracción de eventos en tiempo real para NYC.

Obtiene los eventos del día de hoy desde tres fuentes:
  - SeatGeek      → conciertos y eventos musicales
  - NYC Open Data → eventos públicos de alto impacto (desfiles, carreras, etc.)
  - ESPN          → partidos en casa de equipos NYC

Por cada evento calcula las paradas de metro afectadas en un radio de 500m
usando la fórmula de Haversine. Al final fusiona las tres fuentes en un único
DataFrame, deduplica eventos que aparezcan en más de una fuente y ordena
por score descendente.

Variables de entorno necesarias:
  - CLIENT_ID_SEATGEEK
  - NYC_OPEN_DATA_TOKEN
  - (ESPN no requiere API key, es pública)
"""




import os
import calendar
import requests
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ─────────────────────────────────────────────
#  Constantes ESPN
# ─────────────────────────────────────────────

# URL base de la API de ESPN
BASE_URL_ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# Equipos NYC por deporte, con el slug que usa ESPN en su API
NYC_TEAMS = {
    'basketball/nba': ['knicks', 'nets'],
    'baseball/mlb':   ['yankees', 'mets'],
    'hockey/nhl':     ['rangers', 'islanders', 'devils'],
    'football/nfl':   ['giants', 'jets'],
    'soccer/usa.1':   ['new-york-city-fc', 'new-york-red-bulls'],
}

# Duración estimada en horas de cada tipo de partido,
# usada para calcular la hora de salida estimada
DURACIONES_ESPN = {
    'nba':   2.5,
    'mlb':   3.0,
    'nhl':   2.5,
    'nfl':   3.5,
    'usa.1': 2.0,
}

# Coordenadas [longitud, latitud] de los principales estadios de NYC.
# Se usan para evitar llamadas a la API de geocodificación cuando el venue ya es conocido
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

# Ciudades del área metropolitana de NYC donde pueden jugarse partidos "locales"
CIUDADES_NYC = {'New York', 'Elmont', 'Newark', 'East Rutherford', 'Harrison'}




# Funciones compartidas con los scripts históricos, definidas en utils_eventos.py
from src.eventos.utils_eventos import (
    fusionar_lista_estaciones,
    cargar_paradas_df,
    obtener_paradas_afectadas,
)


#  SeatGeek


def extraccion_actual(fecha, CLIENT_ID, manana):
    """
    Llama a la API de SeatGeek y devuelve los eventos de NYC
    entre fecha (hoy) y manana, ordenados por score descendente.
    """
    url = "https://api.seatgeek.com/2/events"
    params = {
        "client_id": CLIENT_ID,
        "venue.city": "New York",
        "sort": "score.desc",
        "per_page": 100,
        "datetime_local.gte": fecha,   # desde el inicio del día
        "datetime_local.lte": manana,  # hasta el inicio de mañana
    }
    response = requests.get(url, params=params)
    assert response.status_code == 200, "Error en la extracción de eventos"
    return response.json()


def calcular_salida(fila, tiempos_salida):
    """
    Calcula la hora de salida estimada sumando la duración del tipo de evento
    a la hora de inicio.
    """
    horas_duracion = tiempos_salida.get(fila['tipo'], 2.5)  # 2.5h por defecto si el tipo no está en el dic
    hora_fin = pd.to_datetime(fila['hora_inicio']) + timedelta(hours=horas_duracion)
    return hora_fin.strftime('%H:%M')


def api_seatgeek(df_paradas):
    """
    Extrae conciertos y eventos musicales de NYC para hoy desde SeatGeek,
    calcula paradas de metro afectadas y devuelve un DataFrame limpio.
    """
    # Calculamos el rango de fechas: hoy y mañana
    fecha_hoy_obj = datetime.now()
    manana_obj = fecha_hoy_obj + timedelta(days=1)
    fecha_hoy_str = fecha_hoy_obj.strftime('%Y-%m-%d')
    manana_str = manana_obj.strftime('%Y-%m-%d')

    API_KEY = os.getenv('CLIENT_ID_SEATGEEK')
    assert API_KEY is not None, "Falta la variable de entorno CLIENT_ID_SEATGEEK"

    # Solo nos interesan eventos de tipo musical
    TIPOS_CONCIERTO = {'concert', 'music_festival', 'classical', 'opera', 'ballet'}

    data = extraccion_actual(fecha_hoy_str, API_KEY, manana_str)
    eventos_limpios = []

    for e in data['events']:
        # Ignoramos eventos que no sean musicales
        if e.get('type') not in TIPOS_CONCIERTO:
            continue
        eventos_limpios.append({
            'nombre_evento':     e.get('title'),
            'tipo':              e.get('type'),
            'hora_inicio':       e.get('datetime_local'),
            'lugar':             e['venue'].get('name'),
            'direccion':         e['venue'].get('address', 'Dirección no disponible'),
            'latitud':           e['venue']['location'].get('lat'),
            'longitud':          e['venue']['location'].get('lon'),
            'capacidad':         e['venue'].get('capacity'),
            'popularidad_score': e.get('score'),
            'venue_score':       e['venue'].get('score'),
        })

    df = pd.DataFrame(eventos_limpios)
    if df.empty:
        return df

    # Capacidad 0 no tiene sentido, la tratamos como dato desconocido
    df['capacidad'] = df['capacidad'].replace(0, np.nan)

    # Calculamos hora de salida estimada según el tipo de evento
    df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
    df['hora_inicio_str'] = df['hora_inicio'].dt.strftime('%H:%M')

    tiempos_salida = {
        'concert': 3.0, 'music_festival': 8.0, 'classical': 2.5,
        'opera': 3.0, 'ballet': 2.5,
    }

    df['hora_salida_estimada'] = df.apply(lambda fila: calcular_salida(fila, tiempos_salida), axis=1)
    df['hora_inicio'] = df['hora_inicio_str']
    df = df.drop(columns=['hora_inicio_str'])

    # Agrupamos lon/lat en una sola columna coordinates y eliminamos las originales
    df["coordinates"] = df.apply(lambda fila: [fila['longitud'], fila['latitud']], axis=1)
    df = df.drop(['longitud', 'latitud', 'lugar', 'direccion'], axis=1)

    # Eliminamos eventos sin coordenadas válidas
    coords_invalidas = df["coordinates"].apply(lambda c: c == [0, 0] or None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

    # Calculamos las paradas de metro afectadas por cada evento
    df["paradas_afectadas"] = df["coordinates"].apply(
        lambda cor: obtener_paradas_afectadas(cor, df_paradas)
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)
    df = df.drop(columns=["coordinates", "tipo"])

    return df


# ─────────────────────────────────────────────
#  NYC Open Data
# ─────────────────────────────────────────────

def desde_fecha(fecha_str):
    """Formatea la fecha como inicio del día para la query de NYC Open Data."""
    return f'{fecha_str}T00:00:00.000'


def hasta_fecha(fecha_str):
    """Formatea la fecha como fin del día para la query de NYC Open Data."""
    return f'{fecha_str}T23:59:59.000'


def extraer_intersecciones(localizacion, barrio):
    """
    Parsea el campo event_location de NYC Open Data, que describe ubicaciones
    en formato "Calle X between Calle A and Calle B", y las convierte en
    intersecciones geocodificables como "Calle X & Calle A, Barrio, New York".
    """
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
    # Si no se detecta el patrón "between", usamos la localización completa tal cual
    return intersecciones if intersecciones else [localizacion + f", {barrio}, New York"]


def extraer_coord(localizacion, barrio, geocode):
    """
    Devuelve las coordenadas (longitud, latitud) de un evento a partir
    de su campo de localización textual. Si hay varias intersecciones,
    devuelve el centroide de todas ellas.
    Devuelve (None, None) si no se puede geocodificar.
    """
    if pd.isna(localizacion):
        return None, None

    # Si contiene ":", es un nombre de lugar (ej: "Central Park: Great Lawn")
    # tomamos solo la parte antes de los dos puntos
    if ":" in localizacion:
        resultado = geocode(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        if resultado:
            return resultado.longitude, resultado.latitude
        return None, None

    # Para el formato "between", geocodificamos cada intersección y promediamos
    coords = []
    for intersection in extraer_intersecciones(localizacion, barrio):
        try:
            resultado = geocode(intersection)
            if resultado:
                coords.append((resultado.latitude, resultado.longitude))
        except Exception:
            continue

    if coords:
        # Centroide de todas las intersecciones encontradas
        return np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])
    return None, None


def api_nycopendata(df_paradas):
    """
    Extrae eventos públicos de NYC del día de hoy desde NYC Open Data,
    filtra por nivel de impacto en el tráfico, geocodifica y calcula
    paradas de metro afectadas.
    """
    urlbase = "https://data.cityofnewyork.us/resource/"
    url_eventos = f"{urlbase}tvpp-9vvx.json"

    load_dotenv()
    token = os.getenv('NYC_OPEN_DATA_TOKEN')
    assert token is not None, "Falta la variable de entorno NYC_OPEN_DATA_TOKEN"

    # Filtramos por el día de hoy completo
    fecha_hoy_str = datetime.now().strftime('%Y-%m-%d')
    param = {
        "$where": f"start_date_time >= '{desde_fecha(fecha_hoy_str)}' AND start_date_time <= '{hasta_fecha(fecha_hoy_str)}'",
    }

    header = {"X-App-Token": token}
    eventos = requests.get(url=url_eventos, params=param, headers=header)
    if eventos.status_code != 200:
        print(f"Error Parques: {eventos.text}")
    assert eventos.status_code == 200, "Error en la extracción de eventos"

    df = pd.DataFrame(eventos.json())
    if df.empty:
        return df

    # Convertimos las fechas a solo hora HH:MM
    df['start_date_time'] = pd.to_datetime(df['start_date_time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce').dt.strftime('%H:%M')
    df['end_date_time'] = pd.to_datetime(df['end_date_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f').dt.strftime('%H:%M')

    # Eliminamos columnas que no aportan valor al pipeline
    df = df.drop(["event_id", "event_agency", "street_closure_type", 'community_board',
                  'police_precinct', 'cemsid', 'event_street_side'], axis=1)

    # Mapeamos cada tipo de evento a un nivel de impacto en el tráfico del metro (1-10)
    riesgo_map = {
        'Parade': 10, 'Athletic Race / Tour': 10, 'Street Event': 8,
        'Stationary Demonstration': 7, 'Street Festival': 7, 'Special Event': 6,
        'Single Block Festival': 6, 'Bike the Block': 6, 'BID Multi-Block': 6,
        'Plaza Event': 6, 'Plaza Partner Event': 6, 'Block Party': 5,
        'Theater Load in and Load Outs': 5, 'Open Culture': 4, 'Religious Event': 3,
        'Press Conference': 3, 'Health Fair': 3, 'Rigging Permit': 3,
        'Farmers Market': 2, 'Sidewalk Sale': 2, 'Shooting Permit': 2,
        'Filming/Photography': 2, 'Open Street Partner Event': 2, 'Production Event': 1,
        'Sport - Adult': 1, 'Sport - Youth': 1, 'Miscellaneous': 1,
        'Stickball': 1, 'Clean-Up': 1,
    }

    df['nivel_riesgo_tipo'] = df['event_type'].map(riesgo_map).fillna(1)
    df = df.sort_values(by="nivel_riesgo_tipo", ascending=False)
    # Solo conservamos eventos con impacto alto (> 6)
    df = df[df.nivel_riesgo_tipo > 6]

    # Geocodificación con rate limiter para respetar los límites de Nominatim
    geolocator = Nominatim(user_agent="nyc_events_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

    df["coordenadas"] = df.apply(
        lambda row: list(extraer_coord(row["event_location"], row["event_borough"], geocode)), axis=1
    )

    # Descartamos eventos cuya ubicación no se pudo geocodificar
    coords_invalidas = df["coordenadas"].apply(lambda c: None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

    # Calculamos paradas afectadas para cada evento
    df["paradas_afectadas"] = df["coordenadas"].apply(
        lambda cor: obtener_paradas_afectadas(cor, df_paradas)
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)

    # Limpiamos columnas intermedias y renombramos para unificar con el resto
    df = df.drop(columns=["coordenadas", "event_location", "event_type", "event_borough"])
    df = df.rename(columns={
        'event_name': 'nombre_evento',
        'start_date_time': 'hora_inicio',
        'end_date_time': 'hora_salida_estimada',
    })

    return df


# ─────────────────────────────────────────────
#  ESPN (partidos de equipos NYC en casa)
# ─────────────────────────────────────────────

def extraer_scoreboard_espn(session, sport, fecha_gte, fecha_lte):
    """Llama al endpoint scoreboard de ESPN para un deporte y rango de fechas."""
    url = f"{BASE_URL_ESPN}/{sport}/scoreboard"
    respuesta = session.get(url, params={"dates": f"{fecha_gte}-{fecha_lte}"}, timeout=(10, 60))
    respuesta.raise_for_status()
    return respuesta.json()


def es_partido_en_casa_nyc(evento, equipos_nyc):
    """
    Comprueba si el equipo local (homeAway == 'home') es uno de los
    equipos NYC que nos interesan, comparando por slug y nombre.
    """
    for competidor in evento.get("competitions", [{}])[0].get("competitors", []):
        if competidor.get("homeAway") == "home":
            equipo = competidor.get("team", {})
            slug = equipo.get("slug", "").lower()
            nombre = equipo.get("displayName", "").lower()
            if any(nyc in slug or nyc in nombre for nyc in equipos_nyc):
                return True
    return False


def es_venue_nyc(competicion):
    """
    Verifica que el partido se juega en un estadio del área de NYC,
    comprobando tanto la ciudad como el nombre del venue.
    """
    if not competicion:
        return False
    venue = competicion.get("venue", {})
    ciudad = venue.get("address", {}).get("city", "")
    nombre_venue = venue.get("fullName", "")
    return (ciudad in CIUDADES_NYC) or (nombre_venue in VENUES_NYC)


def geocodificar_venue(nombre_venue, funcion_geocode):
    """
    Devuelve (latitud, longitud) de un venue. Primero busca en el diccionario
    VENUES_NYC para evitar llamadas innecesarias a la API de geocodificación.
    Si no está, lo geocodifica via Nominatim.
    """
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


def api_espn(df_paradas):
    """
    Extrae partidos en casa de equipos NYC para el día de hoy desde ESPN.
    Calcula hora de salida estimada según la duración del deporte y
    busca paradas de metro afectadas por cada estadio.
    """
    hoy = date.today()
    # ESPN usa formato YYYYMMDD, y como es solo hoy fecha_gte == fecha_lte
    fecha_gte = hoy.strftime("%Y%m%d")
    fecha_lte = fecha_gte

    session = requests.Session()
    geolocator = Nominatim(user_agent="espn_nyc_geocoder")
    funcion_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

    filas = []

    for sport, equipos in NYC_TEAMS.items():
        try:
            data = extraer_scoreboard_espn(session, sport, fecha_gte, fecha_lte)
            for ev in data.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                if es_partido_en_casa_nyc(ev, equipos) and es_venue_nyc(comp):
                    liga = sport.split("/")[1]
                    nombre_venue = comp.get("venue", {}).get("fullName", "")
                    latitud, longitud = geocodificar_venue(nombre_venue, funcion_geocode)

                    # Convertimos la fecha UTC del evento a hora local de NY
                    dt_ny = pd.to_datetime(ev.get("date")).tz_convert('America/New_York')
                    duracion = DURACIONES_ESPN.get(liga, 2.5)
                    hora_salida = (dt_ny + pd.to_timedelta(duracion, unit='h')).strftime('%H:%M')

                    coordinates = [longitud, latitud] if (longitud and latitud) else []

                    paradas = []
                    if coordinates and df_paradas is not None:
                        paradas = fusionar_lista_estaciones(
                            obtener_paradas_afectadas(coordinates, df_paradas)
                        )

                    filas.append({
                        'nombre_evento':        ev.get("name"),
                        'hora_inicio':          dt_ny.strftime('%H:%M'),
                        'hora_salida_estimada': hora_salida,
                        'score':                1.0,  # score fijo alto: partido en casa
                        'paradas_afectadas':    paradas,
                    })
        except Exception:
            pass

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────
#  Fusión final
# ─────────────────────────────────────────────

def fusionar_dataframes(df_seat_geek, df_nyc, df_espn):
    """
    Combina los DataFrames de las tres fuentes en uno solo.
    Normaliza los scores de cada fuente a una escala común y
    deduplica eventos que aparezcan en más de una fuente,
    conservando el score máximo y fusionando sus paradas afectadas.
    """
    dfs = []

    if df_seat_geek is not None and not df_seat_geek.empty:
        # Score de SeatGeek: media entre popularidad del evento y del venue (ya en escala 0-1)
        df_seat_geek['score'] = (df_seat_geek['popularidad_score'] + df_seat_geek['venue_score']) / 2
        df_seat_geek = df_seat_geek.drop(columns=['popularidad_score', 'venue_score', 'capacidad'])
        dfs.append(df_seat_geek)

    if df_nyc is not None and not df_nyc.empty:
        # Score de NYC Open Data: normalizamos el nivel de riesgo (1-10) a escala 0-1
        df_nyc['score'] = df_nyc['nivel_riesgo_tipo'] / 10
        df_nyc = df_nyc.drop(columns=['nivel_riesgo_tipo'])
        dfs.append(df_nyc)

    if df_espn is not None and not df_espn.empty:
        # ESPN ya viene con score = 1.0 fijo
        dfs.append(df_espn)

    if not dfs:
        return pd.DataFrame()

    # Concatenamos solo las columnas comunes a las tres fuentes
    cols_comunes = ['nombre_evento', 'hora_inicio', 'hora_salida_estimada', 'score', 'paradas_afectadas']
    df_final = pd.concat([d[cols_comunes] for d in dfs], ignore_index=True)

    def fusionar_grupo(grupo):
        """
        Para eventos duplicados (mismo nombre y hora de inicio en varias fuentes):
        - Conserva la primera hora de salida estimada
        - Se queda con el score más alto
        - Se queda con una
        """
        paradas_unidas = []
        for p in grupo['paradas_afectadas']:
            if isinstance(p, list):
                paradas_unidas.extend(p)
        return pd.Series({
            'hora_salida_estimada': grupo['hora_salida_estimada'].iloc[0],
            'score':                grupo['score'].max(),
            'paradas_afectadas':    fusionar_lista_estaciones(paradas_unidas),
        })

    # Agrupamos por nombre y hora de inicio para detectar duplicados entre fuentes
    df_final = (
        df_final
        .groupby(['nombre_evento', 'hora_inicio'], as_index=False)
        .apply(fusionar_grupo, include_groups=False)
        .reset_index(drop=True)
    )

    # Ordenamos por score descendente para que los eventos más relevantes queden primero
    df_final = df_final.sort_values('score', ascending=False).reset_index(drop=True)

    return df_final


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()

    print("\nCargando paradas de metro desde el CSV...")
    df_paradas = cargar_paradas_df()

    if df_paradas is not None:
        df_seat_geek = None
        df_nyc = None
        df_espn = None

        # Cada fuente se extrae de forma independiente para que un fallo
        # en una no impida obtener datos de las demás
        try:
            print("\nExtrayendo eventos de SeatGeek...")
            df_seat_geek = api_seatgeek(df_paradas)
            print(f"  {len(df_seat_geek)} eventos extraídos de SeatGeek")
        except Exception as e:
            print(f"  Error en SeatGeek: {e}")

        try:
            print("\nExtrayendo eventos de NYC Open Data...")
            df_nyc = api_nycopendata(df_paradas)
            print(f"  {len(df_nyc)} eventos extraídos de NYC Open Data")
        except Exception as e:
            print(f"  Error en NYC Open Data: {e}")

        try:
            print("\nExtrayendo partidos ESPN (equipos NYC en casa)...")
            df_espn = api_espn(df_paradas)
            print(f"  {len(df_espn)} partidos extraídos de ESPN")
        except Exception as e:
            print(f"  Error en ESPN: {e}")

        df_final = fusionar_dataframes(df_seat_geek, df_nyc, df_espn)
        print(df_final)
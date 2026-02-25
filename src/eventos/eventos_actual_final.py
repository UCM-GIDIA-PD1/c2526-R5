import os
import calendar
import requests
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ─────────────────────────────────────────────
#  Constantes ESPN
# ─────────────────────────────────────────────
BASE_URL_ESPN = "https://site.api.espn.com/apis/site/v2/sports"

NYC_TEAMS = {
    'basketball/nba': ['knicks', 'nets'],
    'baseball/mlb':   ['yankees', 'mets'],
    'hockey/nhl':     ['rangers', 'islanders', 'devils'],
    'football/nfl':   ['giants', 'jets'],
    'soccer/usa.1':   ['new-york-city-fc', 'new-york-red-bulls'],
}

DURACIONES_ESPN = {
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
#  Helpers compartidos y Cálculo Espacial
# ─────────────────────────────────────────────
def fusionar_lista_estaciones(lista_tuplas):
    """Fusiona líneas con el mismo nombre de estación."""
    if not isinstance(lista_tuplas, list):
        return lista_tuplas

    estaciones_fusionadas = defaultdict(set)
    for nombre, lineas in lista_tuplas:
        estaciones_fusionadas[nombre].update(str(lineas).split())

    return [
        (nombre, " ".join(sorted(lineas_set)))
        for nombre, lineas_set in estaciones_fusionadas.items()
    ]

def cargar_paradas_df():
    """Descarga el CSV del metro de NY y lo prepara como DataFrame."""
    url = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"
    try:
        df = pd.read_csv(url)
        columnas_utiles = ['Stop Name', 'Daytime Routes', 'GTFS Longitude', 'GTFS Latitude']
        df_limpio = df[columnas_utiles].copy()
        
        df_limpio = df_limpio.rename(columns={
            'Stop Name': 'nombre',
            'Daytime Routes': 'lineas',
            'GTFS Longitude': 'lon',
            'GTFS Latitude': 'lat'
        })
        return df_limpio
    except Exception as e:
        print(f"Error descargando paradas: {e}")
        return None

def obtener_paradas_afectadas(coords, df_paradas, max_metros=500):
    """Calcula la distancia Haversine y devuelve paradas a menos de max_metros."""
    if not coords or None in coords or df_paradas is None or df_paradas.empty:
        return []
    
    lon_evento, lat_evento = coords
    
    # Haversine vectorizado con NumPy
    lat1, lon1 = np.radians(lat_evento), np.radians(lon_evento)
    lat2, lon2 = np.radians(df_paradas['lat']), np.radians(df_paradas['lon'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radio de la Tierra en metros
    
    distancias = c * r
    
    # Filtramos las que están dentro del radio (500 metros)
    cercanas = df_paradas[distancias <= max_metros]
    return [(row['nombre'], row['lineas']) for _, row in cercanas.iterrows()]


# ─────────────────────────────────────────────
#  SeatGeek
# ─────────────────────────────────────────────
def extraccion_actual(fecha, CLIENT_ID, manana):
    url = "https://api.seatgeek.com/2/events"
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
    horas_duracion = tiempos_salida.get(fila['tipo'], 2.5)
    hora_fin = pd.to_datetime(fila['hora_inicio']) + timedelta(hours=horas_duracion)
    return hora_fin.strftime('%H:%M')


def api_seatgeek(df_paradas):
    fecha_hoy_obj = datetime.now()
    manana_obj = fecha_hoy_obj + timedelta(days=1)
    fecha_hoy_str = fecha_hoy_obj.strftime('%Y-%m-%d')
    manana_str = manana_obj.strftime('%Y-%m-%d')

    API_KEY = os.getenv('CLIENT_ID_SEATGEEK')
    assert API_KEY is not None, "Falta la variable de entorno CLIENT_ID_SEATGEEK"

    TIPOS_CONCIERTO = {'concert', 'music_festival', 'classical', 'opera', 'ballet'}

    data = extraccion_actual(fecha_hoy_str, API_KEY, manana_str)
    eventos_limpios = []

    for e in data['events']:
        if e.get('type') not in TIPOS_CONCIERTO:
            continue
        eventos_limpios.append({
            'nombre_evento':    e.get('title'),
            'tipo':             e.get('type'),
            'hora_inicio':      e.get('datetime_local'),
            'lugar':            e['venue'].get('name'),
            'direccion':        e['venue'].get('address', 'Dirección no disponible'),
            'latitud':          e['venue']['location'].get('lat'),
            'longitud':         e['venue']['location'].get('lon'),
            'capacidad':        e['venue'].get('capacity'),
            'popularidad_score': e.get('score'),
            'venue_score':      e['venue'].get('score'),
        })

    df = pd.DataFrame(eventos_limpios)
    if df.empty:
        return df

    df['capacidad'] = df['capacidad'].replace(0, np.nan)

    df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
    df['hora_inicio_str'] = df['hora_inicio'].dt.strftime('%H:%M')

    tiempos_salida = {
        'concert': 3.0, 'music_festival': 8.0, 'classical': 2.5,
        'opera': 3.0, 'ballet': 2.5,
    }

    df['hora_salida_estimada'] = df.apply(lambda fila: calcular_salida(fila, tiempos_salida), axis=1)
    df['hora_inicio'] = df['hora_inicio_str']
    df = df.drop(columns=['hora_inicio_str'])

    df["coordinates"] = df.apply(lambda fila: [fila['longitud'], fila['latitud']], axis=1)
    df = df.drop(['longitud', 'latitud', 'lugar', 'direccion'], axis=1)

    coords_invalidas = df["coordinates"].apply(lambda c: c == [0, 0] or None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

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
    return f'{fecha_str}T00:00:00.000'


def hasta_fecha(fecha_str):
    return f'{fecha_str}T23:59:59.000'


def extraer_intersecciones(localizacion, barrio):
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
    return intersecciones if intersecciones else [localizacion + f", {barrio}, New York"]


def extraer_coord(localizacion, barrio, geocode):
    if pd.isna(localizacion):
        return None, None

    if ":" in localizacion:
        resultado = geocode(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        if resultado:
            return resultado.longitude, resultado.latitude
        return None, None

    coords = []
    for intersection in extraer_intersecciones(localizacion, barrio):
        try:
            resultado = geocode(intersection)
            if resultado:
                coords.append((resultado.latitude, resultado.longitude))
        except Exception:
            continue

    if coords:
        return np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])
    return None, None


def api_nycopendata(df_paradas):
    urlbase = "https://data.cityofnewyork.us/resource/"
    url_eventos = f"{urlbase}tvpp-9vvx.json"

    load_dotenv()
    token = os.getenv('NYC_OPEN_DATA_TOKEN')
    assert token is not None, "Falta la variable de entorno NYC_OPEN_DATA_TOKEN"

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

    df['start_date_time'] = pd.to_datetime(df['start_date_time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce').dt.strftime('%H:%M')
    df['end_date_time'] = pd.to_datetime(df['end_date_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f').dt.strftime('%H:%M')

    df = df.drop(["event_id", "event_agency", "street_closure_type", 'community_board',
                  'police_precinct', 'cemsid', 'event_street_side'], axis=1)

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
    df = df[df.nivel_riesgo_tipo > 6]

    geolocator = Nominatim(user_agent="nyc_events_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

    df["coordenadas"] = df.apply(
        lambda row: list(extraer_coord(row["event_location"], row["event_borough"], geocode)), axis=1
    )

    coords_invalidas = df["coordenadas"].apply(lambda c: None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

    df["paradas_afectadas"] = df["coordenadas"].apply(
        lambda cor: obtener_paradas_afectadas(cor, df_paradas)
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)
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
    url = f"{BASE_URL_ESPN}/{sport}/scoreboard"
    respuesta = session.get(url, params={"dates": f"{fecha_gte}-{fecha_lte}"}, timeout=(10, 60))
    respuesta.raise_for_status()
    return respuesta.json()


def es_partido_en_casa_nyc(evento, equipos_nyc):
    for competidor in evento.get("competitions", [{}])[0].get("competitors", []):
        if competidor.get("homeAway") == "home":
            equipo = competidor.get("team", {})
            slug = equipo.get("slug", "").lower()
            nombre = equipo.get("displayName", "").lower()
            if any(nyc in slug or nyc in nombre for nyc in equipos_nyc):
                return True
    return False


def es_venue_nyc(competicion):
    if not competicion:
        return False
    venue = competicion.get("venue", {})
    ciudad = venue.get("address", {}).get("city", "")
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


def api_espn(df_paradas):
    """Extrae partidos en casa de equipos NYC para el día de hoy desde ESPN."""
    hoy = date.today()
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
                        'nombre_evento':       ev.get("name"),
                        'hora_inicio':         dt_ny.strftime('%H:%M'),
                        'hora_salida_estimada': hora_salida,
                        'score':               1.0,   # score fijo alto: partido en casa
                        'paradas_afectadas':   paradas,
                    })
        except Exception:
            pass

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────
#  Fusión final
# ─────────────────────────────────────────────
def fusionar_dataframes(df_seat_geek, df_nyc, df_espn):
    dfs = []

    if df_seat_geek is not None and not df_seat_geek.empty:
        df_seat_geek['score'] = (df_seat_geek['popularidad_score'] + df_seat_geek['venue_score']) / 2
        df_seat_geek = df_seat_geek.drop(columns=['popularidad_score', 'venue_score', 'capacidad'])
        dfs.append(df_seat_geek)

    if df_nyc is not None and not df_nyc.empty:
        df_nyc['score'] = df_nyc['nivel_riesgo_tipo'] / 10
        df_nyc = df_nyc.drop(columns=['nivel_riesgo_tipo'])
        dfs.append(df_nyc)

    if df_espn is not None and not df_espn.empty:
        dfs.append(df_espn)

    if not dfs:
        return pd.DataFrame()

    cols_comunes = ['nombre_evento', 'hora_inicio', 'hora_salida_estimada', 'score', 'paradas_afectadas']
    df_final = pd.concat([d[cols_comunes] for d in dfs], ignore_index=True)

    # Fusionar duplicados con mismo nombre y hora de inicio
    def fusionar_grupo(grupo):
        paradas_unidas = []
        for p in grupo['paradas_afectadas']:
            if isinstance(p, list):
                paradas_unidas.extend(p)
        return pd.Series({
            'hora_salida_estimada': grupo['hora_salida_estimada'].iloc[0],
            'score':                grupo['score'].max(),
            'paradas_afectadas':    fusionar_lista_estaciones(paradas_unidas),
        })

    df_final = (
        df_final
        .groupby(['nombre_evento', 'hora_inicio'], as_index=False)
        .apply(fusionar_grupo, include_groups=False)
        .reset_index(drop=True)
    )

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
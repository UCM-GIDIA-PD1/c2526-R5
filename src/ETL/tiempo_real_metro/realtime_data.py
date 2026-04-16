"""
Script para la extracción y procesamiento de datos en tiempo real del metro
de Nueva York (MTA) con el objetivo de calcular el retraso de los trenes
respecto a sus horarios previstos.

FUENTES DE DATOS:
    - API GTFS-Realtime MTA: tiempos de llegada/salida actuales de los trenes
      para todas las líneas del metro de Nueva York (A/C/E, B/D/F/M, G, J/Z,
      N/Q/R/W, L, 1-7/S y SIR).
    - GTFS Supplemented (S3 MTA): horarios previstos oficiales de cada tren
      en cada parada, descargado automáticamente desde un ZIP en la nube.

PROCESO:
    1. Se extraen los datos en tiempo real de la API para cada línea y se
       construye un DataFrame con el viaje, parada, hora de llegada/salida
       real y timestamp de la extracción.
    2. Se descargan los horarios previstos y se adaptan para que sean
       compatibles con los datos en tiempo real.
    3. Se cruzan ambos DataFrames y se calcula el retraso en segundos,
       filtrando predicciones futuras y ajustando viajes que cruzan la
       medianoche.

OUTPUT:
    DataFrame con el retraso real (en segundos) de cada tren en cada parada,
    junto con información del viaje, línea, dirección y tipo de día.
"""


import requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from google.transit import gtfs_realtime_pb2
import urllib.request
import zipfile
import io
import math
import time
from src.ETL.tiempo_real_metro.aggregations import agregar_por_ventana


# ─────────────────────────────────────────────
#  Fuentes de datos MTA Real Time
# ─────────────────────────────────────────────
FUENTES = {
    "ACES": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
        "lineas": ["A", "C", "E", "Sr"]
    },
    "BDFMS": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
        "lineas": ["B", "D", "F", "M", "Sf"]
    },
    "G": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
        "lineas": ["G"]
    },
    "JZ": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz",
        "lineas": ["J", "Z"]
    },
    "NQRW": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
        "lineas": ["N", "Q", "R", "W"]
    },
    "L": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
        "lineas": ["L"]
    },
    "1234567S": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
        "lineas": ["1", "2", "3", "4", "5", "6", "7", "S"]
    },
    "SIR": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
        "lineas": ["SIR"]
    }
}


# ─────────────────────────────────────────────
#  Datos a DataFrame
# ─────────────────────────────────────────────

def extraccion_linea(url, linea, reintentos=3):
    """
    Extrae los datos de una línea
    """

    for intento in range(reintentos):
        try:
            response = requests.get(url, timeout = 10)
            fuentes = gtfs_realtime_pb2.FeedMessage()
            fuentes.ParseFromString(response.content)

            datos_linea = []
            for entity in fuentes.entity:
                if entity.HasField('trip_update'):
                    trayecto = entity.trip_update

                    if trayecto.trip.route_id == linea:
                        for stop in trayecto.stop_time_update:
                            campos = {
                                'viaje_id': trayecto.trip.trip_id,
                                'linea_id': trayecto.trip.route_id,
                                'parada_id': stop.stop_id,
                                'hora_llegada': datetime.fromtimestamp(stop.arrival.time, tz=timezone.utc) if stop.HasField('arrival') else None,
                                'hora_partida': datetime.fromtimestamp(stop.departure.time, tz=timezone.utc) if stop.HasField('departure') else None,
                                'timestamp': datetime.now(tz=timezone.utc),                       
                            }

                            datos_linea.append(campos)
            return datos_linea
        
        except Exception as e:
            if intento == reintentos - 1:
                print(f"  [ERROR] Línea {linea} fallida tras {reintentos} intentos: {e}")
                return []
            espera = 2 ** intento
            print(f"  [WARN] Línea {linea} intento {intento + 1} fallido, reintentando en {espera}s...")
            time.sleep(espera)


def extraccion_datos():
    """
    Repite la función anterior para cada linea y unifica la información
    de cada una de ellas en una dataframe
    """

    todas_las_lineas = []
    for info in FUENTES.values():
        todas_las_lineas.extend(info['lineas'])
    
    todos_los_datos = []
    for linea in todas_las_lineas:
        for grupo, info in FUENTES.items():
            if linea in info['lineas']:
                fuentes_url = info['url']
            todos_los_datos.extend(extraccion_linea(fuentes_url, linea))  

    return pd.DataFrame(todos_los_datos)


# ─────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────


def conversion_hora_NYC(df):

    """
    Para las variables de tipo datetime, modifica el valor a la hora local de NY
    """
    
    for col in ['hora_llegada', 'hora_partida', 'timestamp']:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert('America/New_York')
    return df

def dia_segun_fecha_y_formato(df):

    """
    Según el dia en el que se ha hecho la extracción, crea una nueva variable
    que lo clasifica en 3 grupos (Weekday, Saturday, Sunday).

    También se extrae si es fin de semana y que día de la semana es numéricamente

    Posteriormente cambia el formato de las horas y lo convierte a string.
    """

    df['dia'] = df['timestamp'].dt.strftime("%A").apply(
        lambda x: 'Weekday' if x not in ('Saturday', 'Sunday') else x
    )

    df['dow']        = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    return df

def direccion_tren(df):

    """
    Según el id de cada parada, se crea una nueva columna que contiene la dirección del tren (0,1)
    """

    norte = (df['parada_id'].str[-1] == 'N')
    sur = (df['parada_id'].str[-1] == 'S')

    df.loc[norte, 'direccion'] = 1 #Dirección Norte
    df.loc[sur, 'direccion'] = 0 #Dirección Sur

    df['direccion'] = df['direccion'].astype('Int64')

    return df


def normalizar_horas(columna):

    """
    Para horas mayores a 24 horas, se convierte a hora del día siguiente
    """
    def ajustar(hora):
        if pd.isna(hora):
            return hora
        partes = hora.split(':')
        h = int(partes[0]) % 24
        return f"{h:02d}:{partes[1]}:{partes[2]}"
 
    return columna.apply(ajustar)

def hora_a_segundos(hora):

    """
    Dado un string con una hora, se calculan los segundos totales
    """
    if pd.isna(hora): 
        return np.nan
    
    partes = hora.split(':')

    return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])


def hora_posterior(hora1, hora2):

    """
    Comprueba si la hora dada como primer parámetro es mayor
    segunda.
    """
    
    s1 = hora_a_segundos(hora1)
    s2 = hora_a_segundos(hora2)
    dif = s1 - s2

    # Tenemos en cuenta posibles primeras horas del día siguiente (00:15) y
    # asumimos que si la diferencia supera las 12 horas es cruce de media noche
    if dif > 43200:   
        dif -= 86400
    elif dif < -43200:
        dif += 86400

    return dif > 0

def filter_delay_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro suave de outliers: delays fuera de +/- 2.5h suelen ser ruido (pero ajustable)
    """
    antes = len(df)
    df = df[df["delay"].between(-9000, 9000)]
    descartadas = antes - len(df)
    if descartadas:
        print(f"  Outliers de delay eliminados: {descartadas} filas ({descartadas/antes*100:.1f}%)")
    return df


def hora_ciclica(df):
    """Codifica la hora de llegada como coordenadas cíclicas (sin/cos).""" 
    hour_float = df["hora_llegada"].dt.hour.astype(float)
    df["hour_sin"] = hour_float.apply(lambda h: math.sin(2 * math.pi * h / 24) if pd.notna(h) else None)
    df["hour_cos"] = hour_float.apply(lambda h: math.cos(2 * math.pi * h / 24) if pd.notna(h) else None)

    return df


# ─────────────────────────────────────────────
#  DataFrame tiempo real
# ─────────────────────────────────────────────


def creacion_df_tiempo_real():

    """
    Creación de dataframe de tiempo real
    """
    df = extraccion_datos()

    if df.empty:
        raise ValueError("No se obtuvieron datos de tiempo real de ninguna línea.")
    
    df = conversion_hora_NYC(df)
    df = dia_segun_fecha_y_formato(df)
    df = direccion_tren(df)
    df = df.dropna()

    #Eliminación de filas con nulos en alguna columna
    df = df.dropna()

    df['segundos_reales'] = (df['hora_llegada'].dt.hour * 3600 + 
                             df['hora_llegada'].dt.minute * 60 + 
                             df['hora_llegada'].dt.second)
    print(f"  DataFrame tiempo real: {len(df)} filas, {df['linea_id'].nunique()} líneas")

    return df


# ─────────────────────────────────────────────
#  DataFrame horarios previstos
# ─────────────────────────────────────────────


def creacion_df_previsto():

    """
    Creación de dataframe de horarios previstos
    """

    url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"

    with urllib.request.urlopen(url) as response:
        zip_data = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_data, 'r') as z:
        with z.open("stop_times.txt") as f:
            df = pd.read_csv(f)

    #Día en el que se lleva a cabo el servivio, viene dado como parte del trip_id
    df['day'] = df['trip_id'].str.split('-').str[-2]

    #Modificamos trip_id para que tenga el mismo formato que el id del otro dataframe
    df['trip_id'] = df['trip_id'].str.split('_', n=1).str[-1]

    df['arrival_time'] = normalizar_horas(df['arrival_time'])
    df['departure_time'] = normalizar_horas(df['departure_time'])

    df['segundos_previstos'] = df['arrival_time'].apply(hora_a_segundos)

    print(f"  DataFrame previsto: {len(df)} filas")

    return df


# ─────────────────────────────────────────────
#  Unión DataFrames
# ─────────────────────────────────────────────

def union_dataframes(df1, df2):

    """
    Une los DataFrames de tiempo real y horarios previstos, calcula el delay
    y aplica las transformaciones finales.
    """

    df = pd.merge(df1, df2, left_on=['viaje_id', 'parada_id', 'dia'], right_on=['trip_id', 'stop_id', 'day'])

    
    #Calcula el retraso de los trenes restando el tiempo de llegada actual menos el tiempo de llegada previsto
    df['delay'] = df['segundos_reales']-df['segundos_previstos']

    #Ajuste para viajes que deberían llegar al final del día (23:00) pero por retraso llega al día siguiente
    df.loc[df['delay'] > 43200, 'delay'] -= 86400
    df.loc[df['delay'] < -43200, 'delay'] += 86400

    #Comprueba que los datos dados son de trenes que ya han realizado sus paradas y no son predicciones que realiza la
    # api para el futuro de los trayectos. Los que son predicciones marcamos el delay a None
    df['delay'] = np.where(
        df['timestamp'] >= df['hora_llegada'],
        df['delay'],  
        np.nan    
    )

    df = df.dropna(subset=['delay'])
    #Filtro para delays con valores masivos y transformacion del día de la semana a valor numérico
    df = filter_delay_outliers(df)
    df = hora_ciclica(df)

    columnas_a_eliminar = [
        'dia', 'hora_partida', 'timestamp',
        'segundos_reales', 'trip_id', 'stop_id', 'stop_sequence',
        'arrival_time', 'departure_time', 'day', 'segundos_previstos'
    ]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    df = df.dropna()

    return df


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    df_real_time = None
    df_previsto = None
    
    try:
        print("\nExtrayendo horarios de trenes en tiempo real...")
        df_real_time = creacion_df_tiempo_real()
    except Exception as e:
        print(f"  Error en datos tiempo real: {e}")

    try:
        print("\nExtrayendo horarios de trenes previstos...")
        df_previsto = creacion_df_previsto()
    except Exception as e:
        print(f"  Error en datos previstos: {e}")

    if df_real_time is None or df_previsto is None:
        print("\n[FATAL] No se puede continuar: uno o ambos DataFrames no se pudieron obtener.")
        exit(1)

    try:
        print("\nUniendo DataFrames...")
        df_final = union_dataframes(df_real_time, df_previsto)
        df_agregado = agregar_por_ventana(df_final)
    except Exception as e:
        print(f"  [ERROR] Unión de DataFrames: {e}")
        exit(1)

    print(df_agregado)
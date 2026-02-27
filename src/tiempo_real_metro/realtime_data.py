import requests
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from google.transit import gtfs_realtime_pb2
import urllib.request
import zipfile
import io


"""
Para poder crear el segundo dataframe es necesario tener descargado un archivo de las
paradas previstas de los trenes. Este se puede encontrar con este link: https://www.mta.info/developers
Habría que descargar el Supplemented GTFS y se nos descargará un zip. El que nos interesa se llama: stop_times.txt
"""


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

def extraccion_linea(url, linea):
    """
    Extrae los datos de una línea
    """
    response = requests.get(url)
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
                        'hora_llegada': datetime.fromtimestamp(stop.arrival.time) if stop.HasField('arrival') else None,
                        'hora_partida': datetime.fromtimestamp(stop.departure.time) if stop.HasField('departure') else None,
                        'timestamp': datetime.now(),                       
                    }

                    datos_linea.append(campos)
    return datos_linea


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
    
    df['hora_llegada'] = df['hora_llegada'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')  
    df['hora_partida'] = df['hora_partida'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

    return df

def dia_segun_fecha_y_formato(df):

    """
    Según el dia en el que se ha hecho la extracción, crea una nueva variable
    que lo clasifica en 3 grupos (Weekday, Saturday, Sunday).

    Posteriormente cambia el formato de las horas y lo convierte a string.
    """

    df['dia'] = df['timestamp'].dt.strftime("%A")
    df['dia'] = df['dia'].apply(
        lambda x: 'Weekday' if x not in ('Saturday', 'Sunday') else x
    )

    df['hora_llegada'] = df['hora_llegada'].dt.strftime('%H:%M:%S')
    df['hora_partida'] = df['hora_partida'].dt.strftime('%H:%M:%S')
    df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')

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
    columna = columna.str.replace('24:', '00:', regex=False)
    columna = columna.str.replace('25:', '01:', regex=False)
    columna = columna.str.replace('26:', '02:', regex=False)
    columna = columna.str.replace('27:', '03:', regex=False)
    return columna

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
    
    partes1 = hora1.split(':')
    partes2 = hora2.split(':')

    return (
        (int(partes1[0]) > int(partes2[0])) | 
        ((int(partes1[0]) == int(partes2[0])) & (int(partes1[1]) > int(partes2[1]))) |
        ((int(partes1[0]) == int(partes2[0])) & (int(partes1[1]) == int(partes2[1])) & ((int(partes1[2]) > int(partes2[2]))))
    )


# ─────────────────────────────────────────────
#  DataFrame tiempo real
# ─────────────────────────────────────────────


def creacion_df_tiempo_real():

    """
    Creación de dataframe de tiempo real
    """

    df = extraccion_datos()
    conversion_hora_NYC(df)
    dia_segun_fecha_y_formato(df)
    direccion_tren(df)

    #Eliminación de filas con nulos en alguna columna
    df = df.dropna()

    df['segundos_reales'] = df['hora_llegada'].apply(hora_a_segundos)

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

    return df


# ─────────────────────────────────────────────
#  Unión DataFrames
# ─────────────────────────────────────────────

def union_dataframes(df1, df2):

    """
    Une los dos dataframes anteriores
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
    df.apply(lambda row: hora_posterior(row['timestamp'], row['hora_llegada']), axis=1),
    df['delay'],  # valor si True
    None    # valor si False
    )

    df = df.drop(['timestamp', 'segundos_reales', 'trip_id', 'stop_id', 'arrival_time', 'departure_time', 'day', 'segundos_previstos'], axis=1)
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

    df_final = union_dataframes(df_real_time, df_previsto)
    print(df_final)
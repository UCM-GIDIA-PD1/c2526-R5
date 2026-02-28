from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry
import os
import io

def extraccion(fechaini,fechafin):
    url = "https://archive-api.open-meteo.com/v1/archive"
    #cache_session = requests_cache.CachedSession('.cache', expire_after=-1) #Guarda en un archivo local .cache para no tener que pedirlo de nuevo
    #retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    parametros = {
        "latitude": 40.47,
        "longitude" : -73.58, #coordenadas de Central Park
        "start_date" : fechaini,
        "end_date" : fechafin,
        "hourly" : ["temperature_2m", "rain", "precipitation", "wind_speed_10m", "snowfall", "cloud_cover"],
        "timezone" : "auto"
    }
    try:
        respuestas = openmeteo.weather_api(url, params = parametros)
        df = transformar_a_df(respuestas, )
        return df
    except Exception as e:
        print("Error", e)

def transformar_a_df(respuestas):
    respuesta = respuestas[0]
    hourly = respuesta.Hourly()
    temp_array = hourly.Variables(0).ValuesAsNumpy()
    lluvia_array = hourly.Variables(1).ValuesAsNumpy()
    prec_array = hourly.Variables(2).ValuesAsNumpy()
    wind_speed_array = hourly.Variables(3).ValuesAsNumpy()
    nieve_array = hourly.Variables(4).ValuesAsNumpy()
    cloud_cover_array = hourly.Variables(5).ValuesAsNumpy()




    dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
    )
    
    datos = {
        "Date" : dates,
        "Temperature" : temp_array,
        "Rain" : lluvia_array,
        "Precipitation": prec_array,
        "Wind Speed" : wind_speed_array,
        "Snow" : nieve_array,
        "Cloud Cover" : cloud_cover_array
    }
    df = pd.DataFrame(datos)
    return df

def separar_dias(df):
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    assert ACCESS_KEY is not None, 'La variable de entorno MINIO_ACCESS_KEY no está definida.'
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    assert SECRET_KEY is not None, 'La variable de entorno MINIO_SECRET_KEY no está definida.'
    from minio import Minio
    client = Minio(endpoint='minio.fdi.ucm.es', access_key=ACCESS_KEY, secret_key=SECRET_KEY)
    for dia, df_dia in df.groupby(df['Date'].dt.date):
        print(df_dia)
        subir_a_MinIO(dia, df_dia, client)
    print("Todo subido con exito")
        
def subir_a_MinIO(dia, df_dia, client):
    buffer = io.BytesIO()
    df_dia.to_parquet(buffer)
    name = 'grupo5/processed/Clima/Clima_Historico/' + str(dia) + '/Clima_Historico_' + str(dia) +'.parquet'
    buffer.seek(0)  # Volver al inicio del buffer para que se lea correctamente
    client.put_object(bucket_name='pd1', object_name=name,
    data=buffer, length=buffer.getbuffer().nbytes, content_type='application/octet-stream')
    print("Archivo subido con exito a" + name)

<<<<<<< HEAD:src/clima/Extraccion_clima_historico.py
def extraccion_historico(fechaini , fechafin):
    return extraccion(fechaini , fechafin)
def ingest_climas_historicos(fechaini = "2024-12-31", fechafin = "2026-01-01"):
=======
def extraccion_historico(fechaini = "2024-12-31", fechafin = "2026-01-01"):
    return extraccion(fechaini, fechafin)

def ingest_clima_historico(fechaini, fechafin):
>>>>>>> 02751b9869a40ad701848de6bdb8ae8291c4280e:src/clima/extraccion_historico_clima.py
    df_historico = extraccion_historico(fechaini, fechafin)
    separar_dias(df_historico)
    
if __name__ == "__main__":
    ingest_climas_historicos()
    

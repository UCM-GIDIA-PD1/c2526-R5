from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry

def extraccion(url, fechaini,fechafin):
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

def extraccion_historico(fechaini = "2025-01-01", fechafin = "2026-01-01"):
    url_historico = "https://archive-api.open-meteo.com/v1/archive"
    return extraccion(url_historico, fechaini, fechafin )

def extraccion_actual():
    url_actual = "https://api.open-meteo.com/v1/forecast"
    return extraccion(url_actual, )

if __name__ == "__main__":
    
    
    df_historico = extraccion_historico()
    df_actual = extraccion()
    print(df)

import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry

#cache_session = requests_cache.CachedSession('.cache', expire_after=-1) #Guarda en un archivo local .cache para no tener que pedirlo de nuevo
#retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

session = requests.Session()
retry_session = retry(session, retries=5, backoff_factor=0.2)

openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

parametros = {
    "latitude": 40.47,
    "longitude" : -73.58, #coordenadas de Central Park
    "start_date" : "2025-01-01",
    "end_date" : "2026-01-01",
    "hourly" : "temperature_2m,rain,snowfall,precipitation,snow_depth,visibility",
    "timezone" : "auto"
}
try:
    respuestas = openmeteo.weather_api(url, params = parametros)
    respuesta = respuestas[0]
    hourly = respuesta.Hourly()
    temp_array = hourly.Variables(0).ValuesAsNumpy()
    lluvia_array = hourly.Variables(1).ValuesAsNumpy()
    nieve_array = hourly.Variables(2).ValuesAsNumpy()
    prec_array = hourly.Variables(3).ValuesAsNumpy()
    nieve_prof_array = hourly.Variables(4).ValuesAsNumpy()
    visib_array = hourly.Variables(5).ValuesAsNumpy()




    dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
    )
    
    datos = {
        "Fecha" : dates,
        "Temperatura" : temp_array,
        "Lluvia" : lluvia_array,
        "Nieve": nieve_array,
        "Precipitacion" : prec_array,
        "Profundidad de nieve" : nieve_prof_array,
        "Visibilidad" : visib_array
    }
    df = pd.DataFrame(datos)
    
    for columna in df.columns:
        nulos = df[columna].isnull().sum()
        print(columna, " tiene ", nulos, "con porcentaje de nulos del ", nulos / df[columna].count())

    df.to_parquet("clima_2025_2_.parquet", index = False)
    print("Guardado")
except Exception as e:
    print("Error", e)

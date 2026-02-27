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

url = "https://archive-api.open-meteo.com/v1/archive"

parametros = {
    "latitude": 40.47,
    "longitude" : -73.58, #coordenadas de Central Park
    "start_date" : "2025-01-01",
    "end_date" : "2026-01-01",
    "hourly" : ["temperature_2m", "rain", "precipitation", "wind_speed_10m", "snowfall", "cloud_cover"],
    "timezone" : "auto"
}
try:
    respuestas = openmeteo.weather_api(url, params = parametros)
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
    print(df)
    
    import os
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    assert ACCESS_KEY is not None, 'La variable de entorno MINIO_ACCESS_KEY no está definida.'
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    assert SECRET_KEY is not None, 'La variable de entorno MINIO_SECRET_KEY no está definida.'

    from minio import Minio
    client = Minio(endpoint='minio.fdi.ucm.es', access_key=ACCESS_KEY, secret_key=SECRET_KEY)
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)  # Volver al inicio del buffer para que se lea correctamente
    client.put_object(bucket_name='pd1', object_name='grupo5/processed/Clima/ClimaHistorico2025.parquet',
    data=buffer, length=buffer.getbuffer().nbytes, content_type='application/octet-stream')
    print(f'DataFrame guardado como parquet y subido a MinIO como grupo5/processed/Clima/ClimaHistorico2025.parquet en el bucket pd1.')
    




except Exception as e:
    print("Error", e)

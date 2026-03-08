'''
clima_realtime.py - Extracción de datos del clima actual
Fuente: https://api.open-meteo.com/v1/forecast
Destino: grupo5/processed/Clima/DataFrame_Clima_TiempoReal.parquet

'''




import openmeteo_requests

import pandas as pd
import requests_cache
import requests
from retry_requests import retry
import datetime
import io
import os

import sys
from src.ETL.common.minio_client import upload_df_parquet



def extraer_clima_actual():
	# Setup the Open-Meteo API client with cache and retry on error
	#cache_session = requests_cache.CachedSession('.cache', expire_after = 3600) #Crea un .cache para no tener que solicitar varias veces lo mismo
	#retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)

	session = requests.Session()
	retry_session = retry(session, retries=5, backoff_factor=0.2) #solicita sin importar lo que se ha solicitado antes

	openmeteo = openmeteo_requests.Client(session = retry_session)


	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": 40.47,
		"longitude": -73.58,
		"hourly": ["temperature_2m", "rain", "precipitation", "wind_speed_10m", "snowfall", "cloud_cover"],
		"current": ["wind_speed_10m", "temperature_2m", "precipitation", "rain", "snowfall", "cloud_cover"],
	}
	responses = openmeteo.weather_api(url, params=params)


	response = responses[0]
	print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation: {response.Elevation()} m asl")
	print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")


	current = response.Current()
	current_wind_speed_10m = current.Variables(0).Value()
	current_temperature_2m = current.Variables(1).Value()
	current_precipitation = current.Variables(2).Value()
	current_rain = current.Variables(3).Value()
	current_snowfall = current.Variables(4).Value()
	current_cloud_cover = current.Variables(5).Value()
	print(current_snowfall)
	current_time = current.Time()
	fecha_utc = datetime.datetime.fromtimestamp(current_time, datetime.timezone.utc)


	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_rain = hourly.Variables(1).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
	hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["rain"] = hourly_rain
	hourly_data["precipitation"] = hourly_precipitation
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
	hourly_data["snowfall"] = hourly_snowfall
	hourly_data["cloud_cover"] = hourly_cloud_cover

	hourly_dataframe = pd.DataFrame(data = hourly_data)

	hourly_dataframe.loc[len(hourly_dataframe)] = [fecha_utc, current_temperature_2m, current_rain, current_precipitation, current_wind_speed_10m, current_snowfall, current_cloud_cover]
	#La última fila del df es el current
	ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
	SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
	upload_df_parquet(ACCESS_KEY, SECRET_KEY, 'grupo5/processed/Clima/DataFrame_Clima_TiempoReal.parquet', hourly_dataframe)
	

if __name__ == "__main__":
	df = extraer_clima_actual()


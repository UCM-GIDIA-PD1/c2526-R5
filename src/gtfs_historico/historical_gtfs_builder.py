import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import tarfile
import shutil

# Descarga de datos realtime
def download_realtime_data(target_date):
    """
    Descarga el archivo .tar.xz diario de SubwayData NYC y extrae los CSV de trips y stop_times.
    """
    tar_filename = f"subwaydatanyc_{target_date}_csv.tar.xz"
    url = f"https://subwaydata.nyc/data/{tar_filename}"
    
    print(f"Descargando Realtime (comprimido): {url}...")
    
    # Descargar el archivo .tar.xz
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        raise Exception(f"Error 404: El archivo {tar_filename} no está disponible para esta fecha. Verifica la URL.")
    response.raise_for_status()
    
    with open(tar_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Extrayendo {tar_filename}...")
    
    # Extraer el contenido
    with tarfile.open(tar_filename, "r:xz") as tar:
        # Esto extraerá los archivos en el directorio actual
        tar.extractall(filter='data')
        
    # Limpieza: Borrar el .tar.xz para ahorrar espacio en el disco
    if os.path.exists(tar_filename):
        os.remove(tar_filename)
        print(f"Archivo {tar_filename} eliminado tras la extracción.")

    # Definir y verificar los nombres de los archivos extraídos
    trips_file = f"subwaydatanyc_{target_date}_trips.csv"
    stops_file = f"subwaydatanyc_{target_date}_stop_times.csv"
    
    if not os.path.exists(trips_file) or not os.path.exists(stops_file):
        raise FileNotFoundError("Los CSVs esperados no se encontraron tras descomprimir el archivo.")
        
    return trips_file, stops_file

# Descarga de datos static
def download_static_data(target_date):
    """
    1. Obtiene un access_token usando el refresh_token.
    2. Consulta la API de Mobility Database autenticada para buscar el estático.
    3. Descarga el ZIP y extrae trips.txt y stop_times.txt.
    """

    # OBTENER EL ACCESS TOKEN
    refresh_token = os.getenv("MOBILITY_DATABASE_REFRESH_TOKEN")
    assert refresh_token is not None, "La variable de entorno MOBILITY_DATABASE_REFRESH_TOKEN no está definida."
    print("Autenticando con Mobility Database...")
    token_url = 'https://api.mobilitydatabase.org/v1/tokens'
    token_payload = { "refresh_token": refresh_token }

    token_response = requests.post(token_url, json=token_payload)

    if token_response.status_code != 200:
        raise Exception(f"Fallo al obtener el access_token. HTTP {token_response.status_code}: {token_response.text}")
        
    # Extraemos el access token de la respuesta JSON
    access_token = token_response.json().get("access_token")
    
   # CONSULTAR LA API CON EL TOKEN
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    api_url = "https://api.mobilitydatabase.org/v1/gtfs_feeds/mdb-511/datasets"

    # Añadimos el Access Token a las cabeceras (Headers)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    print(f"Consultando API Mobility Database para fecha: {target_date}...")
    # Pasamos los headers en la petición GET
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 401:
        raise Exception("Error 401 No Autorizado: El Access Token es inválido o el Refresh Token ha expirado/es incorrecto.")
    response.raise_for_status()

    # BUSCAR Y DESCARGAR EL ARCHIVO
    datasets = response.json()

    # Filtrar para encontrar el dataset publicado justo ANTES o EN nuestra fecha
    valid_datasets = []
    for ds in datasets:
        # Extraemos la fecha de publicación (downloaded_at)
        extracted_dt = datetime.strptime(ds['downloaded_at'][:10], "%Y-%m-%d")
        if extracted_dt <= target_dt:
            valid_datasets.append((extracted_dt, ds['hosted_url']))
            
    if not valid_datasets:
        raise ValueError(f"No se encontró un GTFS estático anterior a {target_date}.")
    
    # Ordenar descendente para coger la versión más reciente válida
    valid_datasets.sort(key=lambda x: x[0], reverse=True)
    best_url = valid_datasets[0][1]
    best_date_str = valid_datasets[0][0].strftime("%Y-%m-%d")

    print(f"Feed estático seleccionado: Versión extraída el {best_date_str}")
    
    zip_filename = f"static_mta_{best_date_str}.zip"
    if not os.path.exists(zip_filename):
        print(f"Descargando ZIP estático desde {best_url}...")
        res = requests.get(best_url, stream=True) # La descarga del archivo ZIP suele ser pública y no requerir Headers
        res.raise_for_status()
        with open(zip_filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"El archivo {zip_filename} ya existe. Saltando descarga.")
    
    # Extraer solo lo que necesitamos
    extract_dir = f"static_gtfs_{best_date_str}"
    os.makedirs(extract_dir, exist_ok=True)
    
    print("Extrayendo trips.txt y stop_times.txt...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Algunos feeds meten los archivos dentro de una subcarpeta. Aquí asumimos que están en la raíz.
        if 'trips.txt' in zip_ref.namelist() and 'stop_times.txt' in zip_ref.namelist():
            zip_ref.extract('trips.txt', extract_dir)
            zip_ref.extract('stop_times.txt', extract_dir)
        else:
            raise FileNotFoundError("El archivo ZIP descargado no contiene trips.txt y/stop_times.txt en la raíz.")
    
    # Borrar el archivo ZIP tras extraer los txt
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    return os.path.join(extract_dir, 'trips.txt'), os.path.join(extract_dir, 'stop_times.txt')

# 3. PROCESAMIENTO Y CRUCE
def build_delay_datalake(static_trips_path, static_stops_path, rt_trips_path, rt_stops_path):
    # 1. Load Data
    df_static_trips = pd.read_csv(static_trips_path)
    df_static_st = pd.read_csv(static_stops_path)
    
    df_rt_trips = pd.read_csv(rt_trips_path)
    df_rt_st = pd.read_csv(rt_stops_path)

    # Prepare Static Data
    # 
    # Extract the matching key from static trip_id 
    # Example: "SIR-FA2017-SI017-Weekday-08_147100_SI..N03R" -> "147100_SI..N03R"
    df_static_trips['match_key'] = df_static_trips['trip_id'].str.extract(r'_(\d{6}_.*)$')
    
    # Merge static trips and stop_times
    static_merged = pd.merge(
        df_static_st, 
        df_static_trips[['trip_id', 'match_key', 'route_id']], 
        on='trip_id'
    )

    # Function to convert GTFS string time "HH:MM:SS" to total seconds
    def gtfs_time_to_seconds(t):
        if pd.isna(t): return np.nan
        h, m, s = map(int, str(t).split(':'))
        return h * 3600 + m * 60 + s

    # Calculate scheduled seconds past midnight
    static_merged['scheduled_seconds'] = static_merged['arrival_time'].apply(gtfs_time_to_seconds)

    # Prepare Realtime Data
    # 
    # In subwaydatanyc datasets, the 'trip_id' column in trips.csv IS the match key
    rt_merged = pd.merge(
        df_rt_st, 
        df_rt_trips[['trip_uid', 'trip_id']], 
        on='trip_uid'
    )
    # Rename for clarity during join with static
    rt_merged.rename(columns={'trip_id': 'match_key'}, inplace=True)

    # Function to convert UNIX timestamp to seconds past midnight in Local NYC time
    nyc_tz = pytz.timezone('America/New_York')
    def unix_to_seconds_past_midnight(ts):
        if pd.isna(ts): return np.nan
        dt = datetime.fromtimestamp(ts, nyc_tz)
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    # Calculate actual seconds past midnight
    rt_merged['actual_seconds'] = rt_merged['arrival_time'].apply(unix_to_seconds_past_midnight)

    # Join and Calculate Delay
    # 
    # LEFT join keeps "Unscheduled/Added" trains
    final_df = pd.merge(
        rt_merged, 
        static_merged[['match_key', 'stop_id', 'scheduled_seconds', 'route_id']], 
        on=['match_key', 'stop_id'], 
        how='left'
    )
    '''
    # INNER join deletes "Unscheduled/Added" trains
    final_df = pd.merge(
        rt_merged, 
        static_merged[['match_key', 'stop_id', 'scheduled_seconds', 'route_id']], 
        on=['match_key', 'stop_id'], 
        how='inner'
    )'''

    # Calculate final delays
    final_df['delay_seconds'] = final_df['actual_seconds'] - final_df['scheduled_seconds']

    # Apply corrections: Arreglar los falsos retrasos de +24h (Trenes programados 00:XX que llegan 23:XX)
    # Si el retraso es mayor a 12 horas (43200s), le restamos 24 horas (86400s)
    final_df.loc[final_df['delay_seconds'] > 43200, 'delay_seconds'] -= 86400

    # Arreglar los falsos adelantos de -24h (Trenes programados 23:XX que llegan 00:XX)
    # Si el supuesto adelanto es menor a -12 horas (-43200s), le sumamos 24 horas (86400s)
    final_df.loc[final_df['delay_seconds'] < -43200, 'delay_seconds'] += 86400
    
    final_df['delay_minutes'] = final_df['delay_seconds'] / 60

    # Marcar trenes no programados
    final_df['is_unscheduled'] = final_df['scheduled_seconds'].isna()

    datalake_ready_df = final_df[[
        'trip_uid', 'match_key', 'route_id', 'stop_id', 'is_unscheduled',
        'scheduled_seconds', 'actual_seconds', 'delay_seconds', 'delay_minutes'
    ]]  
    
    return datalake_ready_df


def process_mta_date(target_date):
    """
    Orquesta la descarga, procesamiento y limpieza para un solo día.
    Devuelve el DataFrame final para que el orquestador lo suba a MinIO.
    """
    # Descargar archivos temporales (tiempos teóricos y reales) al disco local
    rt_trips, rt_stops = download_realtime_data(target_date)
    st_trips, st_stops = download_static_data(target_date)
    
    # Cargar datos en memoria y calcular delays
    print(f"Procesando cruce de datos para {target_date}...")
    df_final = build_delay_datalake(st_trips, st_stops, rt_trips, rt_stops)
    
    # Guardar el parquet en una carpeta temporal
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_file = f"{tmp_dir}/mta_delays_{target_date}.parquet"
    df_final.to_parquet(output_file, engine="pyarrow")

    # Borrar todos los archivos crudos del disco local
    for f in [rt_trips, rt_stops]:
        if os.path.exists(f): 
            os.remove(f)
    
    # Borrar la carpeta estática entera (con los txt dentro)
    static_dir = os.path.dirname(st_trips)
    if os.path.exists(static_dir): 
        shutil.rmtree(static_dir)
        
    # Devolvemos la ruta del parquet temporal para que el orquestador lo suba
    return output_file
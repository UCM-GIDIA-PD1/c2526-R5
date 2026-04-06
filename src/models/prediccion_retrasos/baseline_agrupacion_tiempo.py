import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.minio_client import download_df_parquet



def baseline(tiempo = '60'):
    """Funcion principal que orquesta la carga de datos, el entrenamiento y el registro de resultados."""
    ### 1. Cargar los datos ###
    INPUT_PATH = 'grupo5/aggregations/DataFrameGroupedByMin={tiempo}.parquet'
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    print('Cargando datos desde MinIO...')
    df = download_df_parquet(ACCESS_KEY, SECRET_KEY, INPUT_PATH.format(tiempo=tiempo))

    ### 2. Seleccionar columnas ###
    # Retraso actual (nuestra predicción)
    columna_actual = 'delay_seconds_mean' 
    
    columna_target = f'target_delay_{tiempo}m_mean' 

    ### 3. Limpiar datos ###
    # Eliminamos filas que tengan valores nulos en estas columnas para poder calcular el error
    df_clean = df.dropna(subset=[columna_actual, columna_target])

    ### 4. Definir valores reales y predichos ###
    y_true = df_clean[columna_target]
    y_pred = df_clean[columna_actual] # El baseline asume que dentro de 30 min el retraso será el mismo de ahora

    ### 5. Calcular métricas de error ###
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print("--- Resultados del Baseline (Naïve) ---")
    print(f"MAE (Error Absoluto Medio): {mae:.2f} segundos")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f} segundos")

if __name__ == "__main__":
    # 1. Inicializamos el parser
    parser = argparse.ArgumentParser(description="Script para calcular el baseline de retrasos.")
    
    # 2. Añadimos el argumento --time
    parser.add_argument(
        '--time', 
        type=str, 
        default='60', 
        help='Minutos hacia el futuro para calcular el target (ej: 10, 20, 30, 45, 60)'
    )
    
    # 3. Parseamos los argumentos que vengan de la terminal
    args = parser.parse_args()
    
    # 4. Ejecutamos la función pasándole el valor recogido
    baseline(tiempo=args.time)

import requests
import pandas as pd
import time # Importante para pausar entre peticiones y no saturar el servidor

API_KEY = "382a48b6-7477-429f-b0b4-d2e6ca45b84e"
url = "https://www.jambase.com/jb-api/v1/events"

cabeceras = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

eventos_totales = []
pagina_actual = 1

# ⚠️ Límite de seguridad temporal. Ponlo más alto cuando veas que funciona bien.
LIMITE_PAGINAS = 5 

print("⏳ Comenzando a buscar eventos de 2025...")

while pagina_actual <= LIMITE_PAGINAS:
    # 1. Añadimos las fechas y la página a los parámetros
    parametros = {
        "apikey": API_KEY,
        "eventDateFrom": "2026-01-01",
        "eventDateTo": "2026-12-31",
        "geoStateIso": "US-NY",
        "page": pagina_actual
    }
    
    respuesta = requests.get(url, params=parametros, headers=cabeceras)
    
    if respuesta.status_code == 200:
        datos_json = respuesta.json()
        lista_eventos = datos_json.get("events", [])
        
        # Si la lista viene vacía, ya no hay más eventos en 2025
        if not lista_eventos:
            print("No hay más eventos, hemos llegado al final.")
            break
            
        eventos_totales.extend(lista_eventos) # Juntamos los nuevos eventos con los anteriores
        print(f"✅ Página {pagina_actual} descargada con {len(lista_eventos)} eventos.")
        
        # Si nos devuelve menos de 50 eventos (el límite habitual por página), es la última
        if len(lista_eventos) < 50: 
            break
            
        pagina_actual += 1
        time.sleep(1) # Descansamos 1 segundo entre llamadas para que la API no nos bloquee
        
    else:
        print(f"❌ Error {respuesta.status_code} en la página {pagina_actual}.")
        print(respuesta.text)
        break

# 2. Procesamos TODOS los eventos que hayamos acumulado en la lista
if eventos_totales:
    df = pd.json_normalize(eventos_totales)
    
    columnas_interes = ['name', 'startDate'] 
    columnas_disponibles = [col for col in columnas_interes if col in df.columns]
    df_final = df[columnas_disponibles].copy()
    
    if 'startDate' in df_final.columns:
        # Aquí mantenemos la corrección del formato de fecha (ISO8601)
        df_final['fecha_hora_completa'] = pd.to_datetime(df_final['startDate'], format='ISO8601')
        df_final['fecha'] = df_final['fecha_hora_completa'].dt.date
        df_final['hora'] = df_final['fecha_hora_completa'].dt.time
        
    print(f"\n¡Éxito! DataFrame final creado con {len(df_final)} eventos.")
    print(df_final)
else:
    print("No se encontraron eventos.")
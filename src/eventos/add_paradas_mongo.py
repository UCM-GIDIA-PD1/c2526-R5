import pandas as pd
from pymongo import MongoClient

url_servidor = 'mongodb://127.0.0.1:27017/'


client = MongoClient(url_servidor)

# código para ver si se ha conectado bien
try:
    s = client.server_info() # si hay error tendremos una excepción
    print("Conectado a MongoDB, versión",s["version"])
    db = client["PD1"]
except:
    print ("Error de conexión ¿está arrancado el servidor?")

df = pd.read_csv("MTA_Subway_Stations.csv")

columnas_utiles = ['GTFS Stop ID', 'Stop Name', 'Daytime Routes', 'GTFS Longitude', 'GTFS Latitude']
df_limpio = df[columnas_utiles].copy()


df_limpio = df_limpio.rename(columns={
    'GTFS Stop ID': 'GTFS_id_estacion',
    'Stop Name': 'nombre',
    'Daytime Routes': 'lineas'
})


df_limpio['ubicacion'] = df_limpio.apply(
    lambda fila: {
        "type": "Point",
        "coordinates": [fila['GTFS Longitude'], fila['GTFS Latitude']]
    }, axis=1
)


df_limpio = df_limpio.drop(columns=['GTFS Longitude', 'GTFS Latitude'])


documentos_para_mongo = df_limpio.to_dict(orient='records')
db.subway.drop()
db.subway.insert_many(documentos_para_mongo)
db.subway.drop_indexes()
db.subway.create_index({"ubicacion":"2dsphere"})
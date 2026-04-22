from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import asyncio
import json
import random
from datetime import datetime
import pandas as pd
import os

# --- Mock para la futura Integración con MinIO y Modelos ---
class SystemManager:
    def __init__(self):
        self.latest_predictions = []
        self.stations = []
        self.load_stations()
        
    def load_stations(self):
        remote_url = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"
        local_csv_path = os.path.join(os.path.dirname(__file__), "MTA_Subway_Stations.csv")
        
        df = None
        
        # Intentamos cargar desde la API remota
        print(f"[LOADER] Intentando cargar estaciones desde la API remota...")
        try:
            df = pd.read_csv(remote_url)
            print("[LOADER] Datos descargados correctamente desde la API remota.")
        except Exception as e:
            print(f"[WARNING] No se pudo cargar desde la API remota: {e}")
            print(f"[LOADER] Intentando cargar desde el archivo local: {local_csv_path}")
            try:
                if os.path.exists(local_csv_path):
                    df = pd.read_csv(local_csv_path)
                    print("[LOADER] Datos cargados correctamente desde el archivo local.")
                else:
                    print(f"[ERROR] El archivo local no existe: {local_csv_path}")
            except Exception as e_local:
                print(f"[ERROR] Error al cargar el CSV local: {e_local}")

        if df is not None:
            try:
                # Limpiamos y preparamos los datos
                for _, row in df.iterrows():
                    self.stations.append({
                        "id": str(row["Station ID"]),
                        "name": row["Stop Name"],
                        "lat": float(row["GTFS Latitude"]),
                        "lon": float(row["GTFS Longitude"]),
                        "routes": row["Daytime Routes"] if pd.notna(row["Daytime Routes"]) else ""
                    })
                print(f"[LOADER] {len(self.stations)} estaciones cargadas correctamente.")
            except Exception as e_parse:
                print(f"[ERROR] Error al procesar los datos del CSV: {e_parse}")
        else:
            print("[ERROR] No se pudo cargar ninguna fuente de datos para las estaciones.")
        
    def generate_mock_prediction(self):
        # Simulamos un modelo que predice si hay retraso en paradas reales
        if not self.stations:
            return

        self.latest_predictions = []
        # Elegimos 5 estaciones al azar de las cargadas
        sampled_stations = random.sample(self.stations, min(5, len(self.stations)))
        
        for station in sampled_stations:
            delay_minutes = random.randint(1, 15)
            self.latest_predictions.append({
                "station_id": station["id"],
                "status": "delayed" if delay_minutes > 5 else "alert",
                "delay_minutes": delay_minutes,
                "timestamp": datetime.now().isoformat()
            })
            
    def dump_to_minio(self):
        # Aquí borramos o movimos los datos al bucket de MinIO periódicamente
        print("[MOCK MinIO] Subiendo predicciones al Cloud y limpiando buffer ligero...")
        self.latest_predictions.clear()


sys_manager = SystemManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Iniciando servicio de modelos mock...")
    # app.state.model = load('models/final_model.joblib') # Descomentar cuando esté listo
    
    # Tarea en background para generar datos cada minuto
    async def periodic_task():
        while True:
            # Generar la predicción del modelo
            sys_manager.generate_mock_prediction()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Predicciones generadas.")
            
            # Avisar por WebSockets
            await socket_manager.broadcast(json.dumps({
                "type": "update",
                "data": sys_manager.latest_predictions
            }))
            
            # Guardar histórico en MinIO (cada 2 minutos ej. / aquí lo hacemos rápido)
            sys_manager.dump_to_minio()
            
            # Esperamos 60 s
            await asyncio.sleep(60)

    app.state.bg_task = asyncio.create_task(periodic_task())
    yield
    # Shutdown
    app.state.bg_task.cancel()
    print("Apagando...")


app = FastAPI(lifespan=lifespan)

# Montar las plantillas y archivos estáticos
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- MANEJADOR DE WEBSOCKETS ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Enviar estado actual nada más conectar
        if sys_manager.latest_predictions:
            await websocket.send_text(json.dumps({
                "type": "initial",
                "data": sys_manager.latest_predictions
            }))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass # Se limpiará pronto el socket desconectado

socket_manager = ConnectionManager()

# --- RUTAS ---
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, name="index.html")

@app.get("/api/stations")
def get_stations():
    return sys_manager.stations

@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    await socket_manager.connect(websocket)
    try:
        while True:
            # Mantener conexión viva
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket)

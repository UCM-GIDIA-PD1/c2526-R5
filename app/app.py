"""
Express-Bound — Inference API
Serves real-time MTA subway delay predictions from models stored in wandb.
Data source: 8 × 15-min sliding windows stored in Google Drive.

Run from project root:
    uv run fastapi dev app/app.py        # development
    uv run fastapi run app/app.py        # production
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
from app.cache import TTLCache
from app.config import settings
from app.models.registry import ModelRegistry
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Express-Bound inference API…")

    # Model registry
    registry = ModelRegistry()

    entity = settings.wandb_entity
    await asyncio.gather(
        asyncio.to_thread(
            registry.load_dcrnn,
            entity, settings.wandb_project_dcrnn, settings.dcrnn_artifact,
        ),
        asyncio.to_thread(
            registry.load_delay,
            entity, settings.wandb_project_delay, settings.delay_artifact,
        ),
        asyncio.to_thread(
            registry.load_alertas,
            entity, settings.wandb_project_alertas, settings.alertas_artifact,
        ),
    )

    if registry.errors:
        logger.warning("Some models failed to load: %s", list(registry.errors.keys()))
    else:
        logger.info("All models loaded successfully")

    app.state.registry = registry
    app.state.cache = TTLCache(ttl_seconds=settings.data_cache_ttl)
    app.state.stations_meta = _load_stations_meta()

    # Background task: broadcast live updates to WebSocket clients every 60 s
    app.state.ws_manager = _ConnectionManager()
    app.state.bg_task = asyncio.create_task(_live_broadcast(app))

    yield

    app.state.bg_task.cancel()
    logger.info("Express-Bound API stopped")


def _load_stations_meta() -> dict:
    """Load station coordinates from CSV if available."""
    csv_path = os.path.join(os.path.dirname(__file__), "MTA_Subway_Stations.csv")
    if not os.path.exists(csv_path):
        logger.info("MTA_Subway_Stations.csv not found – coordinates unavailable")
        return {}
    try:
        df = pd.read_csv(csv_path)
        meta = {}
        for _, row in df.iterrows():
            meta[str(row["Station ID"])] = {
                "name": row.get("Stop Name", ""),
                "lat": float(row["GTFS Latitude"]),
                "lon": float(row["GTFS Longitude"]),
                "routes": row.get("Daytime Routes", ""),
            }
        logger.info("Loaded %d station records from CSV", len(meta))
        return meta
    except Exception as exc:
        logger.warning("Could not load stations CSV: %s", exc)
        return {}


async def _live_broadcast(app: FastAPI) -> None:
    """Periodically push the latest /predict/all result to WebSocket clients."""
    while True:
        await asyncio.sleep(60)
        try:
            from app.routers.predict import _get_windows
            from app.models.alertas_infer import run_alerts
            from app.models.delay_infer import run_delays
            from app.models.dcrnn_infer import run_propagation

            registry = app.state.registry
            cache = app.state.cache
            windows = cache.get("windows")
            if windows is None:
                continue

            payload: dict = {"type": "update", "predicted_at": datetime.now(timezone.utc).isoformat()}

            if registry.alertas:
                alerts = await asyncio.to_thread(
                    run_alerts,
                    entry=registry.alertas,
                    windows=windows,
                    threshold=settings.alert_threshold,
                )
                payload["alerts"] = alerts.model_dump()

            await app.state.ws_manager.broadcast(json.dumps(payload))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Live broadcast error (non-fatal): %s", exc)


# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, message: str) -> None:
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Express-Bound",
    description="Real-time MTA subway delay prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(health_router)
app.include_router(predict_router, prefix="/api")


# ── UI & WebSocket ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, name="index.html")


@app.get("/api/stations")
def get_stations(request: Request):
    return list(request.app.state.stations_meta.values())


@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    manager: _ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

"""
Express-Bound — Inference API
Serves real-time MTA subway delay predictions from models stored in wandb.
Data source: 8 × 15-min sliding windows stored in Google Drive (MTA_Realtime_Windows/).

Run from project root:
    uv run fastapi dev app/app.py        # development
    uv run fastapi run app/app.py        # production
"""
import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

    registry = ModelRegistry()
    entity = settings.wandb_entity

    await asyncio.gather(
        asyncio.to_thread(registry.load_dcrnn,
            entity, settings.wandb_project_dcrnn, settings.dcrnn_artifact),
        asyncio.to_thread(registry.load_lgbm_delay_30m,
            entity, settings.wandb_project_delay, settings.lgbm_delay_30m_artifact),
        asyncio.to_thread(registry.load_lgbm_delay_end,
            entity, settings.wandb_project_delay, settings.lgbm_delay_end_artifact),
        asyncio.to_thread(registry.load_delta_10m,
            entity, settings.wandb_project_delay, settings.delta_10m_artifact),
        asyncio.to_thread(registry.load_delta_20m,
            entity, settings.wandb_project_delay, settings.delta_20m_artifact),
        asyncio.to_thread(registry.load_delta_30m,
            entity, settings.wandb_project_delay, settings.delta_30m_artifact),
        asyncio.to_thread(registry.load_alertas,
            entity, settings.wandb_project_alertas, settings.alertas_artifact),
    )

    if registry.errors:
        logger.warning("Some models failed to load: %s", list(registry.errors.keys()))
    else:
        logger.info("All models loaded successfully")

    app.state.registry = registry
    app.state.cache = TTLCache(ttl_seconds=settings.data_cache_ttl)
    app.state.stations_meta = _load_stations_meta()
    app.state.ws_manager = _ConnectionManager()
    app.state.bg_task = asyncio.create_task(_live_broadcast(app))

    yield

    app.state.bg_task.cancel()
    logger.info("Express-Bound API stopped")


def _load_stations_meta() -> dict:
    """Load station coordinates from local CSV or NY.gov API."""
    local = os.path.join(os.path.dirname(__file__), "MTA_Subway_Stations.csv")
    remote = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"

    df = None
    for source in (local, remote):
        try:
            df = pd.read_csv(source)
            logger.info("Station metadata loaded from %s (%d rows)", source, len(df))
            break
        except Exception as exc:
            logger.debug("Could not load stations from %s: %s", source, exc)

    if df is None:
        logger.warning("Station metadata unavailable — coordinates will be omitted")
        return {}

    meta = {}
    try:
        for _, row in df.iterrows():
            meta[str(row["Station ID"])] = {
                "name": row.get("Stop Name", ""),
                "lat": float(row["GTFS Latitude"]),
                "lon": float(row["GTFS Longitude"]),
                "routes": row.get("Daytime Routes", ""),
            }
    except Exception as exc:
        logger.warning("Error parsing station CSV: %s", exc)
    return meta


async def _live_broadcast(app: FastAPI) -> None:
    """Push latest alert predictions to WebSocket clients every 60 s."""
    while True:
        await asyncio.sleep(60)
        try:
            from app.models.alertas_infer import run_alerts

            registry = app.state.registry
            windows = app.state.cache.get("windows")
            if windows is None or registry.alertas is None:
                continue

            alerts = await asyncio.to_thread(
                run_alerts,
                entry=registry.alertas,
                windows=windows,
                threshold=settings.alert_threshold,
            )
            payload = json.dumps({
                "type": "update",
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "alerts": alerts.model_dump(),
            })
            await app.state.ws_manager.broadcast(payload)
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

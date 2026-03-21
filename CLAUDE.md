# Express-Bound — CLAUDE.md

Proyecto de Datos I, Grupo 5 · Facultad de Informática, UCM
Predicción de retrasos e incidencias en el metro de Nueva York (MTA).

## Stack

- **Python 3.13+**, gestor de dependencias: `uv`
- **ML**: LightGBM, XGBoost, PyTorch, torch-geometric-temporal (DCRNN, STGCN)
- **Datos**: pandas, pyarrow, MinIO (S3-compatible data lake)
- **Experimentos**: Weights & Biases (`wandb`)
- **Calidad**: ruff (linting), mypy (tipos), pre-commit, pytest

## Instalación

```bash
uv sync
```

## Variables de entorno necesarias

```
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MOBILITY_DATABASE_REFRESH_TOKEN
NYC_OPEN_DATA_TOKEN
CLIENT_ID_SEATGEEK
SETLIST_API_KEY
WANDB_API_KEY
```

También se necesitan credenciales de Gmail (`credentials.json` / `token.json`) para la ingesta de alertas MTA.

## Estructura

```
src/
├── common/                        # MinIO client y utilidades compartidas
├── ETL/
│   ├── alertas_oficiales_tiempo_real/
│   ├── clima/
│   ├── eventos/
│   ├── gtfs_historico/
│   ├── tiempo_real_metro/
│   └── pipelines/                 # Orquestadores principales
└── models/
    ├── prediccion_retrasos/       # LightGBM, XGBoost por parada
    ├── prediccion_propagacion/    # DCRNN, STGCN (graph neural nets)
    └── modelos_alertas/           # Detección de incidencias
notebooks/                         # EDA y análisis exploratorio
```

## Ejecutar pipelines

```bash
# Extracción (fuentes externas → raw/)
uv run python -m src.ETL.pipelines.run_extraccion --source all --start 2025-01-01 --end 2025-01-31

# Transformación (raw/ → processed/ → cleaned/)
uv run python -m src.ETL.pipelines.run_transform --source all --start 2025-01-01 --end 2025-01-31
```

`--source` acepta `all` o el nombre de una fuente concreta (gtfs, clima, eventos, alertas).

## Data lake (MinIO)

Bucket: `pd1`, raíz: `grupo5/`

| Capa | Contenido |
|------|-----------|
| `raw/` | Datos originales sin tratar |
| `processed/` | Datos estructurados en Parquet, sin limpieza exhaustiva |
| `cleaned/` | Datos limpios, validados y con features derivados |

Convención de rutas: `grupo5/<capa>/<dataset>/date=YYYY-MM-DD/<archivo>.parquet`

## Modelos

- **Predicción de retrasos por parada**: LightGBM / XGBoost con features temporales (lags, rolling, codificación cíclica hora), meteorológicos y de eventos.
- **Propagación de retrasos**: DCRNN y STGCN sobre grafos de la red de metro.
- **Detección de alertas**: Regresión logística, Random Forest, XGBoost.

División temporal: train en meses 01–09 de 2025, validación en 10–12.

## Notebooks

- **nbstripout**: git filter instalado — los outputs de los notebooks se eliminan automáticamente antes de cada commit. No es necesario limpiarlos manualmente.
- **nbdime**: diffs y merges legibles para `.ipynb`. Configurado como driver de git (`diff=jupyternotebook`, `merge=jupyternotebook` en `.gitattributes`). Usar `nbdiff` / `nbdiff-web` para revisar cambios.
- Ambos están en `dev` dependencies y se activan con `uv sync`.

## Convenciones

- Documentación y nombres de variables en **español**.
- Usar `uv run python -m <módulo>` para ejecutar scripts (nunca `python` directamente).
- No almacenar datos en el repositorio; todo va a MinIO.
- Los notebooks en `src/models/` son el entorno de desarrollo de modelos; los de `notebooks/` son EDA.

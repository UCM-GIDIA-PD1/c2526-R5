<img width="512" height="512" alt="Logo1PD" src="https://github.com/user-attachments/assets/44ea9a4a-ce36-4497-9ec0-779366090aa4" />

# Express-Bound
Proyecto de Datos I – Grupo 5

Facultad de Informática – UCM

## Descripción del proyecto

Express-Bound integra datos operativos y contextuales del metro de Nueva York para estimar retrasos a corto plazo y anticipar incidencias.

El proyecto se centra en tres líneas principales:

1. Predicción del retraso de un tren concreto.
2. Modelado de la propagación de los retrasos por la red.
3. Detección temprana de incidencias operativas.

El enfoque es de predicción a corto horizonte (10–30 minutos), utilizando tanto el estado actual de la red como información contextual (clima, calendario, estructura de la red).

El sistema está diseñado siguiendo una arquitectura tipo data lake (raw → processed → cleaned) sobre almacenamiento en MinIO, garantizando trazabilidad y reproducibilidad del pipeline.

## Estructura del proyecto
```
├── src/
│   ├── common/                        # Utilidades compartidas (MinIO client, etc.)
│   ├── ETL/                           # Scripts de ingestión, limpieza y generación de features
│   │   ├── alertas_oficiales_tiempo_real/   # Alertas MTA (histórico y tiempo real)
│   │   ├── clima/                     # Datos meteorológicos (Open-Meteo)
│   │   ├── eventos/                   # Eventos NYC (deportes, conciertos, oficiales)
│   │   ├── gtfs_historico/            # GTFS histórico (instancias de trenes pasando por estaciones)
│   │   ├── pipelines/                 # Orquestadores run_extraccion y run_transform, generacion de dataset final y agregaciones
│   │   └── tiempo_real_metro/         # GTFS en tiempo real (MTA feeds)
│   └── models/                        
│       ├── common/                    # Agregaciones temporales adicionales
│       ├── modelos_alertas/           # Modelos entrenados para anticipar las alertas oficiales de la MTA
│       ├── prediccion_retrasos/       # Modelos entrenados para predecir el retraso de trenes
│       ├── propagacion_estacion/      # Modelos entrenados para modelar la propagación del retraso por la red 
│       └── seleccion_variables.md     # Explicación de las variables que mantenemos en la fase de modelado a partir de los resultados de los notebooks de análisis
│
├── notebooks/                         # Análisis exploratorio y visualizaciones
├── docs/                              # Documentación adicional
├── pyproject.toml                     # Configuración del entorno
├── .gitignore
└── README.md
```

## Almacenamiento en MinIO

Los datos del proyecto se almacenan en un bucket S3-compatible (MinIO),
siguiendo una arquitectura tipo data lake organizada en distintas capas
según su nivel de procesamiento.

No se almacenan datos en GitHub.

Bucket utilizado: `pd1`
Raíz del proyecto: `grupo5/`

### Estructura 

```
pd1/
└── grupo5/
    ├── raw/
    │   ├── avisos_oficiales_historico_2025/
    │   └── eventos_nyc/
    │
    ├── processed/
    │   ├── eventos_nyc/
    │   ├── gtfs_realtime/
    │   ├── gtfs_with_delays/
    │   ├── clima/
    │   └── official_alerts/
    │
    ├── final/
    │   ├── year=2025/
    │   └── year=2026/
    │
    ├── aggregations/
    │   └── lines/
    │
    └── cleaned/
        ├── clima_clean/
        ├── eventos_nyc/
        ├── gtfs_clean_scheduled/
        ├── gtfs_clean_unscheduled/
        └── official_alerts/
```
## Descripción de cada capa

### raw/
Contiene los datos originales descargados de las fuentes externas y sin tratar.
No se modifican una vez almacenados.

### processed/
Datos transformados a un formato estructurado (principalmente Parquet),
unidos de distintas fuentes pero todavía sin limpieza exhaustiva.

### cleaned/
Datos limpios y validados. Incluye:
- Eliminación de duplicados
- Corrección de tipos
- Control de outliers
- Reportes de calidad

También contiene features derivados y agregaciones temporales (p.ej. lagged_delay_1).

## Convención de nombres
Los objetos se almacenan siguiendo la convención:

grupo5/processed/nombre_fuente/date=YYYY-MM-DD/nombre_archivo.parquet

Lo cual permite:
- Filtrado eficiente por fecha
- Procesamiento incremental
- Re-ejecución parcial del pipeline en caso de fallo

## Configuración del entorno de desarrollo

El proyecto utiliza Python y el gestor de dependencias `uv`.

### Requisitos previos

- Python >= 3.13
- uv instalado
- Acceso a MinIO (credenciales proporcionadas al grupo)

### Configuración de variables de entorno

Se recomienda utilizar variables de entorno del sistema (se podría utilizar .env con python-dotenv)
```
export MINIO_ACCESS_KEY=...
export MINIO_SECRET_KEY=...
export MOBILITY_DATABASE_REFRESH_TOKEN=...
export NYC_OPEN_DATA_TOKEN=...
export CLIENT_ID_SEATGEEK=...
export SETLIST_API_KEY=...
```

Credenciales y tokens
```
Gmail credentials
Gmail token
```

### Crear entorno e instalar dependencias

uv sync

---


## Ejecución de los pipelines

El proyecto está automatizado mediante dos orquestadores principales ubicados en:

```
src/ETL/pipelines/
```

Estos permiten ejecutar la ingesta y transformación de datos de forma parametrizable y reproducible.

---

### Extracción de datos

Script principal:

```
src/ETL/pipelines/run_extraccion.py
```

Este orquestador ejecuta la descarga de datos desde las distintas fuentes externas (GTFS, clima, eventos, alertas oficiales, etc.) y los almacena en la capa `raw/` de MinIO.

#### Parámetros disponibles

- `--source`: nombre de la fuente específica o `all`
- `--start`: fecha de inicio (formato YYYY-MM-DD)
- `--end`: fecha de fin (formato YYYY-MM-DD)

#### Ejemplo de ejecución

```bash
uv run python src/ETL/pipelines/run_extraccion --source all --start 2025-01-01 --end 2025-01-03
```

Este comando descargará los datos del rango indicado y los almacenará en:

```
raw/
```

---

### Transformación de datos

Script principal:

```
src/ETL/pipelines/run_transform.py
```

Este orquestador procesa los datos almacenados en `raw/ y/o processed`, realiza limpieza, integración y generación de variables, y los mueve a capas superiores del data lake.

#### Parámetros disponibles

- `--source`
- `--start`
- `--end`
- `--continue_on_error`

#### Ejemplo de ejecución

```bash
uv run python src/ETL/pipelines/run_transform --source all --start 2025-01-01 --end 2025-01-03
```

Tras su ejecución, los datos seguirán el flujo:

```
raw/ → processed/ → cleaned/
```
---

### Flujo típico de trabajo

1. Ejecutar pipelines (extracción + transformación).
2. Abrir notebook de análisis o modelado.
3. Cargar datos desde `cleaned/`.
4. Validar features generadas.

---

## Uso de notebooks

Los notebooks del proyecto están ubicados en:

```
notebooks/
```

Se utilizan para:

- Análisis exploratorio de datos (EDA)
- Validación de variables derivadas
- Visualización de resultados

---

### Cómo ejecutarlos en VS Code

1. Abrir la carpeta raíz del proyecto en VS Code.
2. Navegar hasta la carpeta `notebooks/`.
3. Abrir el notebook deseado.
4. Seleccionar el kernel correspondiente al entorno creado con `uv`.
5. Ejecutar las celdas en orden.

Es importante ejecutar primero las celdas de:

- Carga de variables de entorno
- Configuración de credenciales
- Conexión a MinIO
- Importación de librerías comunes


---
# Uso de modelos de ML

Todos los modelos se almacenan en la carpeta models, la cual está dividida por problemas:

### 1. Modelos de anticipación de alertas

Se almacenan en la carpeta

```
modelos_alertas/
```

En ella encontramos los modelos de Regresión Logística, Random Forest y XGBoost, cuyos hiperparámetros se han obtenido de búsquedas con Optuna y Random Search
almacenados en:

```
optuna/
random/
```

Los entrenamientos y evaluación se almacenan en 

```
common/
```

### 2. Modelos de predicción de retrasos a nivel de tren

Se almacenan en la carpeta

```
prediccion_retrasos/
```

Este modelo se divide en 4 subproblemas

- Predicción de retrasos a 30 minutos vista para los trenes cuyo que estarán en funcionamiento más de 30 minutos.

```
delay_30m/
```

- Predicción del retraso final para trenes que acabarán su viaje antes de 30 minutos.

```
delay_end/
```
- Predicción de comportamiento del retraso (mejora o empeora).

```
delta/
```

- Predicción del retraso por intervalos.

```
prediccion_por_intervalos/
```

Todos ellos a su vez se organizan en carpetas análogas a las anteriores.

```
optuna/
random/
test/
```

### 3. Modelos propagación de retrasos

Se almacenan en la carpeta

```
propagacion_estacion/
```

Que almacena los modelos en

```
models/
```

# Para resumir esta es la estructura de la carpeta models/

```
models/
├── modelos_alertas/                 # 1. Modelos de anticipación de alertas
│   ├── common/                      # Entrenamientos y evaluación
│   ├── optuna/                      # Búsquedas de hiperparámetros con Optuna
│   └── random/                      # Búsquedas con Random Search
│
├── prediccion_retrasos/             # 2. Modelos de predicción de retrasos
│   ├── delay_30m/                   # Trenes en funcionamiento > 30 mins
│   │   ├── optuna/
│   │   ├── random/
│   │   └── test/
│   ├── delay_end/                   # Trenes que acaban viaje < 30 mins
│   │   ├── optuna/
│   │   ├── random/
│   │   └── test/
│   ├── delta/                       # Comportamiento del retraso (mejora/empeora)
│   │   ├── optuna/
│   │   ├── random/
│   │   └── test/
│   └── prediccion_por_intervalos/   # Predicción por intervalos
│       ├── optuna/
│       ├── random/
│       └── test/
│
└── propagacion_estacion/            # 3. Modelos propagación de retrasos
    └── models/                      # Modelos almacenados
```

---


## Autores
- Alex García
- David Rodríguez
- Iván García
- Chiara Gómez
- Julia Huergo
- Mario González
- Sergio Dueñas
- Juan Jurado

Curso 2025-2026

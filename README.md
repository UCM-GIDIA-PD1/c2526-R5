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
## Uso de modelos de ML

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

### 3. Modelos de propagación de retrasos

Se almacenan en la carpeta

```
propagacion_estacion/
```

Modela cómo un retraso en una estación se propaga por la red de metro usando *Graph Neural Networks* (GNN). La red está representada como un grafo de 899 nodos (paradas en estaciones) con pesos calculados mediante Gaussian Kernel sobre los tiempos medianos de viaje entre ellas.

Se implementan y comparan tres arquitecturas GNN:

- **DCRNN** (*Diffusion Convolutional Recurrent Neural Network*): combina convolución difusiva sobre el grafo con GRU para capturar dependencias espaciales y temporales.
- **STGCN** (*Spatio-Temporal Graph Convolutional Network*): bloques de convolución espacial y temporal en paralelo.
- **ASTGCN** (*Attention-based STGCN*): añade mecanismos de atención espacial y temporal sobre STGCN.

#### Scripts del pipeline (ejecución en orden)

| Script | Descripción |
|--------|-------------|
| `01_generar_grafo.py` | Procesa el GTFS histórico desde MinIO y construye el grafo de la red: `edge_index`, `edge_weight` y lista de nodos. Guarda `artefactos/grafo.pt`. |
| `02_generar_tensores.py` | Genera los tensores espacio-temporales X e Y a partir del dataset final, aplica el split cronológico train/val/test y escala con `StandardScaler`. Guarda `artefactos/tensores.pt`. |
| `03_baseline_naive.py` | Baseline 1 — predice para todos los horizontes el último valor observado de retraso. Sirve de cota inferior de referencia. |
| `04_baseline_ha.py` | Baseline 2 — Historical Average. Calcula la media de retraso histórica por (estación, día de semana, hora) en train y la aplica como predicción en test. Captura patrones cíclicos sin aprendizaje. |
| `05_ablacion_features.py` | Entrena variantes de DCRNN con subconjuntos de features (retraso base, contexto, calendario) durante pocas épocas para identificar el conjunto óptimo. Guarda `artefactos/ablacion.pt`. |
| `06_tuning_hpo_dcrnn.py` | HPO de DCRNN con Optuna y Random Search. Guarda el mejor conjunto de hiperparámetros en `artefactos/hpo.pt`. |
| `07_tuning_stgcn.py` | HPO de STGCN con Optuna. Guarda en `artefactos/stgcn_hpo.pt`. |
| `08_tuning_astgcn.py` | HPO de ASTGCN con Optuna. Guarda en `artefactos/astgcn_hpo.pt`. |
| `09_entrenamiento_final_dcrnn.py` | Entrena DCRNN con los mejores hiperparámetros cargando los artefactos pre-calculados. Guarda `artefactos/dcrnn_final.pth`. |
| `10_entrenamiento_final_stgcn.py` | Entrena STGCN final. Guarda `artefactos/stgcn_final.pth`. |
| `11_entrenamiento_final_astgcn.py` | Entrena ASTGCN final. Guarda `artefactos/astgcn_final.pth`. |
| `12_evaluacion_modelos.py` | **Único script que toca el conjunto de test.** Evaluación comparativa de los tres modelos: métricas principales (MAE, RMSE, R²), análisis por segmentos (fin de semana, clima extremo) y Permutation Feature Importance. |

#### Módulos de soporte

```
models/        — implementaciones de DCRNN, STGCN y ASTGCN
utils/         — dataset.py (carga y ventanas deslizantes), metrics.py (MAE, RMSE, R²)
artefactos/    — artefactos generados por los scripts anteriores
```

## Estructura resumida de la carpeta models/

```
models/
├── common/                          # Utilidades compartidas entre modelos
│
├── modelos_alertas/                 # 1. Modelos de anticipación de alertas
│   ├── common/                      # Entrenamiento y evaluación compartidos
│   ├── Optuna/                      # Búsquedas de hiperparámetros con Optuna
│   └── Random/                      # Búsquedas con Random Search
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
└── propagacion_estacion/            # 3. Modelos de propagación de retrasos (GNN)
    ├── 01_generar_grafo.py          # Construye el grafo de la red desde GTFS
    ├── 02_generar_tensores.py       # Genera tensores X/Y, split y escalado
    ├── 03_baseline_naive.py         # Baseline: último valor observado
    ├── 04_baseline_ha.py            # Baseline: media histórica por estación/hora
    ├── 05_ablacion_features.py      # Ablación de subconjuntos de features
    ├── 06_tuning_hpo_dcrnn.py       # HPO de DCRNN con Optuna/Random Search
    ├── 07_tuning_stgcn.py           # HPO de STGCN con Optuna
    ├── 08_tuning_astgcn.py          # HPO de ASTGCN con Optuna
    ├── 09_entrenamiento_final_dcrnn.py  # Entrenamiento final DCRNN
    ├── 10_entrenamiento_final_stgcn.py  # Entrenamiento final STGCN
    ├── 11_entrenamiento_final_astgcn.py # Entrenamiento final ASTGCN
    ├── 12_evaluacion_modelos.py     # Evaluación comparativa en test
    ├── models/                      # Implementaciones: dcrnn.py, stgcn.py, astgcn.py
    ├── utils/                       # dataset.py, metrics.py
    └── artefactos/                  # Pesos y tensores generados por los scripts
```

---

## Resultados de los mejores modelos

### 1. Anticipación de alertas

El modelo ganador es **XGBoost**, con un PR-AUC de **0.795** sobre el conjunto de test (baseline aleatorio: 0.145).

Distribuye el peso predictivo entre múltiples variables, siendo `seg_desde_ultima_alerta_linea` la más relevante (importancia ~0.37), a diferencia de Random Forest y Regresión Logística que dependen de ella de forma excesiva (~0.73).

| Modelo                | PR-AUC |
|----------------------|-------:|
| Baseline aleatorio   | 0.145  |
| Regresión Logística  | 0.44   |
| Random Forest        | 0.755  |
| **XGBoost**          | **0.795** |

---

### 2. Predicción de retrasos a nivel de tren

Los modelos seleccionados para producción son:

- **LightGBM (regresión)** → `delay_30m` y `delay_end`
- **LightGBM (clasificación binaria)** → comportamiento del retraso (`delta_delay`)

Los modelos por intervalos se descartaron por su escasa mejora respecto al baseline.

#### Regresión por horizonte temporal

| Modelo                 | MAE delay_30m (s) | MAE delay_end (s) | R² delay_end |
|-----------------------|------------------:|------------------:|-------------:|
| Baseline media        | 220               | 232               | ~0           |
| Baseline persistencia | 160               | 156               | 0.453        |
| **LightGBM**          | **134**           | **109**           | **0.696**    |
| MLP                   | 137               | 121               | 0.686        |

#### Clasificación binaria del `delta_delay` (media entre horizontes 10–60 min)

| Modelo               | ROC-AUC |
|---------------------|--------:|
| Baseline mayoritario| 0.50    |
| Random Forest       | 0.801   |
| **LightGBM**        | **0.828** |

Las variables más influyentes en todos los modelos son:

- `delay_seconds` (retraso actual)  
- `lagged_delay_1` (retraso retardado)  
- `delay_velocity` (velocidad de cambio del retraso)

---

### 3. Propagación de retrasos a nivel de estación (GNN)

El modelo ganador es **DCRNN**, con un MAE global de **81.24 segundos** en test, superando a STGCN y ASTGCN y a los dos baselines.

Destaca especialmente durante los fines de semana (**MAE de 50.58 s**).

| Modelo                  | MAE (s) | RMSE (s) | R²     |
|--------------------------|--------:|----------:|-------:|
| Baseline media           | 102.27  | 185.32    | 0.051  |
| Baseline persistencia    | 105.79  | 225.00    | -0.400 |
| STGCN                    | 95.08   | 189.09    | 0.021  |
| ASTGCN                   | 93.66   | 175.83    | 0.151  |
| **DCRNN**                | **81.24** | 177.60  | 0.134  |

El análisis de importancia (*Permutation Feature Importance*) confirma que las variables más relevantes son:

- `delay_seconds` (retraso actual)  
- hora del día  
- `route_rolling_delay` (congestión de red)

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

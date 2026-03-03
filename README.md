<img width="512" height="512" alt="Logo1PD" src="https://github.com/user-attachments/assets/44ea9a4a-ce36-4497-9ec0-779366090aa4" />

# Express-Bound
Proyecto de Datos I – Grupo 5

Facultad de Informática – UCM

## Descripción del proyecto

Express-Bound integra datos operativos y contextuales del metro de Nueva York para detectar patrones anómalos y estimar retrasos a corto plazo.

El proyecto se centra en tres líneas principales:

1. Predicción del retraso en una parada concreta.
2. Modelado de la propagación de retrasos a lo largo de una línea.
3. Detección temprana de incidencias operativas mediante análisis estadístico en tiempo real.

El enfoque es de predicción a corto horizonte (15–30 minutos), utilizando tanto el estado actual de la red como información contextual (clima, calendario, estructura de la red).

El sistema está diseñado siguiendo una arquitectura tipo data lake (raw → processed → cleaned) sobre almacenamiento en MinIO, garantizando trazabilidad y reproducibilidad del pipeline.

## Estructura del proyecto
```
├── src/ # Scripts de ingestión, limpieza y generación de features
├── notebooks/ # Análisis exploratorio y visualizaciones
├── docs/ # Documentación adicional
├── pyproject.toml # Configuración del entorno
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
unidos de distints fuentes pero todavía sin limpieza exhaustiva.

### cleaned/
Datos limpios y validados. Incluye:
- Eliminación de duplicados
- Corrección de tipos
- Control de outliers
- Reportes de calidad

También contiene features derivados y agregaciones temporales (p.ej. lagged_delay_1).

## Convención de nombres
Los objetos se almacenan siguiendo la convención:

grupo5/processed/gtfs_with_delays/date=YYYY-MM-DD/nombre_archivo.parquet

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

### Crear entorno, instalar dependencias y ejecutar scripts

uv sync

uv run python src/...


---

---

## Ejecución de los pipelines

El proyecto está automatizado mediante dos orquestadores principales ubicados en:

```
src/pipelines/
```

Estos permiten ejecutar la ingesta y transformación de datos de forma parametrizable y reproducible.

---

### Extracción de datos

Script principal:

```
src/pipelines/run_extraccion.py
```

Este orquestador ejecuta la descarga de datos desde las distintas fuentes externas (GTFS, clima, eventos, alertas oficiales, etc.) y los almacena en la capa `raw/` de MinIO.

#### Parámetros disponibles

- `--source`: nombre de la fuente específica o `all`
- `--start`: fecha de inicio (formato YYYY-MM-DD)
- `--end`: fecha de fin (formato YYYY-MM-DD)

#### Ejemplo de ejecución

```bash
uv run python -m src.pipelines.run_extraccion --source all --start 2025-01-01 --end 2025-01-03
```

Este comando descargará los datos del rango indicado y los almacenará en:

```
raw/
```

---

### Transformación de datos

Script principal:

```
src/pipelines/run_transform.py
```

Este orquestador procesa los datos almacenados en `raw/ y/o processed`, realiza limpieza, integración y generación de variables, y los mueve a capas superiores del data lake.

#### Parámetros disponibles

- `--source`
- `--start`
- `--end`
- `--continue_on_error`

#### Ejemplo de ejecución

```bash
uv run python -m src.pipelines.run_transform --source all --start 2025-01-01 --end 2025-01-03
```

Tras su ejecución, los datos seguirán el flujo:

```
raw/ → processed/ → cleaned/
```


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

### Flujo típico de trabajo

1. Ejecutar pipelines (extracción + transformación).
2. Abrir notebook de análisis o modelado.
3. Cargar datos desde `cleaned/`.
4. Validar features generadas.

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

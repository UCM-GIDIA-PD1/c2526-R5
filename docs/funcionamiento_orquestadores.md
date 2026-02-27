# Contrato de ingesta: `run_extraccion` y `run_transform`

Este documento define el “contrato” mínimo que deben cumplir los módulos de ingesta y transformación
por fuente para que los orquestadores `run_extraccion` y `run_transform` funcionen correctamente.

---

## 1. Orquestadores de pipeline

En este repositorio contamos con dos scripts que actúan como **orquestadores** de los
procesos de datos. Su responsabilidad es coordinar y delegar la lógica específica de cada
fuente, manteniendo la cara común de la línea de comandos y el flujo de ejecución.

- `run_extraccion` controla la fase de **extracción/ingesta**.
- `run_transform` controla la fase de **transformación**.

Los dos orquestadores comparten la misma arquitectura básica descrita a continuación,
solo difieren en el registro de funciones y el tipo de operación que ejecutan.

### 1.1 Qué hacen

1. Se invocan desde la CLI como módulos de Python (con `python -m` o `uv run` para el manejo
   de entornos). Esto garantiza que el paquete `src` esté en el `PYTHONPATH` sin necesidad de
   instalaciones adicionales.
2. Parsean argumentos con `argparse`:
   - `--source`: nombre de la fuente a procesar o `all` para incluir todas las registradas.
   - `--start` y `--end`: rango de fechas (YYYY-MM-DD) que marca el intervalo de trabajo.
   - `--continue_on_error` (opcional): si una fuente falla no se detiene el conjunto.
3. Construyen la lista de fuentes a ejecutar a partir del registro interno (un diccionario).
4. Iteran sobre las fuentes y llaman a la función asociada pasando las fechas.
5. Imprimen mensajes estándar `START`, `OK`, `FAIL` para facilitar el seguimiento y el
   análisis de logs.
6. Devuelven un código de salida 0 si todo sale bien o 1 cuando hay fallos, haciendo que
   herramientas de orquestación externas (Airflow, Make, etc.) puedan reaccionar.

### 1.2 Uso y forma de ejecución

```bash
uv run python -m src.pipelines.run_extraccion --source all --start YYYY-MM-DD --end YYYY-MM-DD
uv run python -m src.pipelines.run_transform --source gtfs_historico --start 2025-12-01 --end 2025-12-31
```

> **Nota:** Siempre deben ejecutarse como módulos para evitar problemas de importación
> cuando el directorio `src` no está instalado en el entorno.

---

## 2. Orquestador `run_extraccion`

### 2.1 Estructura

El archivo `src/pipelines/run_extraccion.py` define:

- Un tipo `IngestFn` que es simplemente `Callable[[str, str], None]`.
- Un `REGISTRY` donde cada clave es el nombre de la fuente y el valor la función que
  realizará la ingesta (`process_and_store_gtfs_range` para GTFS histórico, por ejemplo).

### 2.2 Contrato de los módulos de ingesta

Cada módulo de fuente debe exportar una función de la forma:

```python
# src/<source>/ingest.py

def nombre_func(start: str, end: str) -> None:
    """Extrae y almacena los datos crudos entre las fechas proporcionadas."""
    ...
```

El orquestador no valida las fechas ni la lógica, sólo se encarga de invocarla.

---

## 3. Orquestador `run_transform`

Este funciona de manera análoga a `run_extraccion`, pero el tipo registrado es
`TransformFn` y las funciones se encuentran en `src/<source>/transform.py`.

Un módulo de transformación tiene la firma:

```python
# src/<source>/transform.py

def run_transform(start: str, end: str) -> None:
    """Aplica transformaciones a los datos raw y genera outputs procesados."""
    ...
```

### 3.1 Flujo de trabajo

1. `run_transform` se lanza con el rango de fechas.
2. Crea una lista de fuentes y llama a cada función registrada.
3. Cada función transforma los datos brutos ubicados bajo `data/raw/...` y escribe
   resultados en `data/processed/...`. O bien transforma de `data/processed/...` a `data/cleaned/...`

> El contrato es el mismo que con la ingesta: el orquestador sólo coordina.

---

## 4. Extensión futura

Para añadir nuevas fuentes o etapas basta con:

1. Crear el módulo correspondiente (`<source>/ingest.py` o `<source>/transform.py`).
2. Implementar una función `run_transform` u otro nombre con la firma adecuada.
3. Registrar la función en el `REGISTRY` del orquestador.

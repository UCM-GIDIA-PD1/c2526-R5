"""
ingest.py — Orquestador de la ingesta de eventos para NYC.

Llama secuencialmente a los tres subscripts:
  - deportes      → ESPN
  - conciertos    → Setlist.fm
  - eventos_nyc   → NYC Open Data


Desde run_extraccion, se llama a:
  ingest_eventos(start_date, end_date)
"""

import sys

from .conciertos  import ingest_conciertos
from .deportes    import ingest_deportes
from .eventos_nyc import ingest_eventos_nyc


#  Registro de subscripts

SUBSCRIPTS = {
    "deportes":    ingest_deportes,
    "conciertos":  ingest_conciertos,
    "eventos_nyc": ingest_eventos_nyc,
}



#  Función pública (llamada desde run_extraccion)

def ingest_eventos(start_date, end_date):
    """
    Ejecuta la ingesta completa de eventos (deportes + conciertos + NYC Open Data)
    para el rango [start_date, end_date].
    """
    failed = []

    for name, fn in SUBSCRIPTS.items():
        print(f"\n[eventos] ── START {name} ──────────────────────────")
        try:
            fn(start_date, end_date)
            print(f"[eventos] ── OK    {name}")
        except Exception as exc:
            print(f"[eventos] ── FAIL  {name}: {exc}", file=sys.stderr)
            failed.append(name)

    if failed:
        raise RuntimeError(f"Fallaron los siguientes subscripts de eventos: {failed}")

    print("\n[eventos] Todos los subscripts completados correctamente.")
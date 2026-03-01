"""
Orquestador de la etapa de extracción (raw) para todas las fuentes.

Uso:
  uv run python -m src.pipelines.run_extraccion --source all --start 2025-02-01 --end 2025-02-07
  uv run python -m src.pipelines.run_extraccion --source clima --start 2025-02-01 --end 2025-02-07
"""

import argparse
import sys
from typing import Callable, Dict, List

from src.clima.extraccion_historico_clima import ingest_clima_historico
from src.eventos.ingest import ingest_eventos
from src.gtfs_historico.ingest import process_and_store_gtfs_range as ingest_gtfs_historico
from src.alertas_oficiales_tiempo_real.extraccion_historico_2025 import ingest_alertas



IngestFn = Callable[[str, str], None]

REGISTRY: Dict[str, IngestFn] = {
    "eventos": ingest_eventos,
    "clima": ingest_clima_historico,
    "gtfs_historico": ingest_gtfs_historico,
    #"tiempo_real_metro": ingest_tiempo_real,
    "alertas_oficiales_tiempo_real": ingest_alertas,
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta la ingesta (raw) para una o varias fuentes."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(REGISTRY.keys()) + ["all"],
        help="Fuente a ingestar, o 'all' para todas.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Fecha inicio (YYYY-MM-DD), inclusive.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Fecha fin (YYYY-MM-DD), inclusive.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Si una fuente falla, continúa con las demás.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    sources = list(REGISTRY.keys()) if args.source == "all" else [args.source]

    failed: List[str] = []

    for src_name in sources:
        fn = REGISTRY[src_name]
        print(f"[run_ingest] START source={src_name} start={args.start} end={args.end}")
        try:
            fn(args.start, args.end)
            print(f"[run_ingest] OK    source={src_name}")
        except Exception as e:
            print(f"[run_ingest] FAIL  source={src_name} error={repr(e)}", file=sys.stderr)
            failed.append(src_name)
            if not args.continue_on_error:
                break

    if failed:
        print(f"[run_ingest] Completed with failures: {failed}", file=sys.stderr)
        return 1

    print("[run_ingest] Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

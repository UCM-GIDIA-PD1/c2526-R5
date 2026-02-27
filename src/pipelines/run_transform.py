"""
Orquestador de la etapa de transformación para una o varias fuentes.

Uso:
  [uv run] python -m src.pipelines.run_transform --source gtfs_historico --start 2025-12-01 --end 2025-12-31
"""

import argparse
import sys
from typing import Callable, Dict, List

# importar funciones de transformación de cada fuente
from src.gtfs_historico.transform import run_transform as transform_gtfs_historico
from src.eventos.transform import run_transform as transform_eventos

TransformFn = Callable[[str, str], None]

REGISTRY: Dict[str, TransformFn] = {
    "gtfs_historico": transform_gtfs_historico,
    "eventos":  transform_eventos
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta la transformación para una o varias fuentes."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(REGISTRY.keys()) + ["all"],
        help="Fuente a transformar, o 'all' para todas.",
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
        print(f"[run_transform] START source={src_name} start={args.start} end={args.end}")
        try:
            fn(args.start, args.end)
            print(f"[run_transform] OK    source={src_name}")
        except Exception as e:
            print(f"[run_transform] FAIL  source={src_name} error={repr(e)}", file=sys.stderr)
            failed.append(src_name)
            if not args.continue_on_error:
                break

    if failed:
        print(f"[run_transform] Completed with failures: {failed}", file=sys.stderr)
        return 1

    print("[run_transform] Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
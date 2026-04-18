"""
Lanzador nocturno — búsqueda ventana × peso para delay_30m y delay_end
=======================================================================

Ejecuta los dos scripts en secuencia. Cada uno pre-carga los datos una sola
vez y luego entrena las 28 combinaciones (7 ventanas × 4 esquemas de pesos).

Duración estimada total: ~3-5h

Uso:
    uv run python run_search_overnight.py
"""

import time
import sys
from datetime import datetime

from src.models.prediccion_retrasos.delay_30m.search.window_weight_search import run_search as run_30m
from src.models.prediccion_retrasos.delay_end.search.window_weight_search  import run_search as run_end

CSV_30M = "window_weight_search_30m_results.csv"
CSV_END = "window_weight_search_end_results.csv"


def banner(msg):
    print(f"\n{'═'*70}\n  {msg}\n{'═'*70}\n")


def print_best(csv, label):
    import pandas as pd
    try:
        df = pd.read_csv(csv).sort_values("test_mae_s")
        best = df.iloc[0]
        print(f"  {label}:")
        print(f"    Mejor  : {best['window']}  +  {best['weight_scheme']}")
        print(f"    Test   MAE = {best['test_mae_s']:.1f}s  ({best['test_mae_min']:.2f} min)  R²={best['test_r2']:.4f}")
        print(f"    Train  MAE = {best['train_mae_s']:.1f}s")
        print(f"\n    Top 5:")
        cols = ["window", "weight_scheme", "test_mae_s", "test_r2", "train_mae_s"]
        print(df[cols].head(5).to_string(index=False))
    except FileNotFoundError:
        print(f"  {label}: CSV no encontrado ({csv})")


def main():
    t_start = time.time()
    print(f"\nInicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("56 combinaciones en total (28 por modelo)")

    # ── delay_30m ────────────────────────────────────────────────────────────
    banner("PARTE 1 / 2 — delay_30m  (>= 30 min restantes)  |  28 combinaciones")
    t0 = time.time()
    try:
        run_30m()
        print(f"\n✓ delay_30m completado en {(time.time()-t0)/3600:.1f}h")
    except Exception as e:
        print(f"\n✗ delay_30m falló: {e}")
        sys.exit(1)

    # ── delay_end ────────────────────────────────────────────────────────────
    banner("PARTE 2 / 2 — delay_end  (<  30 min restantes)  |  28 combinaciones")
    t0 = time.time()
    try:
        run_end()
        print(f"\n✓ delay_end completado en {(time.time()-t0)/3600:.1f}h")
    except Exception as e:
        print(f"\n✗ delay_end falló: {e}")
        sys.exit(1)

    # ── Resumen ───────────────────────────────────────────────────────────────
    banner(f"RESUMEN FINAL  —  tiempo total: {(time.time()-t_start)/3600:.1f}h")
    print_best(CSV_30M, "delay_30m")
    print()
    print_best(CSV_END,  "delay_end")
    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

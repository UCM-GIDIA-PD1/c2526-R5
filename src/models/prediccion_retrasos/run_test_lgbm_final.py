"""
Orquestador de evaluación final — Entrega 4
============================================

Ejecuta secuencialmente los scripts de evaluación de los dos modelos:
    1. delay_30m  →  src/models/prediccion_retrasos/delay_30m/test/eval_lgbm.py
    2. delay_end  →  src/models/prediccion_retrasos/delay_end/test/eval_lgbm.py

Cada script re-entrena con la configuración final (ventana desde_jul25,
pesos exponenciales, 7 meses) y sube el modelo como artifact a W&B.

Se lanzan en subprocesos separados para aislar la memoria entre ejecuciones.

Uso:
    uv run python src/models/prediccion_retrasos/run_test_lgbm_final.py

Variables de entorno necesarias:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    WANDB_API_KEY  (o haber hecho `wandb login` previamente)
"""

import subprocess
import sys
import time


SCRIPTS = [
    (
        "delay_30m",
        "src/models/prediccion_retrasos/delay_30m/test/eval_lgbm.py",
    ),
    (
        "delay_end",
        "src/models/prediccion_retrasos/delay_end/test/eval_lgbm.py",
    ),
]


def run_script(name: str, path: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  Iniciando: {name}  ({path})")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        check=False,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [{name}] OK  ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"\n  [{name}] FALLÓ con código {result.returncode}  ({elapsed/60:.1f} min)")
        return False


def main():
    print("\nEvaluación final — delay_30m + delay_end")
    print(f"Scripts a ejecutar: {len(SCRIPTS)}\n")

    t_total = time.time()
    results: dict[str, bool] = {}

    for name, path in SCRIPTS:
        results[name] = run_script(name, path)

    elapsed_total = time.time() - t_total

    print(f"\n{'='*70}")
    print(f"Resumen  (tiempo total: {elapsed_total/60:.1f} min)")
    print(f"{'='*70}")
    for name, ok in results.items():
        status = "OK" if ok else "FALLÓ"
        print(f"  {name:<15} {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Baseline 1:

Asume que la predicción para todos los horizontes es el último valor observado
de delay_seconds en el instante actual.

Carga:  artefactos/tensores.pt
Imprime: MAE real (segundos) por horizonte y MAE medio.

Uso
---
    uv run python src/models/propagacion_estacion/03_baseline_naive.py
"""
from pathlib import Path

import numpy as np
import torch

from utils.metrics import mae_por_horizonte, rmse_por_horizonte, imprimir_metricas

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"

# Índices en la dimensión F=14 de X_* (FEATURE_COLS de 02_generar_tensores.py)
IDX_DELAY_SECONDS = 0
IDX_TEMP_EXTREME  = 4
HORIZONTES        = ['10m', '20m', '30m']


def _imprimir_segmentos(
    preds_real: np.ndarray,
    trues_real: np.ndarray,
    dow_te: np.ndarray,
    temp_te: np.ndarray,
    nombre: str,
) -> None:
    """Imprime MAE por segmento (fin de semana / entre semana / clima extremo / normal)."""
    mask_finde   = dow_te >= 5
    mask_diario  = dow_te <  5
    mask_extremo = temp_te >= 0.5
    mask_normal  = temp_te <  0.5

    def _mae(mask: np.ndarray) -> float:
        if not mask.any():
            return float('nan')
        return float(np.abs(preds_real[mask] - trues_real[mask]).mean())

    fds = _mae(mask_finde)
    dia = _mae(mask_diario)
    ext = _mae(mask_extremo)
    nor = _mae(mask_normal)

    print(f"\n{'='*70}")
    print(f"TABLA 2 — EVALUACIÓN POR SEGMENTOS — {nombre}")
    print(f"{'='*70}")
    print(f"{'Modelo':<25} {'FdS MAE':>10} {'Diario MAE':>12} {'Extremo MAE':>13} {'Normal MAE':>11}")
    print(f"{'-'*70}")
    print(f"{nombre:<25} {fds:>10.2f} {dia:>12.2f} {ext:>13.2f} {nor:>11.2f}")
    print(f"{'='*70}")
    print("Unidades: MAE en segundos reales. FdS = Fines de semana.")


def main():
    print("03: Baseline Naive (Persistencia)")

    datos    = torch.load(RUTA_TENSORES, weights_only=False)
    X_test   = datos['X_test']   # (T, N, 14)
    Y_test   = datos['Y_test']   # (T, N, 3)
    scaler_Y = datos['scaler_Y']
    times    = datos['times']    # DatetimeIndex completo

    T, N, H = Y_test.shape
    times_te = times[-T:]        # últimos T timestamps = período de test

    # La predicción naive replica el último delay observado para todos los horizontes.
    # X_test[:, :, IDX_DELAY_SECONDS] está en espacio escalado (scaler_X);
    # para comparar en el mismo espacio que Y_test (escalado con scaler_Y),
    # desescalamos Y_test y generamos predicciones en espacio real.
    scaler_X = datos['scaler_X']

    # Desescalar X para recuperar delay_seconds en segundos reales
    X_test_2d    = X_test.reshape(-1, X_test.shape[-1])
    X_test_real  = scaler_X.inverse_transform(X_test_2d).reshape(T, N, -1)
    delay_actual = X_test_real[:, :, IDX_DELAY_SECONDS]  # (T, N)

    # Predicción naive: mismo valor para los 3 horizontes → (T, N, 3)
    preds_real = np.stack([delay_actual] * H, axis=-1)

    # Ground truth en espacio real
    Y_test_2d  = Y_test.reshape(-1, H)
    trues_real = scaler_Y.inverse_transform(Y_test_2d).reshape(T, N, H)

    # Métricas directamente en espacio real (scaler identidad)
    from sklearn.preprocessing import StandardScaler as _SS
    _identity = _SS()
    _identity.mean_  = np.zeros(H)
    _identity.scale_ = np.ones(H)
    _identity.var_   = np.ones(H)
    _identity.n_features_in_ = H

    mae  = {
        f'mae_{h}': float(np.abs(preds_real[:, :, i] - trues_real[:, :, i]).mean())
        for i, h in enumerate(HORIZONTES)
    }
    mae['mae_mean'] = float(np.mean(list(mae.values())))

    rmse = {
        f'rmse_{h}': float(np.sqrt(np.mean((preds_real[:, :, i] - trues_real[:, :, i]) ** 2)))
        for i, h in enumerate(HORIZONTES)
    }
    rmse['rmse_mean'] = float(np.mean(list(rmse.values())))

    imprimir_metricas(mae, rmse)
    print(f"\nBaseline naive MAE medio: {mae['mae_mean']:.1f} s")
    print("(El modelo DCRNN debe superar este valor para aportar valor predictivo.)")

    # ── Evaluación por segmentos ──────────────────────────────────────────────
    dow_te  = np.array(times_te.dayofweek)                                # (T,)
    temp_te = (X_test_real[:, :, IDX_TEMP_EXTREME] > 0.5).mean(axis=1)   # (T,)
    _imprimir_segmentos(preds_real, trues_real, dow_te, temp_te, "Naive (Persistencia)")


if __name__ == "__main__":
    main()

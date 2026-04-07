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

# Índice de delay_seconds en la dimensión F=14 de X_*
IDX_DELAY_SECONDS = 0
HORIZONTES        = ['10m', '20m', '30m']


def main():
    print("03: Baseline Naive (Persistencia)")

    datos    = torch.load(RUTA_TENSORES, weights_only=False)
    X_test   = datos['X_test']   # (T, N, 14)
    Y_test   = datos['Y_test']   # (T, N, 3)
    scaler_Y = datos['scaler_Y']

    T, N, H = Y_test.shape

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


if __name__ == "__main__":
    main()

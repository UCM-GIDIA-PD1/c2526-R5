"""
Baseline 2: Historical Average

Calcula la media histórica de retraso por (estación, día_semana, hora) usando
los datos de entrenamiento, y aplica esa media como predicción en el conjunto
de test para todos los horizontes.

Este baseline captura patrones cíclicos (hora punta, diferencias fin de semana),
pero sin ningún componente de aprendizaje.

Carga:  artefactos/tensores.pt
Imprime: MAE real (segundos) por horizonte y MAE medio.

Uso
---
    uv run python src/models/prediccion_propagacion/04_baseline_ha.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.metrics import imprimir_metricas

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"

IDX_DELAY_SECONDS = 0   # índice de delay_seconds en X (dimensión F=14)
HORIZONTES        = ['10m', '20m', '30m']


def main():
    print("=== 04 Baseline Historical Average (estación / día_semana / hora) ===")

    datos    = torch.load(RUTA_TENSORES, weights_only=False)
    X_train  = datos['X_train']  # (T_tr, N, 14)
    Y_train  = datos['Y_train']  # (T_tr, N, 3)
    X_test   = datos['X_test']   # (T_te, N, 14)
    Y_test   = datos['Y_test']   # (T_te, N, 3)
    times    = datos['times']    # pd.DatetimeIndex completo (T_total,)
    nodes    = datos['nodes']    # list[str]
    scaler_X = datos['scaler_X']
    scaler_Y = datos['scaler_Y']

    T_tr = X_train.shape[0]
    T_te = X_test.shape[0]
    N    = X_train.shape[1]
    H    = Y_train.shape[-1]

    # Reconstruir índices temporales de cada split
    # times cubre todo el período; los splits son los primeros T_tr, luego T_va, luego T_te
    T_total  = len(times)
    T_va     = T_total - T_tr - T_te
    times_tr = times[:T_tr]
    times_te = times[T_tr + T_va:]

    # Desescalar X_train para obtener delay_seconds en segundos reales
    X_tr_2d    = X_train.reshape(-1, X_train.shape[-1])
    X_tr_real  = scaler_X.inverse_transform(X_tr_2d).reshape(T_tr, N, -1)
    delay_tr   = X_tr_real[:, :, IDX_DELAY_SECONDS]  # (T_tr, N)

    # Desescalar Y_train para calcular la media de los targets en segundos reales
    Y_tr_2d   = Y_train.reshape(-1, H)
    Y_tr_real = scaler_Y.inverse_transform(Y_tr_2d).reshape(T_tr, N, H)

    # Construir DataFrame de train con claves temporales
    filas = []
    for t, ts in enumerate(times_tr):
        dow  = ts.dayofweek
        hora = ts.hour
        for n in range(N):
            filas.append({
                'nodo': n,
                'dow':  dow,
                'hora': hora,
                **{f'y{h}': Y_tr_real[t, n, h] for h in range(H)},
            })
    df_tr = pd.DataFrame(filas)

    # Media histórica por (nodo, día_semana, hora)
    medias = df_tr.groupby(['nodo', 'dow', 'hora'])[[f'y{h}' for h in range(H)]].mean()

    # Ground truth en espacio real
    Y_te_2d   = Y_test.reshape(-1, H)
    trues_real = scaler_Y.inverse_transform(Y_te_2d).reshape(T_te, N, H)

    # Predicciones: lookup en la tabla de medias históricas
    preds_real = np.zeros((T_te, N, H), dtype=np.float32)
    for t, ts in enumerate(times_te):
        dow  = ts.dayofweek
        hora = ts.hour
        for n in range(N):
            key = (n, dow, hora)
            if key in medias.index:
                for h in range(H):
                    preds_real[t, n, h] = medias.loc[key, f'y{h}']
            else:
                # Fallback: media global del nodo
                for h in range(H):
                    preds_real[t, n, h] = df_tr[df_tr['nodo'] == n][f'y{h}'].mean()

    # Métricas
    mae = {
        f'mae_{hz}': float(np.abs(preds_real[:, :, i] - trues_real[:, :, i]).mean())
        for i, hz in enumerate(HORIZONTES)
    }
    mae['mae_mean'] = float(np.mean(list(mae.values())))

    rmse = {
        f'rmse_{hz}': float(np.sqrt(np.mean((preds_real[:, :, i] - trues_real[:, :, i]) ** 2)))
        for i, hz in enumerate(HORIZONTES)
    }
    rmse['rmse_mean'] = float(np.mean(list(rmse.values())))

    imprimir_metricas(mae, rmse)
    print(f"\nBaseline HA MAE medio: {mae['mae_mean']:.1f} s")
    print("(El modelo DCRNN debe superar este valor para justificar la complejidad del grafo.)")


if __name__ == "__main__":
    main()

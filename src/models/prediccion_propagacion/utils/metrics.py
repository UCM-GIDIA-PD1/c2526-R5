"""
Métricas de evaluación para el modelo DCRNN.

Todas las funciones operan sobre arrays NumPy en espacio escalado y devuelven
las métricas en el espacio original (segundos reales) usando el scaler_Y ajustado
sobre el conjunto de entrenamiento.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _desescalar(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Aplica inverse_transform de StandardScaler a un array (T, N, H).

    Devuelve un array de la misma forma en unidades originales (segundos).
    """
    T, N, H = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, H)).reshape(T, N, H)


def mae_por_horizonte(
    preds: np.ndarray,
    trues: np.ndarray,
    scaler: StandardScaler,
    horizontes: list[str] | None = None,
) -> dict[str, float]:
    """
    Calcula el MAE (segundos reales) para cada horizonte de predicción.

    Parámetros
    ----------
    preds : (T, N, H)  predicciones escaladas
    trues : (T, N, H)  valores reales escalados
    scaler : StandardScaler ajustado sobre Y_train
    horizontes : nombres de los horizontes; por defecto ['10m', '20m', '30m']

    Devuelve
    --------
    dict  con claves 'mae_<horizonte>' y 'mae_mean'
    """
    if horizontes is None:
        horizontes = ['10m', '20m', '30m']

    preds_inv = _desescalar(preds, scaler)
    trues_inv = _desescalar(trues, scaler)

    resultado: dict[str, float] = {}
    for i, h in enumerate(horizontes):
        resultado[f'mae_{h}'] = float(np.abs(preds_inv[:, :, i] - trues_inv[:, :, i]).mean())

    resultado['mae_mean'] = float(np.mean(list(resultado.values())))
    return resultado


def rmse_por_horizonte(
    preds: np.ndarray,
    trues: np.ndarray,
    scaler: StandardScaler,
    horizontes: list[str] | None = None,
) -> dict[str, float]:
    """
    Calcula el RMSE (segundos reales) para cada horizonte de predicción.

    Parámetros
    ----------
    preds : (T, N, H)  predicciones escaladas
    trues : (T, N, H)  valores reales escalados
    scaler : StandardScaler ajustado sobre Y_train
    horizontes : nombres de los horizontes; por defecto ['10m', '20m', '30m']

    Devuelve
    --------
    dict  con claves 'rmse_<horizonte>' y 'rmse_mean'
    """
    if horizontes is None:
        horizontes = ['10m', '20m', '30m']

    preds_inv = _desescalar(preds, scaler)
    trues_inv = _desescalar(trues, scaler)

    resultado: dict[str, float] = {}
    for i, h in enumerate(horizontes):
        resultado[f'rmse_{h}'] = float(
            np.sqrt(np.mean((preds_inv[:, :, i] - trues_inv[:, :, i]) ** 2))
        )

    resultado['rmse_mean'] = float(np.mean(list(resultado.values())))
    return resultado


def imprimir_metricas(mae: dict[str, float], rmse: dict[str, float]) -> None:
    """Imprime un resumen formateado de MAE y RMSE."""
    print("Horizonte │      MAE (s) │     RMSE (s)")
    print("──────────┼─────────────┼─────────────")
    for key in mae:
        if key == 'mae_mean':
            continue
        h = key.replace('mae_', '')
        print(f"  {h:>6}  │  {mae[key]:>10.1f} │  {rmse.get('rmse_' + h, float('nan')):>10.1f}")
    print("──────────┼─────────────┼─────────────")
    print(f"   Media  │  {mae['mae_mean']:>10.1f} │  {rmse.get('rmse_mean', float('nan')):>10.1f}")

"""
Evaluación comparativa de los tres modelos GNN en el conjunto de Test.

ÚNICO script que toca el conjunto de Test. Implementa el Apartado 4 de la
rúbrica:
  1. Métricas principales y complementarias  (MAE, RMSE, R²)
  2. Evaluación por segmentos relevantes      (fin de semana vs. diario,
                                               clima extremo vs. normal)
  3. Influencia de las variables de entrada   (Permutation Feature Importance)

Carga:
  artefactos/dcrnn_final.pth   — pesos DCRNN + metadata
  artefactos/tensores.pt       — X_test, Y_test, scaler_X, scaler_Y, times, nodes
  artefactos/grafo.pt          — edge_index, edge_weight
  artefactos/stgcn_final.pth   — pesos STGCN + test data + metadata
  artefactos/astgcn_final.pth  — pesos ASTGCN + test data + metadata

Uso
---
    uv run python src/models/propagacion_estacion/12_evaluacion_modelos.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

# ── Importar arquitecturas de modelos ─────────────────────────────────────────
DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from models.dcrnn import SubwayDCRNN
from models.stgcn import STGCN_Metro
from models.astgcn import ASTGCN_Metro, calcular_scaled_laplacian, calcular_polinomios_chebyshev

# ── Rutas a los artefactos ────────────────────────────────────────────────────
RUTA_DCRNN_PTH  = DIR / "artefactos" / "dcrnn_final.pth"
RUTA_TENSORES   = DIR / "artefactos" / "tensores.pt"
RUTA_GRAFO      = DIR / "artefactos" / "grafo.pt"
RUTA_STGCN_PTH  = DIR / "artefactos" / "stgcn_final.pth"
RUTA_ASTGCN_PTH = DIR / "artefactos" / "astgcn_final.pth"

# Nombres de features del pipeline DCRNN (02_generar_tensores.py — FEATURE_COLS)
FEATURE_COLS_DCRNN = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
    'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay', 'actual_headway_seconds',
    'hour_sin', 'hour_cos', 'dow', 'afecta_previo', 'afecta_durante', 'afecta_despues',
]
# Nombres de features del pipeline STGCN/ASTGCN
FEATURE_COLS_STGCN = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
    'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay',
    'actual_headway_seconds', 'hour_sin', 'hour_cos', 'dow',
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset de ventana deslizante genérico
# ─────────────────────────────────────────────────────────────────────────────

class VentanaDataset(Dataset):
    """Genera ventanas (history_len, N, F) → etiqueta (N, C)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, history_len: int):
        self.X  = torch.as_tensor(X, dtype=torch.float32)
        self.Y  = torch.as_tensor(Y, dtype=torch.float32)
        self.hl = history_len

    def __len__(self):
        return len(self.X) - self.hl

    def __getitem__(self, idx):
        return self.X[idx: idx + self.hl], self.Y[idx + self.hl]


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia y desescalado
# ─────────────────────────────────────────────────────────────────────────────

def inferir(model_fn, X_scaled: np.ndarray, Y_scaled: np.ndarray,
            history_len: int, batch_size: int, device: torch.device,
            scaler_Y) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta el modelo sobre todo el conjunto de test y devuelve predicciones y
    valores reales desescalados.

    Parámetros
    ----------
    model_fn   : callable que acepta un tensor (B, hl, N, F) y devuelve (B, N, C).
    X_scaled   : (T_test, N, F)  — features escaladas
    Y_scaled   : (T_test, N, C)  — targets escalados
    history_len: longitud de la ventana de entrada
    batch_size : tamaño de lote para inferencia
    device     : dispositivo de cómputo
    scaler_Y   : StandardScaler ajustado sobre Y_train

    Devuelve
    --------
    preds_real : (T_test - history_len, N, C) — predicciones en segundos reales
    trues_real : (T_test - history_len, N, C) — valores reales en segundos
    """
    loader = DataLoader(
        VentanaDataset(X_scaled, Y_scaled, history_len),
        batch_size=batch_size, shuffle=False,
    )
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds_list.append(model_fn(xb.to(device)).cpu().numpy())
            trues_list.append(yb.numpy())

    preds = np.concatenate(preds_list, axis=0)   # (M, N, C)
    trues = np.concatenate(trues_list, axis=0)

    M, N, C = preds.shape
    preds_real = scaler_Y.inverse_transform(preds.reshape(-1, C)).reshape(M, N, C)
    trues_real = scaler_Y.inverse_transform(trues.reshape(-1, C)).reshape(M, N, C)
    return preds_real, trues_real


# ─────────────────────────────────────────────────────────────────────────────
# Métricas globales
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(preds: np.ndarray, trues: np.ndarray) -> dict:
    """
    Calcula MAE, RMSE y R² promediados sobre todos los nodos y horizontes.

    preds / trues : (M, N, C)
    """
    p = preds.reshape(-1)
    t = trues.reshape(-1)
    mae  = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    return {'MAE (s)': mae, 'RMSE (s)': rmse, 'R²': r2}


def imprimir_tabla_metricas(resultados: dict[str, dict]) -> None:
    """Imprime tabla comparativa de métricas lista para copiar a la memoria."""
    print("\n" + "=" * 60)
    print("TABLA 1 — MÉTRICAS COMPARATIVAS EN TEST")
    print("=" * 60)
    print(f"{'Modelo':<10} {'MAE (s)':>10} {'RMSE (s)':>10} {'R²':>8}")
    print("-" * 60)
    for nombre, m in resultados.items():
        print(f"{nombre:<10} {m['MAE (s)']:>10.2f} {m['RMSE (s)']:>10.2f} {m['R²']:>8.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación por segmentos
# ─────────────────────────────────────────────────────────────────────────────

def mae_segmento(preds: np.ndarray, trues: np.ndarray,
                 mascara: np.ndarray) -> float:
    """
    MAE sobre el subconjunto de pasos temporales indicados por mascara.

    preds / trues : (M, N, C)
    mascara       : (M,)  booleano — True = incluir en el segmento
    """
    p = preds[mascara].reshape(-1)
    t = trues[mascara].reshape(-1)
    return float(mean_absolute_error(t, p))


def evaluar_segmentos(resultados_modelos: dict, etiquetas_por_modelo: dict) -> None:
    """
    Compara MAE para cada modelo en cuatro segmentos:
      · Fines de semana (dow ∈ {5, 6})  vs. Días de diario (dow ∈ {0..4})
      · Clima extremo (temp_extreme == 1) vs. Clima normal (temp_extreme == 0)

    etiquetas_por_modelo : {
        nombre: {
            'dow'  : np.ndarray (M_nombre,) — day-of-week promediado sobre nodos,
            'temp' : np.ndarray (M_nombre,) — temp_extreme promediado sobre nodos,
        }, ...
    }
    Cada modelo puede tener distinta longitud M si los test sets difieren (p.ej.
    cuando los modelos se entrenaron con distinto número de meses de datos).
    Las máscaras se calculan por modelo para que coincidan siempre con la forma
    de sus predicciones.
    """
    print("\n" + "=" * 70)
    print("TABLA 2 — EVALUACIÓN POR SEGMENTOS RELEVANTES")
    print("=" * 70)
    cabecera = f"{'Modelo':<10} {'FdS MAE':>10} {'Diario MAE':>12} {'Extremo MAE':>13} {'Normal MAE':>11}"
    print(cabecera)
    print("-" * 70)

    maes_acum: dict[str, dict[str, float]] = {
        'finde': {}, 'diario': {}, 'extremo': {}, 'normal': {}
    }
    for nombre, (preds, trues) in resultados_modelos.items():
        etiq = etiquetas_por_modelo[nombre]
        dow  = etiq['dow']   # (M_nombre,)
        temp = etiq['temp']  # (M_nombre,)

        mask_finde   = dow  >= 5.0
        mask_diario  = dow  <  5.0
        mask_extremo = temp >= 0.5
        mask_normal  = temp <  0.5

        fds  = mae_segmento(preds, trues, mask_finde)   if mask_finde.any()   else float('nan')
        dia  = mae_segmento(preds, trues, mask_diario)  if mask_diario.any()  else float('nan')
        ext  = mae_segmento(preds, trues, mask_extremo) if mask_extremo.any() else float('nan')
        nor  = mae_segmento(preds, trues, mask_normal)  if mask_normal.any()  else float('nan')
        print(f"{nombre:<10} {fds:>10.2f} {dia:>12.2f} {ext:>13.2f} {nor:>11.2f}")

        maes_acum['finde'][nombre]   = fds
        maes_acum['diario'][nombre]  = dia
        maes_acum['extremo'][nombre] = ext
        maes_acum['normal'][nombre]  = nor

    print("=" * 70)
    print("Unidades: MAE en segundos reales. FdS = Fines de semana.")

    # Indicar qué modelo es mejor en cada segmento
    for segmento, key in [('Fin de semana', 'finde'), ('Diario', 'diario'),
                           ('Clima extremo', 'extremo'), ('Clima normal', 'normal')]:
        maes = {n: v for n, v in maes_acum[key].items() if not np.isnan(v)}
        if not maes:
            continue
        mejor = min(maes, key=maes.get)
        print(f"  → Mejor en '{segmento}': {mejor} (MAE={maes[mejor]:.2f}s)")

    return maes_acum


# ─────────────────────────────────────────────────────────────────────────────
# Permutation Feature Importance (PFI)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_pfi(
    model_fn,
    X_test_scaled: np.ndarray,
    Y_test_scaled: np.ndarray,
    history_len: int,
    batch_size: int,
    device: torch.device,
    scaler_Y,
    feature_names: list[str],
    grupos: dict[str, list[int]],
    n_rep: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """
    Permutation Feature Importance (PFI) sobre el conjunto de Test.

    Para cada grupo de features, baraja sus valores a lo largo del eje temporal
    (manteniendo la coherencia intra-ventana) y mide la degradación del MAE.

    Parámetros
    ----------
    model_fn      : callable (B, hl, N, F) → (B, N, C) en CPU
    X_test_scaled : (T, N, F)
    Y_test_scaled : (T, N, C)
    history_len   : longitud de la ventana de entrada
    batch_size    : tamaño de lote
    device        : dispositivo de cómputo
    scaler_Y      : StandardScaler
    feature_names : lista de nombres de features en el mismo orden que el eje F
    grupos        : {nombre_grupo: [índices de features]} — los índices dentro del
                    mismo grupo se barajan con la misma permutación para preservar
                    la correlación interna (p. ej. hour_sin y hour_cos)
    n_rep         : número de repeticiones para reducir varianza
    seed          : semilla aleatoria

    Devuelve
    --------
    dict {nombre_grupo: importancia_media}  — mayor valor = más importante
    """
    rng = np.random.default_rng(seed)

    # MAE de línea base
    preds_base, trues_base = inferir(model_fn, X_test_scaled, Y_test_scaled,
                                     history_len, batch_size, device, scaler_Y)
    mae_base = float(mean_absolute_error(trues_base.reshape(-1), preds_base.reshape(-1)))

    importancias = {}
    for grupo_nombre, feat_indices in grupos.items():
        maes_rep = []
        for _ in range(n_rep):
            X_perm = X_test_scaled.copy()   # (T, N, F)
            # Barajar los pasos temporales de las features del grupo con la MISMA
            # permutación (preserva correlación entre features agrupadas)
            perm = rng.permutation(X_perm.shape[0])
            for fi in feat_indices:
                X_perm[:, :, fi] = X_perm[perm, :, fi]
            preds_p, trues_p = inferir(model_fn, X_perm, Y_test_scaled,
                                       history_len, batch_size, device, scaler_Y)
            maes_rep.append(float(mean_absolute_error(trues_p.reshape(-1), preds_p.reshape(-1))))
        importancias[grupo_nombre] = float(np.mean(maes_rep)) - mae_base

    return importancias


def imprimir_ranking_pfi(importancias: dict[str, float], nombre_modelo: str) -> None:
    """Imprime el ranking de importancia de variables listo para la memoria."""
    print(f"\n{'=' * 60}")
    print(f"TABLA 3 — PFI SOBRE MODELO GANADOR: {nombre_modelo}")
    print(f"{'=' * 60}")
    print(f"{'Rank':<5} {'Variable / Grupo':<30} {'ΔMAE (s)':>10}")
    print("-" * 60)
    ranking = sorted(importancias.items(), key=lambda x: x[1], reverse=True)
    for rk, (nombre, delta) in enumerate(ranking, start=1):
        barra = "█" * max(0, int(delta / max(v for v in importancias.values()) * 20))
        print(f"{rk:<5} {nombre:<30} {delta:>10.3f}  {barra}")
    print("=" * 60)
    print("ΔMAE > 0 indica que la variable es relevante (mayor degradación al barajarla).")


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de grupos de features para PFI
# ─────────────────────────────────────────────────────────────────────────────

def construir_grupos_pfi(feature_names: list[str]) -> dict[str, list[int]]:
    """
    Construye el diccionario de grupos PFI a partir de los nombres de features.

    Agrupa hour_sin + hour_cos como 'Hora del día' y lagged_delay_1 +
    lagged_delay_2 como 'Retrasos previos' para evitar problemas de
    colinealidad. El resto se tratan individualmente.
    """
    grupos_especiales = {
        'Hora del día':    ['hour_sin', 'hour_cos'],
        'Retrasos previos': ['lagged_delay_1', 'lagged_delay_2'],
    }
    # Features que ya están cubiertas por un grupo especial
    ya_agrupadas = {f for fs in grupos_especiales.values() for f in fs}

    grupos: dict[str, list[int]] = {}

    # Añadir grupos especiales si las features están presentes
    for nombre_grupo, feats in grupos_especiales.items():
        indices = [feature_names.index(f) for f in feats if f in feature_names]
        if len(indices) == len(feats):       # solo si el grupo está completo
            grupos[nombre_grupo] = indices

    # Añadir features individuales (no agrupadas)
    for i, f in enumerate(feature_names):
        if f not in ya_agrupadas:
            grupos[f] = [i]

    return grupos


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "evaluacion-final-gnn"


def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 32

    print("=" * 60)
    print("=== 12 Evaluación de Modelos GNN en Test ===")
    print(f"Device: {device}")
    print("=" * 60)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={'batch_size': batch_size, 'device': str(device)},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CARGAR MODELOS Y DATOS DE TEST
    # ══════════════════════════════════════════════════════════════════════════

    # ── DCRNN ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Cargando DCRNN...")
    dcrnn_ckpt = torch.load(RUTA_DCRNN_PTH, weights_only=False)
    tensores   = torch.load(RUTA_TENSORES,  weights_only=False)
    grafo      = torch.load(RUTA_GRAFO,     weights_only=False)

    feature_set_dcrnn = dcrnn_ckpt['feature_set']
    hl_dcrnn          = dcrnn_ckpt['history_len']
    scaler_Y_dcrnn    = dcrnn_ckpt['scaler_Y']

    X_test_dcrnn = tensores['X_test'][:, :, feature_set_dcrnn]   # (T, N, nf)
    Y_test_dcrnn = tensores['Y_test']                             # (T, N, 3)

    edge_index  = grafo['edge_index'].to(device)
    edge_weight = grafo['edge_weight'].to(device)

    modelo_dcrnn = SubwayDCRNN(
        in_channels=dcrnn_ckpt['n_features'],
        hidden_channels=dcrnn_ckpt['hidden_channels'],
        out_horizons=dcrnn_ckpt['out_horizons'],
        K=dcrnn_ckpt['K'],
        dropout=0.0,                    # dropout desactivado en evaluación
    ).to(device)
    modelo_dcrnn.load_state_dict(dcrnn_ckpt['model_state_dict'])
    modelo_dcrnn.eval()

    # Wrapper: DCRNN necesita edge_index y edge_weight en la llamada
    def infer_dcrnn(xb: torch.Tensor) -> torch.Tensor:
        out = modelo_dcrnn(xb, edge_index, edge_weight)  # (B, 1, N, H)
        return out.squeeze(1)                             # (B, N, H)

    # Extraer dow y temp_extreme para DCRNN desde tensores.pt
    # times tiene T timestamps para el dataset completo; el test son los últimos T_test
    T_test_dcrnn = X_test_dcrnn.shape[0]
    times_all    = tensores['times']
    times_test   = times_all[-T_test_dcrnn:]
    N_dcrnn      = X_test_dcrnn.shape[1]
    # dow: shape (T_test,) → broadcast a (T_test, N)
    dow_test_dcrnn  = np.tile(
        times_test.dayofweek.values[:, np.newaxis], (1, N_dcrnn)
    ).astype(float)
    # temp_extreme: desescalar la columna 4 del tensor completo (14 features)
    T_d, N_d, F_d   = tensores['X_test'].shape
    sc_X_dcrnn       = tensores['scaler_X']
    x_test_raw_all   = sc_X_dcrnn.inverse_transform(
        tensores['X_test'].reshape(-1, F_d)
    ).reshape(T_d, N_d, F_d)
    temp_extreme_test_dcrnn = (x_test_raw_all[:, :, FEATURE_COLS_DCRNN.index('temp_extreme')] > 0.5).astype(float)

    # ── STGCN ─────────────────────────────────────────────────────────────────
    print("[2/3] Cargando STGCN...")
    stgcn_ckpt    = torch.load(RUTA_STGCN_PTH, weights_only=False)
    hl_stgcn      = stgcn_ckpt['history_len']
    scaler_Y_stgcn = stgcn_ckpt['scaler_Y']
    X_test_stgcn  = stgcn_ckpt['X_test_scaled']
    Y_test_stgcn  = stgcn_ckpt['Y_test_scaled']
    dow_test_stgcn          = stgcn_ckpt['dow_test_raw']
    temp_extreme_test_stgcn = stgcn_ckpt['temp_extreme_test_raw']

    A_stgcn = torch.tensor(stgcn_ckpt['adj_matrix'], dtype=torch.float32)
    bp_stgcn = stgcn_ckpt['best_params']
    modelo_stgcn = STGCN_Metro(
        num_nodes=stgcn_ckpt['n_nodes'],
        num_features=stgcn_ckpt['num_features'],
        num_targets=stgcn_ckpt['num_targets'],
        history_len=hl_stgcn,
        adj_matrix=A_stgcn,
        hidden1=bp_stgcn['hidden1'],
        hidden2=bp_stgcn['hidden2'],
        dropout=0.0,
    ).to(device)
    modelo_stgcn.load_state_dict(stgcn_ckpt['model_state_dict'])
    modelo_stgcn.eval()

    def infer_stgcn(xb: torch.Tensor) -> torch.Tensor:
        return modelo_stgcn(xb)

    # ── ASTGCN ────────────────────────────────────────────────────────────────
    print("[3/3] Cargando ASTGCN...")
    astgcn_ckpt    = torch.load(RUTA_ASTGCN_PTH, weights_only=False)
    hl_astgcn      = astgcn_ckpt['history_len']
    scaler_Y_astgcn = astgcn_ckpt['scaler_Y']
    X_test_astgcn  = astgcn_ckpt['X_test_scaled']
    Y_test_astgcn  = astgcn_ckpt['Y_test_scaled']
    dow_test_astgcn          = astgcn_ckpt['dow_test_raw']
    temp_extreme_test_astgcn = astgcn_ckpt['temp_extreme_test_raw']

    bp_astgcn = astgcn_ckpt['best_params']
    K_cheb    = bp_astgcn['K_cheb']
    cheb_polys = calcular_polinomios_chebyshev(
        calcular_scaled_laplacian(astgcn_ckpt['A_weighted']), K_cheb
    )
    modelo_astgcn = ASTGCN_Metro(
        num_nodes=astgcn_ckpt['n_nodes'],
        num_features=astgcn_ckpt['num_features'],
        num_targets=astgcn_ckpt['num_targets'],
        history_len=hl_astgcn,
        cheb_polynomials=cheb_polys,
        K=K_cheb,
        hidden_channels=bp_astgcn['hidden_channels'],
        dropout=0.0,
    ).to(device)
    modelo_astgcn.load_state_dict(astgcn_ckpt['model_state_dict'])
    modelo_astgcn.eval()

    def infer_astgcn(xb: torch.Tensor) -> torch.Tensor:
        return modelo_astgcn(xb)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MÉTRICAS PRINCIPALES Y COMPLEMENTARIAS  (MAE, RMSE, R²)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerando predicciones sobre Test...")

    preds_dcrnn,  trues_dcrnn  = inferir(infer_dcrnn,  X_test_dcrnn,  Y_test_dcrnn,  hl_dcrnn,  batch_size, device, scaler_Y_dcrnn)
    preds_stgcn,  trues_stgcn  = inferir(infer_stgcn,  X_test_stgcn,  Y_test_stgcn,  hl_stgcn,  batch_size, device, scaler_Y_stgcn)
    preds_astgcn, trues_astgcn = inferir(infer_astgcn, X_test_astgcn, Y_test_astgcn, hl_astgcn, batch_size, device, scaler_Y_astgcn)

    metricas = {
        'DCRNN':  calcular_metricas(preds_dcrnn,  trues_dcrnn),
        'STGCN':  calcular_metricas(preds_stgcn,  trues_stgcn),
        'ASTGCN': calcular_metricas(preds_astgcn, trues_astgcn),
    }
    imprimir_tabla_metricas(metricas)

    # Registrar métricas globales en W&B
    tabla_metricas_wb = wandb.Table(columns=['Modelo', 'MAE (s)', 'RMSE (s)', 'R²'])
    for nombre, m in metricas.items():
        tabla_metricas_wb.add_data(nombre, m['MAE (s)'], m['RMSE (s)'], m['R²'])
        wandb.log({
            f'{nombre}/MAE_s':  m['MAE (s)'],
            f'{nombre}/RMSE_s': m['RMSE (s)'],
            f'{nombre}/R2':     m['R²'],
        })
    wandb.log({'tabla_metricas': tabla_metricas_wb})

    # Determinar el modelo ganador (menor MAE)
    ganador = min(metricas, key=lambda m: metricas[m]['MAE (s)'])
    print(f"\n→ Modelo con menor MAE global: {ganador} ({metricas[ganador]['MAE (s)']:.2f}s)")
    wandb.log({'ganador': ganador})

    # ══════════════════════════════════════════════════════════════════════════
    # 3. EVALUACIÓN POR SEGMENTOS RELEVANTES
    # ══════════════════════════════════════════════════════════════════════════
    # Las etiquetas deben estar alineadas con las predicciones:
    # las predicciones empiezan en el índice history_len del tensor de test.

    def alinear_etiquetas(etiq: np.ndarray, hl: int) -> np.ndarray:
        """
        Toma la media espacial (eje nodos) del array (T, N) y devuelve
        los T-hl valores alineados con las predicciones.
        """
        return etiq[hl:].mean(axis=1)   # (T - hl,)

    etiquetas_segmento = {
        'DCRNN':  {
            'dow':  alinear_etiquetas(dow_test_dcrnn,          hl_dcrnn),
            'temp': alinear_etiquetas(temp_extreme_test_dcrnn, hl_dcrnn),
        },
        'STGCN':  {
            'dow':  alinear_etiquetas(dow_test_stgcn,           hl_stgcn),
            'temp': alinear_etiquetas(temp_extreme_test_stgcn,  hl_stgcn),
        },
        'ASTGCN': {
            'dow':  alinear_etiquetas(dow_test_astgcn,           hl_astgcn),
            'temp': alinear_etiquetas(temp_extreme_test_astgcn,  hl_astgcn),
        },
    }

    # Alinear los arrays de predicción al período temporal común para que las
    # máscaras booleanas (de longitud M = T_test - hl) sean aplicables a todos
    # los modelos. Si los HPO eligieron history_len distintos, M difiere y la
    # indexación booleana fallaría. Recortamos cada array por las primeras
    # (max_hl - hl_model) filas para que todos comiencen en t_test + max_hl.
    max_hl = max(hl_dcrnn, hl_stgcn, hl_astgcn)
    hl_map = {'DCRNN': hl_dcrnn, 'STGCN': hl_stgcn, 'ASTGCN': hl_astgcn}

    # Para DCRNN las predicciones tienen 3 targets, para STGCN/ASTGCN 6 targets;
    # comparamos solo los primeros 3 (station_delay_10/20/30m) si hay discrepancia.
    min_C = min(preds_dcrnn.shape[2], preds_stgcn.shape[2], preds_astgcn.shape[2])
    resultados_segmentos = {
        'DCRNN':  (preds_dcrnn[max_hl - hl_dcrnn:, :, :min_C],   trues_dcrnn[max_hl - hl_dcrnn:, :, :min_C]),
        'STGCN':  (preds_stgcn[max_hl - hl_stgcn:, :, :min_C],   trues_stgcn[max_hl - hl_stgcn:, :, :min_C]),
        'ASTGCN': (preds_astgcn[max_hl - hl_astgcn:, :, :min_C], trues_astgcn[max_hl - hl_astgcn:, :, :min_C]),
    }
    # Etiquetas de cada modelo recortadas al mismo desplazamiento que sus preds
    # (max_hl - hl_modelo filas, para que coincidan con resultados_segmentos).
    # Cada modelo puede tener distinta longitud total de test set, por lo que
    # las máscaras se calculan por separado dentro de evaluar_segmentos.
    etiquetas_trim = {
        nombre: {
            'dow':  etiquetas_segmento[nombre]['dow'][max_hl - hl_map[nombre]:],
            'temp': etiquetas_segmento[nombre]['temp'][max_hl - hl_map[nombre]:],
        }
        for nombre in hl_map
    }
    maes_segmentos = evaluar_segmentos(resultados_segmentos, etiquetas_trim)

    # Registrar tabla de segmentos en W&B
    tabla_segmentos_wb = wandb.Table(
        columns=['Modelo', 'FdS MAE (s)', 'Diario MAE (s)', 'Extremo MAE (s)', 'Normal MAE (s)']
    )
    for nombre in hl_map:
        tabla_segmentos_wb.add_data(
            nombre,
            maes_segmentos['finde'].get(nombre, float('nan')),
            maes_segmentos['diario'].get(nombre, float('nan')),
            maes_segmentos['extremo'].get(nombre, float('nan')),
            maes_segmentos['normal'].get(nombre, float('nan')),
        )
    wandb.log({'tabla_segmentos': tabla_segmentos_wb})

    # ══════════════════════════════════════════════════════════════════════════
    # 4. PERMUTATION FEATURE IMPORTANCE (PFI) — sobre el modelo ganador
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nCalculando PFI para el modelo ganador: {ganador}...")
    print("(Este proceso puede tardar varios minutos según el hardware)")

    if ganador == 'DCRNN':
        feature_names_ganador = [FEATURE_COLS_DCRNN[i] for i in feature_set_dcrnn]
        X_pfi  = X_test_dcrnn
        Y_pfi  = Y_test_dcrnn
        hl_pfi = hl_dcrnn
        sc_pfi = scaler_Y_dcrnn
        fn_pfi = infer_dcrnn
    elif ganador == 'STGCN':
        feature_names_ganador = FEATURE_COLS_STGCN
        X_pfi  = X_test_stgcn
        Y_pfi  = Y_test_stgcn
        hl_pfi = hl_stgcn
        sc_pfi = scaler_Y_stgcn
        fn_pfi = infer_stgcn
    else:  # ASTGCN
        feature_names_ganador = FEATURE_COLS_STGCN   # mismas features que STGCN
        X_pfi  = X_test_astgcn
        Y_pfi  = Y_test_astgcn
        hl_pfi = hl_astgcn
        sc_pfi = scaler_Y_astgcn
        fn_pfi = infer_astgcn

    grupos_pfi = construir_grupos_pfi(feature_names_ganador)
    importancias = calcular_pfi(
        model_fn=fn_pfi,
        X_test_scaled=X_pfi,
        Y_test_scaled=Y_pfi,
        history_len=hl_pfi,
        batch_size=batch_size,
        device=device,
        scaler_Y=sc_pfi,
        feature_names=feature_names_ganador,
        grupos=grupos_pfi,
        n_rep=3,
        seed=42,
    )
    imprimir_ranking_pfi(importancias, ganador)

    # Registrar PFI en W&B
    tabla_pfi_wb = wandb.Table(columns=['Variable / Grupo', 'ΔMAE (s)'])
    for nombre_feat, delta in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
        tabla_pfi_wb.add_data(nombre_feat, delta)
    wandb.log({'tabla_pfi': tabla_pfi_wb})
    wandb.log({f'pfi/{k.replace(" ", "_")}': v for k, v in importancias.items()})

    wandb.finish()
    print("\n✓ Evaluación completada. Resultados listos para la memoria del proyecto.")


if __name__ == "__main__":
    main()

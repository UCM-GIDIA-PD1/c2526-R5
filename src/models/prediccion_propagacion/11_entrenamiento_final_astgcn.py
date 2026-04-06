import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_START_METHOD"] = "thread"

import wandb

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.minio_client import download_df_parquet

RUTA_HPO = Path(__file__).parent / "artefactos" / "astgcn_hpo.pt"
RUTA_MODELO = Path(__file__).parent / "artefactos" / "astgcn_modelo_final.pt"
WANDB_PROJECT = "pd1-c2526-team5"
SEED = 42
NUM_EPOCHS = 50
ES_PATIENCE = 7
FREQ = "15min"

VARIABLES_ENTRADA = [
    "delay_seconds", "lagged_delay_1", "lagged_delay_2", "is_unscheduled",
    "temp_extreme", "n_eventos_afectando", "route_rolling_delay",
    "actual_headway_seconds", "hour_sin", "hour_cos", "dow",
]
VARIABLES_OBJETIVO = [
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DatasetASTGCN(Dataset):
    def __init__(self, X, Y, history_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.history_len = history_len

    def __len__(self):
        return len(self.X) - self.history_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.history_len], self.Y[idx + self.history_len]


def calcular_scaled_laplacian(adj_matrix):
    adj = adj_matrix.astype(np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    laplacian_norm = d_mat_inv_sqrt @ laplacian @ d_mat_inv_sqrt
    lambda_max = np.linalg.eigvals(laplacian_norm).real.max()
    if lambda_max == 0 or np.isnan(lambda_max):
        lambda_max = 1.0
    return ((2.0 / lambda_max) * laplacian_norm - np.eye(adj.shape[0], dtype=np.float32)).astype(np.float32)


def calcular_polinomios_chebyshev(scaled_laplacian, K):
    N = scaled_laplacian.shape[0]
    cheb_polynomials = [np.eye(N, dtype=np.float32)]
    if K > 1:
        cheb_polynomials.append(scaled_laplacian)
    for k in range(2, K):
        cheb_polynomials.append(2 * scaled_laplacian @ cheb_polynomials[k - 1] - cheb_polynomials[k - 2])
    return [torch.tensor(p, dtype=torch.float32) for p in cheb_polynomials]


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, history_len):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj = nn.Linear(in_channels, 1, bias=False)
        self.scale = np.sqrt(max(num_nodes, 1))

    def forward(self, x):
        q = self.query_proj(x).squeeze(-1)
        k = self.key_proj(x).squeeze(-1)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        return torch.softmax(scores, dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, history_len):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj = nn.Linear(in_channels, 1, bias=False)
        self.scale = np.sqrt(max(history_len, 1))

    def forward(self, x):
        q = self.query_proj(x).squeeze(-1).transpose(1, 2)
        k = self.key_proj(x).squeeze(-1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        return torch.softmax(scores, dim=-1)


class ChebConvWithSpatialAttention(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)])
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)
        self.register_buffer("cheb_polynomials", torch.stack(cheb_polynomials, dim=0))

    def forward(self, x, spatial_attention):
        B, T, N, _ = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[:, t, :, :]
            output_t = torch.zeros((B, N, self.out_channels), device=x.device, dtype=x.dtype)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_at = T_k.unsqueeze(0) * spatial_attention
                rhs = torch.einsum("bij,bjf->bif", T_k_at, graph_signal)
                output_t = output_t + torch.matmul(rhs, self.Theta[k])
            outputs.append(output_t.unsqueeze(1))
        return F.relu(torch.cat(outputs, dim=1))


class ASTGCNBlock(nn.Module):
    def __init__(self, in_channels, K, cheb_polynomials, num_nodes, history_len, out_channels, temporal_kernel=3):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, num_nodes, history_len)
        self.spatial_attention = SpatialAttention(in_channels, num_nodes, history_len)
        self.cheb_conv = ChebConvWithSpatialAttention(K, cheb_polynomials, in_channels, out_channels)
        self.time_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(temporal_kernel, 1), padding=(temporal_kernel // 2, 0))
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        temporal_attention = self.temporal_attention(x)
        x_ta = torch.einsum("bts,bsnf->btnf", temporal_attention, x)
        spatial_attention = self.spatial_attention(x_ta)
        x_gc = self.cheb_conv(x_ta, spatial_attention)
        x_gc_perm = x_gc.permute(0, 3, 1, 2)
        x_tc = self.time_conv(x_gc_perm)
        residual = self.residual_conv(x.permute(0, 3, 1, 2))
        x_out = F.relu(x_tc + residual)
        x_out = x_out.permute(0, 2, 3, 1)
        return self.layer_norm(x_out)


class ASTGCN_Metro(nn.Module):
    def __init__(self, num_nodes, num_features, num_targets, history_len, cheb_polynomials, K, hidden_channels, dropout):
        super().__init__()
        self.block1 = ASTGCNBlock(num_features, K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.block2 = ASTGCNBlock(hidden_channels, K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.final_conv = nn.Conv2d(in_channels=history_len, out_channels=1, kernel_size=(1, hidden_channels))
        self.fc = nn.Linear(num_nodes, num_nodes * num_targets)
        self.num_nodes = num_nodes
        self.num_targets = num_targets

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        x = x.squeeze(-1).squeeze(1)
        x = self.fc(x)
        return x.view(-1, self.num_nodes, self.num_targets)


def descargar_datos():
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Faltan MINIO_ACCESS_KEY y/o MINIO_SECRET_KEY")
    ruta_archivo = "grupo5/final/year=2025/month=01/dataset_final.parquet"
    df_final = download_df_parquet(access_key, secret_key, ruta_archivo)
    dates = pd.date_range(start="2025-01-01", end="2025-01-31").strftime("%Y-%m-%d").tolist()
    dfs = []
    for date in dates:
        try:
            df_gtfs = download_df_parquet(access_key, secret_key, f"grupo5/cleaned/gtfs_clean_scheduled/date={date}/gtfs_scheduled_{date}.parquet")
            dfs.append(df_gtfs)
        except Exception:
            print(f"Could not download data for date: {date}")
    if not dfs:
        raise RuntimeError("No se pudo descargar GTFS")
    return df_final, pd.concat(dfs, ignore_index=True)


def construir_grafo(df):
    df = df.sort_values(by=["trip_uid", "scheduled_seconds"]).reset_index(drop=True)
    df["next_stop_id"] = df.groupby("trip_uid")["stop_id"].shift(-1)
    df["next_scheduled_seconds"] = df.groupby("trip_uid")["scheduled_seconds"].shift(-1)
    edges_df = df.dropna(subset=["next_stop_id"]).copy()
    edges_df["travel_time"] = edges_df["next_scheduled_seconds"] - edges_df["scheduled_seconds"]
    edges_df = edges_df[edges_df["travel_time"] > 0]
    graph_df = edges_df.groupby(["stop_id", "next_stop_id"]).agg(median_travel_time=("travel_time", "median"), trip_count=("trip_uid", "count")).reset_index()
    graph_df = graph_df[graph_df["trip_count"] > 5]
    nodes = sorted(list(set(df["stop_id"].unique()) | set(df["next_stop_id"].dropna().unique())))
    node_to_idx = {stop_id: idx for idx, stop_id in enumerate(nodes)}
    n_nodes = len(nodes)
    A_weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    sigma = graph_df["median_travel_time"].std()
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    for _, row in graph_df.iterrows():
        i = node_to_idx[row["stop_id"]]
        j = node_to_idx[row["next_stop_id"]]
        dist = row["median_travel_time"]
        peso = np.exp(- (dist ** 2) / (sigma ** 2))
        A_weighted[i, j] = peso
        A_weighted[j, i] = peso
    return node_to_idx, A_weighted


def crear_tensores(df, mapa_nodos, features, targets, freq=FREQ):
    nodos_validos = list(mapa_nodos.keys())
    df = df[df["stop_id"].isin(nodos_validos)].copy()
    df["time_bin"] = pd.to_datetime(df["merge_time"]).dt.floor(freq)
    reglas_agregacion = {
        "delay_seconds": "mean", "lagged_delay_1": "mean", "lagged_delay_2": "mean", "is_unscheduled": "sum",
        "temp_extreme": "max", "n_eventos_afectando": "max", "route_rolling_delay": "mean", "actual_headway_seconds": "mean",
        "target_delay_10m": "mean", "target_delay_20m": "mean", "target_delay_30m": "mean",
        "station_delay_10m": "mean", "station_delay_20m": "mean", "station_delay_30m": "mean",
    }
    df_agrupado = df.groupby(["time_bin", "stop_id"]).agg(reglas_agregacion)
    todos_los_tiempos = pd.date_range(start=df_agrupado.index.get_level_values("time_bin").min(), end=df_agrupado.index.get_level_values("time_bin").max(), freq=freq)
    indice_completo = pd.MultiIndex.from_product([todos_los_tiempos, nodos_validos], names=["time_bin", "stop_id"])
    df_completo = df_agrupado.reindex(indice_completo).reset_index()
    cols_retrasos = ["delay_seconds", "lagged_delay_1", "lagged_delay_2", "is_unscheduled", "route_rolling_delay", "actual_headway_seconds"]
    df_completo[cols_retrasos] = df_completo[cols_retrasos].fillna(0)
    cols_contexto = ["temp_extreme", "n_eventos_afectando"]
    df_completo[cols_contexto] = df_completo.groupby("stop_id")[cols_contexto].ffill().bfill().fillna(0)
    df_completo["hour_sin"] = np.sin(2 * np.pi * df_completo["time_bin"].dt.hour / 24)
    df_completo["hour_cos"] = np.cos(2 * np.pi * df_completo["time_bin"].dt.hour / 24)
    df_completo["dow"] = df_completo["time_bin"].dt.dayofweek.astype(float)
    df_completo["nodo_idx"] = df_completo["stop_id"].map(mapa_nodos)
    df_completo = df_completo.sort_values(["time_bin", "nodo_idx"])
    T = len(todos_los_tiempos)
    N = len(nodos_validos)
    F_in = len(features)
    C_out = len(targets)
    X_tensor = np.nan_to_num(df_completo[features].values.reshape(T, N, F_in), nan=0.0)
    Y_tensor = np.nan_to_num(df_completo[targets].values.reshape(T, N, C_out), nan=0.0)
    return X_tensor, Y_tensor


def split_y_escalar(X_full, Y_full):
    num_tiempos = X_full.shape[0]
    limite_corte = int(num_tiempos * 0.8)
    X_train = X_full[:limite_corte]
    X_test = X_full[limite_corte:]
    Y_train = Y_full[:limite_corte]
    Y_test = Y_full[limite_corte:]
    limite_val = int(X_train.shape[0] * 0.8)
    X_train_final = X_train[:limite_val]
    X_val = X_train[limite_val:]
    Y_train_final = Y_train[:limite_val]
    Y_val = Y_train[limite_val:]
    T_train, N, F_in = X_train_final.shape
    T_val = X_val.shape[0]
    T_test = X_test.shape[0]
    C_out = Y_train_final.shape[2]
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_final.reshape(-1, F_in)).reshape(T_train, N, F_in)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, F_in)).reshape(T_val, N, F_in)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, F_in)).reshape(T_test, N, F_in)
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train_final.reshape(-1, C_out)).reshape(T_train, N, C_out)
    Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, C_out)).reshape(T_val, N, C_out)
    Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, C_out)).reshape(T_test, N, C_out)
    return {
        'X_train_scaled': X_train_scaled, 'Y_train_scaled': Y_train_scaled,
        'X_val_scaled': X_val_scaled, 'Y_val_scaled': Y_val_scaled,
        'X_test_scaled': X_test_scaled, 'Y_test_scaled': Y_test_scaled,
        'scaler_Y': scaler_Y,
    }


def metricas_reales(model, loader, scaler_Y, device):
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device))
            preds_list.append(preds.cpu().numpy())
            trues_list.append(yb.numpy())
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    T_eval, N_eval, C_eval = preds.shape
    preds_real = scaler_Y.inverse_transform(preds.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)
    trues_real = scaler_Y.inverse_transform(trues.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)
    metricas = {}
    for i, objetivo in enumerate(VARIABLES_OBJETIVO):
        diff = preds_real[:, :, i] - trues_real[:, :, i]
        metricas[f'MAE_{objetivo}'] = float(np.mean(np.abs(diff)))
        metricas[f'RMSE_{objetivo}'] = float(np.sqrt(np.mean(diff ** 2)))
    return metricas


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    hpo = torch.load(RUTA_HPO, weights_only=False)
    best_params = hpo['best_params']

    df_final, df_gtfs = descargar_datos()
    node_to_idx, A_weighted = construir_grafo(df_gtfs)
    X_full, Y_full = crear_tensores(df_final, node_to_idx, VARIABLES_ENTRADA, VARIABLES_OBJETIVO)
    splits = split_y_escalar(X_full, Y_full)

    cheb_polynomials = calcular_polinomios_chebyshev(calcular_scaled_laplacian(A_weighted), best_params['K_cheb'])

    train_dataset = DatasetASTGCN(splits['X_train_scaled'], splits['Y_train_scaled'], best_params['history_len'])
    val_dataset = DatasetASTGCN(splits['X_val_scaled'], splits['Y_val_scaled'], best_params['history_len'])
    test_dataset = DatasetASTGCN(splits['X_test_scaled'], splits['Y_test_scaled'], best_params['history_len'])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    model = ASTGCN_Metro(
        num_nodes=len(node_to_idx),
        num_features=len(VARIABLES_ENTRADA),
        num_targets=len(VARIABLES_OBJETIVO),
        history_len=best_params['history_len'],
        cheb_polynomials=cheb_polynomials,
        K=best_params['K_cheb'],
        hidden_channels=best_params['hidden_channels'],
        dropout=best_params['dropout'],
    ).to(device)

    criterion = nn.MSELoss() if best_params['loss_name'] == 'mse' else nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    wandb.init(project=WANDB_PROJECT, name='astgcn-entrenamiento-final', mode='offline', config=best_params)

    best_val = float('inf')
    best_epoch = 0
    best_state = None
    sin_mejora = 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_acc = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_acc += loss.item()
        train_loss = train_loss_acc / len(train_loader)

        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device))
                val_loss_acc += criterion(pred, yb.to(device)).item()
        val_loss = val_loss_acc / len(val_loader)
        scheduler.step(val_loss)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
        print(f"Época {epoch:02d}/{NUM_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
        if sin_mejora >= ES_PATIENCE:
            print(f"Early stopping en época {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metricas_test = metricas_reales(model, test_loader, splits['scaler_Y'], device)
    train_time = time.time() - t0
    wandb.log({'best_val_loss': best_val, 'best_epoch': best_epoch, 'train_time_sec': train_time, **metricas_test})
    wandb.finish()

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'metricas_test': metricas_test,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'variables_entrada': VARIABLES_ENTRADA,
        'variables_objetivo': VARIABLES_OBJETIVO,
    }, RUTA_MODELO)
    print(f"Modelo final ASTGCN guardado en: {RUTA_MODELO}")
    print(metricas_test)


if __name__ == '__main__':
    main()

import copy
import gc
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

RUTA_HPO = Path(__file__).parent / "artefactos" / "stgcn_hpo.pt"
RUTA_MODELO = Path(__file__).parent / "artefactos" / "stgcn_modelo_final.pt"
WANDB_PROJECT = "pd1-c2526-team5"
SEED = 42
NUM_EPOCHS = 50
ES_PATIENCE = 7
FREQ = "15min"

VARIABLES_ENTRADA = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
    'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay',
    'actual_headway_seconds', 'hour_sin', 'hour_cos', 'dow'
]
VARIABLES_OBJETIVO = [
    'target_delay_10m', 'target_delay_20m', 'target_delay_30m',
    'station_delay_10m', 'station_delay_20m', 'station_delay_30m'
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DatasetSTGCN(Dataset):
    def __init__(self, X, Y, history_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.history_len = history_len

    def __len__(self):
        return len(self.X) - self.history_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.history_len], self.Y[idx + self.history_len]


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x_transformed = torch.matmul(x, self.weight)
        salida = torch.einsum('vw,btwd->btvd', adj, x_transformed)
        return salida + self.bias


class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.s_conv = GraphConv(out_channels, out_channels)
        self.t_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x, adj):
        x_perm = x.permute(0, 3, 1, 2)
        x_t1 = F.relu(self.t_conv1(x_perm))
        x_t1_perm = x_t1.permute(0, 2, 3, 1)
        x_s = F.relu(self.s_conv(x_t1_perm, adj))
        x_s_perm = x_s.permute(0, 3, 1, 2)
        x_out = F.relu(self.t_conv2(x_s_perm))
        return x_out.permute(0, 2, 3, 1)


class STGCN_Metro(nn.Module):
    def __init__(self, num_nodes, num_features, num_targets, history_len, adj_matrix, hidden1, hidden2, dropout):
        super().__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.block1 = STConvBlock(num_features, hidden1)
        self.block2 = STConvBlock(hidden1, hidden2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2 * history_len, num_targets)

    def forward(self, x):
        batch_size, _, nodos, _ = x.shape
        x = self.block1(x, self.adj_matrix)
        x = self.dropout(x)
        x = self.block2(x, self.adj_matrix)
        x = self.dropout(x)
        x = x.reshape(batch_size, nodos, -1)
        return self.fc(x)



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
    df_gtfs_month = pd.concat(dfs, ignore_index=True)
    return df_final, df_gtfs_month


def construir_grafo(df):
    df = df.sort_values(by=['trip_uid', 'scheduled_seconds']).reset_index(drop=True)
    df['next_stop_id'] = df.groupby('trip_uid')['stop_id'].shift(-1)
    df['next_scheduled_seconds'] = df.groupby('trip_uid')['scheduled_seconds'].shift(-1)
    edges_df = df.dropna(subset=['next_stop_id']).copy()
    edges_df['travel_time'] = edges_df['next_scheduled_seconds'] - edges_df['scheduled_seconds']
    edges_df = edges_df[edges_df['travel_time'] > 0]
    graph_df = edges_df.groupby(['stop_id', 'next_stop_id']).agg(
        median_travel_time=('travel_time', 'median'),
        trip_count=('trip_uid', 'count')
    ).reset_index()
    graph_df = graph_df[graph_df['trip_count'] > 5]
    nodes = sorted(list(set(df['stop_id'].unique()) | set(df['next_stop_id'].dropna().unique())))
    node_to_idx = {stop_id: idx for idx, stop_id in enumerate(nodes)}
    n_nodes = len(nodes)
    A_weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    sigma = graph_df['median_travel_time'].std()
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    for _, row in graph_df.iterrows():
        i = node_to_idx[row['stop_id']]
        j = node_to_idx[row['next_stop_id']]
        dist = row['median_travel_time']
        peso = np.exp(- (dist ** 2) / (sigma ** 2))
        A_weighted[i, j] = peso
        A_weighted[j, i] = peso
    np.fill_diagonal(A_weighted, 1.0)
    grados = np.sum(A_weighted, axis=1)
    grados_inv_raiz = np.power(grados, -0.5, where=(grados != 0))
    grados_inv_raiz[np.isinf(grados_inv_raiz)] = 0.0
    matriz_diagonal = np.diag(grados_inv_raiz)
    A_norm = matriz_diagonal @ A_weighted @ matriz_diagonal
    return node_to_idx, torch.tensor(A_norm, dtype=torch.float32)


def crear_tensores(df, mapa_nodos, features, targets, freq=FREQ):
    nodos_validos = list(mapa_nodos.keys())
    df = df[df['stop_id'].isin(nodos_validos)].copy()
    df['time_bin'] = pd.to_datetime(df['merge_time']).dt.floor(freq)
    reglas_agregacion = {
        'delay_seconds': 'mean', 'lagged_delay_1': 'mean', 'lagged_delay_2': 'mean', 'is_unscheduled': 'sum',
        'temp_extreme': 'max', 'n_eventos_afectando': 'max', 'route_rolling_delay': 'mean', 'actual_headway_seconds': 'mean',
        'target_delay_10m': 'mean', 'target_delay_20m': 'mean', 'target_delay_30m': 'mean',
        'station_delay_10m': 'mean', 'station_delay_20m': 'mean', 'station_delay_30m': 'mean',
    }
    df_agrupado = df.groupby(['time_bin', 'stop_id']).agg(reglas_agregacion)
    todos_los_tiempos = pd.date_range(start=df_agrupado.index.get_level_values('time_bin').min(), end=df_agrupado.index.get_level_values('time_bin').max(), freq=freq)
    indice_completo = pd.MultiIndex.from_product([todos_los_tiempos, nodos_validos], names=['time_bin', 'stop_id'])
    df_completo = df_agrupado.reindex(indice_completo).reset_index()
    cols_retrasos = ['delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled', 'route_rolling_delay', 'actual_headway_seconds']
    df_completo[cols_retrasos] = df_completo[cols_retrasos].fillna(0)
    cols_contexto = ['temp_extreme', 'n_eventos_afectando']
    df_completo[cols_contexto] = df_completo.groupby('stop_id')[cols_contexto].ffill().bfill().fillna(0)
    df_completo['hour_sin'] = np.sin(2 * np.pi * df_completo['time_bin'].dt.hour / 24)
    df_completo['hour_cos'] = np.cos(2 * np.pi * df_completo['time_bin'].dt.hour / 24)
    df_completo['dow'] = df_completo['time_bin'].dt.dayofweek.astype(float)
    df_completo['nodo_idx'] = df_completo['stop_id'].map(mapa_nodos)
    df_completo = df_completo.sort_values(['time_bin', 'nodo_idx'])
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
    T_train, N, num_features_in = X_train_final.shape
    T_val = X_val.shape[0]
    T_test = X_test.shape[0]
    C = Y_train_final.shape[2]
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_final.reshape(-1, num_features_in)).reshape(T_train, N, num_features_in)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, num_features_in)).reshape(T_val, N, num_features_in)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features_in)).reshape(T_test, N, num_features_in)
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train_final.reshape(-1, C)).reshape(T_train, N, C)
    Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, C)).reshape(T_val, N, C)
    Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, C)).reshape(T_test, N, C)
    return {
        'X_train_scaled': X_train_scaled, 'Y_train_scaled': Y_train_scaled,
        'X_val_scaled': X_val_scaled, 'Y_val_scaled': Y_val_scaled,
        'X_test_scaled': X_test_scaled, 'Y_test_scaled': Y_test_scaled,
        'scaler_Y': scaler_Y,
    }


def metricas_reales(model, loader, scaler_Y, device):
    model.eval()
    lista_predicciones, lista_reales = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device))
            lista_predicciones.append(preds.cpu().numpy())
            lista_reales.append(yb.numpy())
    preds = np.concatenate(lista_predicciones, axis=0)
    reales = np.concatenate(lista_reales, axis=0)
    T_eval, N_eval, C_eval = preds.shape
    preds_reales = scaler_Y.inverse_transform(preds.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)
    reales_reales = scaler_Y.inverse_transform(reales.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)
    maes, rmses = {}, {}
    for i, objetivo in enumerate(VARIABLES_OBJETIVO):
        diff = preds_reales[:, :, i] - reales_reales[:, :, i]
        maes[f'MAE_{objetivo}'] = float(np.mean(np.abs(diff)))
        rmses[f'RMSE_{objetivo}'] = float(np.sqrt(np.mean(diff ** 2)))
    return {**maes, **rmses}


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    hpo = torch.load(RUTA_HPO, weights_only=False)
    best_params = hpo['best_params']

    df_final, df_gtfs = descargar_datos()
    node_to_idx, A_tensor = construir_grafo(df_gtfs)
    X_full, Y_full = crear_tensores(df_final, node_to_idx, VARIABLES_ENTRADA, VARIABLES_OBJETIVO)
    splits = split_y_escalar(X_full, Y_full)

    train_dataset = DatasetSTGCN(splits['X_train_scaled'], splits['Y_train_scaled'], best_params['history_len'])
    val_dataset = DatasetSTGCN(splits['X_val_scaled'], splits['Y_val_scaled'], best_params['history_len'])
    test_dataset = DatasetSTGCN(splits['X_test_scaled'], splits['Y_test_scaled'], best_params['history_len'])

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    model = STGCN_Metro(
        num_nodes=len(node_to_idx),
        num_features=len(VARIABLES_ENTRADA),
        num_targets=len(VARIABLES_OBJETIVO),
        history_len=best_params['history_len'],
        adj_matrix=A_tensor,
        hidden1=best_params['hidden1'],
        hidden2=best_params['hidden2'],
        dropout=best_params['dropout'],
    ).to(device)

    criterion = nn.MSELoss() if best_params['loss_name'] == 'mse' else nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    wandb.init(project=WANDB_PROJECT, name='stgcn-entrenamiento-final', mode='offline', config=best_params)

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
    print(f"Modelo final STGCN guardado en: {RUTA_MODELO}")
    print(metricas_test)


if __name__ == '__main__':
    main()

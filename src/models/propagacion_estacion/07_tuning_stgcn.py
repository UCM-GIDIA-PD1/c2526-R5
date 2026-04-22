import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


import wandb

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.minio_client import download_df_parquet

optuna.logging.set_verbosity(optuna.logging.WARNING)

RUTA_SALIDA = Path(__file__).parent / "artefactos" / "stgcn_hpo.pt"
WANDB_PROJECT = "pd1-c2526-team5"
SEED = 42
MAX_EPOCHS = 20
ES_PATIENCE = 5
N_TRIALS = 12
FREQ = "15min"

VARIABLES_ENTRADA = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
    'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay',
    'actual_headway_seconds', 'hour_sin', 'hour_cos', 'dow'
]
VARIABLES_OBJETIVO = [
    'station_delay_10m', 'station_delay_20m', 'station_delay_30m'
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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
    def __init__(self, num_features, num_targets, history_len, adj_matrix, hidden1, hidden2, dropout):
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
    cols_necesarias = [
        'stop_id', 'merge_time',
        'delay_seconds', 'lagged_delay_1', 'lagged_delay_2', 'is_unscheduled',
        'temp_extreme', 'n_eventos_afectando', 'route_rolling_delay', 'actual_headway_seconds',
        'station_delay_10m', 'station_delay_20m', 'station_delay_30m',
    ]
    df_final = df_final[[c for c in cols_necesarias if c in df_final.columns]]

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
    grados_inv_raiz = np.zeros_like(grados)
    mask = grados != 0
    grados_inv_raiz[mask] = np.power(grados[mask], -0.5)
    matriz_diagonal = np.diag(grados_inv_raiz)
    A_norm = matriz_diagonal @ A_weighted @ matriz_diagonal
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    return node_to_idx, A_tensor



def crear_tensores(df, mapa_nodos, features, targets, freq=FREQ):
    nodos_validos = list(mapa_nodos.keys())
    df = df[df['stop_id'].isin(nodos_validos)].copy()
    df['time_bin'] = pd.to_datetime(df['merge_time']).dt.floor(freq)

    reglas_agregacion = {
        'delay_seconds': 'mean',
        'lagged_delay_1': 'mean',
        'lagged_delay_2': 'mean',
        'is_unscheduled': 'sum',
        'temp_extreme': 'max',
        'n_eventos_afectando': 'max',
        'route_rolling_delay': 'mean',
        'actual_headway_seconds': 'mean',
        'station_delay_10m': 'mean',
        'station_delay_20m': 'mean',
        'station_delay_30m': 'mean',
    }

    df_agrupado = df.groupby(['time_bin', 'stop_id']).agg(reglas_agregacion)
    del df
    gc.collect()

    todos_los_tiempos = pd.date_range(
        start=df_agrupado.index.get_level_values('time_bin').min(),
        end=df_agrupado.index.get_level_values('time_bin').max(),
        freq=freq
    )
    indice_completo = pd.MultiIndex.from_product([todos_los_tiempos, nodos_validos], names=['time_bin', 'stop_id'])
    df_completo = df_agrupado.reindex(indice_completo).reset_index()
    del df_agrupado
    gc.collect()

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

    X_tensor = df_completo[features].values.reshape(T, N, F_in)
    Y_tensor = df_completo[targets].values.reshape(T, N, C_out)

    X_tensor = np.nan_to_num(X_tensor, nan=0.0)
    Y_tensor = np.nan_to_num(Y_tensor, nan=0.0)
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
        'X_train_scaled': X_train_scaled,
        'Y_train_scaled': Y_train_scaled,
        'X_val_scaled': X_val_scaled,
        'Y_val_scaled': Y_val_scaled,
        'X_test_scaled': X_test_scaled,
        'Y_test_scaled': Y_test_scaled,
        'scaler_Y': scaler_Y,
    }



def evaluar_scaled(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss_total += loss.item() * xb.size(0)
    return loss_total / len(loader.dataset)



def evaluar_mae_real(model, loader, scaler_Y, device):
    model.eval()
    lista_predicciones = []
    lista_reales = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb)
            lista_predicciones.append(preds.cpu().numpy())
            lista_reales.append(yb.numpy())

    preds = np.concatenate(lista_predicciones, axis=0)
    reales = np.concatenate(lista_reales, axis=0)
    T_eval, N_eval, C_eval = preds.shape
    preds_reales = scaler_Y.inverse_transform(preds.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)
    reales_reales = scaler_Y.inverse_transform(reales.reshape(-1, C_eval)).reshape(T_eval, N_eval, C_eval)

    metricas = {}
    maes = []
    for i, objetivo in enumerate(VARIABLES_OBJETIVO):
        mae = float(np.mean(np.abs(preds_reales[:, :, i] - reales_reales[:, :, i])))
        metricas[f'MAE_{objetivo}'] = mae
        maes.append(mae)
    metricas['MAE_global_real'] = float(np.mean(maes))
    return metricas



def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("Descargando datos...")
    df_final, df_gtfs = descargar_datos()
    print("Construyendo grafo...")
    node_to_idx, A_tensor = construir_grafo(df_gtfs)
    print(f"Grafo construido: {len(node_to_idx)} nodos")
    del df_gtfs
    gc.collect()
    print("Creando tensores...")
    X_full, Y_full = crear_tensores(df_final, node_to_idx, VARIABLES_ENTRADA, VARIABLES_OBJETIVO)
    print(f"Tensores: X={X_full.shape}, Y={Y_full.shape}")
    del df_final
    gc.collect()
    print("Escalando y dividiendo datos...")
    splits = split_y_escalar(X_full, Y_full)
    del X_full, Y_full
    gc.collect()
    print(f"Iniciando HPO con {N_TRIALS} trials...")

    def objective(trial):
        config = {
            'history_len': trial.suggest_categorical('history_len', [8, 12]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 1e-3, log=True),
            'hidden1': trial.suggest_categorical('hidden1', [16, 32, 64]),
            'hidden2': trial.suggest_categorical('hidden2', [32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'loss_name': trial.suggest_categorical('loss_name', ['mse', 'smoothl1']),
        }

        train_dataset = DatasetSTGCN(splits['X_train_scaled'], splits['Y_train_scaled'], config['history_len'])
        val_dataset = DatasetSTGCN(splits['X_val_scaled'], splits['Y_val_scaled'], config['history_len'])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        model = STGCN_Metro(
            num_features=len(VARIABLES_ENTRADA),
            num_targets=len(VARIABLES_OBJETIVO),
            history_len=config['history_len'],
            adj_matrix=A_tensor,
            hidden1=config['hidden1'],
            hidden2=config['hidden2'],
            dropout=config['dropout'],
        ).to(device)

        criterion = nn.MSELoss() if config['loss_name'] == 'mse' else nn.SmoothL1Loss(beta=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val = float('inf')
        sin_mejora = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_loss = evaluar_scaled(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val:
                best_val = val_loss
                sin_mejora = 0
            else:
                sin_mejora += 1
            if sin_mejora >= ES_PATIENCE:
                break

        metricas_val = evaluar_mae_real(model, val_loader, splits['scaler_Y'], device)
        wandb.log({'val_loss': best_val, **metricas_val})
        return metricas_val['MAE_global_real']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    wandb.init(project=WANDB_PROJECT, name='stgcn-tuning', reinit='finish_previous')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    wandb.finish()

    resultados = []
    for t in study.trials:
        resultados.append({
            'trial': t.number,
            'value': t.value,
            'state': t.state.name,
            **t.params,
        })
    df_resultados = pd.DataFrame(resultados).sort_values('value', na_position='last').reset_index(drop=True)

    torch.save({
        'best_params': study.best_trial.params,
        'best_val_metric': study.best_value,
        'df_resultados': df_resultados,
        'variables_entrada': VARIABLES_ENTRADA,
        'variables_objetivo': VARIABLES_OBJETIVO,
    }, RUTA_SALIDA)
    print(f"Mejores hiperparámetros STGCN guardados en: {RUTA_SALIDA}")
    print(study.best_trial.params)
    print(f"Mejor MAE global validación: {study.best_value:.4f}")


if __name__ == '__main__':
    main()

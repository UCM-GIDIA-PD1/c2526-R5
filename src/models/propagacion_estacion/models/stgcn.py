"""
Modelo STGCN (Spatio-Temporal Graph Convolutional Network) para predicción de
retrasos en la red de metro de Nueva York.

Clases exportadas
-----------------
GraphConv      : convolución espectral simple sobre grafo con matriz de adyacencia densa.
STConvBlock    : bloque temporal → espacial → temporal (T-conv + GCN + T-conv).
STGCN_Metro    : red completa con dos bloques STConv y capa FC de salida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Convolución de grafo usando la matriz de adyacencia normalizada densa."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, in_ch)  adj: (N, N)
        x_transformed = torch.matmul(x, self.weight)                    # (B, T, N, out_ch)
        salida = torch.einsum('vw,btwd->btvd', adj, x_transformed)      # (B, T, N, out_ch)
        return salida + self.bias


class STConvBlock(nn.Module):
    """Bloque ST-Conv: t_conv1 → graph_conv → t_conv2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.s_conv  = GraphConv(out_channels, out_channels)
        self.t_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        x_perm  = x.permute(0, 3, 1, 2)                        # (B, F, T, N)
        x_t1    = F.relu(self.t_conv1(x_perm))                 # (B, out_ch, T, N)
        x_t1_p  = x_t1.permute(0, 2, 3, 1)                     # (B, T, N, out_ch)
        x_s     = F.relu(self.s_conv(x_t1_p, adj))              # (B, T, N, out_ch)
        x_s_p   = x_s.permute(0, 3, 1, 2)                      # (B, out_ch, T, N)
        x_out   = F.relu(self.t_conv2(x_s_p))                  # (B, out_ch, T, N)
        return x_out.permute(0, 2, 3, 1)                        # (B, T, N, out_ch)


class STGCN_Metro(nn.Module):
    """
    STGCN para predicción de retrasos en estaciones de metro.

    Parámetros
    ----------
    num_nodes    : número de nodos en el grafo
    num_features : dimensión de features por nodo por paso temporal
    num_targets  : número de salidas por nodo
    history_len  : longitud de la ventana de entrada
    adj_matrix   : tensor (N, N) — matriz de adyacencia normalizada (buffer)
    hidden1      : canales internos del primer bloque STConv
    hidden2      : canales internos del segundo bloque STConv
    dropout      : tasa de dropout antes de la FC
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_targets: int,
        history_len: int,
        adj_matrix: torch.Tensor,
        hidden1: int,
        hidden2: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.block1  = STConvBlock(num_features, hidden1)
        self.block2  = STConvBlock(hidden1, hidden2)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden2 * history_len, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        batch_size, _, nodos, _ = x.shape
        x = self.block1(x, self.adj_matrix)
        x = self.dropout(x)
        x = self.block2(x, self.adj_matrix)
        x = self.dropout(x)
        x = x.reshape(batch_size, nodos, -1)   # (B, N, hidden2*T)
        return self.fc(x)                       # (B, N, num_targets)

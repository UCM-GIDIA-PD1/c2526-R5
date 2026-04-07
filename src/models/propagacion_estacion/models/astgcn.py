"""
Modelo ASTGCN (Attention-based Spatio-Temporal Graph Convolutional Network) para
predicción de retrasos en la red de metro de Nueva York.

Clases exportadas
-----------------
ASTGCN_Metro                    : red completa lista para entrenamiento/inferencia.

Funciones auxiliares exportadas
--------------------------------
calcular_scaled_laplacian        : normaliza el Laplaciano del grafo al rango [-1, 1].
calcular_polinomios_chebyshev    : precalcula los polinomios de Chebyshev T_0..T_{K-1}.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de grafo
# ─────────────────────────────────────────────────────────────────────────────

def calcular_scaled_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """Laplaciano simétrico normalizado escalado a [-1, 1] (para Chebyshev)."""
    adj = adj_matrix.astype(np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    laplacian_norm = d_mat_inv_sqrt @ laplacian @ d_mat_inv_sqrt
    lambda_max = np.linalg.eigvals(laplacian_norm).real.max()
    if lambda_max == 0 or np.isnan(lambda_max):
        lambda_max = 1.0
    return ((2.0 / lambda_max) * laplacian_norm - np.eye(adj.shape[0], dtype=np.float32)).astype(np.float32)


def calcular_polinomios_chebyshev(scaled_laplacian: np.ndarray, K: int) -> list[torch.Tensor]:
    """Precalcula los K polinomios de Chebyshev como tensores."""
    N = scaled_laplacian.shape[0]
    polys = [np.eye(N, dtype=np.float32)]
    if K > 1:
        polys.append(scaled_laplacian)
    for k in range(2, K):
        polys.append(2 * scaled_laplacian @ polys[k - 1] - polys[k - 2])
    return [torch.tensor(p, dtype=torch.float32) for p in polys]


# ─────────────────────────────────────────────────────────────────────────────
# Bloques de atención
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, history_len: int):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj   = nn.Linear(in_channels, 1, bias=False)
        self.scale      = np.sqrt(max(num_nodes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        q = self.query_proj(x).squeeze(-1)                              # (B, T, N)
        k = self.key_proj(x).squeeze(-1)                                # (B, T, N)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale        # (B, T, T)
        return torch.softmax(scores, dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, history_len: int):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj   = nn.Linear(in_channels, 1, bias=False)
        self.scale      = np.sqrt(max(history_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        q = self.query_proj(x).squeeze(-1).transpose(1, 2)             # (B, N, T)
        k = self.key_proj(x).squeeze(-1).transpose(1, 2)               # (B, N, T)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale        # (B, N, N)
        return torch.softmax(scores, dim=-1)


class ChebConvWithSpatialAttention(nn.Module):
    def __init__(self, K: int, cheb_polynomials: list[torch.Tensor], in_channels: int, out_channels: int):
        super().__init__()
        self.K           = K
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)
        ])
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)
        # Almacenar los K polinomios apilados como buffer no entrenable
        self.register_buffer('cheb_polynomials', torch.stack(cheb_polynomials, dim=0))

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)   spatial_attention: (B, N, N)
        B, T, N, _ = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[:, t, :, :]                                # (B, N, F)
            output_t = torch.zeros((B, N, self.out_channels), device=x.device, dtype=x.dtype)
            for k in range(self.K):
                T_k    = self.cheb_polynomials[k]                       # (N, N)
                T_k_at = T_k.unsqueeze(0) * spatial_attention           # (B, N, N)
                rhs    = torch.einsum('bij,bjf->bif', T_k_at, graph_signal)
                output_t = output_t + torch.matmul(rhs, self.Theta[k])
            outputs.append(output_t.unsqueeze(1))
        return F.relu(torch.cat(outputs, dim=1))                        # (B, T, N, out_ch)


class ASTGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        K: int,
        cheb_polynomials: list[torch.Tensor],
        num_nodes: int,
        history_len: int,
        out_channels: int,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, num_nodes, history_len)
        self.spatial_attention  = SpatialAttention(in_channels, num_nodes, history_len)
        self.cheb_conv          = ChebConvWithSpatialAttention(K, cheb_polynomials, in_channels, out_channels)
        self.time_conv          = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1), padding=(temporal_kernel // 2, 0)
        )
        self.residual_conv      = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.layer_norm         = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_attention = self.temporal_attention(x)
        x_ta  = torch.einsum('bts,bsnf->btnf', temporal_attention, x)
        spatial_attention  = self.spatial_attention(x_ta)
        x_gc  = self.cheb_conv(x_ta, spatial_attention)
        x_tc  = self.time_conv(x_gc.permute(0, 3, 1, 2))
        res   = self.residual_conv(x.permute(0, 3, 1, 2))
        x_out = F.relu(x_tc + res).permute(0, 2, 3, 1)
        return self.layer_norm(x_out)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────────────────────

class ASTGCN_Metro(nn.Module):
    """
    ASTGCN con dos bloques de atención espacio-temporal para predicción de
    retrasos en estaciones de metro.

    Parámetros
    ----------
    num_nodes        : número de nodos
    num_features     : features por nodo por paso temporal
    num_targets      : salidas por nodo
    history_len      : longitud de la ventana de entrada
    cheb_polynomials : lista de K tensores (N, N)
    K                : orden de Chebyshev
    hidden_channels  : canales en los bloques ASTGCN
    dropout          : tasa de dropout
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_targets: int,
        history_len: int,
        cheb_polynomials: list[torch.Tensor],
        K: int,
        hidden_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.block1      = ASTGCNBlock(num_features,    K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.block2      = ASTGCNBlock(hidden_channels, K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.dropout     = nn.Dropout(dropout)
        self.final_conv  = nn.Conv2d(history_len, 1, kernel_size=(1, hidden_channels))
        self.fc          = nn.Linear(num_nodes, num_nodes * num_targets)
        self.num_nodes   = num_nodes
        self.num_targets = num_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.final_conv(x)                  # (B, 1, N, 1)  via (B, T, N, hid) perm
        x = x.squeeze(-1).squeeze(1)            # (B, N)
        x = self.fc(x)                          # (B, N * num_targets)
        return x.view(-1, self.num_nodes, self.num_targets)

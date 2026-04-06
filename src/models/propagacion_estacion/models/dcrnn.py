"""
SubwayDCRNN: envuelve BatchedDCRNN de torch-geometric-temporal con una capa
lineal de readout para predecir múltiples horizontes de retraso por estación.

Entrada  : X          (Batch, History, Nodes, in_channels)
           edge_index (2, E)
           edge_weight (E,)
Salida   : y_hat      (Batch, 1, Nodes, out_horizons)
"""
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN


class SubwayDCRNN(nn.Module):
    """
    Parámetros
    ----------
    in_channels     : número de features de entrada (F, p.ej. 14)
    hidden_channels : dimensión oculta del DCRNN
    out_horizons    : número de horizontes de predicción (p.ej. 3 → 10/20/30 min)
    K               : orden de los filtros de difusión en el grafo
    dropout         : tasa de dropout aplicada antes de la capa lineal
    """

    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 64,
        out_horizons: int = 3,
        K: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # BatchedDCRNN maneja el batch de forma nativa y usa row/col swap para
        # el grafo inverso, evitando el bug de to_dense_adj/dense_to_sparse del
        # DCRNN original que requería grafo simétrico.
        self.dcrnn = BatchedDCRNN(
            in_channels=in_channels,
            out_channels=hidden_channels,
            K=K,
        )
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_horizons),
        )

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        X           : (B, History, N, F)
        edge_index  : (2, E)
        edge_weight : (E,)
        retorna     : (B, 1, N, out_horizons)
        """
        # BatchedDCRNN → (B, History, N, hidden_channels)
        out = self.dcrnn(X, edge_index, edge_weight)
        # Último paso temporal como representación del estado oculto
        hidden_last = out[:, -1, :, :]         # (B, N, hidden_channels)
        y_hat = self.readout(hidden_last)       # (B, N, out_horizons)
        return y_hat.unsqueeze(1)               # (B, 1, N, out_horizons)

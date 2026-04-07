"""
SubwayDataset: ventana deslizante (T, N, F) → (history_len, N, F) para BatchedDCRNN.
validar_batch_vs_grafo: comprueba consistencia entre un batch y el grafo antes de forward.
"""
import torch
from torch.utils.data import Dataset


class SubwayDataset(Dataset):
    """
    Genera ventanas deslizantes sobre tensores espaciotemporales escalados.

    Parámetros
    ----------
    X : np.ndarray | torch.Tensor  — (T, N, F)  features escaladas
    Y : np.ndarray | torch.Tensor  — (T, N, H)  targets escalados
    history_len : int              — número de bins de contexto (p.ej. 8 → 2 h a 15 min)

    __getitem__ devuelve
    --------------------
    x_window : (history_len, N, F)
    y_window : (1, N, H)   — el instante inmediatamente posterior a la ventana
    """

    def __init__(self, X, Y, history_len: int = 8):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.history_len = history_len

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X e Y deben tener el mismo tamaño temporal (eje T): "
                f"X.T={self.X.shape[0]}, Y.T={self.Y.shape[0]}"
            )
        if self.history_len < 1:
            raise ValueError("history_len debe ser >= 1.")
        if self.X.shape[0] <= self.history_len:
            raise ValueError(
                f"T ({self.X.shape[0]}) debe ser mayor que history_len ({self.history_len})."
            )

    def __len__(self) -> int:
        return self.X.shape[0] - self.history_len

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.history_len]        # (history_len, N, F)
        y_window = self.Y[idx + self.history_len].unsqueeze(0) # (1, N, H)
        return x_window, y_window


def validar_batch_vs_grafo(
    xb: torch.Tensor,
    yb: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    tag: str = "",
) -> None:
    """
    Lanza RuntimeError si hay desalineación entre el batch y el grafo.

    Comprobaciones
    --------------
    1. edge_index.shape[1] == edge_weight.numel()
    2. xb.shape[2] == n_nodes derivado del grafo
    3. yb.shape[2] == n_nodes derivado del grafo
    """
    prefix = f"[{tag}] " if tag else ""

    if edge_index.shape[1] != edge_weight.numel():
        raise RuntimeError(
            f"{prefix}edge_index.shape[1]={edge_index.shape[1]} "
            f"!= edge_weight.numel()={edge_weight.numel()}"
        )

    n_graph = edge_index.max().item() + 1

    if xb.shape[2] != n_graph:
        raise RuntimeError(
            f"{prefix}xb.shape[2]={xb.shape[2]} != n_graph={n_graph}"
        )
    if yb.shape[2] != n_graph:
        raise RuntimeError(
            f"{prefix}yb.shape[2]={yb.shape[2]} != n_graph={n_graph}"
        )

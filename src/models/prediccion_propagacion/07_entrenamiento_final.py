"""
Entrenamiento final + Evaluación + W&B

Carga la mejor configuración de HPO, entrena a épocas completas con Early
Stopping, evalúa en test y registra todo en Weights & Biases.

Carga:  artefactos/tensores.pt, artefactos/grafo.pt, artefactos/hpo.pt,
        artefactos/ablacion.pt (para df_ablacion en W&B)
Guarda: artefactos/modelo_final.pt
  {
    'model_state_dict': ...,
    'best_params':      dict,
    'feature_set':      list[int],
    'subset_name':      str,
    'metricas_test':    dict,
    'best_val_loss':    float,
    'best_epoch':       int,
    'scaler_Y':         StandardScaler,
  }

Uso
---
    uv run python src/models/prediccion_propagacion/07_entrenamiento_final.py
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from models.dcrnn import SubwayDCRNN
from utils.dataset import SubwayDataset, validar_batch_vs_grafo
from utils.metrics import imprimir_metricas, mae_por_horizonte, rmse_por_horizonte

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"
RUTA_GRAFO    = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_HPO      = Path(__file__).parent / "artefactos" / "hpo.pt"
RUTA_ABLACION = Path(__file__).parent / "artefactos" / "ablacion.pt"
RUTA_MODELO   = Path(__file__).parent / "artefactos" / "modelo_final.pt"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "dcrnn-final"

NUM_EPOCHS   = 50
ES_PATIENCE  = 7
OUT_HORIZONS = 3
HORIZONTES   = ['10m', '20m', '30m']
SEED         = 42


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    print("=== 07 Entrenamiento Final ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    fijar_semilla(SEED)

    # ── Cargar artefactos ─────────────────────────────────────────────────────
    datos      = torch.load(RUTA_TENSORES, weights_only=False)
    X_train    = datos['X_train']
    Y_train    = datos['Y_train']
    X_val      = datos['X_val']
    Y_val      = datos['Y_val']
    X_test     = datos['X_test']
    Y_test     = datos['Y_test']
    scaler_Y   = datos['scaler_Y']

    grafo       = torch.load(RUTA_GRAFO, weights_only=False)
    edge_index  = grafo['edge_index'].to(device)
    edge_weight = grafo['edge_weight'].to(device)
    n_nodes     = grafo['n_nodes']

    hpo           = torch.load(RUTA_HPO, weights_only=False)
    best_params   = hpo['best_params']
    feature_set   = hpo['feature_set']
    subset_name   = hpo['subset_name']
    best_strategy = hpo['best_strategy']

    ablacion    = torch.load(RUTA_ABLACION, weights_only=False)
    df_ablacion = ablacion['df_resultados']

    nf  = len(feature_set)
    hl  = best_params['history_len']
    bs  = best_params['batch_size']
    hc  = best_params['hidden_channels']
    K   = best_params['K']
    drp = best_params['dropout']
    lr  = best_params['lr']
    gc_ = best_params['grad_clip']
    sp  = best_params['scheduler_patience']
    sf  = best_params['scheduler_factor']

    print(f"Subset: {subset_name!r} ({nf} features) | history_len={hl} | hidden={hc} | K={K}")
    print(f"lr={lr:.2e} | dropout={drp:.3f} | batch={bs} | grad_clip={gc_}")

    X_tr = X_train[:, :, feature_set]
    X_vl = X_val[:, :, feature_set]
    X_te = X_test[:, :, feature_set]

    tr_ld = DataLoader(SubwayDataset(X_tr, Y_train, history_len=hl), batch_size=bs, shuffle=True)
    vl_ld = DataLoader(SubwayDataset(X_vl, Y_val,   history_len=hl), batch_size=bs, shuffle=False)
    te_ld = DataLoader(SubwayDataset(X_te, Y_test,   history_len=hl), batch_size=bs, shuffle=False)

    # Modelo 
    modelo = SubwayDCRNN(
        in_channels=nf,
        hidden_channels=hc,
        out_horizons=OUT_HORIZONS,
        K=K,
        dropout=drp,
    ).to(device)

    opt  = torch.optim.Adam(modelo.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=sp, factor=sf, min_lr=1e-5
    )
    crit = torch.nn.L1Loss()

    # W&B init
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            'subset_name':    subset_name,
            'best_strategy':  best_strategy,
            'feature_set':    feature_set,
            'n_features':     nf,
            'n_nodes':        n_nodes,
            'n_edges':        edge_index.shape[1],
            'out_horizons':   OUT_HORIZONS,
            'history_len':    hl,
            'num_epochs':     NUM_EPOCHS,
            'es_patience':    ES_PATIENCE,
            'train_samples':  len(tr_ld.dataset),
            'val_samples':    len(vl_ld.dataset),
            'test_samples':   len(te_ld.dataset),
            **{f'hp_{k}': v for k, v in best_params.items()},
        },
    )
    wandb.log({'ablacion': wandb.Table(dataframe=df_ablacion)})

    # Bucle de entrenamiento con Early Stopping
    best_val_loss  = float('inf')
    best_epoch     = 0
    no_imp         = 0
    best_state     = None
    t0             = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        modelo.train()
        tr_acc = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            validar_batch_vs_grafo(xb, yb, edge_index, edge_weight, tag="train")
            opt.zero_grad()
            loss = crit(modelo(xb, edge_index, edge_weight), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), gc_)
            opt.step()
            tr_acc += loss.item()
        train_loss = tr_acc / len(tr_ld)

        modelo.eval()
        with torch.no_grad():
            vl_acc = sum(
                crit(modelo(xb.to(device), edge_index, edge_weight), yb.to(device)).item()
                for xb, yb in vl_ld
            )
            val_loss = vl_acc / len(vl_ld)

        curr_lr = opt.param_groups[0]['lr']
        sch.step(val_loss)
        new_lr  = opt.param_groups[0]['lr']
        lr_tag  = f"  ↓ lr={new_lr:.2e}" if new_lr < curr_lr else ""

        wandb.log({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         new_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_imp        = 0
            best_state    = copy.deepcopy(modelo.state_dict())

        else:
            no_imp += 1

        print(f"Época {epoch:02d}/{NUM_EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f}{lr_tag}"
              + ("  [best]" if no_imp == 0 else ""))

        if no_imp >= ES_PATIENCE:
            print(f"\nEarly stopping en época {epoch}.")
            break

    tiempo_total = time.time() - t0
    print(f"\nEntrenamiento completado en {tiempo_total:.0f}s | Mejor val: {best_val_loss:.6f} (época {best_epoch})")

    if best_state is not None:
        modelo.load_state_dict(best_state)
    else:
        print("AVISO: best_state es None; se usan los pesos del último epoch.")

    # Evaluación en test
    modelo.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            xb = xb.to(device)
            validar_batch_vs_grafo(xb, yb, edge_index, edge_weight, tag="test")
            preds_list.append(modelo(xb, edge_index, edge_weight).cpu().numpy())
            trues_list.append(yb.numpy())

    preds = np.concatenate(preds_list, axis=0).squeeze(1)  # (T, N, 3)
    trues = np.concatenate(trues_list, axis=0).squeeze(1)

    mae  = mae_por_horizonte(preds, trues, scaler_Y, HORIZONTES)
    rmse = rmse_por_horizonte(preds, trues, scaler_Y, HORIZONTES)

    print("\n=== Métricas en Test (segundos reales) ===")
    imprimir_metricas(mae, rmse)

    # Renombrar claves para W&B
    metricas_wandb = {f'test_{k}': v for k, v in {**mae, **rmse}.items()}
    metricas_wandb['best_val_loss']     = best_val_loss
    metricas_wandb['best_epoch']        = best_epoch
    metricas_wandb['train_time_sec']    = tiempo_total
    wandb.log(metricas_wandb)
    wandb.finish()
    print("\nResultados registrados en W&B.")

    # Guardar modelo
    torch.save(
        {
            'model_state_dict': modelo.state_dict(),
            'best_params':      best_params,
            'feature_set':      feature_set,
            'subset_name':      subset_name,
            'metricas_test':    {**mae, **rmse},
            'best_val_loss':    best_val_loss,
            'best_epoch':       best_epoch,
            'scaler_Y':         scaler_Y,
            'n_features':       nf,
            'out_horizons':     OUT_HORIZONS,
            'K':                K,
            'hidden_channels':  hc,
        },
        RUTA_MODELO,
    )
    print(f"Modelo final guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    main()

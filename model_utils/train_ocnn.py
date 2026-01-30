# model_utils/train_ocnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict, Optional, List

from model_utils.ocnn_utils import (
    hyperplane_loss,
    extract_z,
    compute_violation,
    r_from_quantile,
)


# Train one epoch with r fixed (r is updated outside via quantile)
def train_ocnn_epoch(
    ocnn: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    r: float,
    nu: float,
    encoder: Optional[nn.Module] = None,
    joint: bool = False,
    clip_norm: float = 0.5,
) -> float:
    if joint and encoder is None:
        raise ValueError("joint=True but encoder is None")

    ocnn.train()
    if joint:
        encoder.train()

    # OCNN exposes separate parameter groups (w, V) for the objective regularizer
    params_dict = ocnn.get_trainable()

    total_loss = 0.0
    n_seen = 0

    # joint=False: loader yields (z, y)
    # joint=True : loader yields (x, y) and encoder produces z
    for data, _y in train_loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        z = extract_z(encoder=encoder, data=data, joint=joint)
        scores = ocnn(z)

        loss = hyperplane_loss(
            scores=scores,
            r=r,
            nu=nu,
            w_params=params_dict["w_params"],
            V_params=params_dict["V_params"],
        )

        loss.backward()

        # Clip gradients across all optimizer param groups (OCNN + encoder if joint)
        if clip_norm is not None and clip_norm > 0:
            all_params = [p for pg in optimizer.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_norm)

        optimizer.step()

        total_loss += loss.item() * z.size(0)
        n_seen += z.size(0)

    return total_loss / max(1, n_seen)


# Full OCNN training loop (z-only or joint)
def train_ocnn(
    ocnn: nn.Module,
    train_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    nu: float = 0.1,
    epochs: int = 150,
    lr_init: float = 1e-4,
    lr_finetune: float = 1e-5,
    finetune_start_epoch: int = 50,
    r_init: Optional[float] = None,
    print_every: int = 10,
    encoder: Optional[nn.Module] = None,
    joint: bool = False,
    lr_encoder: float = 0.0,
    clip_norm: float = 0.5,
) -> Tuple[nn.Module, Optional[nn.Module], float, Dict[str, list]]:
    if joint and encoder is None:
        raise ValueError("joint=True but encoder is None")

    ocnn = ocnn.to(device)
    if joint:
        encoder = encoder.to(device)

    # Build OCNN parameter group from the explicit trainable weights
    params_dict = ocnn.get_trainable()
    ocnn_params: List[nn.Parameter] = []
    ocnn_params.extend(params_dict["V_params"])
    ocnn_params.extend(params_dict["w_params"])

    param_groups = [{"params": ocnn_params, "lr": lr_init, "name": "ocnn"}]

    # Optional encoder group for joint training (lr_encoder=0.0 => effectively frozen)
    if joint:
        enc_params = list(encoder.parameters())
        param_groups.append({"params": enc_params, "lr": lr_encoder, "name": "encoder"})

    # No weight decay here: L2 regularization is already part of hyperplane_loss
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)

    history: Dict[str, list] = {"loss": [], "r": [], "violation": [], "lr_ocnn": [], "lr_enc": []}

    print(
        f"[OCNN] mode={'joint' if joint else 'z-only'} | "
        f"lr_ocnn_init={lr_init}, lr_ocnn_finetune={lr_finetune} @ epoch {finetune_start_epoch} | "
        f"lr_encoder={lr_encoder if joint else 'N/A'}"
    )

    # Initialize r
    if r_init is None:
        r = r_from_quantile(encoder, ocnn, train_loader, device, nu, joint)
        print(f"[OCNN] r initialized from nu-quantile: r={r:.6f}")
    else:
        r = float(r_init)

    for epoch in range(1, epochs + 1):
        # Switch OCNN LR at finetune_start_epoch (encoder LR remains unchanged)
        if epoch == finetune_start_epoch:
            for pg in optimizer.param_groups:
                if pg.get("name") == "ocnn":
                    pg["lr"] = lr_finetune
            print(f"[OCNN] Epoch {epoch}: switched OCNN LR -> {lr_finetune} (encoder LR stays {lr_encoder})")

        avg_loss = train_ocnn_epoch(
            ocnn=ocnn,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            r=r,
            nu=nu,
            encoder=encoder,
            joint=joint,
            clip_norm=clip_norm,
        )

        # Update r as nu-quantile of scores
        r = r_from_quantile(encoder, ocnn, train_loader, device, nu, joint)

        # Violation rate: fraction of samples with score < r (target ~ nu)
        viol = compute_violation(encoder, ocnn, train_loader, r, device, joint)

        lr_ocnn = next(pg["lr"] for pg in optimizer.param_groups if pg.get("name") == "ocnn")
        lr_enc = None
        if joint:
            lr_enc = next(pg["lr"] for pg in optimizer.param_groups if pg.get("name") == "encoder")

        history["loss"].append(avg_loss)
        history["r"].append(r)
        history["violation"].append(viol)
        history["lr_ocnn"].append(lr_ocnn)
        history["lr_enc"].append(lr_enc)

        if print_every and (epoch % print_every == 0 or epoch == 1):
            if joint:
                print(
                    f"[OCNN] Epoch {epoch:3d}/{epochs} | "
                    f"loss={avg_loss:.6f} | r={r:.6f} | "
                    f"viol={viol:.4f} (target~{nu:.4f}) | "
                    f"lr_ocnn={lr_ocnn:.2e} | lr_enc={lr_enc:.2e}"
                )
            else:
                print(
                    f"[OCNN] Epoch {epoch:3d}/{epochs} | "
                    f"loss={avg_loss:.6f} | r={r:.6f} | "
                    f"viol={viol:.4f} (target~{nu:.4f}) | lr={lr_ocnn:.2e}"
                )

    print(f"[OCNN] done. final r={r:.6f}")
    return ocnn, encoder, r, history


# Evaluate OCNN and return anomaly scores + labels
@torch.no_grad()
def evaluate_ocnn(
    encoder: Optional[nn.Module],
    ocnn: nn.Module,
    loader: DataLoader,
    r: float,
    device: torch.device,
    joint: bool,
):
    # Convention: anomaly_score = r - score, so larger => more anomalous
    ocnn.to(device).eval()
    if encoder is not None:
        encoder.to(device).eval()

    all_s, all_y = [], []
    for data, y in loader:
        data = data.to(device)
        z = extract_z(encoder, data, joint)
        score = ocnn(z)
        all_s.append(score.cpu())
        all_y.append(y.cpu())

    scores = torch.cat(all_s, dim=0)
    labels = torch.cat(all_y, dim=0)

    anom_scores = r - scores
    return anom_scores, labels


# Compute AUROC from anomaly scores (higher => more anomalous) and labels (1 => anomaly)
def auroc_from_scores(anom_scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float(roc_auc_score(labels.numpy(), anom_scores.numpy()))

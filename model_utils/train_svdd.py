import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, Tuple


def _extract_z(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return latent z as [B, D] (flattening spatial dims if needed)."""
    if hasattr(encoder, "encode_flat"):
        z = encoder.encode_flat(x)
    elif hasattr(encoder, "encode"):
        z = encoder.encode(x)
    else:
        z = encoder(x)

    if z.dim() > 2:
        z = z.flatten(1)
    return z


@torch.no_grad()
def init_center_c(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.1,
) -> torch.Tensor:
    """Initialize center c as mean of z over the training set; push near-zero components away from 0."""
    encoder.eval()
    n = 0
    c = None

    for x, _ in loader:
        x = x.to(device)
        z = _extract_z(encoder, x)
        c = z.sum(dim=0) if c is None else (c + z.sum(dim=0))
        n += z.size(0)

    c = c / max(1, n)

    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c.detach()


def svdd_loss(dist: torch.Tensor, R: torch.Tensor, nu: float, objective: str) -> torch.Tensor:
    """dist is ||z-c||^2 (shape [B])."""
    if objective == "one-class":
        return dist.mean()

    scores = dist - R.pow(2)
    return R.pow(2) + (1.0 / nu) * torch.relu(scores).mean()


@torch.no_grad()
def update_radius_R(dist_all: torch.Tensor, nu: float) -> float:
    """Set R using the (1-nu)-quantile of dist: R = sqrt(quantile(dist))."""
    q = 1.0 - nu
    return torch.quantile(dist_all, q=q).sqrt().item()


def train_deepsvdd(
    encoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    objective: str = "one-class",
    nu: float = 0.1,
    epochs: int = 150,
    lr_encoder: float = 0.0,
    weight_decay: float = 0.0,
    warmup_epochs: int = 10,
    clip_norm: float = 5.0,
    print_every: int = 10,
) -> Tuple[nn.Module, torch.Tensor, float, Dict[str, list]]:
    """Train DeepSVDD; if lr_encoder==0, encoder is frozen (no updates)."""
    encoder = encoder.to(device)
    joint = lr_encoder > 0.0

    if joint:
        for p in encoder.parameters():
            p.requires_grad = True
        encoder.train()
    else:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    c = init_center_c(encoder, train_loader, device, eps=0.1)
    R = torch.tensor(0.0, device=device)

    opt = (
        torch.optim.Adam(encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay)
        if joint else None
    )

    hist = {"loss": [], "R": []}

    for epoch in range(1, epochs + 1):
        encoder.train() if joint else encoder.eval()

        total_loss, n_seen = 0.0, 0
        dist_epoch = []

        for x, _ in train_loader:
            x = x.to(device)

            if joint:
                opt.zero_grad(set_to_none=True)

            z = _extract_z(encoder, x)
            dist = ((z - c) ** 2).sum(dim=1)

            use_obj = objective
            if objective == "soft-boundary" and epoch < warmup_epochs:
                use_obj = "one-class"

            loss = svdd_loss(dist, R, nu, use_obj)

            if joint:
                loss.backward()
                if clip_norm and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_norm)
                opt.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            dist_epoch.append(dist.detach())

        avg_loss = total_loss / max(1, n_seen)

        if objective == "soft-boundary" and epoch >= warmup_epochs:
            dist_all = torch.cat(dist_epoch, dim=0).detach().cpu()
            R_val = update_radius_R(dist_all, nu)
            R = torch.tensor(R_val, device=device)

        hist["loss"].append(avg_loss)
        hist["R"].append(float(R.item()))

        if print_every and (epoch == 1 or epoch % print_every == 0):
            print(f"[SVDD] Epoch {epoch:3d}/{epochs} | loss={avg_loss:.6f} | R={R.item():.6f}")

    return encoder, c, float(R.item()), hist


@torch.no_grad()
def eval_deepsvdd(
    encoder: nn.Module,
    c: torch.Tensor,
    R: float,
    loader: DataLoader,
    device: torch.device,
    normal_class: Optional[int] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Compute AUROC using score = dist - R^2 (higher means more anomalous)."""
    encoder = encoder.to(device).eval()
    c = c.to(device)

    all_scores, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if y.numel() > 0:
            y_unique = torch.unique(y)
            is_binary = (y_unique.numel() <= 2) and torch.all((y_unique == 0) | (y_unique == 1))
        else:
            is_binary = True

        if not is_binary:
            if normal_class is None:
                raise ValueError(
                    "eval_deepsvdd: labels are not binary but normal_class is None. "
                    "Pass normal_class or use a one-class dataset wrapper."
                )
            y = (y != normal_class).long()
        else:
            y = y.long()

        z = _extract_z(encoder, x)
        dist = ((z - c) ** 2).sum(dim=1)
        score = dist - (R ** 2)

        all_scores.append(score.detach().cpu())
        all_labels.append(y.detach().cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    auc = float(roc_auc_score(labels.numpy(), scores.numpy()))
    return auc, scores, labels

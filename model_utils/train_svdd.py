import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, Tuple


def _extract_z(
    encoder: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Ritorna z in forma [B, D] con grafo attivo (se encoder in train e richiede grad).
    """
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
    """
    Inizializza il centro c come media delle rappresentazioni z sul training set.
    Standard DeepSVDD: evita componenti troppo vicine a 0.
    """
    encoder.eval()
    n = 0
    c = None

    for x, _ in loader:
        x = x.to(device)
        z = _extract_z(encoder, x)  # [B, D]
        if c is None:
            c = z.sum(dim=0)
        else:
            c += z.sum(dim=0)
        n += z.size(0)

    c = c / max(1, n)

    # evita componenti troppo vicine a 0 (standard DeepSVDD)
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c.detach()


def svdd_loss(
    dist: torch.Tensor,  # [B] = ||z-c||^2
    R: torch.Tensor,     # scalar tensor
    nu: float,
    objective: str,
) -> torch.Tensor:
    if objective == "one-class":
        return dist.mean()

    # soft-boundary
    scores = dist - R.pow(2)
    return R.pow(2) + (1.0 / nu) * torch.relu(scores).mean()


@torch.no_grad()
def update_radius_R(dist_all: torch.Tensor, nu: float) -> float:
    """
    Paper/common impl: R^2 = (1-nu)-quantile(dist)  =>  R = sqrt(quantile(dist)).
    dist_all: 1D tensor of distances (preferably on CPU).
    """
    q = 1.0 - nu
    return torch.quantile(dist_all, q=q).sqrt().item()


def train_deepsvdd(
    encoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    objective: str = "one-class",
    nu: float = 0.1,
    epochs: int = 50,
    lr_encoder: float = 0.0,          # >0 => joint (aggiorna encoder durante SVDD). 0 => frozen.
    weight_decay: float = 0.0,
    warmup_epochs: int = 10,          # per soft-boundary
    clip_norm: float = 5.0,
    print_every: int = 10,
) -> Tuple[nn.Module, torch.Tensor, float, Dict[str, list]]:
    """
    DeepSVDD training.

    Returns:
      encoder: (potenzialmente aggiornato se joint)
      c: center tensor [D]
      R: float radius
      history: dict con loss e R per epoca
    """
    encoder = encoder.to(device)

    joint = lr_encoder > 0.0

    if joint:
        # assicura che i grad siano abilitati (robusto se encoder era stato congelato altrove)
        for p in encoder.parameters():
            p.requires_grad = True
        encoder.train()
    else:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    # init center c (sempre senza grad)
    c = init_center_c(encoder, train_loader, device, eps=0.1)

    # init R
    R = torch.tensor(0.0, device=device)

    # optimizer solo se joint
    opt = torch.optim.Adam(
        encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay
    ) if joint else None

    hist = {"loss": [], "R": []}

    for epoch in range(1, epochs + 1):
        encoder.train() if joint else encoder.eval()

        total_loss, n_seen = 0.0, 0
        dist_epoch = []

        for x, _ in train_loader:
            x = x.to(device)

            if joint:
                opt.zero_grad(set_to_none=True)

            z = _extract_z(encoder, x)                 # [B, D]
            dist = ((z - c) ** 2).sum(dim=1)          # [B]

            # warmup: se soft-boundary, spesso si ottimizza come one-class prima di aggiornare R
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

        # update R after warmup if soft-boundary
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
    """
    Eval DeepSVDD con AUROC.

    - scores: dist - R^2  (più alto => più anomalo)
    - labels devono essere binarie: 0 normal, 1 anomaly

    Se nel loader y è già binario (0/1), viene usato così.
    Se y è multi-classe (es. CIFAR-10 0..9 o MNIST digit), serve normal_class:
      labels = (y != normal_class).
    """
    encoder = encoder.to(device).eval()
    c = c.to(device)

    all_scores, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # --- ensure binary labels ---
        if y.numel() > 0:
            y_unique = torch.unique(y)
            is_binary = (y_unique.numel() <= 2) and torch.all((y_unique == 0) | (y_unique == 1))
        else:
            is_binary = True

        if not is_binary:
            if normal_class is None:
                raise ValueError(
                    "eval_deepsvdd: labels are not binary but normal_class is None. "
                    "Pass normal_class (e.g., airplane=0 for CIFAR-10) or use a one-class dataset wrapper."
                )
            y = (y != normal_class).long()  # 0 normal, 1 anomaly
        else:
            y = y.long()

        # --- compute scores ---
        z = _extract_z(encoder, x)
        dist = ((z - c) ** 2).sum(dim=1)     # [B]
        score = dist - (R ** 2)

        all_scores.append(score.detach().cpu())
        all_labels.append(y.detach().cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    auc = float(roc_auc_score(labels.numpy(), scores.numpy()))
    return auc, scores, labels

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
import torch.nn.functional as F

# OCNN objective:
#   L = 0.5 (||w||^2 + ||V||^2) + (1/nu) * mean(max(0, r - f(x))) - r
def hyperplane_loss(
    scores: torch.Tensor,
    r: float,
    nu: float,
    w_params: List[nn.Parameter],
    V_params: List[nn.Parameter],
) -> torch.Tensor:
    if not (0.0 < nu <= 1.0):
        raise ValueError(f"nu must be in (0,1], got {nu}")

    # r as tensor on the same device/dtype as scores
    r_t = scores.new_tensor(r)

    # Hinge term: penalize samples with score below r
    hinge = torch.relu(r_t - scores).mean()

    # L2 regularization on w and V (no bias parameters)
    reg = scores.new_tensor(0.0)
    for p in w_params:
        reg += p.pow(2).sum()
    for p in V_params:
        reg += p.pow(2).sum()
    reg = 0.5 * reg

    return reg + (1.0 / nu) * hinge - r_t


# Extract z in a way that supports both:
# - z-only training (data already contains z)
# - joint training (data contains x, encoder produces z)
def extract_z(
    encoder: Optional[nn.Module],
    data: torch.Tensor,
    joint: bool,
) -> torch.Tensor:
    if not joint:
        z = data
        return z.flatten(1) if z.dim() > 2 else z

    if encoder is None:
        raise ValueError("joint=True but encoder is None")

    # Try common encoder interfaces; fallback to forward()
    if hasattr(encoder, "encode_flat"):
        z = encoder.encode_flat(data)
    elif hasattr(encoder, "encode"):
        z = encoder.encode(data)
    else:
        z = encoder(data)
    z = z.flatten(1) if z.dim() > 2 else z

    z = F.normalize(z, p=2, dim=1)
    # Ensure [B, rep_dim]
    return  z

# Collect scores over a loader (no gradient), used for r update / metrics
@torch.no_grad()
def collect_scores(
    encoder: Optional[nn.Module],
    ocnn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    joint: bool,
) -> torch.Tensor:
    ocnn.eval()
    if encoder is not None:
        encoder.eval()

    all_scores = []
    for x, _ in loader:
        x = x.to(device)
        z = extract_z(encoder, x, joint)
        s = ocnn(z)
        all_scores.append(s.detach().cpu())

    if not all_scores:
        return torch.empty(0)

    return torch.cat(all_scores, dim=0)


# Update r as the nu-quantile of the score distribution (no gradient)
@torch.no_grad()
def r_from_quantile(
    encoder: Optional[nn.Module],
    ocnn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    nu: float,
    joint: bool,
) -> float:
    if not (0.0 < nu <= 1.0):
        raise ValueError(f"nu must be in (0,1], got {nu}")

    scores = collect_scores(encoder, ocnn, loader, device, joint)
    if scores.numel() == 0:
        return 0.0

    return torch.quantile(scores, q=nu).item()


# Fraction of samples violating the constraint score >= r (no gradient)
@torch.no_grad()
def compute_violation(
    encoder: Optional[nn.Module],
    ocnn: nn.Module,
    loader: DataLoader,
    r: float,
    device: torch.device,
    joint: bool,
) -> float:
    ocnn.eval()
    if encoder is not None:
        encoder.eval()

    n_total = 0
    n_viol = 0

    for data, _ in loader:
        data = data.to(device)
        z = extract_z(encoder, data, joint)
        scores = ocnn(z)
        n_total += scores.numel()
        n_viol += (scores < r).sum().item()

    return n_viol / max(1, n_total)

# model_utils/train_autoencoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def soft_threshold_numpy(lamda: float, b: np.ndarray) -> np.ndarray:
    th = float(lamda) / 2.0
    x = np.zeros_like(b, dtype=b.dtype)

    k = np.where(b > th)
    x[k] = b[k] - th

    k = np.where(np.abs(b) <= th)
    x[k] = 0.0

    k = np.where(b < -th)
    x[k] = b[k] + th

    return x


def _linear_weight_l2_sum(model: nn.Module) -> torch.Tensor:
    s = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            term = torch.linalg.norm(m.weight)
            s = term if s is None else (s + term)

    if s is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return s


@torch.no_grad()
def _reconstruct_full(
    ae: nn.Module,
    X_in: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    ae.eval().to(device)
    out: List[torch.Tensor] = []

    loader = DataLoader(
        TensorDataset(X_in),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for (xb,) in loader:
        xb = xb.to(device)
        out.append(ae(xb).detach().cpu())

    return torch.cat(out, dim=0)


def train_ae_mse(
    net: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    print_every: int = 1,
) -> nn.Module:
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_seen = 0

        for x, *_ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            x_rec = net(x)
            loss = torch.mean((x - x_rec) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        avg_loss = total_loss / max(1, n_seen)
        if print_every and (epoch % print_every == 0 or epoch in (1, n_epochs)):
            print(f"[AE-MSE] Epoch {epoch:3d}/{n_epochs} | loss={avg_loss:.6f}")

    return net


@dataclass
class RCAEConfig:
    lamda_set: Sequence[float] = (0.1,)
    n_outer_iters: int = 1
    n_epochs_ae: int = 150
    lr_ae: float = 1e-3
    batch_size: int = 128
    mue: float = 0.0
    weight_decay: float = 0.0
    print_every: int = 1
    reconstruct_clean: bool = True
    freeze_encoder: bool = False


def _fit_rcae_inner(
    ae: nn.Module,
    X_noisy: torch.Tensor,
    X_target: torch.Tensor,
    device: torch.device,
    lamda: float,
    N_numpy: np.ndarray,
    cfg: RCAEConfig,
) -> torch.Tensor:
    ae = ae.to(device)
    ae.train()

    X_noisy = X_noisy.to(device)
    X_target = X_target.to(device)

    loader = DataLoader(
        TensorDataset(X_noisy, X_target),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae, weight_decay=cfg.weight_decay)

    # constant term (doesn't affect gradients)
    N_norm = float(np.linalg.norm(N_numpy.reshape(-1)))
    N_norm_t = torch.tensor(N_norm, dtype=torch.float32, device=device)

    for epoch in range(1, cfg.n_epochs_ae + 1):
        epoch_loss = 0.0
        n_seen = 0

        for x_noisy_b, x_tgt_b in loader:
            x_noisy_b = x_noisy_b.to(device)
            x_tgt_b = x_tgt_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            x_hat = ae(x_noisy_b)
            mse = torch.mean((x_hat - x_tgt_b) ** 2)
            reg_linear = _linear_weight_l2_sum(ae)

            loss = mse + (cfg.mue * 0.5 * reg_linear) + (lamda * 0.5 * N_norm_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_noisy_b.size(0)
            n_seen += x_noisy_b.size(0)

        epoch_loss /= max(1, n_seen)
        if cfg.print_every and (epoch % cfg.print_every == 0 or epoch in (1, cfg.n_epochs_ae)):
            print(f"[RCAE-inner] Epoch {epoch:3d}/{cfg.n_epochs_ae} | loss={epoch_loss:.6f}")

    X_hat = _reconstruct_full(ae, X_noisy.detach().cpu(), device=device, batch_size=cfg.batch_size)
    return X_hat


def pretrain_rcae(
    ae: nn.Module,
    train_images: torch.Tensor,   # [N,C,H,W]
    device: torch.device,
    cfg: Optional[RCAEConfig] = None,
) -> Tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
    if cfg is None:
        cfg = RCAEConfig()

    # ✅ PATCH: ora supporta [N,C,H,W] per MNIST e CIFAR10
    if train_images.ndim != 4:
        raise ValueError(f"train_images must be [N,C,H,W], got {tuple(train_images.shape)}")

    # keep on CPU for numpy updates
    X_clean = train_images.clone().detach().cpu()

    # ✅ PATCH: dimensione corretta = C*H*W
    flat_dim = int(X_clean.size(1) * X_clean.size(2) * X_clean.size(3))
    N = np.zeros((X_clean.size(0), flat_dim), dtype=np.float32)

    logs: Dict[str, Any] = {"lambda_logs": []}

    for lamda in cfg.lamda_set:
        print(f"\n[RCAE] lambda={lamda}")

        for outer in range(1, cfg.n_outer_iters + 1):
            X_clean_flat = X_clean.view(X_clean.size(0), -1).numpy()
            X_noisy_flat = X_clean_flat - N
            X_noisy = torch.from_numpy(X_noisy_flat).view_as(X_clean)

            X_target = X_clean if cfg.reconstruct_clean else X_noisy

            X_hat = _fit_rcae_inner(
                ae=ae,
                X_noisy=X_noisy,
                X_target=X_target,
                device=device,
                lamda=lamda,
                N_numpy=N,
                cfg=cfg,
            )

            X_hat_flat = X_hat.view(X_hat.size(0), -1).numpy()
            resid = X_clean_flat - X_hat_flat
            N = soft_threshold_numpy(lamda, resid)

            nz = int(np.count_nonzero(N))
            print(f"[RCAE] Outer {outer:2d}/{cfg.n_outer_iters} | nonzero(N)={nz}")
            logs["lambda_logs"].append({"lambda": float(lamda), "outer": outer, "nonzero_N": nz})

    encoder = ae.encoder if hasattr(ae, "encoder") else None
    if encoder is not None:
        encoder = encoder.to(device)
        if cfg.freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

    info = {"N": N, "cfg": cfg, "logs": logs}
    return ae, encoder, info

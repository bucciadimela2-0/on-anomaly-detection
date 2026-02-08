from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def pretrain_autoencoder_mse(
    ae: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    print_every: int = 1,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    
    #Pretrain an autoencoder using standard MSE reconstruction loss.
    ae = ae.to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history: List[float] = []

    for epoch in range(1, n_epochs + 1):
        ae.train()

        total_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            # Support common loader formats:
            # - (x,)
            # - (x, y) / (x, y, ...)
            # - dict with key "x"
            if isinstance(batch, dict):
                x = batch["x"]
            else:
                x = batch[0]

            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            x_rec = ae(x)
            loss = torch.mean((x_rec - x) ** 2)
            if epoch == 1:
                w0 = None
                for p in ae.parameters():
                    if p.requires_grad:
                        w0 = p.detach().clone()
                        break

            loss.backward()
            optimizer.step()

            if epoch == 1 and w0 is not None:
                for p in ae.parameters():
                    if p.requires_grad:
                        delta = (p.detach() - w0).abs().mean().item()
                        print("[DEBUG][AE] mean |Δw| =", delta)
                        break


            #loss.backward()
            #if grad_clip is not None and grad_clip > 0:
            #    torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=float(grad_clip))
            #optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_seen += bs

        avg_loss = total_loss / max(1, n_seen)
        loss_history.append(avg_loss)

        if print_every and (epoch % print_every == 0 or epoch in (1, n_epochs)):
            print(f"[AE-pretrain] Epoch {epoch:3d}/{n_epochs} | mse={avg_loss:.6f}")

    return {"model": ae, "loss_history": loss_history}

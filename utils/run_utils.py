
import os
from datetime import datetime
from typing import Optional, Union, Tuple
import os
from dataclasses import asdict
from utils.Config import Config, set_seed

import torch
from torch.utils.data import DataLoader, TensorDataset


def ensure_dir(path: str) -> None:
    #Create directory if it does not exist.
    os.makedirs(path, exist_ok=True)


def make_run_dir(
    base_dir: str,
    run_name: str,
    date_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    #Create and return a directory for a specific run.

    timestamp = datetime.now().strftime(date_fmt)
    run_dir = os.path.join(base_dir, run_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_run_name(
    model: str,
    digit: Union[int, str],
    nu: float,
    date_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    timestamp = datetime.now().strftime(date_fmt)
    return f"{model}_digit{digit}_{timestamp}"



class EncoderWrapper(torch.nn.Module):
    """
    Wrap an AE to expose a stable interface: encode(x) -> z [B, rep_dim]
   .
    """
    def __init__(self, ae: torch.nn.Module):
        super().__init__()
        self.ae = ae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.ae, "encode_flat"):
            z = self.ae.encode_flat(x)
        elif hasattr(self.ae, "encode"):
            z = self.ae.encode(x)
        elif hasattr(self.ae, "encoder"):
            z = self.ae.encoder(x)
        else:
            raise AttributeError(
                "EncoderWrapper: AE has no encode_flat(), encode(), or encoder attribute. "
                "Refusing to fall back to ae(x) because that may return reconstructions."
            )

        return z.flatten(1) if z.dim() > 2 else z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


@torch.no_grad()
def infer_rep_dim(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> int:
    encoder.eval().to(device)
    x, _ = next(iter(loader))
    x = x.to(device)
    z = encoder.encode(x) if hasattr(encoder, "encode") else encoder(x)
    z = z.flatten(1) if z.dim() > 2 else z
    return int(z.shape[1])


@torch.no_grad()
def collect_images_and_labels(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Collects x,y from a loader into CPU tensors (for plotting)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

def get_device(prefer_mps: bool = True) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

def _prepare_run_dirs(cfg: Config) -> Tuple[str, str, str, str]:
    mode_tag = f"{cfg.ae_arch}_{cfg.dataset}"
    digit = cfg.normal_digit if cfg.dataset == "mnist" else cfg.normal_class
    run_name = make_run_name(model=mode_tag, digit=digit, nu=cfg.nu)
    run_dir = make_run_dir(cfg.base_runs_dir, run_name)

    ckpt_dir = os.path.join(run_dir, cfg.ckpt_subdir)
    plots_dir = os.path.join(run_dir, cfg.plots_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return run_name, run_dir, ckpt_dir, plots_dir
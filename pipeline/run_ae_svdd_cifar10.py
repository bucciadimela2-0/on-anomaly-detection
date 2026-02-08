# pipeline/run_ae_svdd_cifar10.py
import os
from dataclasses import asdict
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.Config import Config, set_seed
from utils.run_utils import make_run_name, make_run_dir
from utils.cifar_datamodule import CIFAR10OneClassDataModule

from models.ae_factory import build_autoencoder
from model_utils.pretrain_autoencoder import pretrain_autoencoder_mse
from model_utils.train_svdd import train_deepsvdd, eval_deepsvdd

from utils.plot_utils import (
    plot_training_curves,
    plot_score_histogram,
    plot_roc_curve,
    plot_score_boxplot,
    plot_extremes_in_class,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get(cfg: Config, name: str, default):
    """Safe getter for optional fields not present in the Config dataclass."""
    return getattr(cfg, name, default)


def get_device(prefer_mps: bool = True) -> torch.device:
    """Select device with CUDA > MPS > CPU priority."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EncoderWrapper(torch.nn.Module):
    """Uniform wrapper exposing encode(x)->z for any AE; returns flattened z."""
    def __init__(self, ae: torch.nn.Module):
        super().__init__()
        self.ae = ae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.ae, "encode_flat"):
            z = self.ae.encode_flat(x)
        elif hasattr(self.ae, "encode"):
            z = self.ae.encode(x)
        else:
            z = self.ae(x)

        if z.dim() > 2:
            z = z.flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


@torch.no_grad()
def infer_rep_dim(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> int:
    """Infer latent dimension from one batch."""
    encoder.eval().to(device)
    x, _ = next(iter(loader))
    x = x.to(device)
    z = encoder.encode(x) if hasattr(encoder, "encode") else encoder(x)
    if z.dim() > 2:
        z = z.flatten(1)
    return int(z.shape[1])


@torch.no_grad()
def collect_images_and_labels(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a full loader to CPU tensors (used only for plotting)."""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class NormalOnlyLoader:
    """
    Wrap a (x,y) loader and yield only samples with y==0 (normals).
    This is the key fix when training SVDD with pollution_rate > 0.
    """
    def __init__(self, base_loader: DataLoader):
        self.base_loader = base_loader

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for x, y in self.base_loader:
            mask = (y == 0)
            if mask.any():
                yield x[mask], y[mask]

    def __len__(self) -> int:
        # Not exact, but rarely needed by training loops.
        return len(self.base_loader)


# ------------------------------------------------------------
# Phase 1 — AE pretraining (MSE, normals only)
# ------------------------------------------------------------
def _load_or_pretrain_ae_mse(
    cfg: Config,
    dm: CIFAR10OneClassDataModule,
    device: torch.device,
    ckpt_dir: str,
) -> torch.nn.Module:
    ae = build_autoencoder(cfg).to(device)
    ae_ckpt_path = os.path.join(ckpt_dir, "ae_mse.pt")

    if os.path.exists(ae_ckpt_path):
        print(f"[PHASE 1] Loading AE checkpoint: {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae.load_state_dict(ckpt["model"], strict=True)
        return ae

    print("[PHASE 1] Pretraining AE (MSE, normals only)")
    train_imgs = dm.collect_normal_images(device=device)  # [N_norm, 3, 32, 32]

    ae_train_loader = DataLoader(
        TensorDataset(train_imgs),
        batch_size=int(_get(cfg, "ae_batch_size", 200)),
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ae_out = pretrain_autoencoder_mse(
        ae=ae,
        train_loader=ae_train_loader,
        device=device,
        n_epochs=int(_get(cfg, "ae_epochs", 150)),
        lr=float(_get(cfg, "ae_lr", 1e-3)),
        weight_decay=float(_get(cfg, "ae_weight_decay", 0.0)),
        print_every=1,
        grad_clip=_get(cfg, "ae_grad_clip", None),
    )
    ae = ae_out["model"]

    torch.save(
        {
            "model": ae.state_dict(),
            "loss_history": ae_out.get("loss_history", None),
            "cfg": asdict(cfg),
        },
        ae_ckpt_path,
    )
    print(f"[PHASE 1] Saved AE checkpoint: {ae_ckpt_path}")
    return ae


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_pipeline(cfg: Config) -> None:
    # -------------------------
    # System / seed / dirs
    # -------------------------
    set_seed(cfg.seed)
    device = get_device(prefer_mps=True)
    print(f"[SYS] Device: {device}")

    svdd_lr_encoder = float(_get(cfg, "svdd_lr_encoder", 0.0))
    svdd_joint = svdd_lr_encoder > 0.0
    svdd_objective = _get(cfg, "svdd_objective", "one-class")
    svdd_nu = float(_get(cfg, "svdd_nu", 0.1))

    # For CIFAR, normal class id is passed via cfg.normal_digit in your current setup
    normal_class = int(getattr(cfg, "normal_class", cfg.normal_digit))

    mode_tag = f"simple_svdd_{svdd_objective}_{'joint' if svdd_joint else 'zonly'}"
    run_name = make_run_name(model=mode_tag, digit=normal_class, nu=svdd_nu)
    run_dir = make_run_dir(cfg.base_runs_dir, run_name)

    ckpt_dir = os.path.join(run_dir, cfg.ckpt_subdir)
    plots_dir = os.path.join(run_dir, cfg.plots_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[RUN] run_name: {run_name}")
    print(f"[RUN] run_dir : {run_dir}")

    # -------------------------
    # Data (CIFAR-10 one-class; binary labels already)
    # -------------------------
    dm = CIFAR10OneClassDataModule(
        data_dir=cfg.data_dir,
        normal_class=normal_class,
        pollution_rate=cfg.pollution_rate,
        batch_size=int(_get(cfg, "svdd_batch_size", 200)),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        download=True,
        
    )
    dm.setup()
    train_loader_xy = dm.train_dataloader()  # (x,y) with y in {0,1}
    test_loader = dm.test_dataloader()

    print(f"[DATA] normal_class={normal_class} | normals(train_full)={dm.n_normals} | pollution={dm.n_pollution}")
    print(f"[DATA] train_size={dm.n_train} | test_size={dm.n_test}")

    # -------------------------
    # Phase 1 — AE
    # -------------------------
    # Force simple mode for this pipeline (no RCAE)
    setattr(cfg, "ae_mode", "simple")
    ae = _load_or_pretrain_ae_mse(cfg, dm, device, ckpt_dir)

    # -------------------------
    # Phase 2 — encoder wrapper + rep_dim
    # -------------------------
    enc = EncoderWrapper(ae).to(device)
    rep_dim = infer_rep_dim(enc, train_loader_xy, device)
    print(f"[PHASE 2] rep_dim inferred: {rep_dim}")
    print(f"[PHASE 2] SVDD mode: {'JOINT' if svdd_joint else 'Z-ONLY (frozen encoder)'} | lr_encoder={svdd_lr_encoder}")

    # -------------------------
    # Phase 3 — DeepSVDD 
    # -------------------------
    print("[PHASE 3] Training DeepSVDD")

    svdd_epochs = int(_get(cfg, "svdd_epochs", 50))
    svdd_weight_decay = float(_get(cfg, "svdd_weight_decay", 1e-6))
    svdd_warmup_epochs = int(_get(cfg, "svdd_warmup_epochs", 10))
    svdd_clip_norm = float(_get(cfg, "svdd_clip_norm", 5.0))

    train_loader_normals = NormalOnlyLoader(train_loader_xy)

    enc_trained, c, R, svdd_hist = train_deepsvdd(
        encoder=enc,
        train_loader=train_loader_normals,  # <-- critical fix
        device=device,
        objective=svdd_objective,
        nu=svdd_nu,
        epochs=svdd_epochs,
        lr_encoder=svdd_lr_encoder,
        weight_decay=svdd_weight_decay,
        warmup_epochs=svdd_warmup_epochs,
        clip_norm=svdd_clip_norm,
        print_every=10,
    )

    svdd_ckpt_path = os.path.join(ckpt_dir, "svdd.pt")
    torch.save(
        {
            "encoder": enc_trained.state_dict(),
            "ae_state": ae.state_dict(),
            "c": c.detach().cpu(),
            "R": float(R),
            "history": svdd_hist,
            "cfg": asdict(cfg),
            "run_name": run_name,
            "rep_dim": rep_dim,
            "objective": svdd_objective,
            "nu": svdd_nu,
            "svdd_lr_encoder": svdd_lr_encoder,
            "normal_class": normal_class,
        },
        svdd_ckpt_path,
    )
    print(f"[PHASE 3] Saved SVDD checkpoint: {svdd_ckpt_path}")

    # -------------------------
    # Phase 4 — Evaluation
    # -------------------------
    print("[PHASE 4] Evaluation")

    auc, scores, labels = eval_deepsvdd(
        encoder=enc_trained,
        c=c,
        R=float(R),
        loader=test_loader,
        device=device,
    )

    score_name = "dist(x) - R^2"

    _ = plot_roc_curve(-scores, labels, out_dir=plots_dir, filename="roc_curve.png", anomaly_label=1)
    plot_score_histogram(-scores, labels, out_dir=plots_dir, filename="score_hist.png", bins=50, anomaly_label=1, score_name=score_name)
    plot_score_boxplot(-scores, labels, out_dir=plots_dir, filename="score_boxplot.png", anomaly_label=1, score_name=score_name)
   

    X_test, _ = collect_images_and_labels(test_loader)
    plot_extremes_in_class(
        images=X_test,
        anom_scores=-scores,
        labels=labels,
        out_dir=plots_dir,
        filename="extremes_normals.png",
        target_label=0,
        k=8,
        score_name=score_name,
    )

    print(f"[DONE] DeepSVDD CIFAR-10 | AUROC={auc:.4f} | R={float(R):.6f}")
    print(f"[DONE] Saved plots -> {plots_dir}")
    print(f"[DONE] Saved ckpts -> {ckpt_dir}")

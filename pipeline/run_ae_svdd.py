
import os
from dataclasses import asdict
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.Config import Config, set_seed
from utils.run_utils import _prepare_run_dirs, EncoderWrapper, get_device, collect_images_and_labels
from utils.mnist_datamodule import MNISTOneClassDataModule

from models.ae_factory import build_autoencoder

from model_utils.pretrain_autoencoder import pretrain_autoencoder_mse
from model_utils.train_svdd import train_deepsvdd, eval_deepsvdd

from utils.plot_utils import (
    plot_training_curves,
    plot_roc_curve,
    plot_score_histogram,
    plot_score_boxplot,
    plot_extremes_in_class,
)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _get(cfg: Config, name: str, default):
    """Safe getter for optional config fields."""
    return getattr(cfg, name, default)


@torch.no_grad()
def infer_rep_dim(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> int:
    """Infer latent dimensionality from one batch."""
    encoder.eval().to(device)
    x, _ = next(iter(loader))
    x = x.to(device)
    z = encoder(x)
    if z.dim() > 2:
        z = z.flatten(1)
    return int(z.shape[1])


# ------------------------------------------------------------
# Phase 1 — Autoencoder pretraining (MSE, normals only)
# ------------------------------------------------------------
def _load_or_pretrain_ae_mse(
    cfg: Config,
    dm: MNISTOneClassDataModule,
    device: torch.device,
    ckpt_dir: str,
) -> torch.nn.Module:

    ae = build_autoencoder(cfg).to(device)
    ae_ckpt_path = os.path.join(ckpt_dir, "ae_mse.pt")
    #ae_ckpt_path = "runs/autoencoder2_mnist_digit0_20260208_011801/20260208_011801/checkpoints/ae_mse.pt"

    # Reuse checkpoint if available
    if os.path.exists(ae_ckpt_path):
        print(f"[PHASE 1] Loading AE checkpoint: {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae.load_state_dict(ckpt["model"], strict=True)
        return ae

    print("[PHASE 1] Pretraining AE (MSE, normals only)")

    # Collect only normal samples
    train_imgs = dm.collect_normal_images(device=device)

    ae_train_loader = DataLoader(
        TensorDataset(train_imgs),
        batch_size=int(_get(cfg, "ae_batch_size", 128)),
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ae_out = pretrain_autoencoder_mse(
        ae=ae,
        train_loader=ae_train_loader,
        device=device,
        n_epochs=int(_get(cfg, "ae_epochs", 100)),
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
# Phase 2/3 — DeepSVDD training
# ------------------------------------------------------------
def _train_svdd(
    cfg: Config,
    ae: torch.nn.Module,
    train_loader_xy: DataLoader,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.Tensor, float, dict, int]:

    print("[PHASE 3] Training DeepSVDD")

    # Encoder wrapper outputs latent representation z
    encoder = EncoderWrapper(ae).to(device)
    rep_dim = infer_rep_dim(encoder, train_loader_xy, device)
    print(f"[PHASE 2] rep_dim inferred: {rep_dim}")

    # SVDD hyperparameters
    objective = _get(cfg, "svdd_objective", "one-class")
    nu = float(_get(cfg, "svdd_nu", 0.1))
    epochs = int(_get(cfg, "svdd_epochs", 100))
    lr_encoder = float(_get(cfg, "svdd_lr_encoder", 0.0))
    weight_decay = float(_get(cfg, "svdd_weight_decay", 0.0))
    warmup_epochs = int(_get(cfg, "svdd_warmup_epochs", 10))
    clip_norm = float(_get(cfg, "svdd_clip_norm", 5.0))

    joint = lr_encoder > 0.0
    print(f"[PHASE 2] SVDD mode: {'JOINT' if joint else 'Z-ONLY (frozen encoder)'} | lr_encoder={lr_encoder}")

    enc_trained, c, R, hist = train_deepsvdd(
        encoder=encoder,
        train_loader=train_loader_xy,
        device=device,
        objective=objective,
        nu=nu,
        epochs=epochs,
        lr_encoder=lr_encoder,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        clip_norm=clip_norm,
        print_every=10,
    )

    return enc_trained, c, float(R), hist, rep_dim


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_pipeline(cfg: Config) -> None:

    # System setup
    set_seed(cfg.seed)
    device = get_device()
    print(f"[SYS] Device: {device}")

    # Run directories
    run_name, run_dir, ckpt_dir, plots_dir = _prepare_run_dirs(cfg)
    print(f"[RUN] run_name: {run_name}")
    print(f"[RUN] run_dir : {run_dir}")

    # Dataset (binary labels: 0 normal, 1 anomaly)
    dm = MNISTOneClassDataModule(
        data_dir=cfg.data_dir,
        normal_digit=cfg.normal_digit,
        pollution_rate=cfg.pollution_rate,
        batch_size=int(_get(cfg, "svdd_batch_size", _get(cfg, "ocnn_batch_size", 128))),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    print(f"[DATA] normals(train_full)={dm.n_normals} | pollution={dm.n_pollution}")
    print(f"[DATA] train_size={dm.n_train} | test_size={dm.n_test}")
    print("NORMAL DIGIT =", cfg.normal_digit)

    # Phase 1 — AE
    ae = _load_or_pretrain_ae_mse(cfg, dm, device, ckpt_dir)

    # Phase 2/3 — SVDD
    enc_trained, c, R, svdd_hist, rep_dim = _train_svdd(cfg, ae, train_loader, device)

    # Save model
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
            "objective": _get(cfg, "svdd_objective", "one-class"),
            "nu": float(_get(cfg, "svdd_nu", 0.1)),
            "svdd_lr_encoder": float(_get(cfg, "svdd_lr_encoder", 0.0)),
        },
        svdd_ckpt_path,
    )
    print(f"[PHASE 3] Saved SVDD checkpoint: {svdd_ckpt_path}")

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    print("[PHASE 4] Evaluation")

    auc, scores, labels = eval_deepsvdd(
        encoder=enc_trained,
        c=c,
        R=float(R),
        loader=test_loader,
        device=device,
    )

    # For visualization: invert sign so higher = more anomalous
    score_name = "dist(x) - R^2"

    auc_plot = plot_roc_curve(-scores, labels, out_dir=plots_dir, filename="roc_curve.png", anomaly_label=1)

    plot_score_histogram(-scores, labels, out_dir=plots_dir, filename="score_hist.png", bins=50, anomaly_label=1, score_name=score_name)
    plot_score_boxplot(-scores, labels, out_dir=plots_dir, filename="score_boxplot.png", anomaly_label=1, score_name=score_name)
    plot_training_curves(svdd_hist, out_dir=plots_dir, prefix="svdd")

    X_test, _ = collect_images_and_labels(test_loader, device=device)
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

    print(f"[DONE] DeepSVDD | AUROC={auc_plot:.4f} | R={float(R):.6f}")
    print(f"[DONE] Saved plots -> {plots_dir}")
    print(f"[DONE] Saved ckpts -> {ckpt_dir}")

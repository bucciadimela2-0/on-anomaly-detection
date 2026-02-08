# pipeline/run_ae_ocnn.py
import os
from dataclasses import asdict
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.Config import Config, set_seed
from utils.run_utils import _prepare_run_dirs, EncoderWrapper, get_device, collect_images_and_labels
from utils.mnist_datamodule import MNISTOneClassDataModule

from models.ae_factory import build_autoencoder
from models.ocnn import OCNN

from model_utils.pretrain_autoencoder import pretrain_autoencoder_mse
from model_utils.train_ocnn import train_ocnn, evaluate_ocnn

from utils.plot_utils import (
    plot_training_curves,
    plot_roc_curve,
    plot_extremes_in_class,
)


def _load_or_pretrain_ae_mse(
    cfg: Config,
    dm: MNISTOneClassDataModule,
    device: torch.device,
    ckpt_dir: str,
) -> torch.nn.Module:
    """
    PHASE 1: Pretrain AE with MSE on NORMAL samples only.
    Returns: trained AE (nn.Module).
    """
    ae = build_autoencoder(cfg).to(device)
    ae_ckpt_path = os.path.join(ckpt_dir, "ae_mse.pt")
    if os.path.exists(ae_ckpt_path):
        print(f"[PHASE 1] Loading AE checkpoint: {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae.load_state_dict(ckpt["model"], strict=True)
        return ae

    print("[PHASE 1] Pretraining AE (MSE, normals only)")
    train_imgs = dm.collect_normal_images(device=device)
    # DEBUG: statistics of normal images
    print(
        "[DEBUG][AE] normal imgs stats:",
        "min=", train_imgs.min().item(),
        "max=", train_imgs.max().item(),
        "mean=", train_imgs.mean().item(),
        "std=", train_imgs.std().item(),
)

    ae_train_loader = DataLoader(
        TensorDataset(train_imgs),
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ae_out = pretrain_autoencoder_mse(
        ae=ae,
        train_loader=ae_train_loader,
        device=device,
        n_epochs=cfg.ae_epochs,
        lr=cfg.ae_lr,
        weight_decay=cfg.ae_weight_decay,
        print_every=1,
        grad_clip=getattr(cfg, "ae_grad_clip", None),
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


def _train_ocnn_joint(
    cfg: Config,
    ae: torch.nn.Module,
    train_loader_xy: DataLoader,
    device: torch.device,
) -> Tuple[OCNN, torch.nn.Module, float, dict]:
    """
    PHASE 3: OCNN training in JOINT mode.
    train_loader_xy yields (x, y).
    encoder produces z online.
    """
    print("[PHASE 3] Training OCNN (JOINT: x -> encoder -> z -> OCNN)")

    # EncoderWrapper return z (latent), NOT reconstruction.
    encoder = EncoderWrapper(ae).to(device)

    # infer rep_dim from cfg (or from encoder output once)
    rep_dim = int(cfg.rep_dim)

    ocnn = OCNN(indim=rep_dim, outdim=cfg.ocnn_hidden_dim, activation=cfg.activation)

    ocnn, encoder, r_star, hist = train_ocnn(
        ocnn=ocnn,
        train_loader=train_loader_xy,   # (x, y)
        device=device,
        nu=cfg.nu,
        epochs=cfg.ocnn_epochs,
        lr_init=cfg.ocnn_lr_init,
        lr_finetune=cfg.ocnn_lr_finetune,
        finetune_start_epoch=cfg.finetune_start_epoch if hasattr(cfg, "finetune_start_epoch") else 10**9,
        r_init=None,
        print_every=10,
        encoder=encoder,
        joint=True,
        lr_encoder=cfg.lr_encoder_joint if hasattr(cfg, "lr_encoder_joint") else 0.0,
        clip_norm=getattr(cfg, "clip_norm", 0.5),
    )
    return ocnn, encoder, float(r_star), hist


def run_pipeline(cfg: Config) -> None:
    # -------------------------
    # System / seed / dirs
    # -------------------------
    set_seed(cfg.seed)
    device = get_device()
    print(f"[SYS] Device: {device}")

    run_name, run_dir, ckpt_dir, plots_dir = _prepare_run_dirs(cfg)
    print(f"[RUN] run_name: {run_name}")
    print(f"[RUN] run_dir : {run_dir}")

    # -------------------------
    # Data (MNIST one-class)
    # -------------------------
    dm = MNISTOneClassDataModule(
        data_dir=cfg.data_dir,
        normal_digit=cfg.normal_digit,
        pollution_rate=cfg.pollution_rate,
        batch_size=cfg.ocnn_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    dm.setup()
    train_loader = dm.train_dataloader()  # yields (x, y)
    test_loader = dm.test_dataloader()    # yields (x, y)

    print(f"[DATA] normals(train_full)={dm.n_normals} | pollution={dm.n_pollution}")
    print(f"[DATA] train_size={dm.n_train} | test_size={dm.n_test}")

    print("NORMAL DIGIT =", cfg.normal_digit)

    # -------------------------
    # PHASE 1: AE pretraining
    # -------------------------
    ae = _load_or_pretrain_ae_mse(cfg, dm, device, ckpt_dir)

    # -------------------------
    # PHASE 3: train OCNN JOINT
    # -------------------------
    ocnn, encoder_for_ocnn, r_star, ocnn_hist = _train_ocnn_joint(cfg, ae, train_loader, device)

    ocnn_ckpt_path = os.path.join(ckpt_dir, "ocnn.pt")

    torch.save(
        {
            "model": ocnn.state_dict(),
            "ae_state": ae.state_dict(),
            "r": float(r_star),
            "history": ocnn_hist,
            "cfg": asdict(cfg),
            "run_name": run_name,
            "joint": True,
            "lr_encoder_joint": float(getattr(cfg, "lr_encoder_joint", 0.0)),
        },
        ocnn_ckpt_path,
    )
    print(f"[PHASE 3] Saved OCNN checkpoint: {ocnn_ckpt_path}")

    # -------------------------
    # PHASE 4: evaluation JOINT
    # -------------------------
    print("[PHASE 4] Evaluation (JOINT)")
    anom_scores, labels = evaluate_ocnn(
        encoder=encoder_for_ocnn,
        ocnn=ocnn,
        loader=test_loader,   # (x, y)
        r=float(r_star),
        device=device,
        joint=True,
    )

    auc = plot_roc_curve(anom_scores, labels, out_dir=plots_dir, filename="roc_curve.png", anomaly_label=1)
    plot_training_curves(ocnn_hist, out_dir=plots_dir, prefix="ocnn")

    X_test, _ = collect_images_and_labels(test_loader, device=device)
    plot_extremes_in_class(
        images=X_test,
        anom_scores=anom_scores,
        labels=labels,
        out_dir=plots_dir,
        filename="extremes_normals.png",
        target_label=0,
        k=8,
        score_name="r - score",
    )

    print(f"[DONE] r*={r_star:.6f} | AUROC={auc:.4f}")
    print(f"[DONE] Saved plots -> {plots_dir}")
    print(f"[DONE] Saved ckpts -> {ckpt_dir}")

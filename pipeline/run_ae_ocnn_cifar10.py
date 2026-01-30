# pipeline/run_ae_ocnn_cifar.py
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.const import Const
from utils.Config import Config, set_seed
from utils.run_utils import make_run_name, make_run_dir
from utils.wandb_utils import (
    wandb_init_if_enabled,
    wandb_log,
    wandb_finish,
    wandb_log_artifact_dir,
)

from utils.data_cifar import CIFAR10OneClassDataModule, encode_dataset

from models.ae_factory import build_autoencoder
from models.ocnn import OCNN

from model_utils.train_autoencoder import train_ae_mse, pretrain_rcae, RCAEConfig
from model_utils.train_ocnn import train_ocnn, evaluate_ocnn

from utils.plot_utils import (
    plot_training_curves,
    plot_score_histogram,
    plot_roc_curve,
    plot_pr_curve,
    plot_score_boxplot,
    plot_extremes_in_class,
)


# ============================================================
# Helpers
# ============================================================

class EncoderWrapper(torch.nn.Module):
    # Uniform interface: encode(x) -> z [B, rep_dim]
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
def collect_images_and_labels(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(cfg: Config) -> None:
    # -------------------------
    # System / seed
    # -------------------------
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYS] Device: {device}")

    # -------------------------
    # Run dirs
    # -------------------------
    mode_tag = f"cifar10_{cfg.ae_mode}_{'joint' if cfg.ocnn_joint else 'zonly'}"
    run_name = make_run_name(model=mode_tag, digit=getattr(cfg, "normal_class", 0), nu=cfg.nu)
    run_dir = make_run_dir(cfg.base_runs_dir, run_name)

    ckpt_dir = os.path.join(run_dir, cfg.ckpt_subdir)
    plots_dir = os.path.join(run_dir, cfg.plots_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[RUN] run_name: {run_name}")
    print(f"[RUN] run_dir : {run_dir}")

    # -------------------------
    # W&B
    # -------------------------
    wb = wandb_init_if_enabled(cfg.wandb, cfg.wandb_project, run_name, asdict(cfg))

    # -------------------------
    # Data
    # -------------------------
    dm = CIFAR10OneClassDataModule(
        data_dir=cfg.data_dir,
        normal_class=getattr(cfg, "normal_class", 0),
        pollution_rate=cfg.pollution_rate,
        batch_size=cfg.ocnn_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        download=True,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    print(f"[DATA] normals(train_full)={dm.n_normals} | pollution={dm.n_pollution}")
    print(f"[DATA] train_size={dm.n_train} | test_size={dm.n_test}")

    # -------------------------
    # Build AE
    # -------------------------
    ae = build_autoencoder(cfg).to(device)
    #ae_ckpt_path = os.path.join(ckpt_dir, f"ae_{cfg.ae_mode}.pt")
    #ae_ckpt_path = "runs/cifar10_rcae_joint_digit0_nu0.1_20260130_002309/20260130_002309/checkpoints/ae_rcae.pt"
    ae_ckpt_path = "runs/cifar10_rcae_joint_digit1_nu0.1_20260130_082158/20260130_082158/checkpoints/ae_rcae.pt"
    # ============================================================
    # PHASE 1: AE pretraining / loading
    # ============================================================
    if os.path.exists(ae_ckpt_path):
        print(f"[PHASE 1] Loading AE checkpoint: {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae.load_state_dict(ckpt["model"], strict=True)
    else:
        print(f"[PHASE 1] Pretraining AE mode={cfg.ae_mode} (normals only)")
        train_imgs = dm.collect_normal_images(device=device)  # [N,3,32,32]

        if cfg.ae_mode == "rcae":
            rcae_cfg = RCAEConfig(
                lamda_set=cfg.lamda_set,
                n_outer_iters=cfg.rcae_outer_iters,
                n_epochs_ae=cfg.ae_epochs,
                lr_ae=cfg.ae_lr,
                batch_size=cfg.ae_batch_size,
                mue=cfg.mue,
                weight_decay=cfg.ae_weight_decay,
                print_every=1,
                reconstruct_clean=True,
                freeze_encoder=False,
            )
            ae, _enc_maybe, _info = pretrain_rcae(
                ae=ae,
                train_images=train_imgs,
                device=device,
                cfg=rcae_cfg,
            )

        elif cfg.ae_mode == "simple":
            ae_train_ds = TensorDataset(train_imgs, torch.zeros(len(train_imgs)))
            ae_train_loader = DataLoader(
                ae_train_ds,
                batch_size=cfg.ae_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
            ae = train_ae_mse(
                net=ae,
                train_loader=ae_train_loader,
                device=device,
                n_epochs=cfg.ae_epochs,
                lr=cfg.ae_lr,
                weight_decay=cfg.ae_weight_decay,
                print_every=1,
            )
        else:
            raise ValueError(f"Unknown ae_mode='{cfg.ae_mode}' (expected: rcae|simple)")

        torch.save({"model": ae.state_dict(), "cfg": asdict(cfg), "run_name": run_name}, ae_ckpt_path)
        print(f"[PHASE 1] Saved AE checkpoint: {ae_ckpt_path}")

    # ============================================================
    # PHASE 2: z-only features OR joint on-the-fly
    # ============================================================
    if cfg.ocnn_joint:
        print("[PHASE 2] Joint mode: encoder used on-the-fly")
        enc_for_ocnn = EncoderWrapper(ae).to(device)
        rep_dim = infer_rep_dim(enc_for_ocnn, train_loader, device)
        train_feat_loader = None
        test_feat_loader = None
    else:
        print("[PHASE 2] z-only mode: encoding datasets -> latent features")
        ae_enc = EncoderWrapper(ae).to(device)

        train_feat_ds = encode_dataset(ae_enc, train_loader, device=device)
        test_feat_ds = encode_dataset(ae_enc, test_loader, device=device)

        train_feat_loader = DataLoader(
            train_feat_ds,
            batch_size=cfg.ocnn_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        test_feat_loader = DataLoader(
            test_feat_ds,
            batch_size=cfg.ocnn_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        rep_dim = int(train_feat_ds.tensors[0].shape[1])
        enc_for_ocnn = None

    print(f"[PHASE 2] rep_dim inferred: {rep_dim}")

    # ============================================================
    # PHASE 3: Train OCNN
    # ============================================================
    print("[PHASE 3] Training OCNN")
    ocnn = OCNN(indim=rep_dim, outdim=Const.OCNN_HIDDEN_DIM, activation=cfg.activation)

    if cfg.ocnn_joint:
        print(f"[PHASE 3] Joint encoder LR: {cfg.lr_encoder_joint:.2e} (0.0 => frozen)")
        ocnn, _enc_used, r_star, ocnn_hist = train_ocnn(
            ocnn=ocnn,
            train_loader=train_loader,  # (x,y)
            device=device,
            nu=cfg.nu,
            epochs=cfg.ocnn_epochs,
            lr_init=cfg.ocnn_lr_init,
            lr_finetune=cfg.ocnn_lr_finetune,
            finetune_start_epoch=cfg.finetune_start_epoch,
            r_init=None,
            print_every=10,
            encoder=enc_for_ocnn,
            joint=True,
            lr_encoder=cfg.lr_encoder_joint,
            clip_norm=getattr(cfg, "svdd_clip_norm", 0.5),
        )
    else:
        ocnn, _enc_unused, r_star, ocnn_hist = train_ocnn(
            ocnn=ocnn,
            train_loader=train_feat_loader,  # (z,y)
            device=device,
            nu=cfg.nu,
            epochs=cfg.ocnn_epochs,
            lr_init=cfg.ocnn_lr_init,
            lr_finetune=cfg.ocnn_lr_finetune,
            finetune_start_epoch=cfg.finetune_start_epoch,
            r_init=None,
            print_every=10,
            encoder=None,
            joint=False,
            clip_norm=getattr(cfg, "svdd_clip_norm", 0.5),
        )

    ocnn_ckpt_path = os.path.join(ckpt_dir, "ocnn.pt")
    torch.save(
        {
            "model": ocnn.state_dict(),
            "ae_state": ae.state_dict(),
            "r": float(r_star),
            "history": ocnn_hist,
            "cfg": asdict(cfg),
            "run_name": run_name,
            "ocnn_joint": bool(cfg.ocnn_joint),
            "lr_encoder_joint": float(cfg.lr_encoder_joint) if cfg.ocnn_joint else 0.0,
        },
        ocnn_ckpt_path,
    )
    print(f"[PHASE 3] Saved OCNN checkpoint: {ocnn_ckpt_path}")
    wandb_log(wb, {"final_r": float(r_star)})

    # ============================================================
    # PHASE 4: Evaluation + plots
    # ============================================================
    print("[PHASE 4] Evaluation")

    if cfg.ocnn_joint:
        anom_scores, labels = evaluate_ocnn(
            encoder=enc_for_ocnn,
            ocnn=ocnn,
            loader=test_loader,  # (x,y)
            r=float(r_star),
            device=device,
            joint=True,
        )
    else:
        anom_scores, labels = evaluate_ocnn(
            encoder=None,
            ocnn=ocnn,
            loader=test_feat_loader,  # (z,y)
            r=float(r_star),
            device=device,
            joint=False,
        )

    score_name = "r - f(x)"

    auc = plot_roc_curve(
        anom_scores,
        labels,
        out_dir=plots_dir,
        filename="roc_curve.png",
        anomaly_label=1,
    )
    ap = plot_pr_curve(
        anom_scores,
        labels,
        out_dir=plots_dir,
        filename="pr_curve.png",
        anomaly_label=1,
    )

    plot_score_histogram(
        anom_scores,
        labels,
        out_dir=plots_dir,
        filename="score_hist.png",
        bins=50,
        anomaly_label=1,
        score_name=score_name,
    )
    plot_score_boxplot(
        anom_scores,
        labels,
        out_dir=plots_dir,
        filename="score_boxplot.png",
        anomaly_label=1,
        score_name=score_name,
    )
    plot_training_curves(ocnn_hist, out_dir=plots_dir, prefix="ocnn")

    # Extremes among normals
    X_test, y_test = collect_images_and_labels(test_loader)
    plot_extremes_in_class(
        images=X_test,
        anom_scores=anom_scores,
        labels=labels,
        out_dir=plots_dir,
        filename="extremes_normals.png",
        target_label=0,
        k=8,
        score_name=score_name,
    )



    print(f"[DONE] r*={float(r_star):.6f} | AUROC={auc:.4f} | AP={ap:.4f}")
    print(f"[DONE] Saved plots -> {plots_dir}")
    print(f"[DONE] Saved ckpts -> {ckpt_dir}")

    wandb_log(wb, {"auroc": float(auc), "ap": float(ap), "r_star": float(r_star)})
    wandb_log_artifact_dir(wb, run_dir, artifact_name=run_name)
    wandb_finish(wb)

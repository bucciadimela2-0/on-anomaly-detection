# ============================================================
# pipeline/run_ae_svdd.py
# (STILE IDENTICO a run_ae_ocnn.py: run dirs, W&B, 4 fasi, wrapper encoder)
# ============================================================
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
from utils.data import MNISTOneClassDataModule, encode_dataset

from models.ae_factory import build_autoencoder
from models.autoencoder3 import Autoencoder

# AE training (GIÀ ESISTENTE)
from model_utils.train_autoencoder import train_ae_mse, pretrain_rcae, RCAEConfig

# DeepSVDD (NUOVO trainer)
from model_utils.train_svdd import train_deepsvdd, eval_deepsvdd

from utils.plot_utils import (
    plot_training_curves,
    plot_score_histogram,
    plot_roc_curve,
    plot_pr_curve,
    plot_score_boxplot,
    plot_extremes_in_class,
    
)


# -------------------------
# Helpers
# -------------------------
class EncoderWrapper(torch.nn.Module):
    """
    Wrapper uniforme che espone encode(x)->z per qualunque AE.
    - Se AE ha encode_flat: usa quello.
    - Altrimenti se ha encode: usa quello.
    - Altrimenti forward.
    """
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
    encoder.eval().to(device)
    x, _ = next(iter(loader))
    x = x.to(device)
    z = encoder.encode(x) if hasattr(encoder, "encode") else encoder(x)
    if z.dim() > 2:
        z = z.flatten(1)
    return int(z.shape[1])

@torch.no_grad()
def collect_images_and_labels(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _get(cfg: Config, name: str, default):
    # helper per campi nuovi non ancora in Config
    return getattr(cfg, name, default)


def run_pipeline(cfg: Config) -> None:
    # ---------------- System / seed ----------------
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYS] Device: {device}")

    # ---------------- Run dirs ----------------
    # SVDD "joint" qui significa: encoder aggiornato durante SVDD (lr_encoder>0)
    svdd_lr_encoder = float(_get(cfg, "svdd_lr_encoder", 0.0))
    svdd_joint = svdd_lr_encoder > 0.0

    svdd_objective = _get(cfg, "svdd_objective", "one-class")
    svdd_nu = float(_get(cfg, "svdd_nu", 0.1))

    mode_tag = f"{cfg.ae_mode}_svdd_{svdd_objective}_{'joint' if svdd_joint else 'zonly'}"
    run_name = make_run_name(model=mode_tag, digit=cfg.normal_digit, nu=svdd_nu)
    run_dir = make_run_dir(cfg.base_runs_dir, run_name)

    ckpt_dir = os.path.join(run_dir, cfg.ckpt_subdir)
    plots_dir = os.path.join(run_dir, cfg.plots_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[RUN] run_name: {run_name}")
    print(f"[RUN] run_dir : {run_dir}")

    # ---------------- W&B ----------------
    wb = wandb_init_if_enabled(cfg.wandb, cfg.wandb_project, run_name, asdict(cfg))

    # ---------------- DataModule ----------------
    dm = MNISTOneClassDataModule(
        data_dir=cfg.data_dir,
        normal_digit=cfg.normal_digit,
        pollution_rate=cfg.pollution_rate,
        batch_size=_get(cfg, "svdd_batch_size", cfg.ocnn_batch_size),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    dm.setup()
    train_loader = dm.train_dataloader()  # yields (x,y)
    test_loader = dm.test_dataloader()    # yields (x,y)

    print(f"[DATA] normals(train_full)={dm.n_normals} | pollution={dm.n_pollution}")
    print(f"[DATA] train_size={dm.n_train} | test_size={dm.n_test}")

    # ---------------- Build AE ----------------
    if cfg.ae_arch == "autoencoder3":
        ae = Autoencoder().to(device)
    else:
        ae = build_autoencoder(cfg).to(device)

    # =========================================================
    # PHASE 1: AE pretraining / loading (RCAE or simple)
    # =========================================================
    #ae_ckpt_path = os.path.join(ckpt_dir, f"ae_{cfg.ae_mode}.pt")
    ae_ckpt_path = "runs/rcae_svdd_one-class_joint_digit1_nu0.1_20260130_104020/20260130_104020/checkpoints/ae_rcae.pt"
    if os.path.exists(ae_ckpt_path):
        print(f"[PHASE 1] Loading AE checkpoint: {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae.load_state_dict(ckpt["model"], strict=True)

    else:
        print(f"[PHASE 1] Pretraining AE mode={cfg.ae_mode} (normals only)")
        train_imgs = dm.collect_normal_images(device=device)  # [N_norm,1,28,28]

        if cfg.ae_mode == "rcae":
            rcae_cfg = RCAEConfig(
                lamda_set=(float(_get(cfg, "rcae_lambda", 0.1)),),
                n_outer_iters=int(_get(cfg, "rcae_outer_iters", 1)),
                n_epochs_ae=int(_get(cfg, "ae_epochs", 150)),
                lr_ae=float(_get(cfg, "ae_lr", 1e-3)),
                batch_size=int(_get(cfg, "ae_batch_size", 128)),
                mue=float(_get(cfg, "rcae_mue", 0.0)),
                weight_decay=0.0,
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
                batch_size=int(_get(cfg, "ae_batch_size", 128)),
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
            ae = train_ae_mse(
                net=ae,
                train_loader=ae_train_loader,
                device=device,
                n_epochs=int(_get(cfg, "ae_epochs", 150)),
                lr=float(_get(cfg, "ae_lr", 1e-3)),
                weight_decay=0.0,
                print_every=1,
            )

        else:
            raise ValueError(f"Unknown ae_mode='{cfg.ae_mode}' (expected: rcae|simple)")

        torch.save(
            {"model": ae.state_dict(), "cfg": asdict(cfg), "run_name": run_name},
            ae_ckpt_path,
        )
        print(f"[PHASE 1] Saved AE checkpoint: {ae_ckpt_path}")

    # =========================================================
    # PHASE 2: encoder on-the-fly (joint) OR freeze (z-only)
    # =========================================================
    # Per DeepSVDD, anche in "z-only" non serve creare un dataset di z:
    # basta congelare encoder (lr_encoder=0) e calcolare z al volo.
    # Però teniamo la fase 2 simile alla tua per logging/rep_dim.
    enc = EncoderWrapper(ae).to(device)
    rep_dim = infer_rep_dim(enc, train_loader, device)
    print(f"[PHASE 2] rep_dim inferred: {rep_dim}")

    if svdd_joint:
        print("[PHASE 2] SVDD joint: encoder updated during SVDD")
    else:
        print("[PHASE 2] SVDD z-only: encoder frozen during SVDD")

    # =========================================================
    # PHASE 3: Train DeepSVDD
    # =========================================================
    print("[PHASE 3] Training DeepSVDD")

    svdd_epochs = int(_get(cfg, "svdd_epochs", 50))
    svdd_lr = float(_get(cfg, "svdd_lr", 1e-4))
    svdd_weight_decay = float(_get(cfg, "svdd_weight_decay", 0.0))
    svdd_warmup_epochs = int(_get(cfg, "svdd_warmup_epochs", 10))
    svdd_clip_norm = float(_get(cfg, "svdd_clip_norm", 5.0))

    # NOTE: train_deepsvdd allena SOLO encoder (se lr_encoder>0).
    # Se lr_encoder==0, congela e fa forward only.
    enc_trained, c, R, svdd_hist = train_deepsvdd(
        encoder=enc,                 # EncoderWrapper è un nn.Module con parametri del AE dentro
        train_loader=train_loader,   # (x,y)
        device=device,
        objective=svdd_objective,
        nu=svdd_nu,
        epochs=svdd_epochs,
        lr=svdd_lr,                  # (tenuto per compatibilità firma; usato solo se vuoi estendere)
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
            "c": c.detach().cpu(),
            "R": float(R),
            "history": svdd_hist,
            "cfg": asdict(cfg),
            "run_name": run_name,
            "rep_dim": rep_dim,
            "objective": svdd_objective,
            "nu": svdd_nu,
            "svdd_lr_encoder": svdd_lr_encoder,
        },
        svdd_ckpt_path,
    )
    print(f"[PHASE 3] Saved SVDD checkpoint: {svdd_ckpt_path}")
    wandb_log(wb, {"R": float(R)})

    # =========================================================
    # PHASE 4: Evaluation
    # =========================================================
    print("[PHASE 4] Evaluation")

    auc, scores, labels = eval_deepsvdd(
        encoder=enc_trained,
        c=c,
        R=float(R),
        loader=test_loader,
        device=device,
    )

    # Convention: anomaly score = dist - R^2
    score_name = "dist(x) - R^2"

    # plots come OCNN
    auc_plot = plot_roc_curve(
        scores,
        labels,
        out_dir=plots_dir,
        filename="roc_curve.png",
        anomaly_label=1,
    )
    ap = plot_pr_curve(
        scores,
        labels,
        out_dir=plots_dir,
        filename="pr_curve.png",
        anomaly_label=1,
    )

    plot_score_histogram(
        scores,
        labels,
        out_dir=plots_dir,
        filename="score_hist.png",
        bins=50,
        anomaly_label=1,
        score_name=score_name,
    )
    plot_score_boxplot(
        scores,
        labels,
        out_dir=plots_dir,
        filename="score_boxplot.png",
        anomaly_label=1,
        score_name=score_name,
    )
    plot_training_curves(svdd_hist, out_dir=plots_dir, prefix="svdd")

    X_test, y_test = collect_images_and_labels(test_loader)
    plot_extremes_in_class(
        images=X_test,
        anom_scores=scores,
        labels=labels,
        out_dir=plots_dir,
        filename="extremes_normals.png",
        target_label=0,
        k=8,
        score_name=score_name,
    )

    print(f"[DONE] DeepSVDD | AUROC={auc:.4f} | AP={ap:.4f} | R={float(R):.6f}")
    print(f"[DONE] Saved plots -> {plots_dir}")
    print(f"[DONE] Saved ckpts -> {ckpt_dir}")

    wandb_log(
        wb,
        {
            "auroc": float(auc),
            "ap": float(ap),
            "R_final": float(R),
            "objective": svdd_objective,
            "nu": float(svdd_nu),
            "svdd_joint": bool(svdd_joint),
            "svdd_lr_encoder": float(svdd_lr_encoder),
        },
    )
    wandb_log_artifact_dir(wb, run_dir, artifact_name=run_name)
    wandb_finish(wb)

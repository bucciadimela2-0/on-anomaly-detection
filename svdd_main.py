# ============================================================
# svdd_main.py
# (stesso stile di ocnn_main: parse args -> Config -> run_pipeline)
# + CIFAR-10 support: --dataset {mnist,cifar10}
# ============================================================
import argparse

from utils.const import Const
from utils.Config import Config
from pipeline.run_ae_svdd import run_pipeline as run_pipeline_mnist
from pipeline.run_ae_svdd_cifar10 import run_pipeline as run_pipeline_cifar10


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AE (simple/RCAE) + DeepSVDD pipeline (MNIST/CIFAR-10)")

    # Dataset selector
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Which dataset pipeline to run.",
    )

    # One-class (MNIST digit / CIFAR-10 class)
    p.add_argument("--normal-digit", type=int, default=Const.NORMAL_DIGIT,
                   help="MNIST: digit in [0..9]. CIFAR-10: class id in [0..9] (e.g., airplane=0).")
    p.add_argument("--pollution-rate", type=float, default=Const.POLLUTION_RATE)
    p.add_argument("--data-dir", type=str, default=Const.DATA_DIR)
    p.add_argument("--num-workers", type=int, default=0)

    # AE
    p.add_argument(
        "--ae-arch",
        type=str,
        default=Const.ae_arch,
        # aggiunte arch per CIFAR
        choices=["autoencoder1", "autoencoder2", "autoencoder3", "autoencoder_cifar", "autoencoder2_cifar"],
    )
    p.add_argument("--ae-mode", type=str, default="rcae", choices=["rcae", "simple"])
    p.add_argument("--rep-dim", type=int, default=Const.LATENT_DIM)
    p.add_argument("--ae-epochs", type=int, default=Const.AE_EPOCHS)
    p.add_argument("--ae-lr", type=float, default=Const.AE_LR)
    p.add_argument("--mue", type=float, default=Const.MUE)  # (usato da RCAEConfig via cfg)
    p.add_argument("--ae-batch-size", type=int, default=Const.AE_BATCH_SIZE)

    # RCAE extras (se non li hai in Config, li leggiamo via getattr nel run)
    p.add_argument("--rcae-lambda", type=float, default=0.1)
    p.add_argument("--rcae-outer-iters", type=int, default=1)
    p.add_argument("--rcae-mue", type=float, default=0.0)

    # DeepSVDD
    p.add_argument(
        "--svdd-objective",
        type=str,
        default="one-class",
        choices=["one-class", "soft-boundary"],
    )
    p.add_argument("--svdd-nu", type=float, default=0.1)
    p.add_argument("--svdd-epochs", type=int, default=50)
    p.add_argument("--svdd-batch-size", type=int, default=Const.OCNN_BATCH_SIZE)

    # encoder update during SVDD (joint vs z-only)
    p.add_argument(
        "--svdd-lr-encoder",
        type=float,
        default=0.0,
        help="Se >0 abilita joint (aggiorna encoder durante SVDD). Se 0 => encoder frozen.",
    )

    # altri iperparametri SVDD
    p.add_argument("--svdd-lr", type=float, default=1e-4)
    p.add_argument("--svdd-weight-decay", type=float, default=0.0)
    p.add_argument("--svdd-warmup-epochs", type=int, default=10)
    p.add_argument("--svdd-clip-norm", type=float, default=5.0)

    # Runs
    p.add_argument("--seed", type=int, default=Const.SEED)
    p.add_argument("--runs-dir", type=str, default=Const.BASE_RUNS_DIR)

    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=Const.WANDB_PROJECT)

    return p


def main():
    p = build_parser()
    args = p.parse_args()

    # NB: Config è dataclass "chiusa": per i campi SVDD extra usiamo setattr
    cfg = Config(
        normal_digit=args.normal_digit,
        pollution_rate=args.pollution_rate,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        ae_arch=args.ae_arch,
        ae_mode=args.ae_mode,
        rep_dim=args.rep_dim,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        mue=args.mue,
        ae_batch_size=args.ae_batch_size,
        seed=args.seed,
        base_runs_dir=args.runs_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # ---- attach SVDD/RCAE extra params dynamically (run_* usa getattr) ----
    setattr(cfg, "rcae_lambda", args.rcae_lambda)
    setattr(cfg, "rcae_outer_iters", args.rcae_outer_iters)
    setattr(cfg, "rcae_mue", args.rcae_mue)

    setattr(cfg, "svdd_objective", args.svdd_objective)
    setattr(cfg, "svdd_nu", args.svdd_nu)
    setattr(cfg, "svdd_epochs", args.svdd_epochs)
    setattr(cfg, "svdd_batch_size", args.svdd_batch_size)
    setattr(cfg, "svdd_lr_encoder", args.svdd_lr_encoder)
    setattr(cfg, "svdd_lr", args.svdd_lr)
    setattr(cfg, "svdd_weight_decay", args.svdd_weight_decay)
    setattr(cfg, "svdd_warmup_epochs", args.svdd_warmup_epochs)
    setattr(cfg, "svdd_clip_norm", args.svdd_clip_norm)

    # ---- dispatch pipeline ----
    if args.dataset == "mnist":
        run_pipeline_mnist(cfg)
    elif args.dataset == "cifar10":
        run_pipeline_cifar10(cfg)
    else:
        raise ValueError(f"Unknown dataset='{args.dataset}'")


if __name__ == "__main__":
    main()

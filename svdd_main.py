# svdd_main.py
from utils.const import Const
from utils.Config import Config

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("AE (MSE) + DeepSVDD pipeline")

    # ----------------------------
    # Dataset
    # ----------------------------
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to run",
    )
    p.add_argument("--data-dir", type=str, default=Const.DATA_DIR)
    p.add_argument("--num-workers", type=int, default=0)

    # One-class setup
    p.add_argument("--normal-digit", type=int, default=Const.NORMAL_DIGIT)  # MNIST digit OR CIFAR class id
    p.add_argument("--pollution-rate", type=float, default=Const.POLLUTION_RATE)
    p.add_argument("--cifar-transform",
    type=str,
    default="repo",
    choices=["repo", "gcn_l1_minmax", "tensor"],
    help="CIFAR preprocessing mode"
)


    # ----------------------------
    # Autoencoder (MSE only)
    # ----------------------------
    p.add_argument(
        "--ae-arch",
        type=str,
        default=Const.ae_arch,
        choices=["autoencoder1", "autoencoder2", "autoencoder3", "autoencoder_cifar", "autoencoder2_cifar"],
    )
    p.add_argument("--rep-dim", type=int, default=Const.LATENT_DIM)
    p.add_argument("--ae-epochs", type=int, default=Const.AE_EPOCHS)
    p.add_argument("--ae-lr", type=float, default=Const.AE_LR)
    p.add_argument("--ae-batch-size", type=int, default=Const.AE_BATCH_SIZE)
    p.add_argument("--ae-weight-decay", type=float, default=0.0)  # may not exist in Config

    # ----------------------------
    # DeepSVDD
    # ----------------------------
    p.add_argument("--svdd-objective", type=str, default="one-class", choices=["one-class", "soft-boundary"])
    p.add_argument("--svdd-nu", type=float, default=0.1)
    p.add_argument("--svdd-epochs", type=int, default=100)
    p.add_argument("--svdd-batch-size", type=int, default=Const.OCNN_BATCH_SIZE)

    p.add_argument("--svdd-lr-encoder", type=float, default=0.0)
    p.add_argument("--svdd-weight-decay", type=float, default=0.0)
    p.add_argument("--svdd-warmup-epochs", type=int, default=10)
    p.add_argument("--svdd-clip-norm", type=float, default=5.0)

    
    # ----------------------------
    # Run
    # ----------------------------
    p.add_argument("--seed", type=int, default=Const.SEED)
    p.add_argument("--runs-dir", type=str, default=Const.BASE_RUNS_DIR)

    args = p.parse_args()

    # ----------------------------
    # Config (ONLY fields that exist in your Config dataclass)
    # ----------------------------
    cfg = Config(
        normal_digit=args.normal_digit,
        pollution_rate=args.pollution_rate,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        ae_arch=args.ae_arch,
        rep_dim=args.rep_dim,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        ae_batch_size=args.ae_batch_size,
        seed=args.seed,
        base_runs_dir=args.runs_dir,
    )

    # ----------------------------
    # Attach extras dynamically
    # ----------------------------
    setattr(cfg, "dataset", args.dataset)               # if your pipelines want it
    setattr(cfg, "ae_mode", "simple")                   # force: no RCAE
    setattr(cfg, "ae_weight_decay", args.ae_weight_decay)

    setattr(cfg, "svdd_objective", args.svdd_objective)
    setattr(cfg, "svdd_nu", args.svdd_nu)
    setattr(cfg, "svdd_epochs", args.svdd_epochs)
    setattr(cfg, "svdd_batch_size", args.svdd_batch_size)
    setattr(cfg, "svdd_lr_encoder", args.svdd_lr_encoder)
    setattr(cfg, "svdd_weight_decay", args.svdd_weight_decay)
    setattr(cfg, "svdd_warmup_epochs", args.svdd_warmup_epochs)
    setattr(cfg, "svdd_clip_norm", args.svdd_clip_norm)
    


    # ----------------------------
    # Dispatch
    # ----------------------------
    if args.dataset == "mnist":
        from pipeline.run_ae_svdd import run_pipeline
        run_pipeline(cfg)
    else:
        from pipeline.run_ae_svdd_cifar10 import run_pipeline
        run_pipeline(cfg)

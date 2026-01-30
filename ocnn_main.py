# ocnn_main.py
from utils.const import Const
from utils.Config import Config

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("AE/RCAE + OCNN pipeline (MNIST / CIFAR-10)")

    # ----------------------------
    # Dataset selection
    # ----------------------------
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to run (mnist or cifar10)",
    )

    # One-class label
    # MNIST: digit 0..9
    p.add_argument("--normal-digit", type=int, default=Const.NORMAL_DIGIT)
    # CIFAR-10: class id 0..9
    p.add_argument("--normal-class", type=int, default=0)

    p.add_argument("--pollution-rate", type=float, default=Const.POLLUTION_RATE)
    p.add_argument("--data-dir", type=str, default=Const.DATA_DIR)
    p.add_argument("--num-workers", type=int, default=0)

    # ----------------------------
    # AE
    # ----------------------------
    p.add_argument(
        "--ae-arch",
        type=str,
        default=Const.ae_arch,
        choices=["autoencoder1", "autoencoder2", "autoencoder3", "autoencoder_cifar"],
        help="AE architecture (use autoencoder_cifar for CIFAR-10)",
    )

    p.add_argument("--ae-mode", type=str, default="rcae", choices=["rcae", "simple"])
    p.add_argument("--rep-dim", type=int, default=Const.LATENT_DIM)
    p.add_argument("--ae-epochs", type=int, default=Const.AE_EPOCHS)
    p.add_argument("--ae-lr", type=float, default=Const.AE_LR)
    p.add_argument("--mue", type=float, default=Const.MUE)

    # RCAE specific (if you already have these in Config / Const)
    p.add_argument("--rcae-outer-iters", type=int, default=getattr(Const, "RCAE_OUTER_ITERS", 1))
    p.add_argument("--lamda-set", type=float, nargs="*", default=getattr(Const, "LAMDA_SET", [0.1]))

    # ----------------------------
    # OCNN
    # ----------------------------
    p.add_argument("--nu", type=float, default=Const.OCNN_NU)
    p.add_argument(
        "--activation",
        type=str,
        default=Const.OCNN_ACTIVATION,
        choices=["linear", "sigmoid", "relu", "tanh"],
    )
    p.add_argument("--ocnn-epochs", type=int, default=Const.OCNN_EPOCHS)
    p.add_argument("--ocnn-lr-init", type=float, default=Const.OCNN_LR_INIT)
    p.add_argument("--ocnn-lr-finetune", type=float, default=Const.OCNN_LR_FINETUNE)

    p.add_argument(
        "--lr-encoder-joint",
        type=float,
        default=0.0,
        help="Encoder LR during OCNN in joint mode (0 = frozen encoder)",
    )

    p.add_argument("--finetune-start-epoch", type=int, default=Const.OCNN_FINETUNE_START_EPOCH)
    p.add_argument("--batch-size", type=int, default=Const.OCNN_BATCH_SIZE)
    p.add_argument("--ocnn-joint", action="store_true")

    # ----------------------------
    # Runs
    # ----------------------------
    p.add_argument("--seed", type=int, default=Const.SEED)
    p.add_argument("--runs-dir", type=str, default=Const.BASE_RUNS_DIR)

    # ----------------------------
    # W&B
    # ----------------------------
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=Const.WANDB_PROJECT)

    args = p.parse_args()

    # ----------------------------
    # Config
    # ----------------------------
    # Unifichiamo il concetto di "classe normale" in normal_class,
    # ma manteniamo anche normal_digit per compatibilità (MNIST pipeline).
    cfg = Config(
        dataset=args.dataset,
        normal_digit=args.normal_digit,
        normal_class=args.normal_class,
        pollution_rate=args.pollution_rate,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        ae_arch=args.ae_arch,
        ae_mode=args.ae_mode,
        rep_dim=args.rep_dim,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        mue=args.mue,
        rcae_outer_iters=args.rcae_outer_iters,
        lamda_set=args.lamda_set,
        nu=args.nu,
        activation=args.activation,
        ocnn_epochs=args.ocnn_epochs,
        ocnn_lr_init=args.ocnn_lr_init,
        ocnn_lr_finetune=args.ocnn_lr_finetune,
        lr_encoder_joint=args.lr_encoder_joint,
        finetune_start_epoch=args.finetune_start_epoch,
        ocnn_batch_size=args.batch_size,
        ocnn_joint=args.ocnn_joint,
        seed=args.seed,
        base_runs_dir=args.runs_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # ----------------------------
    # Dispatch pipeline
    # ----------------------------
    if args.dataset == "mnist":
        from pipeline.run_ae_ocnn import run_pipeline
        # MNIST pipeline usa cfg.normal_digit
        run_pipeline(cfg)
    else:
        from pipeline.run_ae_ocnn_cifar10 import run_pipeline
        # CIFAR pipeline usa cfg.normal_class
        run_pipeline(cfg)


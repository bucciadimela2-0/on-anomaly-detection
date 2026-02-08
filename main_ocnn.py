# ocnn_main.py
from utils.const import Const
from utils.Config import Config

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("AE (MSE) + OCNN (z-only) pipeline")

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
    p.add_argument("--normal-digit", type=int, default=Const.NORMAL_DIGIT)  # MNIST
    p.add_argument("--normal-class", type=int, default=0)                   # CIFAR-10
    p.add_argument("--pollution-rate", type=float, default=Const.POLLUTION_RATE)

    # ----------------------------
    # Autoencoder (MSE only)
    # ----------------------------
    p.add_argument(
        "--ae-arch",
        type=str,
        default=Const.ae_arch,
        choices=["autoencoder1", "autoencoder2", "autoencoder3", "autoencoder_cifar"],
    )
    p.add_argument("--rep-dim", type=int, default=Const.LATENT_DIM)
    p.add_argument("--ae-epochs", type=int, default=Const.AE_EPOCHS)
    p.add_argument("--ae-lr", type=float, default=Const.AE_LR)
    p.add_argument("--ae-batch-size", type=int, default=Const.AE_BATCH_SIZE)
    p.add_argument("--ae-weight-decay", type=float, default=0.0)

    # ----------------------------
    # OCNN (simple, z-only)
    # ----------------------------
    p.add_argument("--nu", type=float, default=Const.OCNN_NU)
    p.add_argument(
        "--activation",
        type=str,
        default=Const.OCNN_ACTIVATION,
        choices=["linear", "relu", "leaky_relu", "tanh", "sigmoid"],
    )
    p.add_argument("--ocnn-epochs", type=int, default=Const.OCNN_EPOCHS)
    p.add_argument("--ocnn-lr", type=float, default=Const.OCNN_LR_INIT)
    p.add_argument("--ocnn-hidden-dim", type=int, default=Const.OCNN_HIDDEN_DIM)
    p.add_argument("--ocnn-batch-size", type=int, default=Const.OCNN_BATCH_SIZE)
    #p.add_argument("--clip-norm", type=float, default=Const.SVDD_CLIP_NORM)

    # ----------------------------
    # Run
    # ----------------------------
    p.add_argument("--seed", type=int, default=Const.SEED)
    p.add_argument("--runs-dir", type=str, default=Const.BASE_RUNS_DIR)

    args = p.parse_args()

    # ----------------------------
    # Config 
    # ----------------------------
    cfg = Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        normal_digit=args.normal_digit,
        normal_class=args.normal_class,
        pollution_rate=args.pollution_rate,
        ae_arch=args.ae_arch,
        rep_dim=args.rep_dim,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        ae_batch_size=args.ae_batch_size,
        ae_weight_decay=args.ae_weight_decay,
        nu=args.nu,
        activation=args.activation,
        ocnn_epochs=args.ocnn_epochs,
        ocnn_lr_init=args.ocnn_lr,
        ocnn_hidden_dim=args.ocnn_hidden_dim,
        ocnn_batch_size=args.ocnn_batch_size,
       
        seed=args.seed,
        base_runs_dir=args.runs_dir,
    )

    # ----------------------------
    # Dispatch
    # ----------------------------
    if args.dataset == "mnist":
        from pipeline.run_mnist_ocnn import run_pipeline
        run_pipeline(cfg)
    else:
        from pipeline.run_cifar_ocnn import run_pipeline
        
        run_pipeline(cfg)

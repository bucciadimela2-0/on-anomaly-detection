# utils/config.py
from dataclasses import dataclass
from typing import Tuple
import random
import numpy as np
import torch

from utils.const import Const


@dataclass
class Config:
    # Data
    normal_digit: int = Const.NORMAL_DIGIT
    pollution_rate: float = Const.POLLUTION_RATE
    data_dir: str = Const.DATA_DIR
    num_workers: int = 0

    # Pipeline mode
    # If True, OCNN training also updates the encoder parameters
    ocnn_joint: bool = False

    # Autoencoder
    # "autoencoder1" | "autoencoder2" | "autoencoder3"
    ae_arch: str = Const.ae_arch
    rep_dim: int = Const.LATENT_DIM

    # "rcae" | "simple"
    ae_mode: str = "rcae"

    ae_epochs: int = Const.AE_EPOCHS
    ae_lr: float = Const.AE_LR
    ae_weight_decay: float = 0.0
    ae_batch_size: int = Const.AE_BATCH_SIZE

    # Robust CAE
    lamda_set: Tuple[float, ...] = Const.LAMBDA_SET
    rcae_outer_iters: int = 1
    mue: float = Const.MUE

    # OCNN
    nu: float = Const.OCNN_NU
    activation: str = Const.OCNN_ACTIVATION

    ocnn_epochs: int = Const.OCNN_EPOCHS
    ocnn_lr_init: float = Const.OCNN_LR_INIT
    ocnn_lr_finetune: float = Const.OCNN_LR_FINETUNE
    finetune_start_epoch: int = Const.OCNN_FINETUNE_START_EPOCH
    ocnn_batch_size: int = Const.OCNN_BATCH_SIZE

    # Separate encoder LR during joint training (0.0 => frozen encoder)
    lr_encoder_joint: float = 1e-5

    # Runs / output
    seed: int = Const.SEED
    base_runs_dir: str = Const.BASE_RUNS_DIR
    ckpt_subdir: str = Const.CKPT_SUBDIR
    plots_subdir: str = Const.PLOTS_SUBDIR

    # Weights & Biases
    wandb: bool = Const.WANDB_ENABLED
    wandb_project: str = Const.WANDB_PROJECT

    # Deep SVDD
    svdd_objective: str = "one-class"   # "one-class" | "soft-boundary"
    svdd_nu: float = 0.1
    svdd_epochs: int = 50
    svdd_lr: float = 1e-4

    # 0.0 => freeze encoder, >0 => fine-tune encoder
    svdd_lr_encoder: float = 0.0

    svdd_weight_decay: float = 0.0
    svdd_warmup_epochs: int = 10
    svdd_clip_norm: float = 0.5

    # in Config
    dataset: str = "mnist"  # oppure "cifar10"
    normal_class: int = 0   # cifar class id
    rep_dim: int = 128
    ae_arch: str = "autoencoder_cifar"



def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

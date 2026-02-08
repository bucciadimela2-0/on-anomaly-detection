# utils/const.py

class Const:
   

    # Dataset / images
    IMG_HGT: int = 28
    IMG_WDT: int = 28
    IMG_CH: int = 1

    DATA_DIR: str = "./data"

    # One-class MNIST setup
    NORMAL_DIGIT: int = 0
    POLLUTION_RATE: float = 0.1

    # Autoencoder / encoder architecture
    LATENT_DIM: int = 32

    ENC_CONV1_OUT: int = 8
    ENC_CONV2_OUT: int = 4

    LEAKY_RELU_SLOPE: float = 0.1

    # Autoencoder training
    AE_EPOCHS: int = 250
    AE_LR: float = 1e-4
    AE_BATCH_SIZE: int = 128


    # OC-NN (One-Class Neural Network)
    OCNN_HIDDEN_DIM: int = 32
    OCNN_NU: float = 0.1

    OCNN_EPOCHS: int = 150
    OCNN_LR_INIT: float = 1e-4
    OCNN_LR_FINETUNE: float = 1e-5
    OCNN_FINETUNE_START_EPOCH: int = 100
    OCNN_BATCH_SIZE: int = 128

    OCNN_ACTIVATION: str = "linear"

    # Experiment / runs
    SEED: int = 73

    BASE_RUNS_DIR: str = "runs"
    CKPT_SUBDIR: str = "checkpoints"
    PLOTS_SUBDIR: str = "plots"

   

    # Default architecture
    ae_arch: str = "autoencoder1"

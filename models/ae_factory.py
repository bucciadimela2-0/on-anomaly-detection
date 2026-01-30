
import torch
from utils.Config import Config

def build_autoencoder(cfg: Config) -> torch.nn.Module:
    """
    Factory function for autoencoder construction.

    Selects and instantiates the autoencoder architecture specified
    in the configuration object.
    """
     
    if cfg.ae_arch == "autoencoder1":
        # Simple LeNet-like convolutional autoencoder
        from models.autoencoder1 import Autoencoder
        return Autoencoder()
    elif cfg.ae_arch == "autoencoder2":
        # Deep SVDD-style autoencoder with configurable representation dimension
        from models.autoencoder2 import Autoencoder
        return Autoencoder(rep_dim=cfg.rep_dim)
    elif cfg.ae_arch == "autoencoder_cifar":
        from models.autoencoder_cifar import Autoencoder
        return Autoencoder(rep_dim=cfg.rep_dim)
    elif cfg.ae_arch == "autoencoder2_cifar":
        from models.autoencoder2_cifar import Autoencoder
        return Autoencoder(rep_dim=cfg.rep_dim)

    elif cfg.ae_arch == "autoencoder3":
        # OC-NN / repository-style fully convolutional autoencoder
        from models.autoencoder3 import Autoencoder
        return Autoencoder()

    raise ValueError(f"Unknown ae_arch='{cfg.ae_arch}'")

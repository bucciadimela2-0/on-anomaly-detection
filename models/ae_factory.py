
import torch
from utils.Config import Config

def build_autoencoder(cfg: Config) -> torch.nn.Module:
    
    #Factory function for autoencoder construction.
    if cfg.ae_arch == "autoencoder1":
       
        from models.autoencoder1 import Autoencoder
        return Autoencoder()
    elif cfg.ae_arch == "autoencoder_cifar":
        from models.autoencoder_cifar import Autoencoder
        return Autoencoder(rep_dim=cfg.rep_dim)
    
    elif cfg.ae_arch == "autoencoder2":
     from models.autoencoder2 import Autoencoder
     return Autoencoder(rep_dim=cfg.rep_dim)
    
    elif cfg.ae_arch == "autoencoder2_cifar":
       from models.autoencoder2_cifar import Autoencoder
       return Autoencoder(rep_dim=cfg.rep_dim)
    

    raise ValueError(f"Unknown ae_arch='{cfg.ae_arch}'")


from typing import Dict, Type
import torch.nn as nn

from models.autoencoder1 import Autoencoder as Autoencoder1
from models.autoencoder2 import Autoencoder as Autoencoder2
from models.autoencoder3 import Autoencoder as Autoencoder3
from models.ocnn import OCNN

# Autoencoder registry 
AE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "autoencoder1": Autoencoder1,
    "autoencoder2": Autoencoder2,
    "autoencoder3": Autoencoder3,
}

# OCNN registry 
OCNN_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ocnn": OCNN,
}

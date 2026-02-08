
from typing import Dict, Type
import torch.nn as nn

from models.autoencoder1 import Autoencoder as Autoencoder1

from models.ocnn import OCNN

# Autoencoder registry 
AE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "autoencoder1": Autoencoder1,

}

# OCNN registry 
OCNN_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ocnn": OCNN,
}

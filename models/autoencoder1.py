import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.const import *


#LeNet-like convolutional autoencoder for 28x28 grayscale images.
#Encoder: Conv + ReLU + MaxPool + FC → latent space.
#Decoder: FC + Upsampling + Conv → image reconstruction.



class Encoder(nn.Module):
    

    def __init__(self, latent_dim: int = 32, leak: float = 0.1):
        super().__init__()
        self.act = nn.LeakyReLU(leak, inplace=True)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, padding=2)

        self.fc = nn.Linear(7 * 7 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))      # [B,8,28,28]
        x = F.max_pool2d(x, 2, 2)        # [B,8,14,14]

        x = self.act(self.conv2(x))      # [B,4,14,14]
        x = F.max_pool2d(x, 2, 2)        # [B,4,7,7]

        x = x.view(x.size(0), -1)        # [B,196]
        z = self.fc(x)                   # [B,latent_dim]
        return z


class Decoder(nn.Module):
    
    
    def __init__(self, latent_dim: int = 32, leak: float = 0.1):
        super().__init__()
        self.act = nn.LeakyReLU(leak, inplace=True)

        # Mirror of encoder flatten size
        self.fc = nn.Linear(latent_dim, 7 * 7 * 4)

        # Mirror convs with upsampling
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, padding=2)  # 7x7 -> 7x7
        self.conv2 = nn.Conv2d(8, 1, kernel_size=5, padding=2)  # 14/28 -> output

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc(z))                 # [B,196]
        x = x.view(x.size(0), 4, 7, 7)           # [B,4,7,7]

        x = self.act(self.conv1(x))              # [B,8,7,7]
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # [B,8,14,14]

        x = self.act(x)                          

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # [B,8,28,28]
        x = torch.sigmoid(self.conv2(x))         # [B,1,28,28]
        return x


class Autoencoder(nn.Module):
  
    def __init__(self, latent_dim: int = 32, leak: float = 0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, leak=leak)
        self.decoder = Decoder(latent_dim=latent_dim, leak=leak)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def get_encoder(self):
        return self.encoder

  

import torch
import torch.nn as nn
import torch.nn.functional as F


#LeNet-like convolutional autoencoder for 28x28 grayscale images.
#Encoder: Conv + ReLU + MaxPool + FC → latent space.
#Decoder: FC + Upsampling + Conv → image reconstruction.

#Based on the Deep SVDD PyTorch codebase:
#https://github.com/lukasruff/Deep-SVDD-PyTorch


class Encoder(nn.Module):
    
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-4, affine=False)

        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-4, affine=False)

        self.fc1 = nn.Linear(4 * 7 * 7, rep_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))   # [B, 8, 14, 14]
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))   # [B, 4, 7, 7]
        x = x.view(x.size(0), -1)                  # [B, 4*7*7]
        z = self.fc1(x)                            # [B, rep_dim]
        return z

    def get_representation_dim(self) -> int:
        return self.rep_dim


class Decoder(nn.Module):
   
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim

        # rep_dim -> (rep_dim/16) x 4 x 4
        self.deconv_in_channels = rep_dim // 16

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.deconv_in_channels,
            out_channels=4,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(4, eps=1e-4, affine=False)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            padding=3,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(8, eps=1e-4, affine=False)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=1,
            kernel_size=5,
            padding=2,
            bias=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, rep_dim]
        x = z.view(z.size(0), self.deconv_in_channels, 4, 4)     # [B, rep_dim/16, 4, 4]

        x = F.interpolate(F.leaky_relu(x), scale_factor=2)       # [B, *, 8, 8]
        x = self.deconv1(x)

        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)  # ~16x16
        x = self.deconv2(x)

        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)  # ~32x32
        x = self.deconv3(x)

        x = torch.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    
    # Autoencoder = Encoder + Decoder.
    
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(rep_dim=rep_dim)
        self.decoder = Decoder(rep_dim=rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # Getters (
    def get_encoder(self) -> nn.Module:
        return self.encoder

   


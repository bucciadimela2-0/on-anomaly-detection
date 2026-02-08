import torch
import torch.nn as nn
import torch.nn.functional as F


# LeNet-type encoder/decoder for CIFAR-10 (3x32x32)
# Encoder: [Conv + LeakyReLU + MaxPool] x3 -> FC(rep_dim=128)
# Decoder: FC -> upsample -> ConvTranspose (or Conv) back to 3x32x32
#
# Architecture aligned with DeepSVDD description:
# 3 conv modules with 32, 64, 128 filters (5x5), followed by dense layer of 128 units.

class Encoder(nn.Module):
    def __init__(self, rep_dim: int = 128):
        super().__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)

        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4, affine=False)

        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4, affine=False)

        # after 3 pools: 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(128 * 4 * 4, rep_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))  # [B, 32, 16, 16]
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))  # [B, 64,  8,  8]
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))  # [B,128,  4,  4]
        x = x.view(x.size(0), -1)                             # [B, 128*4*4]
        z = self.fc(x)                                        # [B, rep_dim]
        return z

    def get_representation_dim(self) -> int:
        return self.rep_dim


class Decoder(nn.Module):
    def __init__(self, rep_dim: int = 128):
        super().__init__()
        self.rep_dim = rep_dim

        # Map latent to feature map 128x4x4 (mirror of encoder bottleneck)
        self.fc = nn.Linear(rep_dim, 128 * 4 * 4, bias=False)

        # We'll upsample (nearest) + ConvTranspose/Conv to go back to 32x32
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-4, affine=False)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(32, eps=1e-4, affine=False)

        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, padding=2, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, rep_dim]
        x = self.fc(z)                           # [B, 128*4*4]
        x = x.view(z.size(0), 128, 4, 4)         # [B, 128, 4, 4]

        # 4 -> 8
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)          # [B,128, 8, 8]
        x = self.deconv1(x)                                         # [B, 64, 8, 8]

        # 8 -> 16
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)  # [B, 64,16,16]
        x = self.deconv2(x)                                           # [B, 32,16,16]

        # 16 -> 32
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=2)  # [B, 32,32,32]
        x = self.deconv3(x)                                           # [B,  3,32,32]

        # If inputs are in [0,1]
        x = torch.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, rep_dim: int = 128):
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

    def get_encoder(self) -> nn.Module:
        return self.encoder

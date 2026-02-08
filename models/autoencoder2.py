import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-4, affine=False)

        self.conv2 = nn.Conv2d(8, 4, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-4, affine=False)

        self.fc1 = nn.Linear(4 * 7 * 7, rep_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim

        self.fc2 = nn.Linear(rep_dim, 4 * 7 * 7, bias=False)

        self.deconv1 = nn.ConvTranspose2d(4, 8, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-4, affine=False)

        self.deconv2 = nn.ConvTranspose2d(8, 1, 5, padding=2, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc2(z)
        x = x.view(z.size(0), 4, 7, 7)

        x = F.interpolate(F.leaky_relu(x), scale_factor=2)          # 7 -> 14
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2) # 14 -> 28
        x = self.deconv2(x)

        return torch.sigmoid(x)


class Autoencoder(nn.Module):
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(rep_dim=rep_dim)
        self.decoder = Decoder(rep_dim=rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

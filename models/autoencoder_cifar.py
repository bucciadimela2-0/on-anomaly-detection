# models/autoencoder_cifar.py
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        act = nn.LeakyReLU(leak, inplace=True)

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 32 -> 16
            act,
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            act,
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 8 -> 4
            act,
        )
        self.fc = nn.Linear(256 * 4 * 4, rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        act = nn.LeakyReLU(leak, inplace=True)

        self.fc = nn.Linear(rep_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            act,
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 8 -> 16
            act,
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # 16 -> 32
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        self.encoder = Encoder(rep_dim=rep_dim, leak=leak)
        self.decoder = Decoder(rep_dim=rep_dim, leak=leak)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # comodo: alcune pipeline cercano encode_flat
    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return z.flatten(1) if z.dim() > 2 else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

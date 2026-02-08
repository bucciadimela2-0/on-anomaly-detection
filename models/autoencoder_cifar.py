import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder LeNet-style per CIFAR-10 come da paper (Sezione 5.2):
    - 3 moduli convoluzionali: Conv2d + LeakyReLU + MaxPool2d
    - Filters: 32, 64, 128 (tutti 5x5x3)
    - Output: dense layer con rep_dim unità
    """
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        
        self.rep_dim = rep_dim
        
        # Modulo 1: 32 filters (5x5)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4)
        self.leaky1 = nn.LeakyReLU(leak)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Modulo 2: 64 filters (5x5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4)
        self.leaky2 = nn.LeakyReLU(leak)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Modulo 3: 128 filters (5x5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4)
        self.leaky3 = nn.LeakyReLU(leak)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Dense finale
        self.fc = nn.Linear(128 * 4 * 4, rep_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 32, 32)
        Output: (B, rep_dim)
        """
        # Modulo 1
        x = self.conv1(x)        # (B, 32, 32, 32)
        x = self.bn1(x)
        x = self.leaky1(x)
        x = self.pool1(x)        # (B, 32, 16, 16)
        
        # Modulo 2
        x = self.conv2(x)        # (B, 64, 16, 16)
        x = self.bn2(x)
        x = self.leaky2(x)
        x = self.pool2(x)        # (B, 64, 8, 8)
        
        # Modulo 3
        x = self.conv3(x)        # (B, 128, 8, 8)
        x = self.bn3(x)
        x = self.leaky3(x)
        x = self.pool3(x)        # (B, 128, 4, 4)
        
        # Flatten e dense
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.fc(x)             # (B, rep_dim)
        
        return x


class Decoder(nn.Module):
    """
    Decoder simmetrico per CIFAR-10.
    """
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        
        self.rep_dim = rep_dim
        
        # Dense iniziale
        self.fc = nn.Linear(rep_dim, 128 * 4 * 4, bias=False)
        
        # Modulo 1 (inverso): upsample + conv
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 4x4 -> 8x8
        self.conv1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4)
        self.leaky1 = nn.LeakyReLU(leak)
        
        # Modulo 2: upsample + conv
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 8x8 -> 16x16
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4)
        self.leaky2 = nn.LeakyReLU(leak)
        
        # Modulo 3: upsample + conv
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')  # 16x16 -> 32x32
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, bias=False)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, rep_dim)
        Output: (B, 3, 32, 32) in [0, 1]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)  # (B, 128, 4, 4)
        
        # Modulo 1
        x = self.up1(x)          # (B, 128, 8, 8)
        x = self.conv1(x)        # (B, 64, 8, 8)
        x = self.bn1(x)
        x = self.leaky1(x)
        
        # Modulo 2
        x = self.up2(x)          # (B, 64, 16, 16)
        x = self.conv2(x)        # (B, 32, 16, 16)
        x = self.bn2(x)
        x = self.leaky2(x)
        
        # Modulo 3
        x = self.up3(x)          # (B, 32, 32, 32)
        x = self.conv3(x)        # (B, 3, 32, 32)
        x = torch.sigmoid(x)     # [0, 1]
        
        return x


class Autoencoder(nn.Module):
    """
    Autoencoder completo per CIFAR-10 secondo paper OC-NN.
    """
    def __init__(self, rep_dim: int = 128, leak: float = 0.1):
        super().__init__()
        self.encoder = Encoder(rep_dim=rep_dim, leak=leak)
        self.decoder = Decoder(rep_dim=rep_dim, leak=leak)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ricostruzione completa."""
        return self.decode(self.encode(x))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Solo encoding."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Solo decoding."""
        return self.decoder(z)
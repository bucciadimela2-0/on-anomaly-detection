import torch
import torch.nn as nn
import torch.nn.functional as F


#LeNet-like convolutional autoencoder for 28x28 grayscale images.
#Encoder: Conv + ReLU + MaxPool → latent space.
#Decoder: Upsampling + Conv → image reconstruction.

#Adapted from the OC-NN implementation:
#https://github.com/raghavchalapathy/oc-nn



class Encoder(nn.Module):
    
    def __init__(self, latent_dim: int = 32, leak: float = 0.1):
        super().__init__()
        self.act  = nn.LeakyReLU(leak, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))   # [B,16,28,28]
        x = self.pool(x)              # [B,16,14,14]

        x = self.act(self.conv2(x))   # [B,8,14,14]
        x = self.pool(x)              # [B,8,7,7]

        x = self.act(self.conv3(x))   # [B,8,7,7]
        x = self.pool(x)              # [B,8,4,4]

        x = self.act(self.conv4(x))   # [B,8,4,4]
        x = self.pool(x)              # [B,8,2,2]

        return x.view(x.size(0), -1)  # [B,32]


class Decoder(nn.Module):
    
    def __init__(self, latent_dim: int = 32, leak: float = 0.1):
        super().__init__()
        assert latent_dim == 32, "Repo-like MNIST: latent_dim deve essere 32 (=8*2*2)."

        self.act = nn.LeakyReLU(leak, inplace=True)

        self.conv5 = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)

        # "valid" 3x3: 16x16 -> 14x14
        self.conv8 = nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        x = z_flat.view(z_flat.size(0), 8, 2, 2)                 # [B,8,2,2]

        x = self.act(self.conv5(x))                              # [B,4,2,2]
        x = F.interpolate(x, scale_factor=2, mode="nearest")     # [B,4,4,4]

        x = self.act(self.conv6(x))                              # [B,8,4,4]
        x = F.interpolate(x, scale_factor=2, mode="nearest")     # [B,8,8,8]

        x = self.act(self.conv7(x))                              # [B,8,8,8]
        x = F.interpolate(x, scale_factor=2, mode="nearest")     # [B,8,16,16]

        x = self.act(self.conv8(x))                              # [B,16,14,14]
        x = F.interpolate(x, scale_factor=2, mode="nearest")     # [B,16,28,28]

        
        x = F.interpolate(x, size=(32, 32), mode="nearest")      # [B,16,32,32]
        x = self.conv_out(x)                                     # [B,1,32,32]
        x = x[:, :, 2:-2, 2:-2]                                  # [B,1,28,28]

        return torch.sigmoid(x)


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
    
if __name__ == "__main__":
        x = torch.randn(4, 1, 28, 28)
        ae_repo = Autoencoder()
        y2 = ae_repo(x)
        z2 = ae_repo.encode(x)
        print("Repo AE :", y2.shape, z2.shape)

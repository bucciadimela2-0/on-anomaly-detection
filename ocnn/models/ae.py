
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, dataset: str = "mnist"):
        super().__init__()
        dataset = dataset.lower()
        self.dataset = dataset

        if dataset == "mnist":
            in_ch = 1
            self.encoder = nn.Sequential(
                # 28x28 -> 14x14
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),

                # 14x14 -> 7x7
                nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),

                # Rimaniamo su 7x7
                nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.1, inplace=True),
            )

        elif dataset == "cifar10":
            
            in_ch = 3
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
            )
        else:
            raise ValueError(f"Dataset non supportato: {dataset}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, dataset: str = "mnist"):
        super().__init__()
        dataset = dataset.lower()
        self.dataset = dataset

        if dataset == "mnist":
            latent_ch = 8
            out_ch = 1
            # Partiamo da feature map 7x7, vogliamo 28x28

            self.decoder = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(latent_ch, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.1, inplace=True),

                # 14x14 -> 28x28
                nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1, inplace=True),

                # 28x28 -> 28x28 (refine + output)
                nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
                nn.Sigmoid(),  # output in [0,1]
            )

        elif dataset == "cifar10":
            latent_ch = 32
            out_ch = 3

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_ch, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),

                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),

                nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),

                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),

                nn.ConvTranspose2d(128, out_ch, kernel_size=3, stride=2),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Dataset non supportato: {dataset}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ConvAutoencoder(nn.Module):
    
   # Wrapper encoder+decoder, flessibile per MNIST e CIFAR-10.
   
    def __init__(self, dataset: str = "mnist"):
        super().__init__()
        self.dataset = dataset.lower()
        self.encoder = ConvEncoder(dataset=self.dataset)
        self.decoder = ConvDecoder(dataset=self.dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


    def encode(self, x: torch.Tensor) -> torch.Tensor:
       
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        
        return self.decoder(z)

    def get_encoder(self, freeze: bool = False) -> nn.Module:
      
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        return self.encoder

    def get_decoder(self, freeze: bool = False) -> nn.Module:
      
        if freeze:
            for p in self.decoder.parameters():
                p.requires_grad = False
        return self.decoder

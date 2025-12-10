import torch
import torch.nn as nn

class OCNN(nn.Module):
    def __init__(self, encoder, rep_dim=32):
        super().__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(rep_dim, rep_dim, bias = False)
    
    def forward(self,x):
        z_map = self.encoder(x)
        z = self.flatten(z_map)
        y = self.fc(z)
        return y
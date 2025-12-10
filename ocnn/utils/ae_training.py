import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.ae import ConvAutoencoder

def train_autoencoder(
    dataset: str,
    train_loader: DataLoader,
    device: str = "cpu",
    n_epochs: int = 20,
    lr: float = 1e-3

):

 ae=ConvAutoencoder(dataset = dataset).to(device)
 optimizer = torch.optim.Adam(ae.parameters(), lr = lr)
 criterion = nn.MSELoss()

 ae.train()
 for epoch in range(n_epochs):
    epoch_loss = 0.0
    for x,_ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_rec = ae(x)
        loss = criterion(x_rec,x)
        loss.backward()
        optimizer.step()
        epoch_loss+= loss.item()*x.size(0)
    epoch_loss/= len(train_loader.dataset)
    print(f"[AE] Epoch {epoch+1}/{n_epochs} - loss: {epoch_loss:.6f}")

    encoder = ae.get_encoder(freeze = False).to(device)

    encoder = ae.get_encoder(freeze=False).to(device)

 return ae, encoder #restituisce autoencoder, encoder per ocnn






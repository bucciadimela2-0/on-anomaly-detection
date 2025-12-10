import torch
import numpy as np
from torch.utils.data import DataLoader
from models.ocnn import OCNN

@torch.no_grad()

def init_center_c(
    model: OCNN,
    train_loader: DataLoader,
    device: str,
    eps: float =0.1
)-> torch.Tensor:

    model.eval()
    outputs = []

    for x, _ in train_loader:
        x = x.to(device)
        y = model(x)
        outputs.append(y.detach().cpu())
        outputs = torch.cat(outputs, dim = 0)
        c = outputs.mean(dim=0) #calcoliamo la media degli output
        #se minore di epsilon, attribuisce lb ub
        c[(c.abs() < eps)&(c < 0)] = -eps
        c[(c.abs() < eps)&(c > 0)] = eps

        return c.to(device)

@torch.no_grad()
def update_radius_r(
    model: OCNN,
    train_loader: DataLoader,
    c: torch.Tensor, 
    device: str,
    nu: float
)-> float:
    #calcola il raggio r come nu-percentile delle distanze dal centro
    model.eval()
    dists = []

    for x, _ in train_loader:
        x=x.to(device)
        y=model(x)
        dist = torch.sum( (y-c)** 2 , dim = 1)
        dists.append(dist.detach().cpu())
    dists = torch.cat(dists, dim =0).numpy()
    r = np.percentile(dists, nu * 100)
    return float(r)

def ocnn_hypershpere_loss(
    outputs : torch.Tensor,
    c: torch.Tensor,
    r: float,
    nu: float
)-> torch.Tensor:
    #calcola la loss in forma di una ipersfera
    dist = torch.sum ((outputs - c) ** 2, dim =1 )
    term = r**2 + torch.clamp(dist - r**2, min =0.0)
    loss = (1.0/nu) * term.mean()
    return loss




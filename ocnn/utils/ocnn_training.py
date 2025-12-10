import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from models.ae import ConvAutoencoder
from models.ocnn import OCNN

from utils.ocnn_utils import init_center_c, update_radius_r, ocnn_hypershpere_loss
from utils.ae_training import train_autoencoder

def train_ocnn(
    dataset : str,
    train_loader: DataLoader,
    device: str = "cpu",
    nu: float =0.1,
    ae_epochs: int = 20,
    ae_lr: float = 1e-3,
    ocnn_epochs: int = 50,
    ocnn_lr: float = 1e-4
):

#pretraining AE

    ae, encoder = train_autoencoder(
    dataset = dataset,
    train_loader = train_loader,
    device = device,
    n_epochs = ae_epochs,
    lr = ae_lr
)
#calcola rep_dim(dimensione del flatten)
    with torch.no_grad():
        x0,_ = next(iter(train_loader))
        x0 = x0.to(device)
        z_map = encoder(x0)
        rep_dim = z_map.view(z_map.size(0),-1).size(1)
#Costruisci OCNN

    model = OCNN(encoder = encoder, rep_dim=rep_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = ocnn_lr)
#Inizializza centro c e raggio r
    c = init_center_c(model, train_loader, device)
    R = update_radius_r(model, train_loader, c, device, nu)
    print(f"[Init] R = {R:.6f}")
#Training OCNN

    for epoch in range(ocnn_epochs):
        model.train()
        total_loss =0.0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = ocnn_hypershpere_loss(outputs, c, R, nu)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*x.size(0)
        total_loss /= len(train_loader.dataset)
        #aggiorno r a fine epoca
        R =  update_radius_r(model, train_loader,c,device, nu)

        print(f"[OCNN] Epoch {epoch+1}/{ocnn_epochs} - "
              f"loss: {total_loss:.6f} - R: {R:.6f}")
    return model, c, R

@torch.no_grad
def evaluate_ocnn(
        model:OCNN,
        c: torch.Tensor,
        test_loader: DataLoader,
        y_test: np.ndarray,
        device: str ="cpu"

    ):
    model.eval()
    scores = []
    for x,_ in test_loader:
        x = x.to(device)
        outputs = model(x)
        dist = torch.sum((outputs - c)** 2, dim =1)
        scores.append(dist.detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    auc = roc_auc_score(y_test, scores)
    print(f"[EVAL] AUROC OCNN: {auc:.4f}")

    return auc, scores



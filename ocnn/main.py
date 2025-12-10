import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from utils.ocnn_training import train_ocnn, evaluate_ocnn

device = "cuda" if torch.cuda.is_available() else "cpu"

normal_class = 0
nu = 0.1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

idx_train_normal = np.where(np.array(train_dataset.targets) == normal_class)[0]
train_subset = Subset(train_dataset, idx_train_normal)

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

y_test = np.where(np.array(test_dataset.targets) == normal_class, 0, 1)

# Training OCNN
model, c, R = train_ocnn(
    dataset="mnist",
    train_loader=train_loader,
    device=device,
    nu=nu,
    ae_epochs=20,
    ocnn_epochs=50
)

# Evaluation
auc, scores = evaluate_ocnn(model, c, test_loader, y_test, device=device)


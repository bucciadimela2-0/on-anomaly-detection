from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


def global_contrast_normalization(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    
    #Global Contrast Normalization (GCN).
    #Centers the image and rescales by its L2 norm.
    
    x_centered = x - x.mean()
    norm = torch.sqrt(torch.sum(x_centered ** 2)) + eps
    return x_centered / norm


def rescale_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    
    #Min-max rescale to [0, 1].
    
    min_val = x.min()
    max_val = x.max()
    denom = (max_val - min_val)
    if denom <= 0:
        return x - min_val
    return (x - min_val) / denom


class OCNNTransform:
    
    #Transform pipeline:
    #PIL -> Tensor -> GCN -> min-max to [0, 1]
   
    def __call__(self, img) -> torch.Tensor:
        x = transforms.functional.to_tensor(img)
        x = global_contrast_normalization(x)
        x = rescale_to_unit_interval(x)
        return x


class OneClassTrainWrapper(Dataset):
    
    def __init__(self, base: Dataset, indices: List[int], normal_digit: int):
        self.base = base
        self.indices = indices
        self.normal = int(normal_digit)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        label = 0 if int(y) == self.normal else 1
        return x, torch.tensor(label, dtype=torch.long)


class OneClassTestWrapper(Dataset):
   
    def __init__(self, base: Dataset, normal_digit: int):
        self.base = base
        self.normal = int(normal_digit)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        label = 0 if int(y) == self.normal else 1
        return x, torch.tensor(label, dtype=torch.long)


@dataclass
class MNISTOneClassDataModule:
    data_dir: str
    normal_digit: int
    pollution_rate: float
    batch_size: int
    num_workers: int = 0
    seed: int = 42

    # These will be set in setup()
    train_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None

    def setup(self) -> None:
       
        tfm = OCNNTransform()

        train_full = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=tfm)
        test_full = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=tfm)

        # Use targets directly (fast; avoids iterating through the dataset)
        targets = train_full.targets  # shape [60000]
        normal_digit = int(self.normal_digit)

        normal_idx = torch.where(targets == normal_digit)[0].tolist()
        anomaly_idx = torch.where(targets != normal_digit)[0].tolist()

        # Pollution count defined as a fraction of the normal set size
        n_pollution = int(len(normal_idx) * float(self.pollution_rate))

        rng = random.Random(self.seed)
        rng.shuffle(anomaly_idx)
        pollution_idx = anomaly_idx[:n_pollution]

        # Build training indices and shuffle once for good mixing
        train_idx = normal_idx + pollution_idx
        rng.shuffle(train_idx)

        self.train_ds = OneClassTrainWrapper(train_full, train_idx, normal_digit)
        self.test_ds = OneClassTestWrapper(test_full, normal_digit)

        self.n_normals = len(normal_idx)
        self.n_pollution = n_pollution
        self.n_train = len(self.train_ds)
        self.n_test = len(self.test_ds)

        print(
            f"[MNIST-1C] normal={normal_digit} | "
            f"train_normals={self.n_normals} | pollution={self.n_pollution} "
            f"({self.pollution_rate:.3f}) | n_train={self.n_train} | n_test={self.n_test}"
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @torch.no_grad()
    def collect_normal_images(self, device: torch.device) -> torch.Tensor:
        """
        Collect all *normal* training images into a single tensor [N_norm, 1, 28, 28].
        Uses a non-shuffled loader to deterministically gather all normals.
        """
        loader = self.train_dataloader(shuffle=False)
        normals: List[torch.Tensor] = []

        for x, y in loader:
            mask = (y == 0)
            if mask.any():
                normals.append(x[mask])

        if not normals:
            raise RuntimeError("No normal samples found in training loader.")

        return torch.cat(normals, dim=0).to(device, non_blocking=True)


@torch.no_grad()
def encode_dataset(ae: nn.Module, loader: DataLoader, device: torch.device) -> TensorDataset:
    """
    Encode an entire dataset using an autoencoder (or any module exposing encode()).

    Returns:
        TensorDataset(z, y) where:
          z: [N, rep_dim]
          y: [N] one-class labels (0 normal, 1 anomaly)
    """
    ae = ae.to(device).eval()

    all_z: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = ae.encode(x)
        all_z.append(z.detach().cpu())
        all_y.append(y.detach().cpu())

    z = torch.cat(all_z, dim=0)
    y = torch.cat(all_y, dim=0)
    return TensorDataset(z, y)

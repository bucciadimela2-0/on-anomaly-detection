import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


class OCNNTransformCIFAR10:
    """ToTensor only (keeps images in [0,1], compatible with sigmoid decoder)."""
    def __init__(self):
        self.tfm = transforms.ToTensor()

    def __call__(self, img) -> torch.Tensor:
        return self.tfm(img)  # [3,32,32]


class OneClassTrainWrapper(Dataset):
    """Returns (x, label) where label: 0=normal, 1=anomaly."""
    def __init__(self, base: Dataset, indices: List[int], normal_class: int):
        self.base = base
        self.indices = indices
        self.normal = int(normal_class)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        label = 0 if int(y) == self.normal else 1
        return x, torch.tensor(label, dtype=torch.long)


class OneClassTestWrapper(Dataset):
    """Full test set. Returns (x, label) where label: 0=normal, 1=anomaly."""
    def __init__(self, base: Dataset, normal_class: int):
        self.base = base
        self.normal = int(normal_class)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        label = 0 if int(y) == self.normal else 1
        return x, torch.tensor(label, dtype=torch.long)


@dataclass
class CIFAR10OneClassDataModule:
    data_dir: str
    normal_class: int
    pollution_rate: float
    batch_size: int
    num_workers: int = 0
    seed: int = 42
    download: bool = True

    def setup(self) -> None:
        tfm = OCNNTransformCIFAR10()

        train_full = datasets.CIFAR10(
            root=self.data_dir, train=True, download=self.download, transform=tfm
        )
        test_full = datasets.CIFAR10(
            root=self.data_dir, train=False, download=self.download, transform=tfm
        )

        # FAST: read labels without triggering transforms
        targets = train_full.targets  # list[int]

        normal_idx = [i for i, y in enumerate(targets) if y == int(self.normal_class)]
        anomaly_idx = [i for i, y in enumerate(targets) if y != int(self.normal_class)]

        n_pollution = int(len(normal_idx) * float(self.pollution_rate))

        rng = random.Random(self.seed)
        rng.shuffle(anomaly_idx)
        pollution_idx = anomaly_idx[:n_pollution]

        train_idx = normal_idx + pollution_idx

        self.train_ds = OneClassTrainWrapper(train_full, train_idx, self.normal_class)
        self.test_ds = OneClassTestWrapper(test_full, self.normal_class)

        self.n_normals = len(normal_idx)
        self.n_pollution = n_pollution
        self.n_train = len(self.train_ds)
        self.n_test = len(self.test_ds)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def collect_normal_images(self, device: torch.device) -> torch.Tensor:
        loader = self.train_dataloader()
        normals: List[torch.Tensor] = []

        for x, y in loader:
            mask = (y == 0)
            if mask.any():
                normals.append(x[mask])

        if not normals:
            raise RuntimeError("No normal samples found in training loader.")

        return torch.cat(normals, dim=0).to(device)  # [N_norm, 3, 32, 32]


@torch.no_grad()
def encode_dataset(ae, loader: DataLoader, device: torch.device) -> TensorDataset:
    """Returns TensorDataset(z, y) with y in {0 normal, 1 anomaly}."""
    ae.eval()
    all_z: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        z = ae.encode(x)
        all_z.append(z.detach().cpu())
        all_y.append(y.detach().cpu())

    z = torch.cat(all_z, dim=0)
    y = torch.cat(all_y, dim=0)
    return TensorDataset(z, y)

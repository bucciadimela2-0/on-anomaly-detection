import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


class GCN_L1_Transform:
    #Global contrast normalization with L1 norm over the whole image + min-max to [0, 1].
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        x = self.to_tensor(img)                 # (3, 32, 32) in [0, 1]
        x_flat = x.view(-1)                     # (3072,)
        l1 = x_flat.abs().sum().clamp_min(1e-8) # avoid div by 0

        x_norm = (x_flat / l1).view(3, 32, 32)

        x_min = x_norm.min()
        x_max = x_norm.max()
        denom = (x_max - x_min).clamp_min(1e-8)

        return (x_norm - x_min) / denom


class SimpleTransform:
    #Debug transform: just ToTensor.
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        return self.to_tensor(img)


class OneClassTrainWrapper(Dataset):
    #Subset wrapper with binary labels: 0=normal, 1=anomaly.
    def __init__(self, base: Dataset, indices: List[int], normal_class: int):
        self.base = base
        self.indices = list(indices)
        self.normal_class = int(normal_class)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[self.indices[i]]
        label = 0 if int(y) == self.normal_class else 1
        return x, torch.tensor(label, dtype=torch.long)


class OneClassTestWrapper(Dataset):
    #Full test wrapper with binary labels: 0=normal, 1=anomaly.
    def __init__(self, base: Dataset, normal_class: int):
        self.base = base
        self.normal_class = int(normal_class)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[i]
        label = 0 if int(y) == self.normal_class else 1
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
    use_gcn: bool = True

    train_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None
    n_normals: int = 0
    n_pollution: int = 0
    n_train: int = 0
    n_test: int = 0

    def setup(self) -> None:
        tfm = GCN_L1_Transform() if self.use_gcn else SimpleTransform()
        print(f"[DATA] Transform = {'GCN_L1' if self.use_gcn else 'ToTensor'}")

        train_full = datasets.CIFAR10(
            root=self.data_dir, train=True, download=self.download, transform=tfm
        )
        test_full = datasets.CIFAR10(
            root=self.data_dir, train=False, download=self.download, transform=tfm
        )

        targets = train_full.targets
        normal_class = int(self.normal_class)

        normal_idx = [i for i, y in enumerate(targets) if int(y) == normal_class]
        anomaly_idx = [i for i, y in enumerate(targets) if int(y) != normal_class]

        n_pollution = int(len(normal_idx) * float(self.pollution_rate))

        rng = random.Random(self.seed)
        rng.shuffle(anomaly_idx)
        pollution_idx = anomaly_idx[:n_pollution]

        train_idx = normal_idx + pollution_idx
        rng.shuffle(train_idx)

        self.train_ds = OneClassTrainWrapper(train_full, train_idx, normal_class)
        self.test_ds = OneClassTestWrapper(test_full, normal_class)

        self.n_normals = len(normal_idx)
        self.n_pollution = n_pollution
        self.n_train = len(self.train_ds)
        self.n_test = len(self.test_ds)

        print(
            f"[CIFAR-10] normal_class={normal_class} | "
            f"train_normals={self.n_normals} | pollution={self.n_pollution} "
            f"({self.pollution_rate:.1%}) | total_train={self.n_train} | test={self.n_test}"
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        assert self.train_ds is not None, "Call setup() first."
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup() first."
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
        loader = self.train_dataloader(shuffle=False)
        normals: List[torch.Tensor] = []

        for x, y in loader:
            mask = (y == 0)
            if mask.any():
                normals.append(x[mask])

        if not normals:
            raise RuntimeError("No normal samples found in the training set.")

        return torch.cat(normals, dim=0).to(device)


@torch.no_grad()
def encode_dataset(ae, loader: DataLoader, device: torch.device) -> TensorDataset:
    ae.eval()
    all_z: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        z = ae.encode(x)
        all_z.append(z.detach().cpu())
        all_y.append(y.detach().cpu())

    return TensorDataset(torch.cat(all_z, dim=0), torch.cat(all_y, dim=0))


def test_transform():
    import matplotlib.pyplot as plt

    ds = datasets.CIFAR10(root="./data", train=True, download=True)
    img_pil, label = ds[0]

    gcn = GCN_L1_Transform()
    simple = SimpleTransform()

    img_gcn = gcn(img_pil)
    img_simple = simple(img_pil)

    print(f"Label: {label}")
    print(f"GCN: shape={img_gcn.shape}, min={img_gcn.min():.4f}, max={img_gcn.max():.4f}, mean={img_gcn.mean():.4f}")
    print(f"Simple: shape={img_simple.shape}, min={img_simple.min():.4f}, max={img_simple.max():.4f}, mean={img_simple.mean():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_pil); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(img_gcn.permute(1, 2, 0)); axes[1].set_title("GCN (L1)"); axes[1].axis("off")
    axes[2].imshow(img_simple.permute(1, 2, 0)); axes[2].set_title("ToTensor"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("transform_comparison.png")
    print("Saved transform_comparison.png")


if __name__ == "__main__":
    test_transform()

    dm = CIFAR10OneClassDataModule(
        data_dir="./data",
        normal_class=1,
        pollution_rate=0.1,
        batch_size=32,
        num_workers=0,
        seed=42,
        download=True,
        use_gcn=True,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    x, y = next(iter(train_loader))
    print(f"\nTrain batch: x.shape={x.shape}, y.shape={y.shape}")
    print(f"Labels in batch: {y.unique().tolist()}")
    print(f"x range: [{x.min():.4f}, {x.max():.4f}]")

    x_test, y_test = next(iter(test_loader))
    print(f"\nTest batch: x.shape={x_test.shape}, y.shape={y_test.shape}")
    print(f"Labels in batch: {y_test.unique().tolist()}")

    normals = dm.collect_normal_images(device=torch.device("cpu"))
    print(f"\nCollected normals: shape={normals.shape}")
    print(f"Expected: ({dm.n_normals}, 3, 32, 32)")


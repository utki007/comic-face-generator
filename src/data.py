from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .config import NUM_WORKERS


class Face2ComicDataset(Dataset):
    def __init__(self, real, comic):
        self.real = real
        self.comic = comic

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        real_img = self.real[idx].astype(np.float32, copy=True)
        comic_img = self.comic[idx].astype(np.float32, copy=True)
        return torch.from_numpy(real_img), torch.from_numpy(comic_img)


def load_data(data_dir: str = "../data/npy"):
    """Load all npy arrays as memory-mapped files."""
    data_dir = Path(data_dir)
    return {
        "train_real": np.load(data_dir / "train_real.npy", mmap_mode="r"),
        "train_comic": np.load(data_dir / "train_comic.npy", mmap_mode="r"),
        "val_real": np.load(data_dir / "val_real.npy", mmap_mode="r"),
        "val_comic": np.load(data_dir / "val_comic.npy", mmap_mode="r"),
        "test_real": np.load(data_dir / "test_real.npy", mmap_mode="r"),
        "test_comic": np.load(data_dir / "test_comic.npy", mmap_mode="r"),
    }


def make_datasets(data: dict, train_tuning_samples: int = 2000, val_tuning_samples: int = 500):
    """Create full datasets and tuning subsets from loaded data arrays."""
    train_dataset = Face2ComicDataset(data["train_real"], data["train_comic"])
    full_val_dataset = Face2ComicDataset(data["val_real"], data["val_comic"])
    full_test_dataset = Face2ComicDataset(data["test_real"], data["test_comic"])

    train_tuning_dataset = Subset(
        train_dataset, range(min(train_tuning_samples, len(train_dataset)))
    )
    val_tuning_dataset = Subset(
        full_val_dataset, range(min(val_tuning_samples, len(full_val_dataset)))
    )

    return {
        "train": train_dataset,
        "val": full_val_dataset,
        "test": full_test_dataset,
        "train_tuning": train_tuning_dataset,
        "val_tuning": val_tuning_dataset,
    }


def make_loaders(batch_size: int, train_ds, val_ds, device=None, num_workers: int = NUM_WORKERS):
    """Create train and validation DataLoaders."""
    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader

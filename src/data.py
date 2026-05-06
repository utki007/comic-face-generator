from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import NUM_WORKERS

IMAGE_SIZE = 256


def preprocess_for_inference(
    image_or_path: Union[str, Path, Image.Image],
    target_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Load a real-life image and return a (3, target_size, target_size) tensor in [-1, 1].

    Pipeline: center-crop square -> resize -> normalize.
    This avoids aspect-ratio distortion on non-square inputs.

    Args:
        image_or_path: Path to the input image or a PIL image.
        target_size: Output spatial resolution (square).
    """
    if isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
    else:
        img = Image.open(image_or_path).convert("RGB")
    img_np = np.asarray(img)
    h, w = img_np.shape[:2]

    # Center-crop to the largest square.
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize((target_size, target_size), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


class Face2ComicDataset(Dataset):
    def __init__(self, real, comic, augment=None):
        self.real = real
        self.comic = comic
        self.augment = augment

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        real_img = self.real[idx].astype(np.float32, copy=True)
        comic_img = self.comic[idx].astype(np.float32, copy=True)

        if self.augment is not None:
            # Albumentations pipeline expects HWC images in [0, 255].
            real_hwc = (((real_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)).transpose(1, 2, 0)
            comic_hwc = (((comic_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)).transpose(1, 2, 0)
            out = self.augment(image=real_hwc, image0=comic_hwc)
            return out["image"], out["image0"]

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


def make_train_augment():
    return A.Compose(
        [
            # 1. GEOMETRIC (Must be shared between Real and Comic)
            A.SmallestMaxSize(max_size=286),
            A.RandomCrop(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),

            # 2. PHOTOMETRIC (Handling the "Real-Life" lighting/noise)
            # Increased brightness/contrast to handle your classroom lighting
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            
            # Simulated Phone Camera Noise (The fix for "muddy" textures)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.4),

            # Simulated Focus/Blur issues
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),

            # 3. DOMAIN ROBUSTNESS (The fix for background "bleeding")
            # This forces the model to ignore random parts of the background
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),

            # 4. NORMALIZATION
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )


def make_datasets(
    data: dict,
    train_tuning_samples: int = 2000,
    val_tuning_samples: int = 500,
    use_train_augmentation: bool = False,
):
    """Create full datasets and tuning subsets from loaded data arrays."""
    train_augment = make_train_augment() if use_train_augmentation else None
    train_dataset = Face2ComicDataset(
        data["train_real"], data["train_comic"], augment=train_augment
    )
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

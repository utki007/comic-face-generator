"""Paired face-to-comic dataset for pix2pix training."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceComicDataset(Dataset):
    """Loads aligned (real, comic) image pairs for supervised pix2pix training.

    Both images are resized to ``image_size x image_size`` and normalised to [-1, 1].
    Filenames in *real_dir* and *comic_dir* must match (e.g. ``0001.jpg`` <-> ``0001.jpg``).
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, real_dir, comic_dir, image_size=256):
        super().__init__()
        self.real_dir = Path(real_dir)
        self.comic_dir = Path(comic_dir)

        real_files = sorted(
            f for f in self.real_dir.iterdir()
            if f.suffix.lower() in self.IMG_EXTENSIONS
        )

        self.pairs = []
        for rf in real_files:
            cf = self.comic_dir / rf.name
            if cf.exists():
                self.pairs.append((rf, cf))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matching image pairs found in\n"
                f"  real:  {self.real_dir}\n"
                f"  comic: {self.comic_dir}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_path, comic_path = self.pairs[idx]
        real_img = Image.open(real_path).convert("RGB")
        comic_img = Image.open(comic_path).convert("RGB")
        return self.transform(real_img), self.transform(comic_img)

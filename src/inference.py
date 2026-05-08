from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .checkpoints import load_checkpoint
from .config import get_device
from .data import preprocess_for_inference
from .models import Discriminator, Generator
from .visualization import denorm


PresetName = Literal["soft", "balanced", "strong"]


@dataclass(frozen=True)
class EnhancementPreset:
    in_contrast: float
    in_sharpness: float
    out_color: float
    out_contrast: float
    unsharp_radius: float
    unsharp_percent: int
    unsharp_threshold: int


PRESETS: dict[PresetName, EnhancementPreset] = {
    "soft": EnhancementPreset(
        in_contrast=1.06,
        in_sharpness=1.08,
        out_color=1.03,
        out_contrast=1.05,
        unsharp_radius=1.0,
        unsharp_percent=90,
        unsharp_threshold=4,
    ),
    "balanced": EnhancementPreset(
        in_contrast=1.12,
        in_sharpness=1.15,
        out_color=1.08,
        out_contrast=1.10,
        unsharp_radius=1.2,
        unsharp_percent=130,
        unsharp_threshold=3,
    ),
    "strong": EnhancementPreset(
        in_contrast=1.18,
        in_sharpness=1.25,
        out_color=1.15,
        out_contrast=1.20,
        unsharp_radius=1.4,
        unsharp_percent=170,
        unsharp_threshold=2,
    ),
}


def enhance_input_for_model(img: Image.Image, preset: PresetName) -> Image.Image:
    """Mild pre-enhancement to make real photos closer to training-style contrast/sharpness."""
    cfg = PRESETS[preset]
    out = img.convert("RGB")
    out = ImageOps.autocontrast(out, cutoff=1)
    out = ImageEnhance.Contrast(out).enhance(cfg.in_contrast)
    out = ImageEnhance.Sharpness(out).enhance(cfg.in_sharpness)
    return out


def enhance_generated_comic(img: Image.Image, preset: PresetName) -> Image.Image:
    """Post-enhancement to make generated comic edges/details pop a bit more."""
    cfg = PRESETS[preset]
    out = img.convert("RGB")
    out = ImageEnhance.Color(out).enhance(cfg.out_color)
    out = ImageEnhance.Contrast(out).enhance(cfg.out_contrast)
    out = out.filter(
        ImageFilter.UnsharpMask(
            radius=cfg.unsharp_radius,
            percent=cfg.unsharp_percent,
            threshold=cfg.unsharp_threshold,
        )
    )
    return out


def load_model_bundle(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> Tuple[torch.device, Generator]:
    """Load Generator weights from a training checkpoint and return (device, generator).

    Note: the current checkpoint format includes discriminator weights, and
    `load_checkpoint()` expects a discriminator instance.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device() if device is None else device

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    load_checkpoint(checkpoint_path, gen, disc, device=device)

    gen.eval()
    return device, gen


def _tensor_to_pil_rgb_u8(x01: torch.Tensor) -> Image.Image:
    """Convert a (3,H,W) tensor in [0,1] to a PIL RGB image."""
    arr = (
        x01.clamp(0, 1)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")


def run_inference(
    pil_img: Image.Image,
    preset: PresetName,
    gen: Generator,
    device: torch.device,
    use_enhancement: bool = True,
) -> Image.Image:
    """Run Face2Comic inference and return the final enhanced comic as a PIL image."""
    if pil_img is None:
        raise ValueError("No image provided")

    model_input = (
        enhance_input_for_model(pil_img, preset) if use_enhancement else pil_img.convert("RGB")
    )

    # preprocess_for_inference returns a (3,256,256) tensor in [-1,1]
    x = preprocess_for_inference(model_input).unsqueeze(0).to(device)

    with torch.inference_mode():
        fake = gen(x).squeeze(0)

    comic_01 = denorm(fake)
    comic_pil = _tensor_to_pil_rgb_u8(comic_01)

    if not use_enhancement:
        return comic_pil
    comic_enhanced = enhance_generated_comic(comic_pil, preset)
    return comic_enhanced

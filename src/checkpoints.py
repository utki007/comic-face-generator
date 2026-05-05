from pathlib import Path

import torch


def save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, metrics, checkpoint_dir, name="pix2pix"):
    """Save a training checkpoint with model weights, optimizer states, and metrics."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "metrics": metrics,
    }
    path = checkpoint_dir / f"{name}_epoch_{epoch:03d}.pt"
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, gen, disc, opt_gen=None, opt_disc=None, device=None):
    """Load a training checkpoint into existing model/optimizer instances."""
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    gen.load_state_dict(ckpt["gen_state_dict"])
    disc.load_state_dict(ckpt["disc_state_dict"])
    if opt_gen is not None:
        opt_gen.load_state_dict(ckpt["opt_gen_state_dict"])
    if opt_disc is not None:
        opt_disc.load_state_dict(ckpt["opt_disc_state_dict"])
    return ckpt["epoch"], ckpt.get("metrics", {})

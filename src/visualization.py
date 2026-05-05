from pathlib import Path

import torch
import matplotlib.pyplot as plt


def denorm(x):
    """Map tensor from [-1, 1] to [0, 1] for display."""
    return ((x + 1) / 2).clamp(0, 1)


def save_grid_samples(gen, dataset, config_idx, output_dir, device, sample_indices=None):
    """Save a comparison grid (real | generated | ground truth) as a PNG."""
    gen.eval()
    default_indices = [9, 10, 12]
    sample_indices = default_indices if sample_indices is None else sample_indices
    valid_indices = [idx for idx in sample_indices if 0 <= idx < len(dataset)]
    if not valid_indices:
        valid_indices = list(range(min(3, len(dataset))))

    fig, axes = plt.subplots(len(valid_indices), 3, figsize=(9, 3 * len(valid_indices)), squeeze=False)
    axes[0, 0].set_title("Real Face")
    axes[0, 1].set_title("Generated")
    axes[0, 2].set_title("Ground Truth")

    with torch.no_grad():
        for row, idx in enumerate(valid_indices):
            real, target = dataset[idx]
            real_gpu = real.unsqueeze(0).to(device)
            fake = gen(real_gpu).squeeze(0).cpu()

            axes[row, 0].imshow(denorm(real).permute(1, 2, 0).numpy())
            axes[row, 1].imshow(denorm(fake).permute(1, 2, 0).numpy())
            axes[row, 2].imshow(denorm(target).permute(1, 2, 0).numpy())

            for j in range(3):
                axes[row, j].axis("off")

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"config_{config_idx:02d}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_training_curves(history, best_epoch, best_val, output_dir):
    """Plot G/D loss, validation L1, and LR schedule. Saves to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # G loss and D loss
    ax1 = axes[0]
    ax1.plot(epochs, history["g_loss"], label="G Loss", color="tab:blue")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, history["d_loss"], label="D Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Generator Loss", color="tab:blue")
    ax1_twin.set_ylabel("Discriminator Loss", color="tab:orange")
    ax1.set_title("Generator vs Discriminator Loss")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Validation L1
    ax2 = axes[1]
    ax2.plot(epochs, history["val_l1"], color="tab:green")
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    ax2.annotate(
        f"Best: {best_val:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_val),
        fontsize=9,
        color="red",
        ha="left",
        va="bottom",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation L1")
    ax2.set_title("Validation L1 Loss")

    # Learning rate
    ax3 = axes[2]
    ax3.plot(epochs, history["lr"], color="tab:purple")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("LR Schedule (linear decay)")

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=100)
    plt.show()
    plt.close(fig)

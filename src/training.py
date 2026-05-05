import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .models import Generator, Discriminator, init_weights

bce = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


def make_models(lr: float, betas=(0.5, 0.999), device=None):
    """Build generator + discriminator with optimizers, apply weight init."""
    if device is None:
        device = torch.device("cpu")
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    return gen, disc, opt_gen, opt_disc


def train_one_epoch(gen, disc, loader, opt_gen, opt_disc, lambda_l1, device):
    """Run one training epoch; returns (avg_g_loss, avg_d_loss)."""
    gen.train()
    disc.train()

    total_g, total_d = 0.0, 0.0
    loop = tqdm(loader, leave=True)

    for real, target in loop:
        real = real.to(device)
        target = target.to(device)

        fake = gen(real)

        # Train Discriminator
        pred_real = disc(real, target)
        loss_real = bce(pred_real, torch.ones_like(pred_real))

        pred_fake = disc(real, fake.detach())
        loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))

        loss_d = (loss_real + loss_fake) / 2

        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # Train Generator
        pred_fake = disc(real, fake)
        loss_g_gan = bce(pred_fake, torch.ones_like(pred_fake))
        loss_g_l1 = l1_loss(fake, target)

        loss_g = loss_g_gan + lambda_l1 * loss_g_l1

        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        total_g += loss_g.item()
        total_d += loss_d.item()

        loop.set_postfix(G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}")

    return total_g / len(loader), total_d / len(loader)


def validate(gen, loader, device):
    """Run validation pass; returns average L1 loss."""
    gen.eval()
    total_l1 = 0.0

    with torch.no_grad():
        for real, target in loader:
            real = real.to(device)
            target = target.to(device)
            fake = gen(real)
            total_l1 += l1_loss(fake, target).item()

    return total_l1 / len(loader)

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.d1 = UNetDown(in_ch, 64, normalize=False)
        self.d2 = UNetDown(64, 128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512)
        self.d5 = UNetDown(512, 512)
        self.d6 = UNetDown(512, 512)

        self.u1 = UNetUp(512, 512, dropout=True)
        self.u2 = UNetUp(1024, 512, dropout=True)
        self.u3 = UNetUp(1024, 256)
        self.u4 = UNetUp(512, 128)
        self.u5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        u1 = self.u1(d6, d5)
        u2 = self.u2(u1, d4)
        u3 = self.u3(u2, d3)
        u4 = self.u4(u3, d2)
        u5 = self.u5(u4, d1)

        return self.final(u5)


class Discriminator(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, real_input, target_or_fake):
        x = torch.cat([real_input, target_or_fake], dim=1)
        return self.model(x)


def init_weights(m):
    """Standard pix2pix weight initialization: N(0, 0.02) for conv, N(1, 0.02) for BN."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
        )


class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.down1 = Block(3, base)
        self.down2 = Block(base, base*2)
        self.down3 = Block(base*2, base*4)
        self.bridge = Block(base*4, base*8)
        self.up3 = Block(base*12, base*4)
        self.up2 = Block(base*6, base*2)
        self.up1 = Block(base*3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.down3(F.max_pool2d(b, 2))
        z = self.bridge(F.max_pool2d(c, 2))
        for skip, block in [(c, self.up3), (b, self.up2), (a, self.up1)]:
            z = block(torch.cat([F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], dim=1))
        return self.head(z)


def segmentation_loss(logits, target):
    # Per-batch foreground balance is capped to avoid extreme weights on tiny targets.
    count = target.sum().detach()
    weight = ((target.numel()-count)/count.clamp_min(1)).clamp(1, 25)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2*(probability*target).sum(dims)+1)/(probability.sum(dims)+target.sum(dims)+1)
    nonempty = target.sum(dims) > 0
    if nonempty.any():
        return bce + (1-dice[nonempty]).mean()
    return bce

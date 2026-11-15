"""Standard sequential channel and spatial attention (CBAM)."""
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2  # 'same' padding for odd kernels

        # bias=False bc BatchNorm2d will handle bias
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        bottlneck = channels // 2
        self.block = nn.Sequential(
            ConvBlock(channels, bottlneck, kernel_size=1,
                      stride=1, padding=None),
            ConvBlock(bottlneck, channels, kernel_size=3,
                      stride=1, padding=None)
        )

    def forward(self, x):
        return x + self.block(x)  # skip conncetion

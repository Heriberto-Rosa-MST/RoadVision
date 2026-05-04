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
    
class DarknetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(self.downsample(x))
    
class RoadVisionDarknetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial stem
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=2)
        self.stage1 = DarknetStage(32, 64, num_blocks=1)
        self.stage2 = DarknetStage(64, 128, num_blocks=2)
        self.stage3 = DarknetStage(128, 256, num_blocks=8)
        self.stage4 = DarknetStage(256, 512, num_blocks=8)

        # weight initialization
        self._initialize_weights()

    # Kaiming He initialization for conv layers, BatchNorm weights to 1 and bias to 0
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)      # (B, 32, 256, 256)
        x = self.stage1(x)    # (B, 64, 128, 128)
        p3 = self.stage2(x)   # (B, 128, 64, 64)
        p4 = self.stage3(p3)  # (B, 256, 32, 32)
        p5 = self.stage4(p4)  # (B, 512, 16, 16)
        return p3, p4, p5
        
        # check
if __name__ == "__main__":
    backbone = RoadVisionDarknetBackbone()

    # count trainable parameters
    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters in RoadVisionDarknetBackbone: {total_params:,}")

    # verify output shapes with a dummy input
    dummy = torch.zeros(2, 3, 512, 512)  # batch of 2 images
    p3, p4, p5 = backbone(dummy)
    print(f"P3 shape: {p3.shape}")  # Expected: (2, 128, 64, 64)
    print(f"P4 shape: {p4.shape}")  # Expected: (2, 256, 32, 32)
    print(f"P5 shape: {p5.shape}")  # Expected: (2, 512, 16, 16)
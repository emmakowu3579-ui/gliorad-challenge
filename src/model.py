"""
U-Net architecture for TCGA-LGG Brain MRI Glioma Segmentation.
Classic symmetric encoder-decoder with skip connections and batch normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with Skip Connection concatenation"""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for 2D medical image segmentation.
    Input: (B, in_channels, H, W)
    Output: (B, num_classes, H, W) raw logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        init_features: int = 32,
        use_batchnorm: bool = True,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        f = init_features
        self.inc = DoubleConv(in_channels, f, use_batchnorm=use_batchnorm)
        self.down1 = Down(f, f * 2, use_batchnorm=use_batchnorm)
        self.down2 = Down(f * 2, f * 4, use_batchnorm=use_batchnorm)
        self.down3 = Down(f * 4, f * 8, use_batchnorm=use_batchnorm)
        self.down4 = Down(f * 8, f * 16, use_batchnorm=use_batchnorm)

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.up1 = Up(f * 16, f * 8, use_batchnorm=use_batchnorm)
        self.up2 = Up(f * 8, f * 4, use_batchnorm=use_batchnorm)
        self.up3 = Up(f * 4, f * 2, use_batchnorm=use_batchnorm)
        self.up4 = Up(f * 2, f, use_batchnorm=use_batchnorm)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=1, init_features=16)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape} -> Output logits: {out.shape}")

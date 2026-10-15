"""Model definitions matching the supplied trained checkpoints."""
import torch
from torch import nn
from torchvision import models


def build_classifier(num_classes=4):
    # Checkpoint keys confirm torchvision ResNet-18 structure.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ConvBlock3D(nn.Module):
    """Exact convolution block used by the trained BraTS checkpoint."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UnifiedMRI3D(nn.Module):
    """Verified 3D U-Net-like architecture for the supplied segmentation checkpoint."""
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.seg_head = nn.Conv3d(base, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base * 8, num_classes),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.seg_head(d1), self.classifier(b)

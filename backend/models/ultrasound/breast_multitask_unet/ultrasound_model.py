import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = {
    0: "benign",
    1: "malignant",
    2: "normal",
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class MultiTaskUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(1, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = ConvBlock(
            256,
            512
        )

        # Decoder
        self.dec4 = DecoderBlock(
            512,
            256,
            256
        )

        self.dec3 = DecoderBlock(
            256,
            128,
            128
        )

        self.dec2 = DecoderBlock(
            128,
            64,
            64
        )

        self.dec1 = DecoderBlock(
            64,
            32,
            32
        )

        # Segmentation head
        self.segmentation_head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(
                512,
                256
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.dec4(
            b,
            e4
        )

        d3 = self.dec3(
            d4,
            e3
        )

        d2 = self.dec2(
            d3,
            e2
        )

        d1 = self.dec1(
            d2,
            e1
        )

        # Segmentation output
        segmentation = self.segmentation_head(
            d1
        )

        # Classification output
        pooled = F.adaptive_avg_pool2d(
            b,
            output_size=1
        )

        pooled = torch.flatten(
            pooled,
            1
        )

        classification = self.classifier(
            pooled
        )

        return classification, segmentation


def load_ultrasound_model(
    checkpoint_path,
    device=None
):
    """
    Load the trained Breast Ultrasound
    Multi-Task U-Net checkpoint.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = MultiTaskUNet(
        num_classes=3
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model
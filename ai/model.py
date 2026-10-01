import torch
import torch.nn as nn
from torchvision import models


class MedicalCNN(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        # ResNet-18 is our CNN architecture
        self.model = models.resnet18(weights=None)

        # Replace ResNet's final layer
        # with a layer for our medical image classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# Test the CNN when this file is run directly
if __name__ == "__main__":

    num_classes = 3

    model = MedicalCNN(num_classes)

    print(model)
    print(f"Number of classes: {num_classes}")
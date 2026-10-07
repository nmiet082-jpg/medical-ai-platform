import torch
from torchvision import models
import torch.nn as nn

model_path = "backend/models/resnet18_medical_3class_best.pth"

print("Creating ResNet18 model...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

print("Loading trained weights...")

state = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()

print("MODEL LOAD SUCCESSFUL")
print("Classes:")
print("1. Pneumonia")
print("2. Cardiomegaly")
print("3. Pneumothorax")
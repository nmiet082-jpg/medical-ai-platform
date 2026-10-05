from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "resnet18_medical_3class_best.pth"
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

CLASS_NAMES = [
    "Pneumonia",
    "Cardiomegaly",
    "Pneumothorax",
]

THRESHOLDS = {
    "Pneumonia": 0.3,
    "Cardiomegaly": 0.3,
    "Pneumothorax": 0.2,
}


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# Load once when this module is imported
model = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_path: str):

    image = Image.open(image_path).convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)[0]

    results = []

    for index, class_name in enumerate(CLASS_NAMES):

        probability = float(probabilities[index])

        threshold = THRESHOLDS[class_name]

        detected = probability >= threshold

        results.append({
            "class_name": class_name,
            "probability": round(probability, 4),
            "threshold": threshold,
            "flagged": detected,
        })

    return results
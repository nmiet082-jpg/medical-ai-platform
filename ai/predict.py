
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "xray_six_class"
)

MODEL_PATH = MODEL_DIR / "six_class_classifier_best.pt"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"


# Keep this order exactly aligned with the trained checkpoint.
CLASS_NAMES = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Pneumonia",
    "Normal",
    "Fracture",
]


# --------------------------------------------------
# IMAGE PREPROCESSING
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
# LOAD CLASS-SPECIFIC THRESHOLDS
# --------------------------------------------------

def load_thresholds():
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLDS_PATH}"
        )

    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as file:
        thresholds = json.load(file)

    for class_name in CLASS_NAMES:
        if class_name not in thresholds:
            raise ValueError(
                f"Missing threshold for class: {class_name}"
            )

        value = float(thresholds[class_name])

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid threshold for {class_name}: {value}"
            )

        thresholds[class_name] = value

    return thresholds


THRESHOLDS = load_thresholds()


# --------------------------------------------------
# BUILD AND LOAD THE SIX-CLASS RESNET18
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Six-class model checkpoint not found: {MODEL_PATH}"
        )

    # This checkpoint is expected to contain:
    # "labels" and "model_state_dict".
    # Only load a checkpoint from a trusted source.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected checkpoint format: expected a dictionary."
        )

    checkpoint_labels = checkpoint.get("labels")

    if checkpoint_labels is not None:
        if list(checkpoint_labels) != CLASS_NAMES:
            raise ValueError(
                "Checkpoint labels do not match CLASS_NAMES. "
                f"Checkpoint labels: {checkpoint_labels}"
            )

    state_dict = checkpoint.get("model_state_dict")

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint is missing the model_state_dict entry."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# PREDICT AN X-RAY IMAGE
# --------------------------------------------------

def predict_image(image_path: str):
    """
    Return six research-model scores.

    Scores are independent sigmoid outputs, not a
    mutually exclusive softmax distribution.

    Normal is never flagged as an abnormal finding.
    """

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"X-ray image not found: {path}"
        )

    with Image.open(path) as source_image:
        image = source_image.convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0]

    results = []

    for index, class_name in enumerate(CLASS_NAMES):
        probability = float(probabilities[index])
        threshold = THRESHOLDS[class_name]

        is_normal = class_name == "Normal"

        results.append({
            "class_name": class_name,
            "probability": round(probability, 4),
            "threshold": threshold,
            "flagged": (
                probability >= threshold
                and not is_normal
            ),
        })

    return results

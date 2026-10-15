
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultrasound_model import (
    CLASS_NAMES,
    load_ultrasound_model,
)


MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = MODEL_DIR / "multitask_unet_best.pth"

IMAGE_SIZE = (256, 256)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_model = None


def get_ultrasound_model():
    """Load the model once and reuse it for future requests."""
    global _model

    if _model is None:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Ultrasound checkpoint not found: {CHECKPOINT_PATH}"
            )

        _model = load_ultrasound_model(
            CHECKPOINT_PATH,
            device=DEVICE,
        )

    return _model


def preprocess_ultrasound_image(image_path: str):
    """
    Convert an image to a single-channel 256x256 tensor.

    IMPORTANT:
    Confirm the original training preprocessing before relying
    on model predictions. This is a provisional image conversion.
    """
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    tensor = torch.from_numpy(
        image_array.copy()
    ).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


def predict_breast_ultrasound(image_path: str):
    """Return classification scores and a segmentation mask."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Ultrasound image not found: {path}"
        )

    model = get_ultrasound_model()
    image_tensor = preprocess_ultrasound_image(str(path))

    with torch.inference_mode():
        classification_logits, segmentation_logits = model(
            image_tensor
        )

        class_probabilities = torch.softmax(
            classification_logits,
            dim=1,
        )[0]

        predicted_class_id = int(
            torch.argmax(class_probabilities).item()
        )

        segmentation_probabilities = torch.sigmoid(
            segmentation_logits
        )[0, 0]

    segmentation_mask = (
        segmentation_probabilities >= 0.5
    ).to(torch.uint8).cpu().numpy()

    return {
        "model_name": "Breast Ultrasound Multi-Task U-Net",
        "model_checkpoint": CHECKPOINT_PATH.name,
        "predicted_class_id": predicted_class_id,
        "predicted_class": CLASS_NAMES[predicted_class_id],
        "class_probabilities": {
            CLASS_NAMES[index]: round(
                float(class_probabilities[index].item()), 4
            )
            for index in CLASS_NAMES
        },
        "segmentation_mask": segmentation_mask,
        "segmentation_threshold": 0.5,
        "review_note": (
            "Research output only; not a standalone medical diagnosis. "
            "Professional review is required."
        ),
    }

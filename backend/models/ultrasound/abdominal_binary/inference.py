from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_IDX = {"abnormal": 0, "normal": 1}


class AbdominalUltrasoundClassifier:
    """Research-only binary abdominal ultrasound image classifier."""

    def __init__(self, checkpoint_path=None, device=None):
        module_dir = Path(__file__).resolve().parent
        self.checkpoint_path = Path(
            checkpoint_path or module_dir / "best_model.pt"
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.input_size = int(checkpoint["input_size"])
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {
            int(index): label
            for label, index in self.class_to_idx.items()
        }

        if checkpoint["architecture"] != "EfficientNet-B0":
            raise ValueError("Unsupported architecture in checkpoint.")

        if self.class_to_idx != CLASS_TO_IDX:
            raise ValueError("Unexpected class mapping in checkpoint.")

        self.model = models.efficientnet_b0(weights=None)
        features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(features, 2)

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        """Return predicted class and model probabilities for one image."""
        image_path = Path(image_path)

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Unsupported image format. Use JPG, JPEG, PNG, BMP, or WEBP."
            )

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as source:
                source.verify()

            with Image.open(image_path) as source:
                original_size = source.size
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid readable image.") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        predicted_index = int(probabilities.argmax().item())

        return {
            "predicted_class": self.idx_to_class[predicted_index],
            "probabilities": {
                "abnormal": float(probabilities[0]),
                "normal": float(probabilities[1]),
            },
            "input_size": list(original_size),
            "model_input_size": [self.input_size, self.input_size],
            "research_only": True,
            "clinical_diagnosis_supported": False,
        }

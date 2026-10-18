
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Reuse the exact six-class model and preprocessing used by prediction.
from ai.predict import model, CLASS_NAMES, IMAGE_TRANSFORM


MODEL_NAME = "ResNet18 Medical 6-Class"
MODEL_VERSION = "2.0"


class GradCAM:
    """Grad-CAM for the six-class ResNet18 classifier."""

    def __init__(self, classifier):
        # classifier is the full ResNet18 model.
        self.model = classifier
        self.activations = None
        self.gradients = None

        self.target_layer = self.model.layer4[-1]

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index):
        if not 0 <= class_index < len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class index: {class_index}"
            )

        self.model.zero_grad(set_to_none=True)

        # Do not use torch.no_grad() here; Grad-CAM needs gradients.
        output = self.model(image_tensor)
        target_score = output[0, class_index]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM could not capture model gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)[0].cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def generate_gradcam(image_path, class_index, output_path):
    """
    Generate a Grad-CAM overlay for a selected classifier class.

    This visualization is an AI explanation, not a diagnosis.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError(
            f"Invalid class index: {class_index}"
        )

    with Image.open(image_path) as source_image:
        original_image = source_image.convert("RGB")

    original_array = np.asarray(original_image)

    image_tensor = IMAGE_TRANSFORM(
        original_image
    ).unsqueeze(0)

    # Reuse the same model object used by predict_image().
    # The model has six outputs in CLASS_NAMES order.
    gradcam = GradCAM(model)

    try:
        model.eval()
        heatmap = gradcam.generate(
            image_tensor,
            class_index
        )
    finally:
        gradcam.close()
        model.zero_grad(set_to_none=True)

    heatmap = cv2.resize(
        heatmap,
        (original_array.shape[1], original_array.shape[0])
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0.0, 1.0) * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(overlay).save(output_path)

    return {
        "output_path": str(output_path),
        "class_name": CLASS_NAMES[class_index],
        "method": "Grad-CAM",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "clinical_note": (
            "AI explanation only. The heatmap does not "
            "establish a medical diagnosis."
        ),
    }


if __name__ == "__main__":
    print("XAI module loaded successfully.")
    print("Model:", MODEL_NAME)
    print("Supported classes:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")

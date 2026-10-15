from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(classifier, image_path, output_path=None):
    """
    Generate a Grad-CAM overlay for the classifier's predicted class.

    This is a model-attention visualization, not a lesion segmentation.
    """
    image_path = Path(image_path)

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    tensor = classifier.transform(original).unsqueeze(0).to(
        classifier.device
    )

    model = classifier.model
    model.eval()

    activations = []
    gradients = []
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(tensor)
            predicted_index = int(logits.argmax(dim=1).item())
            score = logits[0, predicted_index]
            score.backward()

        feature_maps = activations[0]
        feature_gradients = gradients[0]

        weights = feature_gradients.mean(
            dim=(2, 3), keepdim=True
        )
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)

        # Save the heatmap at the original image dimensions.
        heatmap_image = Image.fromarray(
            np.uint8(np.clip(cam, 0, 1) * 255)
        ).resize(original.size, Image.Resampling.BILINEAR)

        heatmap = np.asarray(heatmap_image).astype(np.float32) / 255.0

        # Create a colored overlay without adding another dependency.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("jet")
        colored_heatmap = cmap(heatmap)[..., :3]
        original_array = np.asarray(original).astype(np.float32) / 255.0

        overlay = np.clip(
            0.60 * original_array + 0.40 * colored_heatmap,
            0, 1
        )
        overlay_image = Image.fromarray(
            np.uint8(overlay * 255)
        )

        if output_path is None:
            output_path = image_path.with_name(
                image_path.stem + "_gradcam.png"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_image.save(output_path)

        probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu()

        return {
            "predicted_class": classifier.idx_to_class[predicted_index],
            "abnormal_probability": float(probabilities[0]),
            "normal_probability": float(probabilities[1]),
            "heatmap_path": str(output_path),
            "research_only": True,
            "interpretation_warning": (
                "Grad-CAM indicates model attention, not a verified lesion "
                "location or clinical diagnosis."
            ),
        }

    finally:
        handle_forward.remove()
        handle_backward.remove()
        model.zero_grad(set_to_none=True)

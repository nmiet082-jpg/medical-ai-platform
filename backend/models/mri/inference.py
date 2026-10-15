
"""Loading, prediction, Grad-CAM and 3D MRI segmentation."""

from pathlib import Path
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    # Package imports when running from the project root.
    from .model_architectures import build_classifier, UnifiedMRI3D
    from .preprocessing import CLASS_NAMES, classifier_tensor
except ImportError:
    # Fallback when running files directly from the MRI folder.
    from model_architectures import build_classifier, UnifiedMRI3D
    from preprocessing import CLASS_NAMES, classifier_tensor


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_models(models_dir, device=None):
    """Load the existing MRI classifier and segmentation checkpoints."""
    models_dir = Path(models_dir)

    if device is None or device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    classifier_path = models_dir / "classifier_best.pt"
    segmentation_path = models_dir / "segmentation_best_epoch10.pt"

    if not classifier_path.is_file():
        raise FileNotFoundError(
            f"MRI classifier checkpoint not found: {classifier_path}"
        )

    if not segmentation_path.is_file():
        raise FileNotFoundError(
            f"MRI segmentation checkpoint not found: {segmentation_path}"
        )

    classifier_ckpt = torch.load(
        classifier_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in classifier_ckpt:
        raise ValueError(
            "Classifier checkpoint is missing model_state_dict."
        )

    classifier = build_classifier(
        num_classes=len(classifier_ckpt.get("classes", CLASS_NAMES))
    )
    classifier.load_state_dict(
        classifier_ckpt["model_state_dict"],
        strict=True,
    )
    classifier.to(device).eval()

    seg_ckpt = torch.load(
        segmentation_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in seg_ckpt:
        raise ValueError(
            "Segmentation checkpoint is missing model_state_dict."
        )

    base_channels = int(seg_ckpt.get("base_channels", 16))

    segmentation = UnifiedMRI3D(
        in_channels=4,
        num_classes=4,
        base=base_channels,
    )
    segmentation.load_state_dict(
        seg_ckpt["model_state_dict"],
        strict=True,
    )
    segmentation.to(device).eval()

    return (
        classifier,
        segmentation,
        classifier_ckpt,
        seg_ckpt,
        device,
    )


# ---------------------------------------------------------
# 2D MRI CLASSIFICATION
# ---------------------------------------------------------

def classify_image(model, pil_image, device, image_size=224):
    """Classify a supported 2D MRI image."""
    x = classifier_tensor(pil_image, image_size).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_names = CLASS_NAMES

    if len(class_names) != len(probabilities):
        raise ValueError(
            "Classifier output count does not match CLASS_NAMES."
        )

    idx = int(np.argmax(probabilities))

    return {
        "predicted_class": class_names[idx],
        "score": float(probabilities[idx]),
        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        },
        "tensor": x,
        "class_index": idx,
    }


# ---------------------------------------------------------
# 2D CLASSIFIER GRAD-CAM
# ---------------------------------------------------------

def gradcam_overlay(model, input_tensor, class_index, pil_image, device):
    """Generate a 2D ResNet Grad-CAM overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    activations = {}
    gradients = {}
    target_layer = model.layer4[-1]

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        x = input_tensor.to(device)
        logits = model(x)

        if not 0 <= class_index < logits.shape[1]:
            raise ValueError("Invalid Grad-CAM class index.")

        logits[0, class_index].backward()

        acts = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * acts).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        base = np.asarray(
            pil_image.resize((224, 224)).convert("RGB")
        ).astype(np.float32)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(base.astype(np.uint8))
        ax.imshow(cam, cmap="jet", alpha=0.38, vmin=0, vmax=1)
        ax.axis("off")
        fig.tight_layout(pad=0)

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        buffer.seek(0)
        with Image.open(buffer) as result:
            return result.convert("RGB").copy()

    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)


# ---------------------------------------------------------
# 3D SLIDING-WINDOW MRI SEGMENTATION
# ---------------------------------------------------------

def _window_starts(length, window, overlap):
    """Generate window starts and ensure the final edge is covered."""
    stride = max(int(round(window * (1.0 - overlap))), 1)

    if length <= window:
        return [0]

    starts = list(range(0, length - window + 1, stride))
    final_start = length - window

    if starts[-1] != final_start:
        starts.append(final_start)

    return sorted(set(starts))


def segment_volume(
    model,
    image_tensor,
    device,
    crop_size=(32, 96, 96),
    overlap=0.5,
):
    """Segment a 3D MRI tensor using overlapping windows.

    Args:
        model: Verified UnifiedMRI3D segmentation model.
        image_tensor: torch.Tensor with shape (4, D, H, W).
        device: torch device or device string.
        crop_size: Trained model window size (D, H, W).
        overlap: Window overlap fraction, from 0 inclusive to 1 exclusive.

    Returns:
        uint8 NumPy array of class labels, shape (D, H, W).

    Important:
        The input must already be loaded, spatially aligned and normalized.
        This function reconstructs a mask over the supplied tensor's grid;
        it does not restore original NIfTI geometry or establish clinical
        validity.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor.")

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 4:
        raise ValueError(
            "Expected image_tensor shape (4, D, H, W); "
            f"received {tuple(image_tensor.shape)}."
        )

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be >= 0 and < 1.")

    crop_size = tuple(int(v) for v in crop_size)

    if len(crop_size) != 3 or any(v <= 0 for v in crop_size):
        raise ValueError(
            "crop_size must contain three positive integers."
        )

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    original_d, original_h, original_w = image_tensor.shape[1:]

    if min(original_d, original_h, original_w) <= 0:
        raise ValueError("Input MRI volume has an empty dimension.")

    image = image_tensor.detach().to(
        device=device,
        dtype=torch.float32,
    )

    if not torch.isfinite(image).all():
        raise ValueError("MRI tensor contains NaN or infinite values.")

    # Pad small dimensions so at least one complete window fits.
    pad_d = max(crop_size[0] - original_d, 0)
    pad_h = max(crop_size[1] - original_h, 0)
    pad_w = max(crop_size[2] - original_w, 0)

    pad_left_d = pad_d // 2
    pad_left_h = pad_h // 2
    pad_left_w = pad_w // 2

    if pad_d or pad_h or pad_w:
        image = F.pad(
            image,
            (
                pad_left_w, pad_w - pad_left_w,
                pad_left_h, pad_h - pad_left_h,
                pad_left_d, pad_d - pad_left_d,
            ),
            mode="constant",
            value=0.0,
        )

    _, depth, height, width = image.shape

    d_starts = _window_starts(depth, crop_size[0], overlap)
    h_starts = _window_starts(height, crop_size[1], overlap)
    w_starts = _window_starts(width, crop_size[2], overlap)

    probability_sum = torch.zeros(
        (4, depth, height, width),
        device=device,
        dtype=torch.float32,
    )

    count_sum = torch.zeros(
        (1, depth, height, width),
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = image[
                        :,
                        d0:d0 + crop_size[0],
                        h0:h0 + crop_size[1],
                        w0:w0 + crop_size[2],
                    ].unsqueeze(0)

                    output = model(patch)
                    logits = (
                        output[0]
                        if isinstance(output, (tuple, list))
                        else output
                    )

                    expected_shape = (1, 4, *crop_size)

                    if tuple(logits.shape) != expected_shape:
                        raise ValueError(
                            "Unexpected segmentation output shape: "
                            f"{tuple(logits.shape)}; expected "
                            f"{expected_shape}."
                        )

                    probabilities = torch.softmax(
                        logits.float(), dim=1
                    )[0]

                    region = (
                        slice(d0, d0 + crop_size[0]),
                        slice(h0, h0 + crop_size[1]),
                        slice(w0, w0 + crop_size[2]),
                    )

                    probability_sum[(slice(None), *region)] += probabilities
                    count_sum[(slice(None), *region)] += 1.0

    if torch.any(count_sum == 0):
        raise RuntimeError(
            "Some voxels were not covered by sliding windows."
        )

    mean_probabilities = probability_sum / count_sum
    prediction = mean_probabilities.argmax(dim=0)

    # Remove only the padding added above.
    prediction = prediction[
        pad_left_d:pad_left_d + original_d,
        pad_left_h:pad_left_h + original_h,
        pad_left_w:pad_left_w + original_w,
    ]

    return prediction.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------
# SEGMENTATION OUTPUT SUMMARY
# ---------------------------------------------------------

def segmentation_summary(prediction):
    """Return voxel counts for the four remapped BraTS labels."""
    prediction = np.asarray(prediction)

    valid_labels = {0, 1, 2, 3}
    found_labels = set(np.unique(prediction).tolist())

    if not found_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected segmentation labels: {sorted(found_labels)}"
        )

    return {
        "shape": tuple(prediction.shape),
        "voxel_counts": {
            class_id: int(np.count_nonzero(prediction == class_id))
            for class_id in range(4)
        },
        "tumor_voxels": int(np.count_nonzero(prediction > 0)),
        "valid_labels": True,
    }

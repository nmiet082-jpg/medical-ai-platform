"""Inference, Grad-CAM, candidate box, and workflow triage for 2D brain CT.

The green box is a candidate region derived from Grad-CAM, NOT validated
hemorrhage localization. This educational/research service is not for clinical
decision-making. It deliberately avoids importing a module named `model` to
prevent collision with the existing 3D lung model.
"""
from functools import lru_cache
from pathlib import Path
import base64
import importlib.util
import io

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F


MODEL_DIR = Path(__file__).resolve().parent
MODEL_DEFINITION = MODEL_DIR / "brain_ct_model.py"
CHECKPOINT_PATH = MODEL_DIR / "brain_ct_hemorrhage_model.pth"
IMAGE_SIZE = 224
CLASS_NAMES = ["No hemorrhage", "Hemorrhage"]
HEMORRHAGE_FLAG_THRESHOLD = 0.20


def _load_model_class():
    if not MODEL_DEFINITION.is_file():
        raise FileNotFoundError(f"Brain CT model definition not found: {MODEL_DEFINITION}")
    spec = importlib.util.spec_from_file_location(
        "brain_ct_hemorrhage_model_definition", MODEL_DEFINITION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the brain CT model definition.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BrainCTCNN


@lru_cache(maxsize=1)
def _get_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Brain CT hemorrhage checkpoint not found. Expected file: "
            f"{CHECKPOINT_PATH}"
        )
    model = _load_model_class()()
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Checkpoint does not contain a valid model state.")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint weights do not match brain_ct_model.py. "
            "Confirm this is a checkpoint trained with BrainCTCNN."
        ) from exc
    model.eval()
    return model


def _png_base64(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _jet_rgb(values: np.ndarray) -> np.ndarray:
    """Small NumPy-only JET-like color map; values expected in [0, 1]."""
    x = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def _largest_component_box(mask: np.ndarray):
    """Return bounding box of the largest 4-connected hot region, or None."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    if not best:
        return None
    ys = [point[0] for point in best]
    xs = [point[1] for point in best]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # Make a square around the hot region, with modest padding.
    side = max(x2 - x1 + 1, y2 - y1 + 1)
    side = max(12, min(side + 12, int(min(h, w) * 0.55)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(w - side, left))
    top = max(0, min(h - side, top))
    return [left, top, left + side, top + side]


def _make_explanations(model, tensor: torch.Tensor, original_gray: Image.Image, predicted_index: int):
    """Compute Grad-CAM for final conv-block ReLU (features[14])."""
    activations = {}
    gradients = {}
    target_layer = model.features[14]

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, predicted_index].backward()
        acts = activations.get("value")
        grads = gradients.get("value")
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM activations or gradients were not captured.")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    gray_arr = np.asarray(original_gray.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
    gray_rgb = np.stack([gray_arr, gray_arr, gray_arr], axis=-1)
    colors = _jet_rgb(cam)
    alpha = (np.clip((cam - 0.25) / 0.75, 0, 1) * 0.42)[..., None]
    overlay_arr = (gray_rgb * (1 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGB")

    # Candidate region comes from the highest-activation heatmap area only.
    # It is a model explanation region, not an anatomical segmentation.
    cutoff = max(0.55, float(np.quantile(cam, 0.90)))
    box = _largest_component_box(cam >= cutoff)
    boxed = overlay.copy()
    if box:
        draw = ImageDraw.Draw(boxed)
        # Green outline, with dark under-stroke for contrast.
        draw.rectangle(tuple(box), outline=(0, 0, 0), width=5)
        draw.rectangle(tuple(box), outline=(0, 255, 80), width=3)

    return {
        "gradcam_overlay_base64": _png_base64(overlay),
        "candidate_box_overlay_base64": _png_base64(boxed),
        "candidate_region_box_xyxy": box,
        "method": "Grad-CAM",
        "note": (
            "The green square marks a candidate region derived from model activation. "
            "It is not validated hemorrhage localization and may not correspond to an abnormality."
        ),
    }


def _triage(probability_hemorrhage: float) -> dict:
    """Use the existing application's workflow-style priority bands, not clinical rules."""
    flagged = probability_hemorrhage >= HEMORRHAGE_FLAG_THRESHOLD
    if not flagged:
        priority = "Routine Review"
        reason = (
            "The hemorrhage probability did not cross the configured 20% workflow flag threshold. "
            "This does not rule out hemorrhage; clinician review remains necessary."
        )
    elif probability_hemorrhage >= 0.70:
        priority = "High Priority"
        reason = (
            f"The model assigned {probability_hemorrhage * 100:.2f}% probability to hemorrhage, "
            "crossing the configured workflow flag threshold. Prioritize qualified review under "
            "the application's research workflow."
        )
    elif probability_hemorrhage >= 0.40:
        priority = "Moderate Priority"
        reason = (
            f"The model assigned {probability_hemorrhage * 100:.2f}% probability to hemorrhage. "
            "This crossed the configured workflow flag threshold; qualified review is required."
        )
    else:
        priority = "Routine Review"
        reason = (
            f"The model assigned {probability_hemorrhage * 100:.2f}% probability to hemorrhage. "
            "The configured workflow flag was crossed, but this prototype's existing priority bands "
            "classify probabilities below 40% as Routine Review. Do not use this label to rule out "
            "an emergency; qualified review is required."
        )
    return {
        "priority": priority,
        "reason": reason,
        "flagged": flagged,
        "threshold": HEMORRHAGE_FLAG_THRESHOLD,
        "note": "Operational research workflow only; not clinical urgency assessment or a diagnosis.",
    }


def predict_brain_ct_hemorrhage(image_bytes: bytes, filename: str) -> dict:
    """Predict one JPG/JPEG/PNG brain CT slice and return explanation images."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Upload a JPG, JPEG, or PNG brain CT image.")
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            original = ImageOps.exif_transpose(source).convert("L")
            image = original.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError("The uploaded file could not be decoded as a valid image.") from exc

    if array.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(array).all():
        raise ValueError("Image preprocessing did not produce a valid 224x224 image.")

    tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
    model = _get_model()
    # Grad-CAM requires gradients, so do not wrap this forward pass in inference_mode.
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probabilities = torch.softmax(logits.detach(), dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())
    probability_no = float(probabilities[0].item())
    probability_yes = float(probabilities[1].item())
    confidence = float(probabilities[predicted_index].item())

    explanation = _make_explanations(model, tensor, image, predicted_index)
    triage = _triage(probability_yes)

    return {
        "success": True,
        "task": "2D brain CT slice-level hemorrhage classification",
        "model": {
            "name": "BrainCTCNN",
            "architecture": "2D grayscale CNN",
            "image_size": IMAGE_SIZE,
            "classes": CLASS_NAMES,
        },
        "input": {
            "filename": Path(filename).name,
            "format": suffix.lstrip(".").upper(),
            "channels": 1,
            "size": [IMAGE_SIZE, IMAGE_SIZE],
        },
        "prediction": {
            "class_id": predicted_index,
            "class": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 4),
            "probability_no_hemorrhage": round(probability_no, 4),
            "probability_hemorrhage": round(probability_yes, 4),
        },
        "probabilities": {
            "No hemorrhage": round(probability_no, 4),
            "Hemorrhage": round(probability_yes, 4),
        },
        "explanation": explanation,
        "triage": triage,
        "clinical_note": (
            "Research-only output for a single 2D slice. This is not a diagnosis, "
            "does not assess an entire CT study, and requires qualified clinical review."
        ),
    }



from pathlib import Path
import base64
import importlib.util
import tempfile
import threading

MODEL_DIR = Path(__file__).resolve().parent
INFERENCE_PATH = MODEL_DIR / "inference.py"
GRADCAM_PATH = MODEL_DIR / "gradcam.py"


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_inference = load_module(
    "_abdominal_binary_inference", INFERENCE_PATH
)
_gradcam = load_module(
    "_abdominal_binary_gradcam", GRADCAM_PATH
)

AbdominalUltrasoundClassifier = (
    _inference.AbdominalUltrasoundClassifier
)
generate_gradcam = _gradcam.generate_gradcam

_model = None
_model_lock = threading.Lock()


def get_abdominal_model():
    """Load the abdominal classifier once per backend process."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = AbdominalUltrasoundClassifier(
                    checkpoint_path=MODEL_DIR / "best_model.pt"
                )

    return _model


def predict_abdominal_ultrasound(image_path: str) -> dict:
    """Run research-only abdominal classification with Grad-CAM."""
    model = get_abdominal_model()

    # Preserve the existing classification response.
    result = model.predict(image_path)

    # Use a unique temporary file for every request.
    with tempfile.TemporaryDirectory(
        prefix="abdominal_gradcam_"
    ) as temp_dir:
        heatmap_path = Path(temp_dir) / "gradcam.png"

        gradcam_result = generate_gradcam(
            model,
            image_path,
            output_path=heatmap_path,
        )

        with heatmap_path.open("rb") as heatmap_file:
            heatmap_base64 = base64.b64encode(
                heatmap_file.read()
            ).decode("ascii")

    return {
        "model_name": "Abdominal Ultrasound EfficientNet-B0",
        "model_checkpoint": "best_model.pt",
        **result,
        "gradcam": {
            "image_base64": heatmap_base64,
            "predicted_class": gradcam_result["predicted_class"],
            "interpretation_warning": (
                "Grad-CAM indicates model attention, not a verified "
                "lesion location or clinical diagnosis."
            ),
            "research_only": True,
        },
        "note": (
            "Research output only. This model is not a diagnosis or "
            "validated clinical assessment. Qualified clinical review "
            "is required."
        ),
    }

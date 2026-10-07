import torch
import torch.nn.functional as F
from pathlib import Path

from model import LungNodule3DCNN


# Location of this model folder
MODEL_DIR = Path(__file__).resolve().parent

# Trained weights
MODEL_PATH = MODEL_DIR / "lung_nodule_3dcnn.pth"


# Model settings from config.json
CLASS_NAMES = {
    0: "Non-nodule candidate",
    1: "Nodule candidate"
}

CLASSIFICATION_THRESHOLD = 0.20


def load_model():
    """
    Load the trained 3D CNN and its weights.
    """

    model = LungNodule3DCNN(num_classes=2)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def predict_patch(model, patch):
    """
    Predict whether a 3D CT patch is a lung-nodule candidate.

    Expected patch shape:
        [1, 1, 32, 32, 32]

    Returns:
        prediction
        probabilities
        nodule_probability
    """

    # Convert input to PyTorch tensor
    if not isinstance(patch, torch.Tensor):
        patch = torch.tensor(patch, dtype=torch.float32)

    patch = patch.float()

    # Make sure the tensor has the expected dimensions
    if patch.ndim == 4:
        patch = patch.unsqueeze(0)

    if patch.shape != (1, 1, 32, 32, 32):
        raise ValueError(
            f"Expected input shape "
            f"[1, 1, 32, 32, 32], got {tuple(patch.shape)}"
        )

    with torch.no_grad():

        logits = model(patch)

        probabilities = F.softmax(logits, dim=1)

        nodule_probability = probabilities[0, 1].item()

    # Model configuration specifies threshold = 0.20
    if nodule_probability >= CLASSIFICATION_THRESHOLD:
        prediction = 1
    else:
        prediction = 0

    return {
        "prediction": prediction,
        "class": CLASS_NAMES[prediction],
        "probability_non_nodule": round(
            probabilities[0, 0].item(), 4
        ),
        "probability_nodule": round(
            nodule_probability, 4
        ),
        "threshold": CLASSIFICATION_THRESHOLD
    }


if __name__ == "__main__":

    print("Loading CT lung nodule model...")

    model = load_model()

    print("Model loaded successfully.")

    print("Model:", type(model).__name__)
    print("Weights:", MODEL_PATH)

    # Create a dummy 3D CT patch for testing.
    # Shape: [batch, channel, depth, height, width]
    test_patch = torch.zeros(
        (1, 1, 32, 32, 32),
        dtype=torch.float32
    )

    print("Testing input shape:", test_patch.shape)

    result = predict_patch(model, test_patch)

    print("Prediction result:")
    print(result)
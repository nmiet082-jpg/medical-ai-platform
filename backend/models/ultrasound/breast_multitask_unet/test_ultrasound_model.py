from pathlib import Path
import torch

from ultrasound_model import load_ultrasound_model


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "multitask_unet_best.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model:", MODEL_PATH)

model = load_ultrasound_model(
    MODEL_PATH,
    device=device
)

print("Model loaded successfully!")

# Test with one grayscale 256x256 ultrasound image
dummy_input = torch.randn(
    1, 1, 256, 256,
    device=device
)

with torch.no_grad():
    classification, segmentation = model(dummy_input)

print("Classification output shape:", classification.shape)
print("Segmentation output shape:", segmentation.shape)
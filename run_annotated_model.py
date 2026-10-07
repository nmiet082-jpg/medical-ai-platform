import sys
import torch
import numpy as np

MODEL_DIR = r"C:\medical-ai-platform\backend\models\ct\lung_nodule_model"
MODEL_FILE = MODEL_DIR + r"\lung_nodule_3dcnn.pth"
PATCH_FILE = r"C:\medical-ai-platform\annotated_nodule_patch.npy"

sys.path.insert(0, MODEL_DIR)

from model import LungNodule3DCNN


print("Loading model...")

model = LungNodule3DCNN()

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove DataParallel prefix if present
checkpoint = {
    k.replace("module.", "", 1): v
    for k, v in checkpoint.items()
}

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")


print()
print("Loading annotated CT patch...")

patch = np.load(PATCH_FILE).astype(np.float32)

print("Patch shape:", patch.shape)


# Model expects:
# batch x channel x depth x height x width

x = torch.from_numpy(patch)
x = x.unsqueeze(0).unsqueeze(0)

print("Model input shape:", x.shape)


with torch.no_grad():
    output = model(x)

print()
print("Raw model output:")
print(output)


# Convert output to nodule probability
if output.numel() == 1:

    probability_nodule = torch.sigmoid(
        output.reshape(-1)[0]
    ).item()

elif output.shape[-1] == 2:

    probabilities = torch.softmax(output, dim=1)
    probability_nodule = probabilities[0, 1].item()

else:
    raise RuntimeError(
        f"Unexpected model output shape: {output.shape}"
    )


probability_non_nodule = 1.0 - probability_nodule

threshold = 0.20

prediction = (
    1 if probability_nodule >= threshold else 0
)

classification = (
    "Nodule candidate"
    if prediction == 1
    else "Non-nodule candidate"
)


print()
print("================================")
print("CT NODULE MODEL RESULT")
print("================================")
print("Nodule probability:", round(probability_nodule, 4))
print("Non-nodule probability:", round(probability_non_nodule, 4))
print("Threshold:", threshold)
print("Prediction:", prediction)
print("Class:", classification)
print("================================")
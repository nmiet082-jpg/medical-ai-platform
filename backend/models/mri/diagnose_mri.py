
from pathlib import Path
import torch
import numpy as np

from inference import load_models
from preprocessing import read_nifti_modalities

MODEL_DIR = Path(__file__).resolve().parent

# Change this to the folder where you extracted the four test volumes.

TEST_DIR = Path(r"C:\medical-ai-platform\backend\models\mri\mri_test_volumes")

def main():
    print("Loading MRI models...")
    _, model, _, checkpoint, device = load_models(MODEL_DIR)

    print("Checkpoint validation metrics:")
    print(checkpoint.get("validation_metrics"))

    files = [
        TEST_DIR / "flair.nii",
        TEST_DIR / "t1.nii",
        TEST_DIR / "t1ce.nii",
        TEST_DIR / "t2.nii",
    ]

    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print("\nMissing test files:")
        for p in missing:
            print(" -", p)
        print("\nUpdate TEST_DIR to your extracted test-volume folder.")
        return

    x = read_nifti_modalities(files)
    print("\nInput shape:", tuple(x.shape))
    print("Input min/max/mean:", x.min().item(), x.max().item(), x.mean().item())

    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(device))
        seg_logits = outputs[0]
        probs = torch.softmax(seg_logits, dim=1)
        prediction = probs.argmax(dim=1)[0].cpu().numpy()

    print("\nSegmentation logits shape:", tuple(seg_logits.shape))
    print("Segmentation logits min/max:",
          seg_logits.min().item(), seg_logits.max().item())
    print("\nVoxel counts by predicted class:")
    for label in range(4):
        print(f"Class {label}: {int(np.count_nonzero(prediction == label))}")

    print("\nMean probability per class:")
    mean_probs = probs.mean(dim=(0, 2, 3, 4)).cpu().numpy()
    for label, value in enumerate(mean_probs):
        print(f"Class {label}: {float(value):.6f}")

if __name__ == "__main__":
    main()

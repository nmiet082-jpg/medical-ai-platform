
from pathlib import Path
import torch

from model_architectures import build_classifier, UnifiedMRI3D

MODEL_DIR = Path(__file__).resolve().parent

def test_checkpoints_load_and_shapes():
    classifier_path = MODEL_DIR / "classifier_best.pt"
    segmentation_path = MODEL_DIR / "segmentation_best_epoch10.pt"

    # Test classifier
    c = torch.load(
        classifier_path, map_location="cpu", weights_only=False
    )
    classifier = build_classifier(num_classes=len(c["classes"]))
    classifier.load_state_dict(c["model_state_dict"])
    classifier.eval()

    with torch.no_grad():
        out = classifier(torch.zeros(1, 3, 224, 224))

    assert tuple(out.shape) == (1, 4)
    print("Classifier loaded; output shape:", tuple(out.shape))

    # Test 3D segmentation model
    s = torch.load(
        segmentation_path, map_location="cpu", weights_only=False
    )
    segmentation = UnifiedMRI3D(4, 4, 16)
    segmentation.load_state_dict(s["model_state_dict"])
    segmentation.eval()

    with torch.no_grad():
        mask_logits, auxiliary_logits = segmentation(
            torch.zeros(1, 4, 32, 96, 96)
        )

    assert tuple(mask_logits.shape) == (1, 4, 32, 96, 96)
    assert tuple(auxiliary_logits.shape) == (1, 4)

    print("Segmentation loaded; mask shape:", tuple(mask_logits.shape))
    print("Auxiliary output shape:", tuple(auxiliary_logits.shape))
    print("Both checkpoint tests passed.")

if __name__ == "__main__":
    test_checkpoints_load_and_shapes()

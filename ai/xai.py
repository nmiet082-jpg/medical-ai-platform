import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Pneumonia",
    "Cardiomegaly",
    "Pneumothorax"
]


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "models"
    / "resnet18_medical_3class_best.pth"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model):

        self.model = model

        self.gradients = None
        self.activations = None

        # ResNet18 final convolutional layer
        self.target_layer = self.model.layer4[-1]

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()


    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        target = output[
            0,
            class_index
        ]

        target.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activation maps
        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam


    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# GENERATE XAI HEATMAP
# ============================================================

def generate_gradcam(
    image_path,
    class_index,
    output_path
):

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")


    original_array = np.array(
        original_image
    )


    # --------------------------------------------------------
    # Prepare image for model
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    gradcam = GradCAM(model)

    heatmap = gradcam.generate(
        image_tensor,
        class_index
    )

    gradcam.close()


    # --------------------------------------------------------
    # Resize heatmap to original image size
    # --------------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original_array.shape[1],
            original_array.shape[0]
        )
    )


    # Convert to 8-bit
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    # --------------------------------------------------------
    # Create colored heatmap
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        heatmap_color,
        0.40,
        0
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )


    return {
        "output_path": str(output_path),
        "class_name": CLASS_NAMES[class_index],
        "method": "Grad-CAM",
        "model": "ResNet18 Medical 3-Class",
        "model_version": "1.0"
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "XAI module loaded successfully."
    )

    print(
        "Model path:"
    )

    print(
        MODEL_PATH
    )

    print(
        "Supported classes:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{index}: {class_name}"
        )
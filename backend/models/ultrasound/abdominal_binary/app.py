import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from inference import AbdominalUltrasoundClassifier
from gradcam import generate_gradcam


st.set_page_config(
    page_title="Abdominal Ultrasound Research Module",
    page_icon="🩻",
    layout="wide"
)

st.title("Abdominal Ultrasound Analysis")
st.caption("AI-based research prototype — not for clinical diagnosis.")

st.warning(
    "This model is not clinically validated. Predictions and Grad-CAM "
    "heatmaps must not be used alone to diagnose or exclude disease."
)

@st.cache_resource
def load_classifier():
    return AbdominalUltrasoundClassifier()

try:
    classifier = load_classifier()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an abdominal ultrasound image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as image:
            image.verify()

        uploaded_file.seek(0)

        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / f"uploaded{suffix}"
            heatmap_path = Path(temp_dir) / "gradcam.png"

            image_path.write_bytes(uploaded_file.getvalue())

            result = classifier.predict(image_path)

            left, right = st.columns(2)

            with left:
                st.subheader("Uploaded image")
                st.image(
                    str(image_path),
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            with right:
                st.subheader("Model output")
                st.write("**Predicted class:**", result["predicted_class"])
                st.metric(
                    "Abnormal probability",
                    f'{result["probabilities"]["abnormal"] * 100:.2f}%'
                )
                st.metric(
                    "Normal probability",
                    f'{result["probabilities"]["normal"] * 100:.2f}%'
                )

            gradcam_result = generate_gradcam(
                classifier, image_path, heatmap_path
            )

            st.subheader("Grad-CAM visualization")
            st.image(
                str(heatmap_path),
                caption="Model attention visualization — not lesion segmentation",
                use_container_width=True
            )

            st.warning(
                "The displayed probabilities are model outputs, not "
                "clinically calibrated confidence scores. Human review "
                "and independent clinical validation are required."
            )

            report = {
                "filename": uploaded_file.name,
                **result,
                "gradcam": {
                    "available": heatmap_path.is_file(),
                    "interpretation": gradcam_result["interpretation_warning"]
                }
            }

            st.download_button(
                "Download research result (JSON)",
                data=__import__("json").dumps(report, indent=2),
                file_name="ultrasound_research_result.json",
                mime="application/json"
            )

    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")

# Abdominal Ultrasound Research Module

Standalone research prototype for binary abdominal ultrasound image
classification using EfficientNet-B0 and Grad-CAM.

## Features
- Upload JPG, JPEG, PNG, BMP, or WEBP images.
- Predict the model's abnormal or normal class.
- Display both model probabilities.
- Generate a Grad-CAM attention visualization.
- Download a JSON research result.

## Run in VS Code on Windows

1. Extract the ZIP into a folder.
2. Open that folder in VS Code.
3. Open the terminal in VS Code.
4. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

7. Open the local address printed by Streamlit in your browser.

## Model information
- Architecture: EfficientNet-B0
- Input: RGB image resized to 224 x 224
- Normalization: ImageNet mean and standard deviation
- Class mapping: abnormal = 0, normal = 1
- Checkpoint selected at epoch 23

## Reported held-out test results
- Accuracy: 78.10%
- Balanced accuracy: 71.39%
- Macro F1: 72.66%
- Abnormal precision: 69.84%
- Abnormal recall: 53.33%
- Abnormal F1: 60.48%
- ROC-AUC: 81.13%

Patient-level independence has not been verified, and external clinical
validation has not been completed.

## Important limitations
This application is for research and educational use only. It is not a
medical device and does not provide a clinical diagnosis. Model probabilities
are not clinically calibrated confidence scores. Grad-CAM visualizes model
attention; it does not establish a lesion boundary or prove disease.
A qualified clinician must not rely on this prototype alone for patient care.

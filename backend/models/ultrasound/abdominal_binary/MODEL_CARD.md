# Abdominal Ultrasound Classifier

## Model
EfficientNet-B0 with an ImageNet-pretrained initialization.
The selected checkpoint is from Experiment 4, epoch 23.

## Intended use
Research prototype for binary classification of ultrasound images
as normal or abnormal according to the dataset labels.

## Limitations
- Not a clinically validated diagnostic system.
- Does not identify specific diseases.
- A normal prediction does not exclude abnormality.
- Grad-CAM, if added, is an explanation and not verified lesion segmentation.
- Patient-level independence has not been established.
- External clinical validation has not been completed.
- Manual clinician review remains necessary.

## Held-out test results
Accuracy: 78.10%
Balanced accuracy: 71.39%
Macro F1: 72.66%
Abnormal precision: 69.84%
Abnormal recall: 53.33%
Abnormal F1: 60.48%
ROC-AUC: 81.13%
Average precision for abnormal: 67.93%

Test images: 525
Checkpoint epoch: 23

These are experimental dataset results and must not be interpreted
as evidence of clinical readiness.

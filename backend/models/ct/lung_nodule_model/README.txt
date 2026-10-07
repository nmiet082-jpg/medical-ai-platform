AI-BASED MEDICAL IMAGING TRIAGE
3D LUNG NODULE CLASSIFICATION MODEL

MODEL
-----

Architecture:
LungNodule3DCNN

Framework:
PyTorch

Input:
1 x 32 x 32 x 32

Classes:
0 = Non-nodule candidate
1 = Nodule candidate


PREPROCESSING
-------------

1. CT volume is loaded.
2. CT is resampled to 1 mm isotropic spacing.
3. A candidate-centered 32 x 32 x 32 3D patch is extracted.
4. CT intensity is clipped to HU range [-1000, 400].
5. Values are normalized to [0, 1].


TRAINING
--------

Optimizer: AdamW
Learning Rate: 0.001
Weight Decay: 0.0001
Batch Size: 16
Epochs: 20
Best Epoch: 15


FINAL TEST RESULTS
------------------

Classification Threshold: 0.20

Accuracy: 92.33%
Precision: 84.68%
Recall / Sensitivity: 94.00%
Specificity: 91.50%
F1 Score: 89.10%
ROC-AUC: 98.60%


CONFUSION MATRIX
----------------

                 Predicted
                 Non-nodule   Nodule

Actual Non-nodule    183        17
Actual Nodule         6         94


MODEL FILE
----------

lung_nodule_3dcnn.pth

This is the trained PyTorch model weights file.

The model architecture is defined in model.py.


IMPORTANT
---------

This project performs candidate-level lung nodule classification
from selected 3D CT patches.

The evaluation was performed on a selected LUNA16 candidate-level
test subset with a 2:1 negative-to-positive ratio.

The results should not be interpreted as clinical diagnostic
performance.
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from typing import Annotated
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from database import get_connection, create_tables
from security import verify_password, hash_password

from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from PIL import Image

from pydantic import BaseModel

import shutil

import uuid

import numpy as np

import sys
import base64
import json
import os
import threading
import tempfile
import jwt
import torch
import SimpleITK as sitk

from datetime import datetime, timedelta, timezone

# --------------------------------------------------

# AI MODEL IMPORT

# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

from ai.predict import predict_image
from ai.xai import generate_gradcam

# ============================================================
# BREAST ULTRASOUND MODEL SERVICE
# ============================================================
BREAST_ULTRASOUND_MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "ultrasound"
    / "breast_multitask_unet"
)
if str(BREAST_ULTRASOUND_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BREAST_ULTRASOUND_MODEL_DIR))

from ultrasound_service import predict_breast_ultrasound
# ============================================================
# ABDOMINAL ULTRASOUND MODEL SERVICE
# ============================================================
ABDOMINAL_ULTRASOUND_MODEL_DIR = (
    PROJECT_ROOT / "backend" / "models" / "ultrasound" / "abdominal_binary"
)
if not ABDOMINAL_ULTRASOUND_MODEL_DIR.is_dir():
    raise RuntimeError(
        f"Abdominal ultrasound model folder not found: {ABDOMINAL_ULTRASOUND_MODEL_DIR}"
    )
if str(ABDOMINAL_ULTRASOUND_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ABDOMINAL_ULTRASOUND_MODEL_DIR))

from abdominal_service import predict_abdominal_ultrasound
# ============================================================
# CT LUNG NODULE MODEL
# ============================================================

CT_MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "ct"
    / "lung_nodule_model"
)

CT_MODEL_FILE = (
    CT_MODEL_DIR
    / "lung_nodule_3dcnn.pth"
)

CT_MODEL_NAME = "Lung Nodule 3D CNN"
CT_MODEL_VERSION = "1.0"

CT_CLASS_NAMES = {
    0: "Non-nodule candidate",
    1: "Nodule candidate",
}

CT_NODULE_THRESHOLD = 0.20
CT_PATCH_SIZE = (32, 32, 32)
CT_TARGET_SPACING = (1.0, 1.0, 1.0)
CT_HU_MIN = -1000.0
CT_HU_MAX = 400.0

if str(CT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CT_MODEL_DIR))

from model import LungNodule3DCNN

# ============================================================
# MRI MODEL SERVICE
# ============================================================
MRI_MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "mri"

if str(MRI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MRI_MODEL_DIR))

from inference import (
    load_models as load_mri_models,
    classify_image as classify_mri_image,
    gradcam_overlay as gradcam_mri_overlay,
)
from preprocessing import (
    MODALITIES as MRI_MODALITIES,
    CLASS_NAMES as MRI_CLASS_NAMES,
    SEGMENTATION_LABELS as MRI_SEGMENTATION_LABELS,
    read_nifti_modalities,
)

_mri_models = None

def get_mri_models():
    """Load MRI checkpoints once per backend process."""
    global _mri_models
    if _mri_models is None:
        required = (
            MRI_MODEL_DIR / "classifier_best.pt",
            MRI_MODEL_DIR / "segmentation_best_epoch10.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MRI checkpoint file(s) missing: " + ", ".join(missing)
            )
        _mri_models = load_mri_models(MRI_MODEL_DIR)
    return _mri_models

def load_ct_model():
    """
    Load the trained 3D CT lung-nodule classifier.

    The model expects:
        [batch, channel, depth, height, width]
        [1, 1, 32, 32, 32]
    """

    if not CT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"CT model weights not found: {CT_MODEL_FILE}"
        )

    model = LungNodule3DCNN(num_classes=2)

    checkpoint = torch.load(
        CT_MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "CT model checkpoint does not contain a valid state_dict."
        )

    checkpoint = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(checkpoint)
    model.eval()

    return model

ct_model = load_ct_model()

print("CT lung nodule model loaded successfully.")

def predict_ct_patch_array(patch: np.ndarray):
    """
    Classify one preprocessed 32x32x32 CT candidate patch.

    The patch must already use the training preprocessing:
        1 mm isotropic spacing
        HU clipped to [-1000, 400]
        normalized to [0, 1]

    This is candidate-level classification. It does not detect
    candidate locations in an entire CT scan.
    """

    patch = np.asarray(patch, dtype=np.float32)

    if patch.shape != CT_PATCH_SIZE:
        raise ValueError(
            "Invalid CT patch shape. "
            f"Expected {CT_PATCH_SIZE}, got {tuple(patch.shape)}."
        )

    if not np.isfinite(patch).all():
        raise ValueError(
            "CT patch contains NaN or infinite values."
        )

    patch_min = float(patch.min())
    patch_max = float(patch.max())

    if patch_min < 0.0 or patch_max > 1.0:
        raise ValueError(
            "CT patch values must be normalized to [0, 1]. "
            f"Received range [{patch_min:.4f}, {patch_max:.4f}]."
        )

    model_input = torch.from_numpy(patch).float()
    model_input = model_input.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = ct_model(model_input)
        probabilities = torch.softmax(logits, dim=1)

    probability_non_nodule = float(probabilities[0, 0].item())
    probability_nodule = float(probabilities[0, 1].item())

    prediction = (
        1
        if probability_nodule >= CT_NODULE_THRESHOLD
        else 0
    )

    return {
        "class_id": prediction,
        "class": CT_CLASS_NAMES[prediction],
        "probability_nodule": round(probability_nodule, 4),
        "probability_non_nodule": round(probability_non_nodule, 4),
        "threshold": CT_NODULE_THRESHOLD,
    }

# ==================================================
# JWT AUTHENTICATION CONFIGURATION
# ==================================================

JWT_SECRET_KEY = os.environ.get(
    "JWT_SECRET_KEY",
    "development-only-change-this-secret"
)

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

bearer_scheme = HTTPBearer()

def create_access_token(user_id: int, username: str, role: str):
    expiration = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expiration
    }

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    return token

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Authentication token has expired"
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                username,
                email,
                role,
                account_status,
                patient_record_id
            FROM users
            WHERE id = ?
            """,
            (int(user_id),)
        )

        user = cursor.fetchone()

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User account not found"
            )

        if user["account_status"] != "ACTIVE":
            raise HTTPException(
                status_code=403,
                detail="User account is not active"
            )

        return dict(user)

    finally:
        connection.close()

# ==================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# Roles follow the project SRS.
# Existing database accounts using "Admin" remain compatible.
# ==================================================

ROLE_ADMIN = "Administrator"
ROLE_RADIOLOGIST = "Radiologist"
ROLE_CLINICIAN = "Clinician"
ROLE_AI_ENGINEER = "AI/ML Engineer"

def normalize_role(role: str) -> str:
    """Normalize the legacy administrator role without changing the database."""
    return ROLE_ADMIN if role == "Admin" else role

def require_roles(*allowed_roles: str):
    """FastAPI dependency factory enforcing least-privilege role access."""
    def role_checker(
        current_user: dict = Depends(get_current_user)
    ):
        user_role = normalize_role(current_user.get("role", ""))
        normalized_allowed = {normalize_role(role) for role in allowed_roles}

        if user_role not in normalized_allowed:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to perform this action."
            )

        return current_user

    return role_checker

def require_admin(
    current_user: dict = Depends(get_current_user)
):
    """Compatibility dependency for existing administrator-only endpoints."""
    return require_roles(ROLE_ADMIN)(current_user)

# --------------------------------------------------

# FastAPI Application

# --------------------------------------------------

app = FastAPI(

    title="AI Medical Imaging Triage & XAI",

    version="1.0.0"

)

# --------------------------------------------------

# Create Database Tables

# --------------------------------------------------

create_tables()

# --------------------------------------------------

# CORS

# --------------------------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        # VS Code Live Server frontend
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        # Vite development frontend
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

# --------------------------------------------------

# Upload Folder

# --------------------------------------------------

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# XAI / Grad-CAM Output
# --------------------------------------------------

XAI_OUTPUT_DIR = Path("xai_outputs")
XAI_OUTPUT_DIR.mkdir(exist_ok=True)
CT_XAI_OUTPUT_DIR = PROJECT_ROOT / "uploads" / "ct_xai"
CT_XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CT_XAI_LOCK = threading.Lock()

# Explanation artifacts can contain patient-related imaging data.
# Serve them only to authenticated clinical users or authorized engineers.
CLINICAL_VIEW_ROLES = (
    ROLE_ADMIN,
    ROLE_RADIOLOGIST,
)
XAI_VIEW_ROLES = (
    ROLE_ADMIN,
    ROLE_RADIOLOGIST,
    ROLE_AI_ENGINEER,
)

@app.get("/xai_outputs/{filename}")
def get_xai_artifact(
    filename: str,
    _rbac_user: dict = Depends(require_roles(*XAI_VIEW_ROLES)),
):
    safe_name = Path(filename).name
    artifact_path = (XAI_OUTPUT_DIR / safe_name).resolve()
    if artifact_path.parent != XAI_OUTPUT_DIR.resolve() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Explanation artifact not found")
    return FileResponse(artifact_path)

@app.get("/ct_xai/{filename}")
def get_ct_xai_artifact(
    filename: str,
    _rbac_user: dict = Depends(require_roles(*XAI_VIEW_ROLES)),
):
    """Serve legacy CT explanation files saved directly in ct_xai/."""
    safe_name = Path(filename).name
    artifact_path = (CT_XAI_OUTPUT_DIR / safe_name).resolve()
    if artifact_path.parent != CT_XAI_OUTPUT_DIR.resolve() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="CT explanation artifact not found")
    return FileResponse(artifact_path)

@app.get("/ct_xai/{series_id}/{filename}")
def get_ct_xai_series_artifact(
    series_id: str,
    filename: str,
    _rbac_user: dict = Depends(require_roles(*XAI_VIEW_ROLES)),
):
    """Serve a CT explanation saved under ct_xai/<series_id>/<filename>."""
    safe_series_id = Path(series_id).name
    safe_name = Path(filename).name
    if safe_series_id != series_id or safe_name != filename:
        raise HTTPException(status_code=404, detail="CT explanation artifact not found")

    base_dir = CT_XAI_OUTPUT_DIR.resolve()
    series_dir = (base_dir / safe_series_id).resolve()
    artifact_path = (series_dir / safe_name).resolve()

    # Keep the requested artifact inside this series' output directory.
    if (
        series_dir.parent != base_dir
        or artifact_path.parent != series_dir
        or not artifact_path.is_file()
    ):
        raise HTTPException(status_code=404, detail="CT explanation artifact not found")

    return FileResponse(artifact_path)

CLASS_INDEX_MAP = {
    "Pneumonia": 0,
    "Cardiomegaly": 1,
    "Pneumothorax": 2
}

# --------------------------------------------------

# Allowed File Types

# --------------------------------------------------

ALLOWED_EXTENSIONS = {

    ".jpg",

    ".jpeg",

    ".png",

    ".dcm"

}

# ==================================================
# AUTHENTICATION
# ==================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class AdminUserCreate(BaseModel):
    full_name: str
    username: str
    email: str
    password: str
    role: str
    patient_record_id: int | None = None

# ==================================================

# PATIENT MODEL

# ==================================================

class PatientCreate(BaseModel):

    patient_id: str

    name: str

    age: int

    gender: str

# ==================================================

# STUDY MODEL

# ==================================================

class StudyCreate(BaseModel):

    patient_id: int

    study_type: str

    study_date: str

# ==================================================

# AI MODEL MANAGEMENT MODEL

# ==================================================

class AIModelCreate(BaseModel):

    model_name: str

    version: str

    model_path: str | None = None

    status: str = "Registered"

# ==================================================

# AI PREDICTION STORAGE MODEL

# ==================================================

class AIPredictionCreate(BaseModel):

    model_id: int

    abnormality: str

    confidence_score: float

    prediction_status: str = "Completed"

# ============================================================

# XAI / EXPLANATION MANAGEMENT

# ============================================================

class ExplanationCreate(BaseModel):

    prediction_id: int

    explanation_type: str

    model_version: str

    explanation_path: str | None = None

    metadata: str | None = None

@app.post("/api/analysis/{study_id}/explanation")

def create_explanation(

    study_id: int,

    explanation: ExplanationCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    conn = get_connection()

    cursor = conn.cursor()

    # Check that the study exists

    cursor.execute(

        "SELECT id FROM studies WHERE id = ?",

        (study_id,)

    )

    study = cursor.fetchone()

    if not study:

        conn.close()

        raise HTTPException(

            status_code=404,

            detail="Study not found"

        )

    # Check that the prediction exists

    cursor.execute(

        """

        SELECT id, study_id

        FROM ai_predictions

        WHERE id = ?

        """,

        (explanation.prediction_id,)

    )

    prediction = cursor.fetchone()

    if not prediction:

        conn.close()

        raise HTTPException(

            status_code=404,

            detail="Prediction not found"

        )

    # Make sure the prediction belongs to this study

    if prediction["study_id"] != study_id:

        conn.close()

        raise HTTPException(

            status_code=400,

            detail="Prediction does not belong to this study"

        )

    cursor.execute(

        """

        INSERT INTO explanations

        (

            prediction_id,

            explanation_type,

            model_version,

            explanation_path,

            metadata

        )

        VALUES (?, ?, ?, ?, ?)

        """,

        (

            explanation.prediction_id,

            explanation.explanation_type,

            explanation.model_version,

            explanation.explanation_path,

            explanation.metadata

        )

    )

    explanation_id = cursor.lastrowid

    conn.commit()

    conn.close()

    return {

        "message": "Explanation created successfully",

        "explanation_id": explanation_id,

        "study_id": study_id,

        "prediction_id": explanation.prediction_id,

        "explanation_type": explanation.explanation_type,

        "model_version": explanation.model_version

    }

@app.get("/api/analysis/{study_id}/explanation")

def get_explanations(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    conn = get_connection()

    cursor = conn.cursor()

    # Check that study exists

    cursor.execute(

        "SELECT id FROM studies WHERE id = ?",

        (study_id,)

    )

    study = cursor.fetchone()

    if not study:

        conn.close()

        raise HTTPException(

            status_code=404,

            detail="Study not found"

        )

    cursor.execute(

        """

        SELECT

            id,

            prediction_id,

            explanation_type,

            model_version,

            explanation_path,

            metadata,

            created_at

        FROM explanations

        WHERE prediction_id IN (

            SELECT id

            FROM ai_predictions

            WHERE study_id = ?

        )

        ORDER BY id DESC

        """,

        (study_id,)

    )

    explanations = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {

        "study_id": study_id,

        "count": len(explanations),

        "explanations": explanations

    }

# ==================================================

# ============================================================

# ============================================================
# AUTOMATIC GRAD-CAM XAI GENERATION
# ============================================================

@app.post("/api/analysis/{study_id}/explanation/generate")
def generate_explanation_for_study(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()
    cursor = connection.cursor()

    try:

        # --------------------------------------------------
        # 1. Check study
        # --------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM studies
            WHERE id = ?
            """,
            (study_id,)
        )

        study = cursor.fetchone()

        if study is None:
            raise HTTPException(
                status_code=404,
                detail="Study not found"
            )

        # --------------------------------------------------
        # 2. Find latest uploaded image
        # --------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                file_id,
                original_filename,
                file_path
            FROM medical_images
            WHERE study_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (study_id,)
        )

        image_record = cursor.fetchone()

        if image_record is None:
            raise HTTPException(
                status_code=404,
                detail="No medical image is associated with this study"
            )

        # --------------------------------------------------
        # 3. Get predictions already produced by the AI model
        # --------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                model_id,
                abnormality,
                confidence_score,
                prediction_status
            FROM ai_predictions
            WHERE study_id = ?
              AND model_id IN (
                  SELECT id
                  FROM ai_models
                  WHERE model_name = 'ResNet18 Medical 3-Class'
              )
            ORDER BY id ASC
            """,
            (study_id,)
        )

        predictions = cursor.fetchall()

        if not predictions:
            raise HTTPException(
                status_code=404,
                detail="No AI predictions found for this study"
            )

        # --------------------------------------------------
        # 4. Class thresholds
        # --------------------------------------------------

        thresholds = {
            "Pneumonia": 0.3,
            "Cardiomegaly": 0.3,
            "Pneumothorax": 0.2
        }

        # --------------------------------------------------
        # 5. Find predictions that crossed their threshold
        # --------------------------------------------------

        flagged_predictions = []

        for prediction in predictions:

            class_name = prediction["abnormality"]
            probability = float(prediction["confidence_score"])
            threshold = thresholds.get(class_name, 0.5)

            if probability >= threshold:
                flagged_predictions.append(
                    {
                        "prediction_id": prediction["id"],
                        "model_id": prediction["model_id"],
                        "class_name": class_name,
                        "probability": probability,
                        "threshold": threshold
                    }
                )

        # --------------------------------------------------
        # 6. Select highest-probability flagged class.
        #    If nothing is flagged, use the highest-probability
        #    class so an explanation can still be generated.
        # --------------------------------------------------

        if flagged_predictions:

            selected_prediction = max(
                flagged_predictions,
                key=lambda item: item["probability"]
            )

        else:

            highest_prediction = max(
                predictions,
                key=lambda row: float(row["confidence_score"])
            )

            class_name = highest_prediction["abnormality"]

            selected_prediction = {
                "prediction_id": highest_prediction["id"],
                "model_id": highest_prediction["model_id"],
                "class_name": class_name,
                "probability": float(
                    highest_prediction["confidence_score"]
                ),
                "threshold": thresholds.get(class_name, 0.5)
            }

        # --------------------------------------------------
        # 7. Check model information
        # --------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                model_name,
                version
            FROM ai_models
            WHERE id = ?
            """,
            (selected_prediction["model_id"],)
        )

        model_record = cursor.fetchone()

        if model_record is None:
            raise HTTPException(
                status_code=404,
                detail="AI model not found"
            )

        # --------------------------------------------------
        # 8. Get Grad-CAM class index
        # --------------------------------------------------

        class_name = selected_prediction["class_name"]

        if class_name not in CLASS_INDEX_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported XAI class: {class_name}"
            )

        class_index = CLASS_INDEX_MAP[class_name]

        # --------------------------------------------------
        # 9. Create output filename
        # --------------------------------------------------

        safe_class_name = class_name.lower().replace(" ", "_")

        output_filename = (
            f"study_{study_id}_"
            f"{safe_class_name}_"
            f"gradcam.png"
        )

        output_path = XAI_OUTPUT_DIR / output_filename

        # --------------------------------------------------
        # 10. Generate Grad-CAM
        # --------------------------------------------------

        xai_result = generate_gradcam(
            str(image_record["file_path"]),
            class_index,
            str(output_path)
        )

        # --------------------------------------------------
        # 11. Store explanation metadata
        # --------------------------------------------------

        metadata = {
            "study_id": study_id,
            "image_file_id": image_record["file_id"],
            "image_filename": image_record["original_filename"],
            "finding": class_name,
            "probability": selected_prediction["probability"],
            "threshold": selected_prediction["threshold"],
            "method": "Grad-CAM",
            "model_name": model_record["model_name"],
            "model_version": model_record["version"],
            "clinical_note": (
                "AI explanation only. The heatmap does not "
                "establish a medical diagnosis."
            )
        }

        cursor.execute(
            """
            INSERT INTO explanations
            (
                prediction_id,
                explanation_type,
                model_version,
                explanation_path,
                metadata
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                selected_prediction["prediction_id"],
                "Grad-CAM",
                model_record["version"],
                str(output_path),
                json.dumps(metadata)
            )
        )

        explanation_id = cursor.lastrowid
        connection.commit()

        # --------------------------------------------------
        # 12. Return result
        # --------------------------------------------------

        return {
            "message": "Grad-CAM explanation generated successfully",
            "study_id": study_id,
            "explanation_id": explanation_id,
            "prediction_id": selected_prediction["prediction_id"],
            "finding": class_name,
            "probability": selected_prediction["probability"],
            "threshold": selected_prediction["threshold"],
            "flagged": (
                selected_prediction["probability"]
                >= selected_prediction["threshold"]
            ),
            "explanation_type": "Grad-CAM",
            "model_name": model_record["model_name"],
            "model_version": model_record["version"],
            "explanation_path": str(output_path),
            "metadata": metadata
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception as e:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate XAI explanation: {str(e)}"
        )

    finally:
        connection.close()
# ============================================================
# AUTOMATIC TRIAGE CALCULATION
# ============================================================

def calculate_triage(predictions):
    """
    Calculate workflow priority from the AI predictions.

    This is a workflow-priority rule, NOT a medical diagnosis.
    """

    if not predictions:
        return {
            "priority": "AI Inconclusive / Manual Review",
            "reason": "No AI predictions are available."
        }

    # Find predictions that crossed their class threshold
    flagged_predictions = [
        prediction
        for prediction in predictions
        if prediction["flagged"] is True
    ]

    # No prediction crossed its threshold
    if not flagged_predictions:
        highest_prediction = max(
            predictions,
            key=lambda item: float(item["probability"])
        )

        return {
            "priority": "Routine Review",
            "reason": (
                "No AI prediction crossed its configured threshold. "
                f"Highest model probability was for "
                f"{highest_prediction['class_name']} "
                f"({float(highest_prediction['probability']) * 100:.2f}%)."
            )
        }

    # Highest probability among flagged findings
    highest_flagged = max(
        flagged_predictions,
        key=lambda item: float(item["probability"])
    )

    probability = float(highest_flagged["probability"])
    class_name = highest_flagged["class_name"]

    # Triage workflow rules
    if probability >= 0.70:
        priority = "High Priority"
    elif probability >= 0.40:
        priority = "Moderate Priority"
    else:
        priority = "Routine Review"

    reason = (
        f"{class_name} was flagged by the AI model with "
        f"{probability * 100:.2f}% probability. "
        f"Triage priority is {priority} based on the configured "
        f"AI workflow rules. Clinician review is required."
    )

    return {
        "priority": priority,
        "reason": reason
    }
# TRIAGE MANAGEMENT

# ============================================================

class TriageCreate(BaseModel):

    priority: str

    reason: str | None = None

# ------------------------------------------------------------

# Create Triage Result

# ------------------------------------------------------------

@app.post("/api/analysis/{study_id}/triage")

def create_triage(

    study_id: int,

    triage: TriageCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    allowed_priorities = [

        "High Priority",

        "Moderate Priority",

        "Routine Review",

        "AI Inconclusive / Manual Review"

    ]

    # Check that the priority is valid

    if triage.priority not in allowed_priorities:

        raise HTTPException(

            status_code=400,

            detail={

                "message": "Invalid triage priority",

                "allowed_priorities": allowed_priorities

            }

        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # Check that the study exists

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # Store triage result

        cursor.execute(

            """

            INSERT INTO triage_results (

                study_id,

                priority,

                reason

            )

            VALUES (?, ?, ?)

            """,

            (

                study_id,

                triage.priority,

                triage.reason

            )

        )

        triage_id = cursor.lastrowid

        connection.commit()

        return {

            "message": "Triage result created successfully",

            "triage_id": triage_id,

            "study_id": study_id,

            "priority": triage.priority,

            "reason": triage.reason

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"Failed to create triage result: {str(e)}"

        )

    finally:

        connection.close()

# ------------------------------------------------------------

# Get Triage Result for a Study

# ------------------------------------------------------------

@app.get("/api/analysis/{study_id}/triage")

def get_triage(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # Check that the study exists

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # Get the latest triage result

        cursor.execute(

            """

            SELECT

                id,

                study_id,

                priority,

                reason,

                created_at

            FROM triage_results

            WHERE study_id = ?

            ORDER BY id DESC

            LIMIT 1

            """,

            (study_id,)

        )

        triage_result = cursor.fetchone()

        # No triage result yet

        if triage_result is None:

            return {

                "study_id": study_id,

                "triage_available": False,

                "message": "No triage result available for this study"

            }

        return {

            "study_id": study_id,

            "triage_available": True,

            "triage": dict(triage_result)

        }

    finally:

        connection.close()

# ============================================================

# CLINICIAN REVIEW MANAGEMENT

# ============================================================

class ClinicianReviewCreate(BaseModel):

    # Kept optional for compatibility with older frontend clients.
    # The server always uses the authenticated account identity.
    clinician_name: str | None = None

    review_status: str = "Completed"

    final_decision: str

    comments: str | None = None

# ------------------------------------------------------------

# Create Clinician Review

# ------------------------------------------------------------

@app.post("/api/analysis/{study_id}/review")

def create_clinician_review(

    study_id: int,

    review: ClinicianReviewCreate,

    current_user: dict = Depends(require_roles('Administrator', 'Radiologist'))

):

    allowed_decisions = [

        "Accept",

        "Modify",

        "Reject"

    ]

    allowed_statuses = [

        "Pending",

        "Completed"

    ]

    # Validate review status

    if review.review_status not in allowed_statuses:

        raise HTTPException(

            status_code=400,

            detail={

                "message": "Invalid review status",

                "allowed_statuses": allowed_statuses

            }

        )

    # Validate final decision

    if review.final_decision not in allowed_decisions:

        raise HTTPException(

            status_code=400,

            detail={

                "message": "Invalid final decision",

                "allowed_decisions": allowed_decisions

            }

        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # ----------------------------------------------------

        # Check whether the study exists

        # ----------------------------------------------------

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # ----------------------------------------------------

        # Store clinician review

        # ----------------------------------------------------

        cursor.execute(

            """

            INSERT INTO clinician_reviews

            (

                study_id,

                clinician_name,

                review_status,

                final_decision,

                comments

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                study_id,

                current_user["full_name"],

                review.review_status,

                review.final_decision,

                review.comments

            )

        )

        review_id = cursor.lastrowid

        # Automatically record the clinician review in the audit log.
        cursor.execute(
            """
            INSERT INTO audit_logs (
                action,
                entity_type,
                entity_id,
                performed_by,
                details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "clinician_review_created",
                "study",
                str(study_id),
                current_user["full_name"],
                (
                    f"Review ID {review_id}; "
                    f"status={review.review_status}; "
                    f"decision={review.final_decision}"
                ),
            ),
        )

        connection.commit()

        return {

            "message": "Clinician review created successfully",

            "review_id": review_id,

            "study_id": study_id,

            "clinician_name": current_user["full_name"],

            "review_status": review.review_status,

            "final_decision": review.final_decision,

            "comments": review.comments

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"Failed to create clinician review: {str(e)}"

        )

    finally:

        connection.close()

# ------------------------------------------------------------

# Get Clinician Reviews for a Study

# ------------------------------------------------------------

@app.get("/api/analysis/{study_id}/review")

def get_clinician_reviews(

    study_id: int,

    current_user: dict = Depends(require_roles('Administrator', 'Radiologist'))

):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # ----------------------------------------------------

        # Check whether the study exists

        # ----------------------------------------------------

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # ----------------------------------------------------

        # Retrieve clinician reviews

        # ----------------------------------------------------

        cursor.execute(

            """

            SELECT

                id,

                study_id,

                clinician_name,

                review_status,

                final_decision,

                comments,

                created_at

            FROM clinician_reviews

            WHERE study_id = ?

            ORDER BY id DESC

            """,

            (study_id,)

        )

        reviews = [dict(row) for row in cursor.fetchall()]

        return {

            "study_id": study_id,

            "count": len(reviews),

            "reviews": reviews

        }

    finally:

        connection.close()

# ============================================================

# REPORTS AND AUDIT LOGS MANAGEMENT

# ============================================================

class ReportCreate(BaseModel):

    report_status: str = "Draft"

    report_text: str | None = None

    generated_by: str | None = None

class AuditLogCreate(BaseModel):

    action: str

    entity_type: str | None = None

    entity_id: str | None = None

    performed_by: str | None = None

    details: str | None = None

# ------------------------------------------------------------

# Create Report

# ------------------------------------------------------------

@app.post("/api/reports/{study_id}")

def create_report(

    study_id: int,

    report: ReportCreate,
    _rbac_user: dict = Depends(require_roles('Administrator'))
):

    allowed_statuses = [

        "Draft",

        "Final"

    ]

    if report.report_status not in allowed_statuses:

        raise HTTPException(

            status_code=400,

            detail={

                "message": "Invalid report status",

                "allowed_statuses": allowed_statuses

            }

        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            "SELECT id FROM studies WHERE id = ?",

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        cursor.execute(

            """

            INSERT INTO reports (

                study_id,

                report_status,

                report_text,

                generated_by

            )

            VALUES (?, ?, ?, ?)

            """,

            (

                study_id,

                report.report_status,

                report.report_text,

                report.generated_by

            )

        )

        report_id = cursor.lastrowid

        # Automatically record report creation in the audit log.
        cursor.execute(
            """
            INSERT INTO audit_logs (
                action,
                entity_type,
                entity_id,
                performed_by,
                details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "report_created",
                "study",
                str(study_id),
                report.generated_by,
                (
                    f"Report ID {report_id}; "
                    f"status={report.report_status}"
                ),
            ),
        )

        connection.commit()

        return {

            "message": "Report created successfully",

            "report_id": report_id,

            "study_id": study_id,

            "report_status": report.report_status,

            "report_text": report.report_text,

            "generated_by": report.generated_by

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"Failed to create report: {str(e)}"

        )

    finally:

        connection.close()

# ------------------------------------------------------------

# Get Reports for a Study

# ------------------------------------------------------------

@app.get("/api/reports/{study_id}")

def get_reports(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            "SELECT id FROM studies WHERE id = ?",

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        cursor.execute(

            """

            SELECT

                id,

                study_id,

                report_status,

                report_text,

                generated_by,

                created_at,

                updated_at

            FROM reports

            WHERE study_id = ?

            ORDER BY id DESC

            """,

            (study_id,)

        )

        reports = [dict(row) for row in cursor.fetchall()]

        return {

            "study_id": study_id,

            "count": len(reports),

            "reports": reports

        }

    finally:

        connection.close()

# ------------------------------------------------------------

# Create Audit Log

# ------------------------------------------------------------

@app.post("/api/audit-logs")

def create_audit_log(
    log: AuditLogCreate,
    current_admin: dict = Depends(require_admin)
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            INSERT INTO audit_logs (

                action,

                entity_type,

                entity_id,

                performed_by,

                details

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                log.action,

                log.entity_type,

                log.entity_id,

                log.performed_by,

                log.details

            )

        )

        log_id = cursor.lastrowid

        connection.commit()

        return {

            "message": "Audit log created successfully",

            "audit_log_id": log_id,

            "action": log.action,

            "entity_type": log.entity_type,

            "entity_id": log.entity_id,

            "performed_by": log.performed_by,

            "details": log.details

        }

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"Failed to create audit log: {str(e)}"

        )

    finally:

        connection.close()

# ------------------------------------------------------------

# Get Audit Logs

# ------------------------------------------------------------

@app.get("/api/audit-logs")

def get_audit_logs(
    current_admin: dict = Depends(require_admin)
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            SELECT

                id,

                action,

                entity_type,

                entity_id,

                performed_by,

                details,

                created_at

            FROM audit_logs

            ORDER BY id DESC

            """

        )

        logs = [dict(row) for row in cursor.fetchall()]

        return {

            "count": len(logs),

            "audit_logs": logs

        }

    finally:

        connection.close()

# ABNORMALITY MANAGEMENT

# ==================================================

# --------------------------------------------------

# Abnormality Request Model

# --------------------------------------------------

class AbnormalityCreate(BaseModel):

    name: str

    description: str | None = None

    status: str = "Active"

# --------------------------------------------------

# Create Abnormality

# --------------------------------------------------

@app.post("/api/abnormalities")

def create_abnormality(abnormality: AbnormalityCreate,
    _rbac_user: dict = Depends(require_roles('Administrator'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            INSERT INTO abnormalities (

                name,

                description,

                status

            )

            VALUES (?, ?, ?)

            """,

            (

                abnormality.name,

                abnormality.description,

                abnormality.status

            )

        )

        connection.commit()

        abnormality_id = cursor.lastrowid

        return {

            "message": "Abnormality created successfully",

            "abnormality_id": abnormality_id,

            "name": abnormality.name,

            "description": abnormality.description,

            "status": abnormality.status

        }

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

# --------------------------------------------------

# Get All Abnormalities

# --------------------------------------------------

@app.get("/api/abnormalities")

def get_abnormalities(
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT *

        FROM abnormalities

        ORDER BY id ASC

        """

    )

    rows = cursor.fetchall()

    connection.close()

    return {

        "count": len(rows),

        "abnormalities": [dict(row) for row in rows]

    }

# --------------------------------------------------

# Get One Abnormality

# --------------------------------------------------

@app.get("/api/abnormalities/{abnormality_id}")

def get_abnormality(abnormality_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT *

        FROM abnormalities

        WHERE id = ?

        """,

        (abnormality_id,)

    )

    row = cursor.fetchone()

    connection.close()

    if row is None:

        raise HTTPException(

            status_code=404,

            detail="Abnormality not found"

        )

    return dict(row)

# ==================================================
# ============================================================
# BREAST ULTRASOUND PREDICTION API
# ============================================================

BREAST_ULTRASOUND_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

@app.post("/api/ultrasound/breast/predict")
async def predict_breast_ultrasound_endpoint(file: UploadFile = File(...),
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """Run research breast-ultrasound inference on an uploaded image."""
    original_name = file.filename or "ultrasound.png"
    extension = Path(original_name).suffix.lower()
    if extension not in BREAST_ULTRASOUND_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG breast ultrasound image.")

    temporary_path = UPLOAD_DIR / f"breast_ultrasound_{uuid.uuid4().hex}{extension}"
    try:
        with temporary_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        result = predict_breast_ultrasound(str(temporary_path))

        mask_array = np.asarray(result.pop("segmentation_mask"), dtype=np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode="L")
        from io import BytesIO
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        result["segmentation_mask_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        result["original_filename"] = Path(original_name).name
        result["note"] = (
            "Research output only. These results are not a diagnosis "
            "and require qualified clinical review."
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Breast ultrasound inference failed: {exc}") from exc
    finally:
        await file.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

# ============================================================
# ABDOMINAL ULTRASOUND PREDICTION API
# ============================================================
ABDOMINAL_ULTRASOUND_ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

@app.post("/api/ultrasound/abdominal/predict")
async def predict_abdominal_ultrasound_endpoint(
    file: UploadFile = File(...),
    _rbac_user: dict = Depends(
        require_roles("Administrator", "Radiologist", "AI/ML Engineer")
    ),
):
    """Run research-only abdominal ultrasound classification."""
    original_name = file.filename or "abdominal_ultrasound.png"
    extension = Path(original_name).suffix.lower()

    if extension not in ABDOMINAL_ULTRASOUND_ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, PNG, BMP, or WEBP abdominal ultrasound image.",
        )

    temporary_path = (
        UPLOAD_DIR / f"abdominal_ultrasound_{uuid.uuid4().hex}{extension}"
    )

    try:
        with temporary_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = predict_abdominal_ultrasound(str(temporary_path))
        result["original_filename"] = Path(original_name).name
        return result

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Abdominal ultrasound inference failed: {exc}",
        ) from exc
    finally:
        await file.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

# BASIC ENDPOINTS

# ==================================================

@app.get("/")

def root():

    return {

        "message": "AI Medical Imaging Triage & XAI",

        "status": "running"

    }

@app.get("/health")

def health():

    return {

        "status": "healthy"

    }

# ==================================================
# AUTHENTICATION
# ==================================================

@app.post("/api/auth/login")
def login_user(login: LoginRequest):

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # Find the user by User ID
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                username,
                email,
                password_hash,
                role,
                account_status,
                patient_record_id
            FROM users
            WHERE username = ?
            """,
            (login.username,)
        )

        user = cursor.fetchone()

        # Do not reveal whether the User ID exists
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid User ID or password"
            )

        # Check whether the account is active
        if user["account_status"] != "ACTIVE":
            raise HTTPException(
                status_code=403,
                detail="This account is not active"
            )

        # Verify the password against the stored hash
        if not verify_password(
            login.password,
            user["password_hash"]
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid User ID or password"
            )

        # Create JWT access token after successful password verification
        access_token = create_access_token(
            user_id=user["id"],
            username=user["username"],
            role=user["role"]
        )

        # Successful login
        return {
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "patient_record_id": user["patient_record_id"]
            }
        }

    finally:
        connection.close()

@app.get("/api/auth/me")
def get_logged_in_user(
    current_user: dict = Depends(get_current_user)
):
    return {
        "authenticated": True,
        "user": {
            "id": current_user["id"],
            "full_name": current_user["full_name"],
            "username": current_user["username"],
            "email": current_user["email"],
            "role": current_user["role"],
            "patient_record_id": current_user.get("patient_record_id")
        }
    }

# ==================================================
# ADMIN USER MANAGEMENT
# ==================================================

@app.post("/api/admin/users")
def create_user_by_admin(
    user_data: AdminUserCreate,
    current_admin: dict = Depends(require_admin)
):
    allowed_roles = ["Radiologist", "Technician", "Clinician", "AI/ML Engineer"]

    if user_data.role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid role",
                "allowed_roles": allowed_roles
            }
        )

    if not user_data.full_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Full Name cannot be empty"
        )

    if not user_data.username.strip():
        raise HTTPException(
            status_code=400,
            detail="User ID cannot be empty"
        )

    if not user_data.email.strip():
        raise HTTPException(
            status_code=400,
            detail="Email cannot be empty"
        )

    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters"
        )

    # In this project, Clinician is the patient-facing login role.
    if user_data.role == ROLE_CLINICIAN and not user_data.patient_record_id:
        raise HTTPException(
            status_code=400,
            detail="Select an existing patient record for a Clinician account"
        )
    if user_data.role != ROLE_CLINICIAN and user_data.patient_record_id is not None:
        raise HTTPException(
            status_code=400,
            detail="Only Clinician accounts can be linked to a patient record"
        )

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            """,
            (user_data.username.strip(),)
        )

        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail="User ID already exists"
            )

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (user_data.email.strip(),)
        )

        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail="Email already exists"
            )

        if user_data.patient_record_id is not None:
            cursor.execute("SELECT id FROM patients WHERE id = ?", (user_data.patient_record_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Selected patient record was not found")
            cursor.execute("SELECT id FROM users WHERE patient_record_id = ?", (user_data.patient_record_id,))
            if cursor.fetchone() is not None:
                raise HTTPException(status_code=409, detail="This patient record is already linked to a login account")

        password_hash = hash_password(user_data.password)

        cursor.execute(
            """
            INSERT INTO users (
                full_name,
                username,
                email,
                password_hash,
                role,
                account_status,
                email_verified,
                failed_login_attempts,
                patient_record_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_data.full_name.strip(),
                user_data.username.strip(),
                user_data.email.strip(),
                password_hash,
                user_data.role,
                "ACTIVE",
                1,
                0,
                user_data.patient_record_id
            )
        )

        new_user_id = cursor.lastrowid
        connection.commit()

        return {
            "message": "User created successfully",
            "created_by": {
                "username": current_admin["username"],
                "role": current_admin["role"]
            },
            "user": {
                "id": new_user_id,
                "full_name": user_data.full_name.strip(),
                "username": user_data.username.strip(),
                "email": user_data.email.strip(),
                "role": user_data.role,
                "account_status": "ACTIVE",
                "patient_record_id": user_data.patient_record_id
            }
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception as e:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create user: {str(e)}"
        )

    finally:
        connection.close()

# ==================================================
# ADMIN USER LIST
# ==================================================

@app.get("/api/admin/users")
def get_admin_users(
    current_admin: dict = Depends(require_admin)
):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                username,
                email,
                role,
                account_status,
                email_verified,
                created_at,
                updated_at,
                patient_record_id
            FROM users
            ORDER BY id DESC
            """
        )

        users = [dict(row) for row in cursor.fetchall()]

        return {
            "count": len(users),
            "users": users
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load users: {str(e)}"
        )

    finally:
        connection.close()

# ==================================================

# ==================================================
# CLINICIAN (PATIENT-FACING) DASHBOARD AND REPORT HISTORY
# ==================================================

@app.get("/api/clinician/me")
def get_clinician_dashboard(current_user: dict = Depends(require_roles('Clinician'))):
    patient_record_id = current_user.get("patient_record_id")
    if not patient_record_id:
        raise HTTPException(status_code=403, detail="This Clinician account is not linked to a patient record")

    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, patient_id, name, age, gender, created_at FROM patients WHERE id = ?",
            (patient_record_id,)
        )
        patient = cursor.fetchone()
        if patient is None:
            raise HTTPException(status_code=404, detail="Linked patient record was not found")
        cursor.execute("""
            SELECT COUNT(*) AS report_count
            FROM reports r
            JOIN studies s ON s.id = r.study_id
            WHERE s.patient_id = ?
        """, (patient_record_id,))
        report_count = cursor.fetchone()["report_count"]
        return {"patient": dict(patient), "report_count": report_count}
    finally:
        connection.close()

@app.get("/api/report-history")
def get_report_history(current_user: dict = Depends(get_current_user)):
    role = normalize_role(current_user.get("role", ""))
    if role not in (ROLE_ADMIN, ROLE_RADIOLOGIST, ROLE_CLINICIAN):
        raise HTTPException(status_code=403, detail="Your role is not permitted to view report history")
    if role == ROLE_CLINICIAN and not current_user.get("patient_record_id"):
        raise HTTPException(status_code=403, detail="This Clinician account is not linked to a patient record")

    connection = get_connection()
    try:
        cursor = connection.cursor()
        query = """
            SELECT r.id AS report_id, p.patient_id AS patient_code, p.name AS patient_name,
                   s.id AS study_id, s.study_type, s.study_date, r.report_status,
                   r.created_at, r.report_text
            FROM reports r
            JOIN studies s ON s.id = r.study_id
            JOIN patients p ON p.id = s.patient_id
        """
        params = ()
        if role == ROLE_CLINICIAN:
            query += " WHERE p.id = ?"
            params = (current_user["patient_record_id"],)
        query += " ORDER BY r.created_at DESC, r.id DESC"
        cursor.execute(query, params)
        reports = [dict(row) for row in cursor.fetchall()]
        return {"count": len(reports), "reports": reports}
    finally:
        connection.close()

# ==================================================

# PATIENT MANAGEMENT

# ==================================================

# --------------------------------------------------

# Create Patient

# --------------------------------------------------

@app.post("/api/patients")

def create_patient(patient: PatientCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            INSERT INTO patients (

                patient_id,

                name,

                age,

                gender

            )

            VALUES (?, ?, ?, ?)

            """,

            (

                patient.patient_id,

                patient.name,

                patient.age,

                patient.gender

            )

        )

        connection.commit()

        return {

            "message": "Patient created successfully",

            "patient_id": patient.patient_id

        }

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

# --------------------------------------------------

# Get All Patients

# --------------------------------------------------

@app.get("/api/patients")

def get_patients(
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT *

        FROM patients

        ORDER BY id DESC

        """

    )

    rows = cursor.fetchall()

    connection.close()

    return {

        "count": len(rows),

        "patients": [dict(row) for row in rows]

    }

# --------------------------------------------------

# Get One Patient

# --------------------------------------------------

@app.get("/api/patients/{patient_id}")

def get_patient(patient_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT *

        FROM patients

        WHERE id = ?

        """,

        (patient_id,)

    )

    row = cursor.fetchone()

    connection.close()

    if row is None:

        raise HTTPException(

            status_code=404,

            detail="Patient not found"

        )

    return dict(row)

# ==================================================

# STUDY MANAGEMENT

# ==================================================

# --------------------------------------------------

# Create Study

# --------------------------------------------------

@app.post("/api/studies")

def create_study(study: StudyCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # Check whether patient exists

        cursor.execute(

            """

            SELECT id

            FROM patients

            WHERE id = ?

            """,

            (study.patient_id,)

        )

        patient = cursor.fetchone()

        if patient is None:

            raise HTTPException(

                status_code=404,

                detail="Patient not found"

            )

        # Create study

        cursor.execute(

            """

            INSERT INTO studies (

                patient_id,

                study_type,

                study_date

            )

            VALUES (?, ?, ?)

            """,

            (

                study.patient_id,

                study.study_type,

                study.study_date

            )

        )

        connection.commit()

        study_id = cursor.lastrowid

        return {

            "message": "Study created successfully",

            "study_id": study_id,

            "patient_id": study.patient_id,

            "study_type": study.study_type,

            "study_date": study.study_date

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

# --------------------------------------------------

# Get All Studies

# --------------------------------------------------

@app.get("/api/studies")

def get_studies(
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT

            studies.id,

            studies.patient_id,

            patients.patient_id AS patient_code,

            patients.name AS patient_name,

            studies.study_type,

            studies.study_date,

            studies.created_at

        FROM studies

        JOIN patients

            ON studies.patient_id = patients.id

        ORDER BY studies.id DESC

        """

    )

    rows = cursor.fetchall()

    connection.close()

    return {

        "count": len(rows),

        "studies": [dict(row) for row in rows]

    }

# ==================================================

# DATABASE IMAGE RECORDS

# IMPORTANT:

# This route must come BEFORE /api/studies/{study_id}

# ==================================================

@app.get("/api/studies/database")

def get_database_images(
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT *

        FROM medical_images

        ORDER BY id DESC

        """

    )

    rows = cursor.fetchall()

    connection.close()

    return {

        "count": len(rows),

        "images": [dict(row) for row in rows]

    }

# --------------------------------------------------

# Get One Study

# --------------------------------------------------

@app.get("/api/studies/{study_id}")

def get_study(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(

        """

        SELECT

            studies.id,

            studies.patient_id,

            patients.patient_id AS patient_code,

            patients.name AS patient_name,

            patients.age,

            patients.gender,

            studies.study_type,

            studies.study_date,

            studies.created_at

        FROM studies

        JOIN patients

            ON studies.patient_id = patients.id

        WHERE studies.id = ?

        """,

        (study_id,)

    )

    row = cursor.fetchone()

    connection.close()

    if row is None:

        raise HTTPException(

            status_code=404,

            detail="Study not found"

        )

    return dict(row)

# ==================================================

# MEDICAL IMAGE UPLOAD

# ==================================================

@app.post("/api/studies/upload")

async def upload_study(

    file: UploadFile = File(...),

    study_id: int | None = None,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    # Check file extension

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(

            status_code=400,

            detail="Unsupported file type"

        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # --------------------------------------------------

        # Check whether study exists

        # --------------------------------------------------

        if study_id is not None:

            cursor.execute(

                """

                SELECT id

                FROM studies

                WHERE id = ?

                """,

                (study_id,)

            )

            study = cursor.fetchone()

            if study is None:

                raise HTTPException(

                    status_code=404,

                    detail="Study not found"

                )

        # --------------------------------------------------

        # Create unique file ID

        # --------------------------------------------------

        file_id = str(uuid.uuid4())

        saved_filename = f"{file_id}{extension}"

        file_path = UPLOAD_DIR / saved_filename

        # --------------------------------------------------

        # Save uploaded file

        # --------------------------------------------------

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(

                file.file,

                buffer

            )

        # --------------------------------------------------

        # Save image information in SQLite

        # --------------------------------------------------

        cursor.execute(

            """

            INSERT INTO medical_images (

                study_id,

                file_id,

                original_filename,

                stored_filename,

                file_path,

                file_type,

                validation_status

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

            """,

            (

                study_id,

                file_id,

                file.filename,

                saved_filename,

                str(file_path),

                extension,

                "Pending"

            )

        )

        connection.commit()

        return {

            "message": "File uploaded successfully",

            "file_id": file_id,

            "study_id": study_id,

            "filename": file.filename,

            "saved_as": saved_filename

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

# ==================================================

# IMAGE VALIDATION

# ==================================================

@app.post("/api/studies/{file_id}/validate")

def validate_image(file_id: str,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    # Find image in database

    cursor.execute(

        """

        SELECT *

        FROM medical_images

        WHERE file_id = ?

        """,

        (file_id,)

    )

    image_record = cursor.fetchone()

    if not image_record:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Image not found"

        )

    file_path = image_record["file_path"]

    # Check that file exists

    if not Path(file_path).exists():

        cursor.execute(

            """

            UPDATE medical_images

            SET validation_status = ?

            WHERE file_id = ?

            """,

            ("Invalid", file_id)

        )

        connection.commit()

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Image file not found"

        )

    # Try to open and validate the image

    try:

        image = Image.open(file_path)

        width, height = image.size

        image_format = image.format

        mode = image.mode

        # Successful validation

        cursor.execute(

            """

            UPDATE medical_images

            SET validation_status = ?

            WHERE file_id = ?

            """,

            ("Valid", file_id)

        )

        connection.commit()

        connection.close()

        return {

            "file_id": file_id,

            "valid": True,

            "validation_status": "Valid",

            "width": width,

            "height": height,

            "format": image_format,

            "mode": mode

        }

    except Exception as error:

        # Failed validation

        cursor.execute(

            """

            UPDATE medical_images

            SET validation_status = ?

            WHERE file_id = ?

            """,

            ("Invalid", file_id)

        )

        connection.commit()

        connection.close()

        return {

            "file_id": file_id,

            "valid": False,

            "validation_status": "Invalid",

            "message": str(error)

        }

# ==================================================

# IMAGE PREPROCESSING

# ==================================================

@app.post("/api/studies/{file_id}/preprocess")

async def preprocess_image(file_id: str,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    matching_files = list(

        UPLOAD_DIR.glob(f"{file_id}.*")

    )

    if not matching_files:

        raise HTTPException(

            status_code=404,

            detail="Uploaded file not found"

        )

    file_path = matching_files[0]

    try:

        with Image.open(file_path) as image:

            # Convert image to RGB

            image = image.convert("RGB")

            # Resize image

            image = image.resize(

                (224, 224)

            )

            # Convert image to NumPy array

            image_array = np.array(image)

            # Normalize pixels

            normalized = (

                image_array.astype(np.float32)

                / 255.0

            )

        # Update preprocessing status in SQLite

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(

            """

            UPDATE medical_images

            SET preprocessing_status = ?

            WHERE file_id = ?

            """,

            ("Preprocessed", file_id)

        )

        connection.commit()

        connection.close()

        return {

            "file_id": file_id,

            "preprocessed": True,

            "preprocessing_status": "Preprocessed",

            "size": [224, 224],

            "channels": 3,

            "min_pixel_value": float(

                normalized.min()

            ),

            "max_pixel_value": float(

                normalized.max()

            )

        }

    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=f"Preprocessing failed: {str(e)}"

        )
# ============================================================
# CT DICOM SERIES UPLOAD + PREPROCESSING + ANALYSIS
# ============================================================

CT_UPLOAD_DIR = PROJECT_ROOT / "uploads" / "ct"
CT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _ct_series_dir(series_id: str) -> Path:
    if not series_id or Path(series_id).name != series_id:
        raise HTTPException(status_code=400, detail="Invalid CT series ID")
    return CT_UPLOAD_DIR / series_id

def _load_ct_series_metadata(series_id: str) -> dict:
    path = _ct_series_dir(series_id) / "series_metadata.json"
    if not path.exists():
        return {"series_id": series_id, "study_id": None}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read CT series metadata: {str(exc)}"
        )

def _save_ct_series_metadata(series_id: str, metadata: dict):
    path = _ct_series_dir(series_id) / "series_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

@app.post("/api/ct/upload-series")
async def upload_ct_series(
    files: Annotated[list[UploadFile], File(...)],
    study_id: int | None = None,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """Upload and persist a complete CT DICOM series."""
    if not files:
        raise HTTPException(status_code=400, detail="No CT DICOM files were uploaded")

    connection = get_connection()
    cursor = connection.cursor()
    series_dir = None

    try:
        if study_id is not None:
            cursor.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Study not found")

        series_id = str(uuid.uuid4())
        series_dir = CT_UPLOAD_DIR / series_id
        series_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for index, file in enumerate(files):
            original_filename = file.filename or f"slice_{index + 1}.dcm"
            if Path(original_filename).suffix.lower() != ".dcm":
                raise HTTPException(
                    status_code=400,
                    detail=f"CT series can contain only .dcm files: {original_filename}"
                )

            saved_filename = f"slice_{index + 1:04d}.dcm"
            file_path = series_dir / saved_filename

            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append({
                "original_filename": original_filename,
                "saved_filename": saved_filename,
                "file_path": str(file_path),
            })

        _save_ct_series_metadata(
            series_id,
            {
                "series_id": series_id,
                "study_id": study_id,
                "slice_count": len(saved_files),
                "files": saved_files,
                "status": "Uploaded",
            },
        )

        connection.commit()

        return {
            "success": True,
            "message": "CT DICOM series uploaded successfully",
            "series_id": series_id,
            "study_id": study_id,
            "slice_count": len(saved_files),
            "series_directory": str(series_dir),
            "next_step": f"/api/ct/preprocess/{series_id}",
        }

    except HTTPException:
        connection.rollback()
        if series_dir is not None and series_dir.exists():
            shutil.rmtree(series_dir, ignore_errors=True)
        raise
    except Exception as exc:
        connection.rollback()
        if series_dir is not None and series_dir.exists():
            shutil.rmtree(series_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"CT series upload failed: {str(exc)}"
        )
    finally:
        connection.close()

@app.post("/api/ct/preprocess/{series_id}")
async def preprocess_ct_series(series_id: str,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """
    DICOM -> 1 mm isotropic -> HU [-1000,400] -> [0,1].
    Saves the complete preprocessed volume as ct_volume.npy.
    """
    series_dir = _ct_series_dir(series_id)

    if not series_dir.exists():
        raise HTTPException(status_code=404, detail="CT series not found")

    try:
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise HTTPException(status_code=400, detail="No DICOM files found in CT series")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(series_dir))

        if series_ids:
            series_file_names = reader.GetGDCMSeriesFileNames(
                str(series_dir), series_ids[0]
            )
        else:
            series_file_names = [str(p) for p in dicom_files]

        if not series_file_names:
            raise HTTPException(status_code=400, detail="Unable to read DICOM series")

        reader.SetFileNames(series_file_names)
        ct_image = reader.Execute()

        original_size = list(ct_image.GetSize())
        original_spacing = list(ct_image.GetSpacing())

        spacing = np.array(ct_image.GetSpacing(), dtype=np.float32)
        size = np.array(ct_image.GetSize(), dtype=np.float32)
        target = np.array(CT_TARGET_SPACING, dtype=np.float32)

        new_size = np.maximum(
            1,
            np.round(size * spacing / target).astype(int)
        ).tolist()

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(CT_TARGET_SPACING)
        resampler.SetSize([int(x) for x in new_size])
        resampler.SetOutputDirection(ct_image.GetDirection())
        resampler.SetOutputOrigin(ct_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(CT_HU_MIN)

        resampled = resampler.Execute(ct_image)

        volume_hu = sitk.GetArrayFromImage(resampled).astype(np.float32)
        volume_hu = np.clip(volume_hu, CT_HU_MIN, CT_HU_MAX)
        volume = (
            (volume_hu - CT_HU_MIN) /
            (CT_HU_MAX - CT_HU_MIN)
        ).astype(np.float32)

        processed_dir = series_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        volume_path = processed_dir / "ct_volume.npy"
        np.save(volume_path, volume)

        series_metadata = _load_ct_series_metadata(series_id)

        preprocessing_metadata = {
            "series_id": series_id,
            "study_id": series_metadata.get("study_id"),
            "slice_count": len(dicom_files),
            "original_size": original_size,
            "original_spacing": original_spacing,
            "resampled_size": list(resampled.GetSize()),
            "resampled_spacing": list(resampled.GetSpacing()),
            "volume_shape": list(volume.shape),
            "target_spacing": list(CT_TARGET_SPACING),
            "hu_clip_min": CT_HU_MIN,
            "hu_clip_max": CT_HU_MAX,
            "normalization": "[0,1]",
            "volume_path": str(volume_path),
        }

        with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(preprocessing_metadata, f, indent=2)

        series_metadata.update({
            "status": "Preprocessed",
            "preprocessing": preprocessing_metadata,
        })
        _save_ct_series_metadata(series_id, series_metadata)

        return {
            "success": True,
            "message": "CT preprocessing completed successfully",
            "series_id": series_id,
            "study_id": series_metadata.get("study_id"),
            "original": {
                "size": original_size,
                "spacing": original_spacing,
            },
            "processed": {
                "size": list(resampled.GetSize()),
                "spacing": list(resampled.GetSpacing()),
                "volume_shape": list(volume.shape),
            },
            "normalization": {
                "hu_min": CT_HU_MIN,
                "hu_max": CT_HU_MAX,
                "min": float(volume.min()),
                "max": float(volume.max()),
            },
            "volume_file": str(volume_path),
            "model_input": {
                "shape": [1, 1, 32, 32, 32],
                "patch_size": list(CT_PATCH_SIZE),
                "channels": 1,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CT preprocessing failed: {str(exc)}"
        )

class CTCandidate(BaseModel):
    z: int
    y: int
    x: int

class CTAnalyzeRequest(BaseModel):
    candidates: list[CTCandidate]

def _extract_ct_patch(volume: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    half = 16
    depth, height, width = volume.shape

    z1, z2 = z - half, z + half
    y1, y2 = y - half, y + half
    x1, x2 = x - half, x + half

    if z1 < 0 or y1 < 0 or x1 < 0 or z2 > depth or y2 > height or x2 > width:
        raise ValueError(
            "Candidate is too close to a volume boundary for a 32x32x32 patch."
        )

    patch = volume[z1:z2, y1:y2, x1:x2]

    if patch.shape != CT_PATCH_SIZE:
        raise ValueError(
            f"Extracted patch has unexpected shape: {tuple(patch.shape)}"
        )

    return patch

# Research candidate-proposal settings.
# These values control a sliding-window proposal pass around the
# existing candidate-level classifier; they are not medically validated
# whole-CT detection parameters.
CT_SCAN_STRIDE = 16
CT_SCAN_BATCH_SIZE = 16
CT_SCAN_MAX_CANDIDATES = 20
CT_SCAN_SUPPRESSION_DISTANCE = 24.0

def _predict_ct_patch_batch(patches: np.ndarray):
    """
    Run the trained candidate classifier on a batch of preprocessed
    32x32x32 CT patches.

    Input shape:
        [N, 32, 32, 32]
    """
    patches = np.asarray(patches, dtype=np.float32)

    if patches.ndim != 4 or tuple(patches.shape[1:]) != CT_PATCH_SIZE:
        raise ValueError(
            "Invalid CT patch batch shape. "
            f"Expected [N, {CT_PATCH_SIZE[0]}, {CT_PATCH_SIZE[1]}, {CT_PATCH_SIZE[2]}], "
            f"got {tuple(patches.shape)}."
        )

    if patches.shape[0] == 0:
        return []

    if not np.isfinite(patches).all():
        raise ValueError("CT patch batch contains NaN or infinite values.")

    patch_min = float(patches.min())
    patch_max = float(patches.max())
    if patch_min < 0.0 or patch_max > 1.0:
        raise ValueError(
            "CT patch values must be normalized to [0, 1]. "
            f"Received range [{patch_min:.4f}, {patch_max:.4f}]."
        )

    model_input = torch.from_numpy(patches).float().unsqueeze(1)

    with torch.no_grad():
        logits = ct_model(model_input)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = []
    for probability_non_nodule, probability_nodule in probabilities:
        probability_non_nodule = float(probability_non_nodule)
        probability_nodule = float(probability_nodule)
        prediction = 1 if probability_nodule >= CT_NODULE_THRESHOLD else 0

        predictions.append({
            "class_id": prediction,
            "class": CT_CLASS_NAMES[prediction],
            "probability_nodule": round(probability_nodule, 4),
            "probability_non_nodule": round(probability_non_nodule, 4),
            "threshold": CT_NODULE_THRESHOLD,
        })

    return predictions

def _scan_window_starts(length: int, patch_length: int, stride: int):
    """Return starts that cover the full axis without partial patches."""
    if length < patch_length:
        return []

    starts = list(range(0, length - patch_length + 1, stride))
    last_start = length - patch_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def _suppress_ct_candidates(candidates, distance_threshold: float):
    """
    Greedy 3D spatial suppression of overlapping/highly nearby proposals.

    Candidates must already be sorted by descending nodule probability.
    This is a research post-processing heuristic, not a clinical rule.
    """
    kept = []
    threshold_sq = float(distance_threshold) ** 2

    for candidate in candidates:
        z, y, x = candidate["center"]["z"], candidate["center"]["y"], candidate["center"]["x"]

        too_close = False
        for existing in kept:
            ez = existing["center"]["z"]
            ey = existing["center"]["y"]
            ex = existing["center"]["x"]
            distance_sq = (
                (z - ez) ** 2 +
                (y - ey) ** 2 +
                (x - ex) ** 2
            )
            if distance_sq <= threshold_sq:
                too_close = True
                break

        if not too_close:
            kept.append(candidate)

    return kept

def _get_or_create_ct_model_record(cursor):
    cursor.execute(
        """
        SELECT id
        FROM ai_models
        WHERE model_name = ? AND version = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (CT_MODEL_NAME, CT_MODEL_VERSION),
    )
    row = cursor.fetchone()

    if row is not None:
        return row["id"]

    cursor.execute(
        """
        INSERT INTO ai_models (
            model_name,
            version,
            model_path,
            status
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            CT_MODEL_NAME,
            CT_MODEL_VERSION,
            str(CT_MODEL_FILE),
            "Active",
        ),
    )
    return cursor.lastrowid

@app.post("/api/ct/scan/{series_id}")
async def scan_ct_series(
    series_id: str,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """
    Research candidate-proposal pass over a preprocessed CT volume.

    This endpoint slides a 32x32x32 window through the complete CT,
    classifies each patch with the existing candidate-level 3D CNN,
    retains patches at/above the configured nodule threshold, and
    applies simple 3D spatial suppression to merge nearby proposals.

    IMPORTANT:
        This is not a clinically validated whole-CT nodule detector.
        The trained model is a candidate-level classifier. Returned
        locations are AI research candidates for further review.
    """
    series_dir = _ct_series_dir(series_id)

    if not series_dir.exists():
        raise HTTPException(status_code=404, detail="CT series not found")

    volume_path = series_dir / "processed" / "ct_volume.npy"
    if not volume_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Run /api/ct/preprocess/{series_id} before CT scanning."
        )

    try:
        volume = np.load(volume_path, allow_pickle=False).astype(np.float32)

        if volume.ndim != 3:
            raise ValueError("Preprocessed CT volume must be 3-dimensional.")

        depth, height, width = volume.shape
        patch_depth, patch_height, patch_width = CT_PATCH_SIZE

        z_starts = _scan_window_starts(depth, patch_depth, CT_SCAN_STRIDE)
        y_starts = _scan_window_starts(height, patch_height, CT_SCAN_STRIDE)
        x_starts = _scan_window_starts(width, patch_width, CT_SCAN_STRIDE)

        if not z_starts or not y_starts or not x_starts:
            raise ValueError(
                "Preprocessed CT volume is smaller than the required 32x32x32 patch."
            )

        window_count = len(z_starts) * len(y_starts) * len(x_starts)

        batch_patches = []
        batch_centers = []
        raw_candidates = []

        def flush_batch():
            if not batch_patches:
                return

            predictions = _predict_ct_patch_batch(np.stack(batch_patches, axis=0))

            for center, prediction in zip(batch_centers, predictions):
                if prediction["probability_nodule"] >= CT_NODULE_THRESHOLD:
                    raw_candidates.append({
                        "center": center,
                        "patch_shape": list(CT_PATCH_SIZE),
                        "prediction": prediction,
                    })

            batch_patches.clear()
            batch_centers.clear()

        for z1 in z_starts:
            for y1 in y_starts:
                for x1 in x_starts:
                    patch = volume[
                        z1:z1 + patch_depth,
                        y1:y1 + patch_height,
                        x1:x1 + patch_width,
                    ]

                    center = {
                        "z": z1 + patch_depth // 2,
                        "y": y1 + patch_height // 2,
                        "x": x1 + patch_width // 2,
                    }

                    batch_patches.append(patch)
                    batch_centers.append(center)

                    if len(batch_patches) >= CT_SCAN_BATCH_SIZE:
                        flush_batch()

        flush_batch()

        raw_candidates.sort(
            key=lambda item: item["prediction"]["probability_nodule"],
            reverse=True,
        )

        suppressed_candidates = _suppress_ct_candidates(
            raw_candidates,
            CT_SCAN_SUPPRESSION_DISTANCE,
        )

        final_candidates = suppressed_candidates[:CT_SCAN_MAX_CANDIDATES]

        series_metadata = _load_ct_series_metadata(series_id)
        study_id = series_metadata.get("study_id")

        scan_payload = {
            "series_id": series_id,
            "study_id": study_id,
            "task": "Research CT candidate proposal using sliding-window candidate classification",
            "model": {
                "name": CT_MODEL_NAME,
                "version": CT_MODEL_VERSION,
                "architecture": "LungNodule3DCNN",
                "classes": CT_CLASS_NAMES,
                "threshold": CT_NODULE_THRESHOLD,
            },
            "scan": {
                "volume_shape": list(volume.shape),
                "patch_size": list(CT_PATCH_SIZE),
                "stride": CT_SCAN_STRIDE,
                "batch_size": CT_SCAN_BATCH_SIZE,
                "windows_evaluated": window_count,
                "raw_candidates_above_threshold": len(raw_candidates),
                "suppression_distance": CT_SCAN_SUPPRESSION_DISTANCE,
                "max_returned_candidates": CT_SCAN_MAX_CANDIDATES,
            },
            "candidates": final_candidates,
            "clinical_note": (
                "AI research output only. This endpoint uses a candidate-level "
                "classifier in a sliding-window proposal workflow and is not a "
                "clinically validated whole-CT lung-nodule detector. A returned "
                "candidate is not a confirmed nodule or cancer diagnosis."
            ),
        }

        result_path = series_dir / "processed" / "ct_scan.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(scan_payload, f, indent=2)

        return {
            "success": True,
            "series_id": series_id,
            "study_id": study_id,
            "task": scan_payload["task"],
            "model": scan_payload["model"],
            "scan": scan_payload["scan"],
            "candidate_count": len(final_candidates),
            "candidates": final_candidates,
            "result_file": str(result_path),
            "clinical_note": scan_payload["clinical_note"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CT candidate scan failed: {str(exc)}"
        )

@app.post("/api/ct/analyze/{series_id}")
async def analyze_ct_series(
    series_id: str,
    request: CTAnalyzeRequest,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """
    Classify supplied candidate locations.

    The supplied 3D CNN is candidate-level, so this endpoint does
    not falsely convert the entire CT into a normal/abnormal result.
    Candidate locations must come from a detector or application.
    """
    series_dir = _ct_series_dir(series_id)

    if not series_dir.exists():
        raise HTTPException(status_code=404, detail="CT series not found")

    if not request.candidates:
        raise HTTPException(status_code=400, detail="At least one CT candidate is required")

    volume_path = series_dir / "processed" / "ct_volume.npy"

    if not volume_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Run /api/ct/preprocess/{series_id} before CT analysis."
        )

    try:
        volume = np.load(volume_path, allow_pickle=False).astype(np.float32)

        if volume.ndim != 3:
            raise ValueError("Preprocessed CT volume must be 3-dimensional.")

        results = []

        for index, candidate in enumerate(request.candidates):
            try:
                patch = _extract_ct_patch(
                    volume,
                    candidate.z,
                    candidate.y,
                    candidate.x,
                )
                prediction = predict_ct_patch_array(patch)

                results.append({
                    "candidate_index": index,
                    "center": {
                        "z": candidate.z,
                        "y": candidate.y,
                        "x": candidate.x,
                    },
                    "patch_shape": list(patch.shape),
                    "prediction": prediction,
                })

            except ValueError as exc:
                results.append({
                    "candidate_index": index,
                    "center": {
                        "z": candidate.z,
                        "y": candidate.y,
                        "x": candidate.x,
                    },
                    "error": str(exc),
                })

        valid_results = [r for r in results if "prediction" in r]

        if not valid_results:
            raise HTTPException(
                status_code=400,
                detail="No valid candidate patches could be extracted."
            )

        series_metadata = _load_ct_series_metadata(series_id)
        study_id = series_metadata.get("study_id")

        analysis_payload = {
            "series_id": series_id,
            "study_id": study_id,
            "task": "Candidate-level lung nodule classification from 3D CT patches",
            "model": {
                "name": CT_MODEL_NAME,
                "version": CT_MODEL_VERSION,
                "architecture": "LungNodule3DCNN",
                "classes": CT_CLASS_NAMES,
                "threshold": CT_NODULE_THRESHOLD,
            },
            "results": valid_results,
            "clinical_note": (
                "AI research output only. A Nodule candidate is not a confirmed "
                "nodule or cancer diagnosis. A Non-nodule candidate does not "
                "establish that the entire CT scan is normal."
            ),
        }

        result_path = series_dir / "processed" / "ct_analysis.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)

        stored_predictions = []

        if study_id is not None:
            connection = get_connection()
            cursor = connection.cursor()

            try:
                cursor.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
                if cursor.fetchone() is None:
                    raise HTTPException(status_code=404, detail="Associated study not found")

                model_id = _get_or_create_ct_model_record(cursor)

                for result in valid_results:
                    prediction = result["prediction"]

                    cursor.execute(
                        """
                        INSERT INTO ai_predictions (
                            study_id,
                            model_id,
                            abnormality,
                            confidence_score,
                            prediction_status
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            study_id,
                            model_id,
                            prediction["class"],
                            prediction["probability_nodule"],
                            "Completed",
                        ),
                    )

                    stored_predictions.append({
                        "prediction_id": cursor.lastrowid,
                        "candidate_index": result["candidate_index"],
                        "class": prediction["class"],
                        "probability_nodule": prediction["probability_nodule"],
                    })

                connection.commit()

            except HTTPException:
                connection.rollback()
                raise
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

        return {
            "success": True,
            "series_id": series_id,
            "study_id": study_id,
            "candidate_count": len(valid_results),
            "results": valid_results,
            "stored_predictions": stored_predictions,
            "result_file": str(result_path),
            "clinical_note": analysis_payload["clinical_note"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CT analysis failed: {str(exc)}"
        )

@app.get("/api/ct/results/{series_id}")
def get_ct_results(
    series_id: str,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):
    series_dir = _ct_series_dir(series_id)

    if not series_dir.exists():
        raise HTTPException(status_code=404, detail="CT series not found")

    result_path = series_dir / "processed" / "ct_analysis.json"

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="No CT analysis results found")

    try:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read CT results: {str(exc)}"
        )

# ============================================================
# CT EXPLAINABILITY + TRIAGE
# ============================================================

def _load_ct_scan_payload(series_id: str) -> dict:
    result_path = _ct_series_dir(series_id) / "processed" / "ct_scan.json"
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Run the CT scan before requesting CT explanation or triage.",
        )
    try:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read CT scan results: {str(exc)}",
        )

def _calculate_ct_triage(scan_payload: dict) -> dict:
    candidates = scan_payload.get("candidates") or []
    if not candidates:
        return {
            "priority": "AI Inconclusive / Manual Review",
            "reason": "No CT candidate exceeded the configured proposal threshold.",
        }

    top = candidates[0]
    probability = float(
        (top.get("prediction") or {}).get("probability_nodule", 0.0)
    )

    if probability >= 0.70:
        priority = "High Priority"
    elif probability >= 0.40:
        priority = "Moderate Priority"
    else:
        priority = "Routine Review"

    return {
        "priority": priority,
        "reason": (
            f"The highest-ranked CT candidate has a model score of "
            f"{probability * 100:.2f}%. The configured research workflow "
            f"assigns {priority}. This is an operational review priority, "
            f"not a medical diagnosis."
        ),
        "top_candidate": top.get("center"),
        "top_probability": round(probability, 4),
        "candidate_count": len(candidates),
        "threshold": scan_payload.get("model", {}).get(
            "threshold", CT_NODULE_THRESHOLD
        ),
    }

def _generate_ct_gradcam_artifact(
    volume: np.ndarray,
    center: dict,
    output_path: Path,
    target_class: int = 1,
):
    patch = _extract_ct_patch(
        volume, int(center["z"]), int(center["y"]), int(center["x"])
    )

    model_input = (
        torch.from_numpy(patch.astype(np.float32))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .requires_grad_(True)
    )

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = getattr(ct_model, "features4", None)
    if target_layer is None:
        raise RuntimeError("CT model does not expose features4 for 3D Grad-CAM.")

    with CT_XAI_LOCK:
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        try:
            ct_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = ct_model(model_input)
                logits[0, target_class].backward()
            activation = activations.get("value")
            gradient = gradients.get("value")
        finally:
            forward_handle.remove()
            backward_handle.remove()

    if activation is None or gradient is None:
        raise RuntimeError("Could not capture CT Grad-CAM activations/gradients.")

    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=CT_PATCH_SIZE, mode="trilinear", align_corners=False
    )
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= float(cam.min())
    cam_max = float(cam.max())
    cam = cam / cam_max if cam_max > 1e-8 else np.zeros_like(cam)

    local_z = CT_PATCH_SIZE[0] // 2
    ct_slice = patch[local_z]
    heat_slice = cam[local_z]

    gray = np.clip(ct_slice * 255.0, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    heat_uint8 = np.clip(heat_slice * 255.0, 0, 255).astype(np.uint8)
    heat_rgb = np.zeros_like(gray_rgb)
    heat_rgb[..., 0] = heat_uint8
    heat_rgb[..., 1] = (heat_uint8 * 0.45).astype(np.uint8)

    alpha = (heat_slice * 0.70)[..., None]
    overlay = (
        gray_rgb.astype(np.float32) * (1.0 - alpha)
        + heat_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    base = Image.fromarray(gray_rgb, mode="RGB").resize(
        (512, 512), Image.Resampling.BICUBIC
    )
    overlay_image = Image.fromarray(overlay, mode="RGB").resize(
        (512, 512), Image.Resampling.BICUBIC
    )

    canvas = Image.new("RGB", (1024, 560), (15, 23, 42))
    canvas.paste(base, (0, 48))
    canvas.paste(overlay_image, (512, 48))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")

    probability = float(torch.softmax(logits, dim=1)[0, target_class].item())

    return {
        "method": "3D Grad-CAM",
        "target_class": target_class,
        "target_class_name": CT_CLASS_NAMES[target_class],
        "probability": round(probability, 4),
        "candidate_center": {
            "z": int(center["z"]),
            "y": int(center["y"]),
            "x": int(center["x"]),
        },
        "slice_index_within_patch": local_z,
        "patch_shape": list(CT_PATCH_SIZE),
    }

@app.post("/api/ct/explanation/{series_id}")
def generate_ct_explanation(
    series_id: str,
    candidate_index: int = 0,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist')),
):
    scan_payload = _load_ct_scan_payload(series_id)
    candidates = scan_payload.get("candidates") or []

    if candidate_index < 0 or candidate_index >= len(candidates):
        raise HTTPException(status_code=400, detail="Invalid CT candidate index.")

    candidate = candidates[candidate_index]
    center = candidate.get("center")
    if not center:
        raise HTTPException(status_code=400, detail="CT candidate has no center.")

    volume_path = _ct_series_dir(series_id) / "processed" / "ct_volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=400, detail="Preprocessed CT volume not found.")

    try:
        volume = np.load(volume_path, allow_pickle=False).astype(np.float32)
        safe_name = f"candidate_{candidate_index + 1}_gradcam.png"
        output_path = CT_XAI_OUTPUT_DIR / series_id / safe_name

        xai = _generate_ct_gradcam_artifact(
            volume, center, output_path, target_class=1
        )
        url = f"/ct_xai/{series_id}/{safe_name}"

        metadata = {
            "series_id": series_id,
            "study_id": _load_ct_series_metadata(series_id).get("study_id"),
            "candidate_index": candidate_index,
            "method": "3D Grad-CAM",
            "model_name": CT_MODEL_NAME,
            "model_version": CT_MODEL_VERSION,
            "candidate_center": center,
            "explanation_url": url,
            "clinical_note": (
                "AI explanation only. The activation heatmap shows regions "
                "that influenced the candidate-level model output. It does "
                "not establish a nodule, cancer, or causal clinical finding."
            ),
        }

        metadata_path = (
            _ct_series_dir(series_id)
            / "processed"
            / f"ct_explanation_candidate_{candidate_index + 1}.json"
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "success": True,
            "series_id": series_id,
            "study_id": metadata["study_id"],
            "candidate_index": candidate_index,
            "candidate": candidate,
            "explanation": {**xai, "url": url},
            "model": {
                "name": CT_MODEL_NAME,
                "version": CT_MODEL_VERSION,
                "architecture": "LungNodule3DCNN",
            },
            "clinical_note": metadata["clinical_note"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CT Grad-CAM generation failed: {str(exc)}",
        )

@app.get("/api/ct/triage/{series_id}")
def get_ct_triage(
    series_id: str,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist')),
):
    scan_payload = _load_ct_scan_payload(series_id)
    return {
        "success": True,
        "series_id": series_id,
        "study_id": _load_ct_series_metadata(series_id).get("study_id"),
        "triage": _calculate_ct_triage(scan_payload),
        "clinical_note": (
            "Triage priority is an operational research workflow output. "
            "It is not a medical diagnosis or a substitute for clinician review."
        ),
    }

@app.post("/api/ct/triage/{series_id}")
def create_ct_triage(
    series_id: str,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist')),
):
    scan_payload = _load_ct_scan_payload(series_id)
    triage = _calculate_ct_triage(scan_payload)
    study_id = _load_ct_series_metadata(series_id).get("study_id")

    if study_id is None:
        raise HTTPException(status_code=400, detail="CT series has no study ID.")

    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Associated study not found.")

        cursor.execute(
            """
            INSERT INTO triage_results (study_id, priority, reason)
            VALUES (?, ?, ?)
            """,
            (study_id, triage["priority"], triage["reason"]),
        )
        triage_id = cursor.lastrowid
        connection.commit()

        return {
            "success": True,
            "triage_id": triage_id,
            "series_id": series_id,
            "study_id": study_id,
            "triage": triage,
            "clinical_note": (
                "Triage is an operational research workflow output and requires "
                "clinician review."
            ),
        }
    except HTTPException:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save CT triage: {str(exc)}",
        )
    finally:
        connection.close()

@app.post("/api/ct/predict-patch")
async def predict_ct_patch(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):
    """Classify an already-preprocessed 32x32x32 .npy CT patch."""
    filename = file.filename or ""

    if not filename.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="CT patch must be a .npy file")

    try:
        import io

        patch = np.load(
            io.BytesIO(await file.read()),
            allow_pickle=False,
        )

        prediction = predict_ct_patch_array(patch)

        return {
            "success": True,
            "model": {
                "name": CT_MODEL_NAME,
                "version": CT_MODEL_VERSION,
                "architecture": "LungNodule3DCNN",
            },
            "input": {
                "shape": list(np.asarray(patch).shape),
                "expected_shape": list(CT_PATCH_SIZE),
                "type": "preprocessed CT patch",
            },
            "prediction": prediction,
            "clinical_note": (
                "AI research output only. This model classifies a candidate "
                "CT patch and does not establish a whole-CT diagnosis."
            ),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CT prediction failed: {str(exc)}"
        )

# AI MODEL MANAGEMENT

# ==================================================

@app.post("/api/models")

def create_ai_model(model: AIModelCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            SELECT id

            FROM ai_models

            WHERE model_name = ? AND version = ?

            """,

            (model.model_name, model.version)

        )

        existing_model = cursor.fetchone()

        if existing_model is not None:

            raise HTTPException(

                status_code=409,

                detail="A model with the same name and version already exists"

            )

        cursor.execute(

            """

            INSERT INTO ai_models (

                model_name,

                version,

                model_path,

                status

            )

            VALUES (?, ?, ?, ?)

            """,

            (

                model.model_name,

                model.version,

                model.model_path,

                model.status

            )

        )

        connection.commit()

        model_id = cursor.lastrowid

        return {

            "message": "AI model registered successfully",

            "model_id": model_id,

            "model_name": model.model_name,

            "version": model.version,

            "model_path": model.model_path,

            "status": model.status

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

@app.get("/api/models")

def get_ai_models(
    _rbac_user: dict = Depends(require_roles('Administrator', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            SELECT

                id,

                model_name,

                version,

                model_path,

                status,

                created_at

            FROM ai_models

            ORDER BY id DESC

            """

        )

        rows = cursor.fetchall()

        return {

            "count": len(rows),

            "models": [dict(row) for row in rows]

        }

    finally:

        connection.close()

@app.get("/api/models/{model_id}")

def get_ai_model(model_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(

            """

            SELECT

                id,

                model_name,

                version,

                model_path,

                status,

                created_at

            FROM ai_models

            WHERE id = ?

            """,

            (model_id,)

        )

        row = cursor.fetchone()

        if row is None:

            raise HTTPException(

                status_code=404,

                detail="AI model not found"

            )

        return dict(row)

    finally:

        connection.close()

# ==================================================

# AI ANALYSIS / PREDICTION STRUCTURE

# ==================================================

#

# IMPORTANT:

# This endpoint prepares the backend for the real AI model.

# It does NOT create or invent a medical prediction.

# The trained model will be connected in a later step.

# ==================================================

@app.post("/api/analysis/{study_id}")

def run_ai_analysis(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # --------------------------------------------------

        # Check whether the study exists

        # --------------------------------------------------

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # --------------------------------------------------

        # Find the most recently uploaded image

        # --------------------------------------------------

        cursor.execute(

            """

            SELECT

                id,

                file_id,

                original_filename,

                file_path,

                validation_status,

                preprocessing_status

            FROM medical_images

            WHERE study_id = ?

            ORDER BY id DESC

            LIMIT 1

            """,

            (study_id,)

        )

        image_record = cursor.fetchone()

        if image_record is None:

            raise HTTPException(

                status_code=404,

                detail="No medical image is associated with this study"

            )

        # --------------------------------------------------

        # Validation must be successful

        # --------------------------------------------------

        if image_record["validation_status"] != "Valid":

            raise HTTPException(

                status_code=400,

                detail="Image must be successfully validated before AI analysis"

            )

        # --------------------------------------------------

        # Preprocessing must be successful

        # --------------------------------------------------

        if image_record["preprocessing_status"] != "Preprocessed":

            raise HTTPException(

                status_code=400,

                detail="Image must be preprocessed before AI analysis"

            )

        # --------------------------------------------------

        # Check that the image file still exists

        # --------------------------------------------------

        image_path = Path(image_record["file_path"])

        if not image_path.exists():

            raise HTTPException(

                status_code=404,

                detail="Image file not found"

            )

        # --------------------------------------------------

        # Run the trained ResNet18 model

        # --------------------------------------------------

        predictions = predict_image(str(image_path))

        # --------------------------------------------------

        # Identify the trained model in the database

        # --------------------------------------------------

        model_name = "ResNet18 Medical 3-Class"

        model_version = "1.0"

        model_path = "backend/models/resnet18_medical_3class_best.pth"

        cursor.execute(

            """

            SELECT id

            FROM ai_models

            WHERE model_name = ?

              AND version = ?

            ORDER BY id DESC

            LIMIT 1

            """,

            (

                model_name,

                model_version

            )

        )

        model_record = cursor.fetchone()

        # Register the trained model automatically if it

        # has not already been registered.

        if model_record is None:

            cursor.execute(

                """

                INSERT INTO ai_models (

                    model_name,

                    version,

                    model_path,

                    status

                )

                VALUES (?, ?, ?, ?)

                """,

                (

                    model_name,

                    model_version,

                    model_path,

                    "Active"

                )

            )

            model_id = cursor.lastrowid

        else:

            model_id = model_record["id"]

        # --------------------------------------------------

        # Store all three model predictions

        # --------------------------------------------------

        stored_predictions = []

        for prediction in predictions:

            cursor.execute(

                """

                INSERT INTO ai_predictions (

                    study_id,

                    model_id,

                    abnormality,

                    confidence_score,

                    prediction_status

                )

                VALUES (?, ?, ?, ?, ?)

                """,

                (

                    study_id,

                    model_id,

                    prediction["class_name"],

                    prediction["probability"],

                    "Completed"

                )

            )

            prediction_id = cursor.lastrowid

            stored_predictions.append(

                {

                    "prediction_id": prediction_id,

                    "class_name": prediction["class_name"],

                    "probability": prediction["probability"],

                    "threshold": prediction["threshold"],

                    "flagged": prediction["flagged"]

                }

            )

        # --------------------------------------------------
        # AUTOMATIC TRIAGE CALCULATION
        # --------------------------------------------------

        triage_result = calculate_triage(stored_predictions)

        # Store the automatically calculated workflow priority.
        # This is a workflow-priority result, NOT a medical diagnosis.
        cursor.execute(
            """
            INSERT INTO triage_results (
                study_id,
                priority,
                reason
            )
            VALUES (?, ?, ?)
            """,
            (
                study_id,
                triage_result["priority"],
                triage_result["reason"]
            )
        )

        triage_id = cursor.lastrowid

        connection.commit()

        # --------------------------------------------------

        # Return the real model output and triage result

        # --------------------------------------------------

        return {

            "study_id": study_id,

            "file_id": image_record["file_id"],

            "filename": image_record["original_filename"],

            "analysis_status": "Completed",

            "prediction_available": True,

            "model": {

                "model_id": model_id,

                "model_name": model_name,

                "version": model_version

            },

            "predictions": stored_predictions,

            "triage": {
                "triage_id": triage_id,
                "priority": triage_result["priority"],
                "reason": triage_result["reason"]
            }

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=500,

            detail=f"AI analysis failed: {str(e)}"

        )

    finally:

        connection.close()

@app.get("/api/analysis/{study_id}")

def get_ai_analysis(study_id: int,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist'))
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # --------------------------------------------------

        # Check whether the study exists

        # --------------------------------------------------

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # --------------------------------------------------

        # Retrieve stored predictions

        # --------------------------------------------------

        cursor.execute(

            """

            SELECT

                ai_predictions.id,

                ai_predictions.study_id,

                ai_predictions.model_id,

                ai_models.model_name,

                ai_models.version AS model_version,

                ai_predictions.abnormality,

                ai_predictions.confidence_score,

                ai_predictions.prediction_status,

                ai_predictions.created_at

            FROM ai_predictions

            LEFT JOIN ai_models

                ON ai_predictions.model_id = ai_models.id

            WHERE ai_predictions.study_id = ?

            ORDER BY ai_predictions.id DESC

            """,

            (study_id,)

        )

        rows = cursor.fetchall()

        return {

            "study_id": study_id,

            "count": len(rows),

            "predictions": [dict(row) for row in rows]

        }

    finally:

        connection.close()

# ==================================================

# AI PREDICTION RECORD MANAGEMENT

# ==================================================

#

# IMPORTANT:

# This endpoint stores a prediction produced by the real AI model.

# It must NOT be used to invent or manually create medical predictions.

# The real model integration will call this logic in a later step.

# ==================================================

@app.post("/api/analysis/{study_id}/predictions")

def store_ai_prediction(

    study_id: int,

    prediction: AIPredictionCreate,
    _rbac_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer'))
):

    if prediction.confidence_score < 0 or prediction.confidence_score > 1:

        raise HTTPException(

            status_code=400,

            detail="confidence_score must be between 0 and 1"

        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        # Check that the study exists

        cursor.execute(

            """

            SELECT id

            FROM studies

            WHERE id = ?

            """,

            (study_id,)

        )

        study = cursor.fetchone()

        if study is None:

            raise HTTPException(

                status_code=404,

                detail="Study not found"

            )

        # Check that the referenced model exists

        cursor.execute(

            """

            SELECT id, model_name, version

            FROM ai_models

            WHERE id = ?

            """,

            (prediction.model_id,)

        )

        model = cursor.fetchone()

        if model is None:

            raise HTTPException(

                status_code=404,

                detail="AI model not found"

            )

        # Store the prediction produced by the model

        cursor.execute(

            """

            INSERT INTO ai_predictions (

                study_id,

                model_id,

                abnormality,

                confidence_score,

                prediction_status

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                study_id,

                prediction.model_id,

                prediction.abnormality,

                prediction.confidence_score,

                prediction.prediction_status

            )

        )

        connection.commit()

        prediction_id = cursor.lastrowid

        return {

            "message": "AI prediction stored successfully",

            "prediction_id": prediction_id,

            "study_id": study_id,

            "model_id": prediction.model_id,

            "model_name": model["model_name"],

            "model_version": model["version"],

            "abnormality": prediction.abnormality,

            "confidence_score": prediction.confidence_score,

            "prediction_status": prediction.prediction_status

        }

    except HTTPException:

        connection.rollback()

        raise

    except Exception as e:

        connection.rollback()

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    finally:

        connection.close()

# ============================================================
# CT FOLDER UPLOAD PAGE
# ============================================================

@app.get("/ct-folder-upload", response_class=HTMLResponse)
def ct_folder_upload_page():
    """
    Browser page for selecting and uploading an entire CT DICOM folder.

    The browser selects all files inside the chosen folder and sends
    them as individual multipart files to the existing CT series
    upload endpoint.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CT DICOM Folder Upload</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .box {
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 24px;
        }
        button {
            margin-top: 16px;
            padding: 10px 18px;
            cursor: pointer;
        }
        #status {
            margin-top: 20px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>CT DICOM Folder Upload</h1>
        <p>Select the folder containing one complete CT DICOM series.</p>
        <p><strong>Only .dcm files will be uploaded.</strong></p>

        <label>
            Optional Study ID:
            <input id="studyId" type="number" min="1" placeholder="Leave empty if not linked">
        </label>
        <br><br>

        <input id="folderInput" type="file" webkitdirectory directory multiple>
        <div id="fileCount">No folder selected.</div>

        <button id="uploadButton" type="button">Upload CT Folder</button>
        <div id="status"></div>
    </div>

<script>
const folderInput = document.getElementById("folderInput");
const fileCount = document.getElementById("fileCount");
const uploadButton = document.getElementById("uploadButton");
const statusBox = document.getElementById("status");

folderInput.addEventListener("change", () => {
    const dcmFiles = Array.from(folderInput.files)
        .filter(file => file.name.toLowerCase().endsWith(".dcm"));

    fileCount.textContent = dcmFiles.length
        ? `${dcmFiles.length} DICOM file(s) selected.`
        : "No .dcm files found in the selected folder.";
});

uploadButton.addEventListener("click", async () => {
    const dcmFiles = Array.from(folderInput.files)
        .filter(file => file.name.toLowerCase().endsWith(".dcm"));

    if (!dcmFiles.length) {
        statusBox.textContent = "Please select a folder containing .dcm files.";
        return;
    }

    const formData = new FormData();

    dcmFiles.forEach(file => {
        formData.append("files", file, file.name);
    });

    const studyId = document.getElementById("studyId").value.trim();
    if (studyId) {
        formData.append("study_id", studyId);
    }

    uploadButton.disabled = true;
    statusBox.textContent = `Uploading ${dcmFiles.length} DICOM file(s)...`;

    try {
        const response = await fetch("/api/ct/upload-series", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        statusBox.textContent = JSON.stringify(data, null, 2);

        if (!response.ok) {
            throw new Error(data.detail || "CT folder upload failed.");
        }
    } catch (error) {
        statusBox.textContent = `Upload failed: ${error.message}`;
    } finally {
        uploadButton.disabled = false;
    }
});
</script>
</body>
</html>
"""

# ============================================================
# SWAGGER / OPENAPI FILE INPUT FIX
# ============================================================

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=getattr(app, "version", "1.0.0"),
        description=app.description,
        routes=app.routes,
    )

    body_name = "Body_upload_ct_series_api_ct_upload_series_post"
    body_schema = schema.get("components", {}).get("schemas", {}).get(body_name)

    if body_schema:
        files_schema = body_schema.get("properties", {}).get("files")
        if files_schema:
            files_schema["items"] = {
                "type": "string",
                "format": "binary"
            }

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

# ============================================================
# MRI DRAFT REPORT PERSISTENCE
# ============================================================
def save_mri_draft_report(study_id: int | None, current_user: dict, report_text: str):
    """Persist MRI research output as a draft without changing other modality workflows."""
    if study_id is None:
        return None

    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, study_type FROM studies WHERE id = ?", (study_id,))
        study = cursor.fetchone()
        if study is None:
            raise HTTPException(status_code=404, detail="MRI study not found.")
        if str(study["study_type"]).strip().lower() != "mri":
            raise HTTPException(status_code=400, detail="The supplied study ID is not an MRI study.")

        generated_by = current_user.get("full_name") or current_user.get("username") or "Unknown user"
        cursor.execute(
            "INSERT INTO reports (study_id, report_status, report_text, generated_by) VALUES (?, ?, ?, ?)",
            (study_id, "Draft", report_text, generated_by),
        )
        report_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO audit_logs (action, entity_type, entity_id, performed_by, details) VALUES (?, ?, ?, ?, ?)",
            (
                "mri_draft_report_created",
                "study",
                str(study_id),
                generated_by,
                f"Report ID {report_id}; status=Draft; MRI research inference output",
            ),
        )
        connection.commit()
        return report_id
    except HTTPException:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        print(f"MRI draft report save failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="MRI results were produced, but the draft report could not be saved.") from exc
    finally:
        connection.close()


# ============================================================
# MRI 2D IMAGE CLASSIFICATION API (research inference)
# ============================================================

@app.post("/api/mri/classify-image")
async def classify_mri_2d_image(
    file: UploadFile = File(...),
    study_id: int | None = None,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer')),
):
    """Classify one PNG/JPG/JPEG brain MRI image with the trained 2D classifier."""
    from io import BytesIO

    filename = (file.filename or "").lower()
    allowed_extensions = (".png", ".jpg", ".jpeg")
    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Upload one brain MRI image in .png, .jpg, or .jpeg format.",
        )

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

        try:
            with Image.open(BytesIO(raw)) as probe:
                if probe.format not in {"PNG", "JPEG"}:
                    raise HTTPException(status_code=400, detail="The file content is not a PNG or JPEG image.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise HTTPException(status_code=400, detail="Only single-frame images are supported.")
                probe.verify()
            with Image.open(BytesIO(raw)) as source_image:
                image = source_image.convert("RGB")
        except HTTPException:
            raise
        except Exception as image_error:
            raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from image_error

        classifier, _segmentation, classifier_ckpt, _seg_ckpt, device = get_mri_models()
        result = classify_mri_image(
            classifier,
            image,
            device,
            image_size=int(classifier_ckpt.get("image_size", 224)),
        )

        # Generate Grad-CAM from the same 2D classifier prediction.
        # Return PNGs as base64 so the frontend can display them without another endpoint.
        import base64
        from io import BytesIO

        cam_image = gradcam_mri_overlay(
            classifier,
            result["tensor"],
            int(result["class_index"]),
            image,
            device,
        )
        cam_buffer = BytesIO()
        cam_image.save(cam_buffer, format="PNG")
        cam_base64 = base64.b64encode(cam_buffer.getvalue()).decode("ascii")

        original_buffer = BytesIO()
        image.resize((224, 224)).save(original_buffer, format="PNG")
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode("ascii")

        note = (
            "Research output only. This classifier is intended for brain MRI images in its "
            "training domain, is not clinically validated, and must not be used as a diagnosis. "
            "Scores are not calibrated probabilities; qualified clinician review is required."
        )
        report_lines = [
            "MRI 2D AI RESEARCH REPORT (DRAFT)",
            "=================================",
            f"Study ID: {study_id if study_id is not None else 'Not linked'}",
            f"Input filename: {Path(file.filename or 'uploaded_image').name}",
            "Modality/workflow: MRI 2D image classification",
            f"Predicted model class: {result['predicted_class']}",
            f"Highest model score (not calibrated): {float(result['score']):.6f}",
            "Class scores:",
        ]
        report_lines.extend(
            f"- {name}: {float(score):.6f}"
            for name, score in (result.get("probabilities") or {}).items()
        )
        report_lines.extend(["", f"Model: ResNet-18 2D brain MRI classifier", f"Research limitation: {note}"])
        report_id = save_mri_draft_report(study_id, current_user, "\n".join(report_lines))

        return {
            "message": "2D MRI research classification completed.",
            "study_id": study_id,
            "report_saved": report_id is not None,
            "report_id": report_id,
            "report_status": "Draft" if report_id is not None else None,
            "input_filename": Path(file.filename or "uploaded_image").name,
            "predicted_class": result["predicted_class"],
            "score_not_calibrated": result["score"],
            "probabilities": result["probabilities"],
            "original_image_png_base64": original_base64,
            "gradcam_overlay_png_base64": cam_base64,
            "gradcam_target_class": result["predicted_class"],
            "model": "ResNet-18 2D brain MRI classifier",
            "model_device": str(device),
            "note": note,
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"2D MRI classification failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="2D MRI classification failed. Check the backend terminal for details.",
        ) from exc
    finally:
        await file.close()

# ============================================================
# MRI 3D SEGMENTATION API (research inference)
# ============================================================

@app.post("/api/mri/segment")
async def segment_mri_volume(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    study_id: int | None = None,
    current_user: dict = Depends(require_roles('Administrator', 'Radiologist', 'AI/ML Engineer')),
):
    """Run research segmentation on FLAIR, T1, T1ce and T2 NIfTI volumes."""
    uploads = [("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2)]
    connection = None

    try:
        # Validate extensions before processing. Do not accept arbitrary files.
        for modality, upload in uploads:
            filename = (upload.filename or "").lower()
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                raise HTTPException(
                    status_code=400,
                    detail=f"{modality.upper()} must be a NIfTI file (.nii or .nii.gz).",
                )

        if study_id is not None:
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Study not found.")

        # Temporary files are deleted automatically when inference finishes.
        with tempfile.TemporaryDirectory(prefix="mri_") as temp_dir:
            temp_dir = Path(temp_dir)
            nifti_paths = []
            for modality, upload in uploads:
                filename = (upload.filename or "").lower()
                suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
                destination = temp_dir / f"{modality}{suffix}"
                with destination.open("wb") as output_file:
                    shutil.copyfileobj(upload.file, output_file)
                nifti_paths.append(destination)

            image_tensor = read_nifti_modalities(nifti_paths)
            classifier, segmentation, classifier_ckpt, seg_ckpt, device = get_mri_models()

            with torch.no_grad():
                outputs = segmentation(image_tensor.unsqueeze(0).to(device))
                if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
                    raise RuntimeError("Unexpected MRI segmentation model output.")
                segmentation_logits, auxiliary_logits = outputs[0], outputs[1]
                predicted_mask = torch.argmax(segmentation_logits, dim=1)[0].cpu().numpy()
                probabilities = torch.softmax(auxiliary_logits, dim=1)[0].cpu().numpy()

            voxel_counts = {
                str(label): int(np.count_nonzero(predicted_mask == label))
                for label in range(len(MRI_SEGMENTATION_LABELS))
            }

            # Create preview images from the model output for the browser UI.
            # This is a center slice of the model's cropped inference tensor, not
            # a reconstructed full-volume clinical segmentation.
            from io import BytesIO

            slice_index = int(predicted_mask.shape[0] // 2)
            flair_slice = np.asarray(image_tensor[0, slice_index].cpu().numpy(), dtype=np.float32)
            finite_values = flair_slice[np.isfinite(flair_slice)]
            if finite_values.size:
                low, high = np.percentile(finite_values, [1, 99])
                if high <= low:
                    high = low + 1.0
                grayscale = np.clip((flair_slice - low) / (high - low), 0, 1)
            else:
                grayscale = np.zeros_like(flair_slice, dtype=np.float32)
            gray_u8 = (grayscale * 255).astype(np.uint8)
            flair_preview = Image.fromarray(gray_u8, mode="L").convert("RGB")

            # Colors are only a visualization key for the model's label IDs.
            label_colors = {
                1: (255, 90, 90),   # label 1: NCR/NET
                2: (60, 180, 255),  # label 2: edema
                3: (255, 210, 60),  # label 3: enhancing tumor
            }
            mask_slice = np.asarray(predicted_mask[slice_index])
            overlay_array = np.asarray(flair_preview).copy()
            for label, color in label_colors.items():
                selected = mask_slice == label
                if np.any(selected):
                    overlay_array[selected] = (
                        0.40 * overlay_array[selected] + 0.60 * np.asarray(color)
                    ).astype(np.uint8)
            overlay_preview = Image.fromarray(overlay_array, mode="RGB")

            def _png_data_url(pil_image):
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG", optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                return "data:image/png;base64," + encoded

            flair_preview_url = _png_data_url(flair_preview)
            overlay_preview_url = _png_data_url(overlay_preview)
            mask_preview_url = _png_data_url(
                Image.fromarray(mask_slice.astype(np.uint8), mode="L")
            )

            auxiliary_probabilities = {
                MRI_CLASS_NAMES[index]: float(probabilities[index])
                for index in range(len(MRI_CLASS_NAMES))
            }

            note = (
                "Research output only, not a diagnosis. The segmentation-label mapping "
                "and center-crop inference preprocessing require validation against the "
                "training pipeline. A qualified clinician must interpret the results."
            )
            report_lines = [
                "MRI 3D AI SEGMENTATION REPORT (DRAFT)",
                "====================================",
                f"Study ID: {study_id if study_id is not None else 'Not linked'}",
                "Modality/workflow: MRI 3D multi-modal segmentation",
                "Input modalities: FLAIR, T1, T1ce, T2",
                f"Input tensor shape: {list(image_tensor.shape)}",
                f"Segmentation output shape: {list(predicted_mask.shape)}",
                f"Preview slice index: {slice_index}",
                "Voxel counts by segmentation label:",
            ]
            report_lines.extend(
                f"- Label {label} ({MRI_SEGMENTATION_LABELS.get(label, 'Background/other')}): {count} voxels"
                for label, count in voxel_counts.items()
            )
            report_lines.append("Auxiliary classification scores:")
            report_lines.extend(
                f"- {name}: {float(score):.6f}"
                for name, score in auxiliary_probabilities.items()
            )
            report_lines.extend(["", f"Research limitation: {note}"])
            report_id = save_mri_draft_report(study_id, current_user, "\n".join(report_lines))

            return {
                "message": "MRI research inference completed.",
                "study_id": study_id,
                "report_saved": report_id is not None,
                "report_id": report_id,
                "report_status": "Draft" if report_id is not None else None,
                "modalities": list(MRI_MODALITIES),
                "input_shape": list(image_tensor.shape),
                "segmentation_shape": list(predicted_mask.shape),
                "segmentation_label_counts_voxels": voxel_counts,
                "segmentation_labels": MRI_SEGMENTATION_LABELS,
                "preview_slice_index": slice_index,
                "flair_slice_png": flair_preview_url,
                "segmentation_overlay_png": overlay_preview_url,
                "segmentation_mask_png": mask_preview_url,
                "auxiliary_class_probabilities": auxiliary_probabilities,
                "model_device": str(device),
                "note": note,
            }

    except HTTPException:
        raise
    except Exception as exc:
        # Keep implementation details in server logs, not API response.
        print(f"MRI inference failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="MRI inference failed. Check the backend terminal for details.",
        ) from exc
    finally:
        if connection is not None:
            connection.close()
        for _, upload in uploads:
            await upload.close()


# ============================================================
# 2D BRAIN CT HEMORRHAGE PREDICTION (SEPARATE FROM 3D CT)
# Add this block near the END of main.py, after existing routes.
# Do not replace or edit the existing DICOM CT endpoints.
# ============================================================

# Reuse the same CT model parent directory already configured above.
BRAIN_CT_MODEL_DIR = CT_MODEL_DIR.parent / "brain_hemorrhage_2d"
if str(BRAIN_CT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_CT_MODEL_DIR))

BRAIN_CT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.post("/api/brain-ct/hemorrhage/predict")
async def predict_brain_ct_hemorrhage_endpoint(
    file: UploadFile = File(...),
    _rbac_user: dict = Depends(
        require_roles("Administrator", "Radiologist", "AI/ML Engineer")
    ),
):
    """Classify one JPG/JPEG/PNG brain CT slice with the separate 2D model."""
    original_name = file.filename or "brain_ct.png"
    extension = Path(original_name).suffix.lower()

    if extension not in BRAIN_CT_ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG brain CT image.",
        )

    try:
        image_bytes = await file.read()
        # Guard against accidentally uploading extremely large files.
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum upload size is 20 MB.",
            )

        # Local import prevents any effect on backend startup when the model
        # checkpoint is not present yet.
        from brain_ct_hemorrhage_service import predict_brain_ct_hemorrhage

        return predict_brain_ct_hemorrhage(image_bytes, original_name)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Brain CT hemorrhage inference failed: {exc}",
        ) from exc
    finally:
        await file.close()


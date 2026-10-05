from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles







from database import get_connection, create_tables







from fastapi.middleware.cors import CORSMiddleware







from pathlib import Path







from PIL import Image







from pydantic import BaseModel







import shutil







import uuid







import numpy as np



import sys
import json



# --------------------------------------------------

# AI MODEL IMPORT

# --------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parent.parent



if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from ai.predict import predict_image
from ai.xai import generate_gradcam









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







    allow_origins=["*"],







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
app.mount(
    "/xai_outputs",
    StaticFiles(directory=str(XAI_OUTPUT_DIR)),
    name="xai_outputs"
)

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







    explanation: ExplanationCreate







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







def get_explanations(study_id: int):







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
def generate_explanation_for_study(study_id: int):

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







    triage: TriageCreate







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







def get_triage(study_id: int):







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







    clinician_name: str







    review_status: str = "Completed"







    final_decision: str







    comments: str | None = None







# ------------------------------------------------------------







# Create Clinician Review







# ------------------------------------------------------------







@app.post("/api/analysis/{study_id}/review")







def create_clinician_review(







    study_id: int,







    review: ClinicianReviewCreate







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







                review.clinician_name,







                review.review_status,







                review.final_decision,







                review.comments







            )







        )







        review_id = cursor.lastrowid







        connection.commit()







        return {







            "message": "Clinician review created successfully",







            "review_id": review_id,







            "study_id": study_id,







            "clinician_name": review.clinician_name,







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







def get_clinician_reviews(study_id: int):







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



    report: ReportCreate



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



def get_reports(study_id: int):







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



def create_audit_log(log: AuditLogCreate):







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



def get_audit_logs():







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







def create_abnormality(abnormality: AbnormalityCreate):







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







def get_abnormalities():







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







def get_abnormality(abnormality_id: int):







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







# PATIENT MANAGEMENT







# ==================================================







# --------------------------------------------------







# Create Patient







# --------------------------------------------------







@app.post("/api/patients")







def create_patient(patient: PatientCreate):







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







def get_patients():







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







def get_patient(patient_id: int):







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







def create_study(study: StudyCreate):







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







def get_studies():







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







def get_database_images():







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







def get_study(study_id: int):







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







    study_id: int | None = None







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







def validate_image(file_id: str):







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







async def preprocess_image(file_id: str):







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







# ==================================================







# AI MODEL MANAGEMENT







# ==================================================







@app.post("/api/models")







def create_ai_model(model: AIModelCreate):







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







def get_ai_models():







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







def get_ai_model(model_id: int):







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

def run_ai_analysis(study_id: int):



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



        connection.commit()



        # --------------------------------------------------

        # Return the real model output

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

            "predictions": stored_predictions

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







def get_ai_analysis(study_id: int):







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







    prediction: AIPredictionCreate







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












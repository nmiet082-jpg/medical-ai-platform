from fastapi import FastAPI, UploadFile, File, HTTPException

from database import get_connection, create_tables

from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from PIL import Image

from pydantic import BaseModel

import shutil

import uuid

import numpy as np





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
        # Find the most recently uploaded image for the study
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
        # Make sure validation has succeeded
        # --------------------------------------------------

        if image_record["validation_status"] != "Valid":
            raise HTTPException(
                status_code=400,
                detail="Image must be successfully validated before AI analysis"
            )

        # --------------------------------------------------
        # Make sure preprocessing has succeeded
        # --------------------------------------------------

        if image_record["preprocessing_status"] != "Preprocessed":
            raise HTTPException(
                status_code=400,
                detail="Image must be preprocessed before AI analysis"
            )

        # --------------------------------------------------
        # Model connection will be added later
        # --------------------------------------------------

        return {
            "study_id": study_id,
            "file_id": image_record["file_id"],
            "filename": image_record["original_filename"],
            "analysis_status": "Model not connected",
            "prediction_available": False,
            "message": "Image is ready for AI analysis. The trained AI model has not been connected yet."
        }

    finally:
        connection.close()


# --------------------------------------------------
# Get AI Predictions for a Study
# --------------------------------------------------


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


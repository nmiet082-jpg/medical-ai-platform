from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import shutil
import uuid
import numpy as np

app = FastAPI(
    title="AI Medical Imaging Triage & XAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where uploaded images will be stored
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initial prototype file types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}


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


@app.post("/api/studies/upload")
async def upload_study(file: UploadFile = File(...)):

    # Check file extension
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    # Create a unique filename
    file_id = str(uuid.uuid4())
    saved_filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename

    # Save the uploaded file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "file_id": file_id,
        "filename": file.filename,
        "saved_as": saved_filename
    }


@app.post("/api/studies/{file_id}/validate")
async def validate_image(file_id: str):

    matching_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found"
        )

    file_path = matching_files[0]

    try:
        with Image.open(file_path) as image:
            image.verify()

        with Image.open(file_path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode

        return {
            "file_id": file_id,
            "valid": True,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode
        }

    except Exception:
        return {
            "file_id": file_id,
            "valid": False,
            "message": "Image is corrupted or cannot be read"
        }


@app.post("/api/studies/{file_id}/preprocess")
async def preprocess_image(file_id: str):

    matching_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found"
        )

    file_path = matching_files[0]

    try:
        with Image.open(file_path) as image:

            # Convert to RGB
            image = image.convert("RGB")

            # Resize to a standard size
            image = image.resize((224, 224))

            # Convert image to NumPy array
            image_array = np.array(image)

            # Normalize pixel values from 0-255 to 0-1
            normalized = image_array.astype(np.float32) / 255.0

            return {
                "file_id": file_id,
                "preprocessed": True,
                "size": [224, 224],
                "channels": 3,
                "min_pixel_value": float(normalized.min()),
                "max_pixel_value": float(normalized.max())
            }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing failed: {str(e)}"
        )
# AI Medical Imaging Triage & Explainability Platform

A research and demonstration application for medical imaging workflows, including AI-assisted image analysis and report management.

## Technology Stack

- Python
- FastAPI
- PyTorch
- SQLite
- HTML, CSS and JavaScript
- Git and Git LFS for version control and model files

## Prerequisites

Install the following before starting:

- Git
- Git LFS
- A compatible Python version for the pinned dependencies
- Visual Studio Code (recommended)

## 1. Clone the Repository

Open PowerShell and run:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd medical-ai-platform
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the repository's actual GitHub URL.

## 2. Download AI Model Files

Initialize Git LFS and download the model files:

```powershell
git lfs install
git lfs pull
git lfs ls-files
```

Confirm that the model files are downloaded before running the application.

## 3. Create a Python Virtual Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, use Command Prompt or consult the project's setup instructions rather than changing system security settings.

## 4. Install Dependencies

Install the project's pinned dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

If installation fails, check the Python version and the package error message. PyTorch and other machine-learning packages may require compatible versions.

## 5. Start the Backend

Open a terminal at the project root and run:

```powershell
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Verify the backend by opening:

http://127.0.0.1:8000/health

The expected response is:

```json
{"status":"healthy"}
```

Keep the backend terminal running.

## 6. Start the Frontend

Open a second terminal at the project root and run:

```powershell
python -m http.server 5500 --directory index
```

Open the frontend in your browser:

http://127.0.0.1:5500

Keep both terminal windows running while using the application.

## 7. Database

The application uses SQLite. The backend initializes its database tables during startup.

Use a separate database for development and testing. Do not copy real patient information or production database files into a public repository.

## Important Notes

- Do not commit passwords, API keys, environment secrets, private medical images, or patient-identifiable information.
- Keep large AI model files tracked through Git LFS.
- Preserve existing model files and workflows when making changes.
- AI outputs are for research and demonstration; they must not be treated as a substitute for professional medical interpretation.
- Test the setup on a fresh clone before considering the installation instructions verified.

import sqlite3
from pathlib import Path


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DATABASE_PATH = Path("medical_ai.db")


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)

    # Allows rows to be accessed like dictionaries
    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

def create_tables():

    connection = get_connection()
    cursor = connection.cursor()


    # =====================================================
    # PATIENTS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)


    # =====================================================
    # STUDIES TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            study_type TEXT,
            study_date TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id)
                REFERENCES patients(id)
        )
    """)


    # =====================================================
    # MEDICAL IMAGES TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER,
            file_id TEXT UNIQUE NOT NULL,
            original_filename TEXT,
            stored_filename TEXT,
            file_path TEXT,
            file_type TEXT,
            validation_status TEXT,
            preprocessing_status TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (study_id)
                REFERENCES studies(id)
        )
    """)


    # =====================================================
    # AI MODELS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            model_path TEXT,
            status TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)


    # =====================================================
    # ABNORMALITIES TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS abnormalities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Active',
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)


    # =====================================================
    # AI PREDICTIONS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            model_id INTEGER,
            abnormality TEXT,
            confidence_score REAL,
            prediction_status TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (study_id)
                REFERENCES studies(id),
            FOREIGN KEY (model_id)
                REFERENCES ai_models(id)
        )
    """)


    # =====================================================
    # EXPLANATIONS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            explanation_type TEXT,
            model_version TEXT,
            explanation_path TEXT,
            metadata TEXT,
            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id)
                REFERENCES ai_predictions(id)
        )
    """)


    # =====================================================
    # SAVE CHANGES
    # =====================================================

    connection.commit()

    # Close database connection
    connection.close()


# =========================================================
# CREATE DATABASE WHEN FILE IS RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    create_tables()

    print(
        "SQLite database and tables created successfully."
    )
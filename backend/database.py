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

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (study_id)
                REFERENCES studies(id)
        )
    """)


    # Save changes
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
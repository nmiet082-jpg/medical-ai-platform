import sqlite3
from pathlib import Path

# Database file remains alongside this module. Existing records are preserved.
DATABASE_PATH = Path(__file__).resolve().parent / "medical_ai.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    # Enforce declared foreign keys for new connections.
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                account_status TEXT NOT NULL DEFAULT 'ACTIVE',
                email_verified INTEGER NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_record_id INTEGER,
                FOREIGN KEY (patient_record_id) REFERENCES patients(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Safe migration for databases created before patient-account linking.
        user_columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(users)").fetchall()
        }
        if "patient_record_id" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN patient_record_id INTEGER")

        # One patient record should not be linked to multiple login accounts.
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_patient_record_unique
            ON users(patient_record_id)
            WHERE patient_record_id IS NOT NULL
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                study_type TEXT,
                study_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_path TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abnormalities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                model_id INTEGER,
                abnormality TEXT,
                confidence_score REAL,
                prediction_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(id),
                FOREIGN KEY (model_id) REFERENCES ai_models(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                explanation_type TEXT,
                model_version TEXT,
                explanation_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES ai_predictions(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triage_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                priority TEXT NOT NULL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinician_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                clinician_name TEXT,
                review_status TEXT NOT NULL,
                final_decision TEXT,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                report_status TEXT NOT NULL,
                report_text TEXT,
                generated_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                performed_by TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    create_tables()
    print("SQLite database tables checked; existing records preserved.")

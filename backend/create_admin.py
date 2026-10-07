import getpass

from database import get_connection, create_tables
from security import hash_password


def create_initial_admin():
    print("=" * 50)
    print("       INITIAL ADMIN SETUP")
    print("=" * 50)

    # Make sure all database tables exist
    create_tables()

    connection = get_connection()
    cursor = connection.cursor()

    # Check whether an Admin already exists
    cursor.execute(
        "SELECT id, username FROM users WHERE role = ? LIMIT 1",
        ("Admin",)
    )

    existing_admin = cursor.fetchone()

    if existing_admin:
        print()
        print("An Admin account already exists.")
        print(f"Admin User ID: {existing_admin['username']}")
        print()
        print("Initial Admin setup is already completed.")
        connection.close()
        return

    print()
    print("Create your first Admin account.")
    print()

    full_name = input("Full Name: ").strip()
    username = input("User ID: ").strip()
    email = input("Email: ").strip()

    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Confirm Password: ")

    # Basic validation
    if not full_name:
        print("Error: Full Name cannot be empty.")
        connection.close()
        return

    if not username:
        print("Error: User ID cannot be empty.")
        connection.close()
        return

    if not email:
        print("Error: Email cannot be empty.")
        connection.close()
        return

    if not password:
        print("Error: Password cannot be empty.")
        connection.close()
        return

    if password != confirm_password:
        print("Error: Passwords do not match.")
        connection.close()
        return

    # Check whether username already exists
    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,)
    )

    if cursor.fetchone():
        print("Error: This User ID already exists.")
        connection.close()
        return

    # Check whether email already exists
    cursor.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,)
    )

    if cursor.fetchone():
        print("Error: This email already exists.")
        connection.close()
        return

    # Securely hash the password
    password_hash = hash_password(password)

    # Create Admin account
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
            failed_login_attempts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            full_name,
            username,
            email,
            password_hash,
            "Admin",
            "ACTIVE",
            1,
            0
        )
    )

    connection.commit()

    connection.close()

    print()
    print("=" * 50)
    print("INITIAL ADMIN CREATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Full Name : {full_name}")
    print(f"User ID   : {username}")
    print(f"Email     : {email}")
    print("Role      : Admin")
    print()
    print("You can now use this account for Admin login.")
    print("=" * 50)


if __name__ == "__main__":
    create_initial_admin()
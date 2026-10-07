from pwdlib import PasswordHash


# =========================================================
# PASSWORD HASHING
# =========================================================

password_hash = PasswordHash.recommended()


# =========================================================
# HASH PASSWORD
# =========================================================

def hash_password(password: str) -> str:
    """
    Convert a plain-text password into a secure password hash.
    """

    return password_hash.hash(password)


# =========================================================
# VERIFY PASSWORD
# =========================================================

def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    """
    Check whether a plain-text password matches
    the stored password hash.
    """

    return password_hash.verify(
        plain_password,
        hashed_password
    )
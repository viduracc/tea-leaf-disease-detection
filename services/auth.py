from passlib.context import CryptContext
from services.database import User, get_session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def register_user(email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message).
    """
    db = get_session()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return False, "Email already registered."
        
        new_user = User(
            email=email,
            hashed_password=hash_password(password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return True, "Registration successful!"
    except Exception as e:
        db.rollback()
        return False, f"Registration failed: {str(e)}"
    finally:
        db.close()


def authenticate_user(email: str, password: str) -> tuple[bool, str, int | None]:
    """
    Authenticate a user by email and password.
    Returns (success, message, user_id).
    """
    db = get_session()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return False, "Invalid email or password.", None
        if not verify_password(password, user.hashed_password):
            return False, "Invalid email or password.", None
        return True, "Login successful!", user.id
    finally:
        db.close()

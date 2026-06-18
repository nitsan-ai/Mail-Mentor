from app.config import SessionLocal
from app.models import User  # Import the User model from its new location

def authenticate_user(email: str, password: str):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user and user.check_password(password):
            return user
        return None
    finally:
        session.close()
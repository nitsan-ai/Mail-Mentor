from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.config import Base
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

class User(Base):
    """
    User model for storing user information and credentials.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    gmail_credentials = Column(Text, nullable=True)

    # This creates the link to the collection of emails owned by the user.
    emails = relationship("Email", back_populates="owner", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Email(Base):
    """
    Email model for storing individual email details.
    """
    __tablename__ = 'emails'

    id = Column(String, primary_key=True, index=True)
    subject = Column(String, index=True)
    sender = Column(String)
    recipient = Column(String)
    body = Column(Text)
    timestamp = Column(DateTime(timezone=True), index=True, default=datetime.datetime.now(datetime.timezone.utc))
    category = Column(String)
    priority = Column(String, default="Normal")
    status = Column(String, default="unread")
    summary = Column(Text, nullable=True)
    ai_response = Column(Text, nullable=True)

    # --- THIS IS THE CRITICAL FIX ---
    # This column links the email to a user in the 'users' table.
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))

    # This creates a direct link from an email object back to its User object.
    owner = relationship("User", back_populates="emails")


# reconbot/backend/app/models/user.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from app.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)  # This is already correct and supports manual login
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    google_id = Column(String, unique=True, nullable=True) # Used for Google OAuth
    created_at = Column(DateTime, default=datetime.utcnow)

    # Optional: Add __repr__ for better debugging
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', full_name='{self.full_name}')>"
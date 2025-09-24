# reconbot/backend/app/models/user.py - FIXED VERSION

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from datetime import datetime
from app.db import Base
import enum

class PlanType(enum.Enum):
    BETA_PRO = "beta_pro"
    LITE = "lite"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)  # This is already correct and supports manual login
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    google_id = Column(String, unique=True, nullable=True) # Used for Google OAuth
    created_at = Column(DateTime, default=datetime.utcnow)

    # NEW: Security fields for password reset
    reset_token_hash = Column(String, nullable=True)  # Stores hashed reset token
    reset_token_expires_at = Column(DateTime, nullable=True)  # Token expiration timestamp

    # ✅ FIXED: Proper default values for beta phase
    # During beta phase, all new users get beta access automatically
    plan_type = Column(Enum(PlanType), default=PlanType.BETA_PRO)
    plan_expires_at = Column(DateTime, nullable=True)
    monthly_reconciliations_count = Column(Integer, default=0)
    monthly_transactions_processed = Column(Integer, default=0)
    monthly_ai_chat_queries = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)
    beta_feedback_submitted = Column(Integer, default=0)

    # ✅ FIXED: Default to beta user for everyone during beta phase
    is_beta_user = Column(Boolean, default=True)  # Everyone gets beta access

    # ✅ FIXED: Only specific admins should be admin (not everyone!)
    is_admin = Column(Boolean, default=False)  # Changed from True to False

    # Optional: Add __repr__ for better debugging
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', full_name='{self.full_name}', is_beta_user={self.is_beta_user}, is_admin={self.is_admin})>"
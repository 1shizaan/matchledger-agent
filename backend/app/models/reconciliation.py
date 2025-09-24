from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from datetime import datetime
from app.db import Base

class Reconciliation(Base):
    __tablename__ = "reconciliations"

    id = Column(Integer, primary_key=True, index=True)
    uploaded_by = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(JSON, nullable=True)
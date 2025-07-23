from sqlalchemy import Column, Integer, String, DateTime, JSON,  ForeignKey
#from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from app.db import Base

#Base = declarative_base()

class Reconciliation(Base):
    __tablename__ = "reconciliations"

    id = Column(Integer, primary_key=True)
    uploaded_by = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(JSON)  # store matched/unmatched JSON
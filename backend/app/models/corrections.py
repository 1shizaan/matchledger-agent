from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.reconciliation import Base  # reuse your Base

class CorrectedMatch(Base):
    __tablename__ = "corrected_matches"

    id = Column(Integer, primary_key=True, index=True)
    reconciliation_id = Column(Integer, ForeignKey("reconciliations.id"), nullable=False)

    ledger_date = Column(String)
    ledger_amount = Column(Float)
    ledger_narration = Column(String)
    ledger_ref = Column(String)

    bank_date = Column(String)
    bank_amount = Column(Float)
    bank_narration = Column(String)
    bank_ref = Column(String)

    corrected_by = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
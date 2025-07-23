from app.auth.dependencies import get_current_user
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.recon_engine import match_transactions
from app.utils.file_parser import parse_csv, parse_excel # <-- ADDED parse_excel here
from app.utils.pdf_parser import parse_bank_pdf_to_df
from app.utils.chat_agent import run_chat_agent
from app.utils.email_utils import send_email_report
from app.models.user import User
from app.models.reconciliation import Reconciliation
from app.db import SessionLocal, get_db
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from pydantic import BaseModel

router = APIRouter()

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Response Models
class ReconciliationResult(BaseModel):
    id: int
    created_at: datetime
    uploaded_by: str
    summary: dict

class ReconciliationHistoryResponse(BaseModel):
    history: List[ReconciliationResult]

def convert_df_for_json(obj):
    """
    Enhanced JSON serializer with security checks
    """
    if isinstance(obj, dict):
        return {k: convert_df_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_df_for_json(elem) for elem in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.where(pd.notnull(obj), None).to_dict(orient='records')
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return convert_df_for_json(obj.__dict__)
    return str(obj)  # Fallback for unhandled types

@router.post("/reconcile", response_model=ReconciliationResult)
async def reconcile_files(
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    bank_is_pdf: bool = Form(False),
    email: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)

):
    """Secure reconciliation endpoint with user validation"""
    try:
        # Validate file types
        if not (ledger_file.filename.lower().endswith('.csv') or ledger_file.filename.lower().endswith('.xlsx')):
            raise HTTPException(400, "Ledger file must be CSV or Excel.")

        if bank_is_pdf and not bank_file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Bank file must be PDF when bank_is_pdf=True.")
        elif not bank_is_pdf and not (bank_file.filename.lower().endswith('.csv') or bank_file.filename.lower().endswith('.xlsx')):
            raise HTTPException(400, "Bank file must be CSV or Excel when bank_is_pdf=False.")

        # Read file content into memory (bytes)
        ledger_content = await ledger_file.read()
        bank_content = await bank_file.read()

        bank_path = None # Initialize bank_path for PDF handling

        try:
            # Parse ledger file based on extension
            if ledger_file.filename.lower().endswith('.csv'):
                ledger_df = parse_csv(ledger_content)
            elif ledger_file.filename.lower().endswith('.xlsx'):
                ledger_df = parse_excel(ledger_content)
            else:
                # This should ideally be caught by initial validation
                raise HTTPException(400, "Unsupported ledger file format.")

            # Parse bank file
            if bank_is_pdf:
                # For PDF, write to temp file then parse
                with tempfile.NamedTemporaryFile(delete=False) as bf_pdf:
                    bf_pdf.write(bank_content)
                    bank_path = bf_pdf.name
                bank_df = parse_bank_pdf_to_df(bank_path)
            else:
                # For CSV/Excel, parse directly from content
                if bank_file.filename.lower().endswith('.csv'):
                    bank_df = parse_csv(bank_content)
                elif bank_file.filename.lower().endswith('.xlsx'):
                    bank_df = parse_excel(bank_content)
                else:
                    # This should ideally be caught by initial validation
                    raise HTTPException(400, "Unsupported bank file format.")

            # Process data
            result = match_transactions(ledger_df, bank_df)
            json_result = convert_df_for_json(result)

            # Save to DB
            record = Reconciliation(
                user_id=current_user.id,
                uploaded_by=current_user.email,
                summary=json_result
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            # Optional email report
            if email:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_email_for_filename = "".join(c if c.isalnum() or c in ['@', '.', '_'] else '_' for c in email)
                temp_dir = tempfile.gettempdir()
                pdf_filename = os.path.join(temp_dir, f"recon_report_{timestamp}_{clean_email_for_filename}.pdf")

                send_email_report(email, result, filename=pdf_filename)
                print(f"✅ PDF generated and sent to {email}: {pdf_filename}")


            return record

        finally:
            # Secure cleanup (only for the bank PDF temp file if it was created)
            if bank_path and os.path.exists(bank_path):
                os.unlink(bank_path)

    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Reconciliation failed: {str(e)}")

@router.get("/history", response_model=ReconciliationHistoryResponse)
def get_reconciliation_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Secure history endpoint with pagination"""
    try:
        records = db.query(Reconciliation)\
            .filter(Reconciliation.user_id == current_user.id)\
            .order_by(Reconciliation.created_at.asc())\
            .offset(skip)\
            .limit(limit)\
            .all()

        history = []
        for r in records:
            try:
                summary = json.loads(r.summary) if isinstance(r.summary, str) else r.summary
                history.append({
                    "id": r.id,
                    "created_at": r.created_at,
                    "uploaded_by": r.uploaded_by,
                    "summary": summary
                })
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Skipping corrupted record {r.id}: {e}")
        print(f"Backend sending history: {{'history': {history}}}")
        return {"history": history}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch history: {str(e)}")

@router.post("/upload")
async def upload_files(
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Protected file preview endpoint"""
    try:
        # Read contents as bytes first
        ledger_content = await ledger_file.read()
        bank_content = await bank_file.read()

        # Determine parser for ledger file
        if ledger_file.filename.lower().endswith('.csv'):
            ledger_df = parse_csv(ledger_content)
        elif ledger_file.filename.lower().endswith('.xlsx'):
            ledger_df = parse_excel(ledger_content)
        else:
            raise HTTPException(400, "Unsupported ledger file format for preview.")

        # Determine parser for bank file
        if bank_file.filename.lower().endswith('.csv'):
            bank_df = parse_csv(bank_content)
        elif bank_file.filename.lower().endswith('.xlsx'):
            bank_df = parse_excel(bank_content)
        else:
            raise HTTPException(400, "Unsupported bank file format for preview.")

        return {
            "ledger_rows": ledger_df.head().to_dict(orient="records"),
            "bank_rows": bank_df.head().to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(400, f"File parsing error: {str(e)}")

@router.post("/chat")
async def chat_with_data(
    query: str = Form(...),
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Secure chat endpoint"""
    try:
        print("✅ /chat hit with:")
        print("   query:", query)
        print("   ledger_file:", ledger_file.filename)
        print("   bank_file:", bank_file.filename)

        ledger_content = await ledger_file.read()
        bank_content = await bank_file.read()

        if ledger_file.filename.lower().endswith('.csv'):
            ledger_df = parse_csv(ledger_content)
        elif ledger_file.filename.lower().endswith('.xlsx'):
            ledger_df = parse_excel(ledger_content)
        else:
            raise HTTPException(400, "Unsupported ledger file format for chat.")

        if bank_file.filename.lower().endswith('.csv'):
            bank_df = parse_csv(bank_content)
        elif bank_file.filename.lower().endswith('.xlsx'):
            bank_df = parse_excel(bank_content)
        else:
            raise HTTPException(400, "Unsupported bank file format for chat.")

        response = run_chat_agent(ledger_df, bank_df, query)
        return {"response": response}

    except Exception as e:
        print("🔥 Chat error:", str(e))
        raise HTTPException(400, f"Chat error: {str(e)}")


@router.delete("/cleanup-db")
def cleanup_database(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin-only cleanup endpoint"""
    if not current_user.is_admin:
        raise HTTPException(403, "Admin access required")

    try:
        corrupted = db.query(Reconciliation)\
            .filter(Reconciliation.user_id == current_user.id)\
            .filter(Reconciliation.summary == None)\
            .delete()
        db.commit()
        return {"cleaned": corrupted}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Cleanup failed: {str(e)}")

# Add health check endpoint
@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
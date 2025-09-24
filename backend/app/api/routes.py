# backend/app/api/routes.py - COMPLETE FIXED VERSION

# --- Existing Imports ---
from app.auth.dependencies import get_current_user
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.utils.file_parser import parse_csv, parse_excel
from app.utils.pdf_parser import parse_bank_pdf_to_df
from app.utils.chat_agent import run_chat_agent
from app.utils.email_utils import send_email_report
from app.models.user import User
from app.models.reconciliation import Reconciliation
from app.db import get_db
from app.services.usage_tracker import UsageTracker
import base64
from app.utils.column_utils import detect_csv_headers, suggest_column_mapping
from app.schemas.user import ColumnDetectionResponse, StartReconciliationRequest
from app.models.feedback import BetaFeedback
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# --- ✅ NEW IMPORTS for Celery Tasks ---
from app.tasks import run_reconciliation_task
from app.celery_app import celery

router = APIRouter()

# --- Your existing Security and Response Models ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- ✅ FIXED: Enhanced models for the history list to prevent duplicates ---
class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    uploaded_by: str
    matched_count: int
    unmatched_count: int
    total_count: int
    match_rate: float

class ReconciliationHistoryResponse(BaseModel):
    history: List[HistoryItem]

# --- ✅ ADD: ErrorReport class definition ---
class ErrorReport(BaseModel):
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    user_agent: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[str] = None

# --- ⭐️ Enhanced JSON converter for better data handling ⭐️ ---
def convert_df_for_json(obj):
    """
    Enhanced JSON serializer with security checks.
    This function is now also imported by tasks.py.
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
    return str(obj)

# --- ✅ FIXED: Enhanced column detection endpoint ---
@router.post("/detect-columns", response_model=ColumnDetectionResponse)
async def detect_columns(
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    ✅ ENHANCED: Endpoint to detect columns and suggest mappings before reconciliation
    """
    try:
        # Read file contents
        ledger_content = await ledger_file.read()
        bank_content = await bank_file.read()

        # Detect headers with enhanced logic
        ledger_headers = detect_csv_headers(ledger_content, ledger_file.filename)
        bank_headers = detect_csv_headers(bank_content, bank_file.filename)

        # Generate smart column suggestions
        ledger_suggestions = suggest_column_mapping(ledger_headers)
        bank_suggestions = suggest_column_mapping(bank_headers)

        print(f"✅ Column Detection Results:")
        print(f"  Ledger headers: {ledger_headers}")
        print(f"  Bank headers: {bank_headers}")
        print(f"  Ledger suggestions: {ledger_suggestions}")
        print(f"  Bank suggestions: {bank_suggestions}")

        # ✅ FIXED: Return consistent response structure
        return {
            "ledger_headers": ledger_headers,
            "bank_headers": bank_headers,
            "ledger_suggestions": ledger_suggestions,
            "bank_suggestions": bank_suggestions,
            "confidence_scores": {
                "ledger_date": ledger_suggestions.get("confidence", {}).get("date", 0),
                "ledger_amount": ledger_suggestions.get("confidence", {}).get("amount", 0),
                "ledger_description": ledger_suggestions.get("confidence", {}).get("description", 0),
                "bank_date": bank_suggestions.get("confidence", {}).get("date", 0),
                "bank_amount": bank_suggestions.get("confidence", {}).get("amount", 0),
                "bank_description": bank_suggestions.get("confidence", {}).get("description", 0),
            }
        }

    except Exception as e:
        print(f"❌ Error in detect_columns: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error detecting columns: {str(e)}")

# --- ✅ FIXED: Enhanced reconciliation with column mapping ---
@router.post("/reconcile/start-with-mapping")
async def start_reconciliation_with_mapping(
    request: StartReconciliationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    ✅ ENHANCED: Reconciliation endpoint with column mapping support
    """
    try:
        # Check usage limits
        tracker = UsageTracker(db)
        tracker.check_limits(current_user)

        # Decode base64 file contents
        ledger_content = base64.b64decode(request.ledger_file_content)
        bank_content = base64.b64decode(request.bank_file_content)

        # ✅ FIXED: Validate required column mappings
        required_columns = ["date", "amount"]
        for col in required_columns:
            if not getattr(request.ledger_column_map, col, None):
                raise HTTPException(400, f"Missing required ledger column mapping: {col}")
            if not getattr(request.bank_column_map, col, None):
                raise HTTPException(400, f"Missing required bank column mapping: {col}")

        print(f"✅ Starting enhanced reconciliation for user {current_user.email}")
        print(f"  Ledger mappings: {request.ledger_column_map.model_dump()}")
        print(f"  Bank mappings: {request.bank_column_map.model_dump()}")

        # Start the enhanced task with column mappings
        task = run_reconciliation_task.delay(
            ledger_content=ledger_content,
            bank_content=bank_content,
            ledger_filename=request.ledger_filename,
            bank_filename=request.bank_filename,
            bank_is_pdf=request.bank_is_pdf,
            email=request.email,
            user_id=current_user.id,
            user_email=current_user.email,
            # ✅ NEW: Pass the column mappings
            ledger_column_map=request.ledger_column_map.model_dump(),
            bank_column_map=request.bank_column_map.model_dump()
        )

        return {"status": "Enhanced reconciliation started", "task_id": task.id}

    except Exception as e:
        print(f"❌ Error in start_reconciliation_with_mapping: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error starting enhanced reconciliation: {str(e)}")

@router.post("/reconcile/start")
async def start_reconciliation_job(
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    bank_is_pdf: bool = Form(False),
    email: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    ✅ FIXED: Basic reconciliation endpoint with usage tracking
    """
    tracker = UsageTracker(db)
    tracker.check_limits(current_user)

    ledger_content = await ledger_file.read()
    bank_content = await bank_file.read()

    print(f"✅ Starting basic reconciliation for user {current_user.email}")

    task = run_reconciliation_task.delay(
        ledger_content=ledger_content,
        bank_content=bank_content,
        ledger_filename=ledger_file.filename,
        bank_filename=bank_file.filename,
        bank_is_pdf=bank_is_pdf,
        email=email,
        user_id=current_user.id,
        user_email=current_user.email
    )
    return {"status": "Basic reconciliation started", "task_id": task.id}

@router.get("/reconcile/status/{task_id}")
def get_reconciliation_status(task_id: str):
    """
    ✅ ENHANCED: Task status checker with better error handling
    """
    try:
        task_result = celery.AsyncResult(task_id)

        # Safely get the task state without triggering exception deserialization
        try:
            state = task_result.state
            info = task_result.info
        except (ValueError, KeyError, Exception) as e:
            print(f"❌ Error accessing task state/info: {e}")
            return {
                'task_id': task_id,
                'state': 'ERROR',
                'info': 'Task result corrupted - unable to retrieve status'
            }

        response = {
            'task_id': task_id,
            'state': state,
            'info': info,
        }

        # Only try to get result if task succeeded
        if state == 'SUCCESS':
            try:
                response['result'] = task_result.get()
                print(f"✅ Task {task_id} completed successfully")
            except (ValueError, KeyError, Exception) as e:
                print(f"❌ Error getting task result: {e}")
                response['result'] = 'Task completed but result data corrupted'
        elif state == 'FAILURE':
            response['info'] = 'Task failed - check worker logs for details'
            print(f"❌ Task {task_id} failed")

        return response

    except Exception as e:
        print(f"❌ Outer exception in get_reconciliation_status: {e}")
        return {
            'task_id': task_id,
            'state': 'ERROR',
            'info': f'Error retrieving task status: {str(e)}'
        }

# --- ✅ FIXED: Enhanced history endpoint to prevent duplicates and 502 errors ---
@router.get("/history", response_model=ReconciliationHistoryResponse)
def get_reconciliation_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    ✅ FIXED: Secure and EFFICIENT history endpoint with duplicate prevention
    """
    try:
        # Query reconciliations with proper ordering and deduplication
        records = db.query(Reconciliation)\
            .filter(Reconciliation.user_id == current_user.id)\
            .filter(Reconciliation.summary.isnot(None))\
            .order_by(Reconciliation.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

        history = []
        seen_ids = set()  # ✅ FIXED: Prevent duplicates

        for r in records:
            # ✅ FIXED: Skip duplicates
            if r.id in seen_ids:
                continue
            seen_ids.add(r.id)

            try:
                summary = json.loads(r.summary) if isinstance(r.summary, str) else r.summary

                matched_count = len(summary.get('matched', []))
                unmatched_ledger_count = len(summary.get('unmatched_ledger', []))
                unmatched_bank_count = len(summary.get('unmatched_bank', []))
                total_count = matched_count + unmatched_ledger_count + unmatched_bank_count
                match_rate = (matched_count / total_count * 100) if total_count > 0 else 0

                history.append({
                    "id": r.id,
                    "created_at": r.created_at,
                    "uploaded_by": r.uploaded_by or current_user.email,
                    "matched_count": matched_count,
                    "unmatched_count": unmatched_ledger_count + unmatched_bank_count,
                    "total_count": total_count,
                    "match_rate": round(match_rate, 1)
                })
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                print(f"⚠️ Skipping corrupted summary for record {r.id}: {e}")

        print(f"✅ Returning {len(history)} history items for user {current_user.email}")
        return {"history": history}

    except Exception as e:
        print(f"❌ Error fetching history: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to fetch history: {str(e)}")

# --- ✅ FIXED: Enhanced chat endpoint with better reconciliation support ---
@router.post("/chat")
async def chat_with_data(
    query: str = Form(...),
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    reconciliation_summary: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    ✅ ENHANCED: Chat endpoint with comprehensive reconciliation support
    """
    try:
        print("=== ✅ ENHANCED CHAT ENDPOINT START ===")
        print(f"Query: '{query}'")
        print(f"User: {current_user.email}")
        print(f"Ledger file: {ledger_file.filename}")
        print(f"Bank file: {bank_file.filename}")

        # Check usage limits
        tracker = UsageTracker(db)
        tracker.check_limits(current_user)
        tracker.record_ai_chat_usage(current_user)

        # ✅ FIXED: Parse reconciliation summary with better error handling
        parsed_reconciliation_summary = None
        if reconciliation_summary:
            try:
                parsed_reconciliation_summary = json.loads(reconciliation_summary)
                print(f"✅ Parsed reconciliation summary successfully!")

                # Log summary statistics
                if isinstance(parsed_reconciliation_summary, dict):
                    matched = parsed_reconciliation_summary.get('matched', [])
                    unmatched_ledger = parsed_reconciliation_summary.get('unmatched_ledger', [])
                    unmatched_bank = parsed_reconciliation_summary.get('unmatched_bank', [])

                    print(f"  Matched: {len(matched)} transactions")
                    print(f"  Unmatched ledger: {len(unmatched_ledger)} transactions")
                    print(f"  Unmatched bank: {len(unmatched_bank)} transactions")

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse reconciliation_summary as JSON: {e}")
                parsed_reconciliation_summary = None
            except Exception as e:
                print(f"❌ Error processing reconciliation_summary: {e}")
                parsed_reconciliation_summary = None
        else:
            print("⚠️ No reconciliation_summary provided")

        # Parse uploaded files
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

        print(f"✅ File parsing successful:")
        print(f"  Ledger DataFrame shape: {ledger_df.shape}")
        print(f"  Bank DataFrame shape: {bank_df.shape}")

        # ✅ CRITICAL FIX: Call chat agent with reconciliation summary
        print("🤖 Calling enhanced chat agent...")
        response = run_chat_agent(
            ledger_df=ledger_df,
            bank_df=bank_df,
            query=query,
            reconciliation_summary=parsed_reconciliation_summary
        )

        print(f"✅ Chat agent response received: {len(str(response))} characters")
        print("=== ✅ ENHANCED CHAT ENDPOINT END ===")

        return {"response": response}

    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(400, f"Chat error: {str(e)}")

# --- All other endpoints ---
@router.post("/upload")
async def upload_files(
    ledger_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Protected file preview endpoint"""
    try:
        ledger_content = await ledger_file.read()
        bank_content = await bank_file.read()

        if ledger_file.filename.lower().endswith('.csv'):
            ledger_df = parse_csv(ledger_content)
        elif ledger_file.filename.lower().endswith('.xlsx'):
            ledger_df = parse_excel(ledger_content)
        else:
            raise HTTPException(400, "Unsupported ledger file format for preview.")

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

@router.get("/usage/stats")
def get_usage_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tracker = UsageTracker(db)
    return tracker.get_usage_stats(current_user)

@router.post("/beta/feedback")
def submit_feedback(message: str = Form(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.beta_feedback_submitted = (current_user.beta_feedback_submitted or 0) + 1

    feedback = BetaFeedback(user_id=current_user.id, message=message.strip())
    db.add(feedback)
    db.commit()
    print(f"�� Feedback received from {current_user.email}: {message}")
    return {"status": "Thanks for your feedback!"}

@router.get("/admin/beta-feedback")
def list_feedback(limit: int = 50, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(403, "Admin access required")

    feedbacks = db.query(BetaFeedback).order_by(BetaFeedback.created_at.desc()).limit(limit).all()
    return [
        {"user_id": f.user_id, "message": f.message, "created_at": f.created_at.isoformat()}
        for f in feedbacks
    ]

@router.get("/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get user profile information"""
    try:
        return {
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.full_name or current_user.email,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "is_active": current_user.is_active,
            "plan": current_user.plan_type.value if current_user.plan_type else "free",
            "is_beta_user": current_user.is_beta_user,
            "is_admin": current_user.is_admin,
            "usage_stats": {
                "reconciliations_this_month": current_user.monthly_reconciliations_count or 0,
                "total_reconciliations": current_user.monthly_reconciliations_count or 0,
                "ai_chat_queries": current_user.monthly_ai_chat_queries or 0
            }
        }
    except Exception as e:
        print(f"❌ Error fetching profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")

@router.post("/errors/report")
async def report_error(
    error_report: ErrorReport,
    request: Request
):
    """Report frontend errors - No authentication required"""
    try:
        # Try to get user info if available, but don't require it
        user_info = "anonymous"
        try:
            # Try to extract token and get user info, but don't fail if not available
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                # You could add optional user extraction here, but for simplicity, we'll keep it anonymous
                pass
        except:
            pass

        # Log the error
        error_data = {
            "user_id": user_info,
            "error_type": error_report.error_type,
            "error_message": error_report.error_message,
            "stack_trace": error_report.stack_trace,
            "url": error_report.url,
            "user_agent": error_report.user_agent,
            "timestamp": error_report.timestamp or datetime.utcnow().isoformat()
        }

        # Use the existing logger or create one for errors
        import logging
        error_logger = logging.getLogger("frontend_errors")
        error_logger.error(f"Frontend Error Report: {error_data}")

        return {
            "status": "success",
            "message": "Error report received",
            "report_id": f"err_{int(datetime.utcnow().timestamp())}"
        }
    except Exception as e:
        print(f"❌ Error reporting error: {str(e)}")
        # Don't raise an exception for error reporting - just log and return success
        return {
            "status": "success",
            "message": "Error report received (with issues)",
            "report_id": f"err_{int(datetime.utcnow().timestamp())}"
        }

@router.delete("/cleanup-db")
def cleanup_database(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin-only cleanup endpoint"""
    if not current_user.is_admin:
        raise HTTPException(403, "Admin access required")

    try:
        num_deleted = db.query(Reconciliation)\
            .filter(Reconciliation.user_id == current_user.id)\
            .filter(Reconciliation.summary == None)\
            .delete(synchronize_session=False)
        db.commit()
        return {"cleaned": num_deleted}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Cleanup failed: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
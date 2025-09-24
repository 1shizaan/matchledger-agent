# reconbot/backend/app/schemas/user.py

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    is_active: bool
    google_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class GoogleAuthRequest(BaseModel):
    token: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class PasswordResetResponse(BaseModel):
    message: str

class ColumnMapping(BaseModel):
    date: str
    narration: str
    debit: Optional[str] = None
    credit: Optional[str] = None
    ref_no: Optional[str] = None

class StartReconciliationRequest(BaseModel):
    ledger_file_content: str  # base64 encoded
    bank_file_content: str    # base64 encoded
    ledger_filename: str
    bank_filename: str
    bank_is_pdf: bool
    email: Optional[str] = None
    ledger_column_map: ColumnMapping
    bank_column_map: ColumnMapping

class ColumnDetectionResponse(BaseModel):
    ledger_headers: List[str]
    bank_headers: List[str]
    ledger_suggestions: Dict[str, Optional[str]]
    bank_suggestions: Dict[str, Optional[str]]
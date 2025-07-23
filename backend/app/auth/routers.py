# reconbot/backend/app/auth/routers.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from app.auth.services import AuthService, generate_password_reset_token, verify_password_reset_token
from app.db import get_db
from app.models.user import User
from app.utils.email_utils import send_password_reset_email

router = APIRouter()
auth_service = AuthService()

class GoogleAuthRequest(BaseModel):
    token: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

@router.post("/google")
async def login_google(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Handles Google OAuth login by verifying the token and getting/creating a user.
    Returns an access token upon successful authentication.
    """
    google_info = await auth_service.verify_google_token(data.token)
    user = auth_service.get_or_create_user(db, google_info)
    access_token = auth_service.create_access_token(user.id, user.email)
    return {"access_token": access_token}

@router.post("/login")
async def login_manual(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Handles manual email and password login.
    Authenticates the user and returns an access token if credentials are valid.
    """
    user = auth_service.authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_service.create_access_token(user.id, user.email)
    return {"access_token": access_token}

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Handles new user registration.
    """
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = auth_service.get_password_hash(user_data.password)

    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "User registered successfully", "user_id": user.id}

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Handles a request to reset a password.
    Generates a token and sends a password reset email.
    """
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        # For security, always respond as if the email was sent to prevent email enumeration.
        raise HTTPException(status_code=status.HTTP_200_OK, detail="If a matching account was found, a password reset email has been sent.")

    # Generate the raw token
    token = generate_password_reset_token(user.email)

    # ✅ CORRECTED: Pass only the raw token.
    # The email utility is responsible for building the full link.
    await send_password_reset_email(user.email, token)

    return {"message": "If a matching account was found, a password reset email has been sent."}

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Resets the user's password using a valid token.
    """
    email = verify_password_reset_token(request.token)
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token.")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    # Hash the new password and update the user record
    user.hashed_password = auth_service.get_password_hash(request.new_password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "Password has been reset successfully."}
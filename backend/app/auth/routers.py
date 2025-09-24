# reconbot/backend/app/auth/routers.py - FIXED VERSION

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
from passlib.context import CryptContext

from app.auth.services import AuthService
from app.db import get_db
from app.models.user import User
from app.schemas.user import (
    GoogleAuthRequest, UserLogin, UserCreate,
    ForgotPasswordRequest, ResetPasswordRequest
)
from app.utils.email_utils import send_password_reset_email
from app.core.config import settings

router = APIRouter()
auth_service = AuthService()

# Password hashing context for reset tokens
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ✅ FIXED: Admin email list (add your admin emails here)
ADMIN_EMAILS = [
    "admin@matchledger.com",
    "optixonai@gmail.com",
    "t.d.m.mohi@gmail.com",  # Add your email here
    # Add more admin emails as needed
]

def generate_reset_token():
    """Generate a cryptographically secure random token"""
    return secrets.token_urlsafe(32)

def hash_reset_token(token: str) -> str:
    """Hash the reset token for secure storage"""
    return pwd_context.hash(token)

def verify_reset_token(token: str, hashed_token: str) -> bool:
    """Verify the reset token against the stored hash"""
    return pwd_context.verify(token, hashed_token)

@router.post("/google")
async def login_google(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Handles Google OAuth login by verifying the token and getting/creating a user.
    Returns an access token upon successful authentication.
    """
    google_info = await auth_service.verify_google_token(data.token)
    user = auth_service.get_or_create_user(db, google_info)
    access_token = auth_service.create_access_token(user)
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
    access_token = auth_service.create_access_token(user)
    return {"access_token": access_token}

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    ✅ FIXED: Handles new user registration with proper admin/beta access control
    """
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = auth_service.get_password_hash(user_data.password)

    # ✅ FIXED: Determine admin and beta status based on email
    is_admin = user_data.email.lower() in [email.lower() for email in ADMIN_EMAILS]
    is_beta_user = True  # During beta phase, all users get beta access

    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_admin=is_admin,  # ✅ FIXED: Only specific emails get admin access
        is_beta_user=is_beta_user  # ✅ FIXED: All users get beta access during beta phase
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    print(f"✅ New user registered: {user.email}, Admin: {is_admin}, Beta: {is_beta_user}")

    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "is_admin": is_admin,
        "is_beta_user": is_beta_user
    }

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Handles a request to reset a password.
    Generates an encrypted token and sends a password reset email.
    """
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        # For security, always respond as if the email was sent to prevent email enumeration.
        return {"message": "If a matching account was found, a password reset email has been sent."}

    # Generate a secure random token
    raw_token = generate_reset_token()

    # Hash the token for database storage
    hashed_token = hash_reset_token(raw_token)

    # Set expiration time
    expiry_time = datetime.utcnow() + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)

    # Store the hashed token and expiry in the database
    user.reset_token_hash = hashed_token
    user.reset_token_expires_at = expiry_time
    db.commit()

    # Send the raw token via email (never store the raw token)
    await send_password_reset_email(user.email, raw_token)

    return {"message": "If a matching account was found, a password reset email has been sent."}

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Resets the user's password using a valid encrypted token.
    """
    # Find user with an active reset token
    users_with_tokens = db.query(User).filter(
        User.reset_token_hash.isnot(None),
        User.reset_token_expires_at.isnot(None)
    ).all()

    valid_user = None

    # Check each user's token hash against the provided token
    for user in users_with_tokens:
        # Check if token is not expired
        if datetime.utcnow() > user.reset_token_expires_at:
            continue

        # Verify the token against the stored hash
        if verify_reset_token(request.token, user.reset_token_hash):
            valid_user = user
            break

    if not valid_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token."
        )

    # Hash the new password and update the user record
    valid_user.hashed_password = auth_service.get_password_hash(request.new_password)

    # Clear the reset token data for security
    valid_user.reset_token_hash = None
    valid_user.reset_token_expires_at = None

    db.commit()

    return {"message": "Password has been reset successfully."}
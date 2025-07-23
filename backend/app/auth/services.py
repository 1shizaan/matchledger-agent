# reconbot/backend/app/auth/services.py

import os
import bcrypt
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException, status
import httpx
from sqlalchemy.orm import Session
from app.models.user import User
from typing import Optional

# --- Password Reset Token Configuration (NEW) ---
_RESET_SECRET_KEY = os.getenv("RESET_PASSWORD_SECRET_KEY", "fallback-reset-secret-change-in-prod") # Use the new env var
_RESET_ALGORITHM = "HS256"
_RESET_TOKEN_EXPIRE_MINUTES = 30 # Token valid for 30 minutes

def generate_password_reset_token(email: str) -> str:
    """Generates a JWT token for password reset."""
    expires_delta = timedelta(minutes=_RESET_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta
    to_encode = {"sub": email, "exp": expire, "type": "password_reset"}
    encoded_jwt = jwt.encode(to_encode, _RESET_SECRET_KEY, algorithm=_RESET_ALGORITHM)
    return encoded_jwt

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verifies a password reset token and returns the email if valid."""
    try:
        payload = jwt.decode(token, _RESET_SECRET_KEY, algorithms=[_RESET_ALGORITHM])
        if payload.get("type") != "password_reset":
            return None # Not a password reset token
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None # Invalid or expired token

# --- AuthService Class (Existing with minor modification for clarity) ---
class AuthService:
    def __init__(self):
        self.SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-change-in-prod")
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE = timedelta(minutes=60)
        self.GOOGLE_CLIENT_ID = os.getenv("VITE_GOOGLE_CLIENT_ID") # Or GOOGLE_CLIENT_ID directly

    def create_access_token(self, user_id: int, email: str) -> str:
        payload = {
            "sub": email,
            "user_id": user_id,
            "exp": datetime.utcnow() + self.ACCESS_TOKEN_EXPIRE,
            "iss": "reconbot-auth"
        }
        return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            if payload.get("iss") != "reconbot-auth":
                raise JWTError("Invalid issuer")
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )

    async def verify_google_token(self, token: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Verify token with Google
                response = await client.get(
                    f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
                )
                response.raise_for_status()
                token_info = response.json()

                # Validate audience
                if token_info.get("aud") != self.GOOGLE_CLIENT_ID:
                    raise HTTPException(status_code=400, detail="Invalid token audience")

                return token_info
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Google token verification failed: {e.response.text}"
                )

    def get_password_hash(self, password: str) -> str:
        """Hashes a plain-text password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain-text password against a hashed password."""
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticates a user by email and password.
        Returns the User object if authenticated, otherwise None.
        """
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.hashed_password:
            return None # User not found or no password set (e.g., Google user without manual password)

        if not self.verify_password(password, user.hashed_password):
            return None # Password does not match
        return user

    def get_or_create_user(self, db: Session, google_info: dict) -> User:
        """
        Gets an existing user by Google ID or email, or creates a new one.
        Primarily used for Google authentication flow.
        """
        # Try to find user by google_id first
        user = db.query(User).filter(User.google_id == google_info["sub"]).first()
        if user:
            return user

        # If not found by google_id, try by email
        user = db.query(User).filter(User.email == google_info["email"]).first()
        if user:
            # Update existing user with google_id if it's a manual user linking Google
            if not user.google_id:
                user.google_id = google_info["sub"]
                db.add(user)
                db.commit()
                db.refresh(user)
            return user

        # If no existing user, create a new one
        user = User(
            email=google_info["email"],
            full_name=google_info.get("name", ""),
            google_id=google_info["sub"],
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        return user
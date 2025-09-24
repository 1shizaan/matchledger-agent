# backend/app/auth/dependencies.py - FIXED VERSION
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.user import User
from app.auth.services import AuthService
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Initialize AuthService globally
auth_service = AuthService()

# ✅ CRITICAL FIX: Change tokenUrl from "auth/login" to "/auth/login"
# This must match your router prefix in main.py: app.include_router(auth_router, prefix="/auth")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Enhanced get_current_user with token content debugging"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        logger.info(f"🔍 Auth Debug - Received token: {token[:20] if token else 'None'}...")

        # Decode the JWT
        payload = jwt.decode(token, auth_service.SECRET_KEY, algorithms=[auth_service.ALGORITHM])

        # ✅ NEW DEBUG: Log the entire token payload
        logger.info(f"�� TOKEN DEBUG - Full payload: {payload}")
        logger.info(f"🔍 TOKEN DEBUG - is_admin in token: {payload.get('is_admin', 'NOT_FOUND')}")
        logger.info(f"🔍 TOKEN DEBUG - Token created at: {payload.get('exp', 'NO_EXP')}")

        email: str = payload.get("sub")
        if email is None:
            logger.error("❌ Auth Debug - No email in token payload")
            raise credentials_exception

        logger.info(f"🔍 Auth Debug - Email from token: {email}")

    except JWTError as e:
        logger.error(f"❌ Auth Debug - JWT decode error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"❌ Auth Debug - Unexpected error: {str(e)}")
        raise credentials_exception

    # Query user from database
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        logger.error(f"❌ Auth Debug - User {email} not found in database")
        raise credentials_exception

    if not user.is_active:
        logger.error(f"❌ Auth Debug - User {email} is inactive")
        raise HTTPException(status_code=400, detail="Inactive user")

    # ✅ NEW DEBUG: Compare token vs database
    logger.info(f"🔍 DATABASE DEBUG - user.is_admin from DB: {user.is_admin}")
    logger.info(f"🔍 COMPARISON DEBUG - Token says admin: {payload.get('is_admin')}, DB says admin: {user.is_admin}")

    logger.info(f"✅ Auth Debug - Successfully authenticated user: {user.email}")
    return user
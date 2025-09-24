# reconbot/backend/app/auth/services.py - FIXED VERSION

import os
import bcrypt
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException, status
import httpx
from sqlalchemy.orm import Session
from app.models.user import User, PlanType
from typing import Optional

# --- Password Reset Token Configuration (NEW) ---
_RESET_SECRET_KEY = os.getenv("RESET_PASSWORD_SECRET_KEY", "fallback-reset-secret-change-in-prod")
_RESET_ALGORITHM = "HS256"
_RESET_TOKEN_EXPIRE_MINUTES = 30

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
            return None
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None

# ✅ FIXED: Updated admin email list with your actual admin emails
ADMIN_EMAILS = [
    "admin@matchledger.com",
    "support@matchledger.com",
    "optixonai@gmail.com",
    "t.d.m.mohi@gmail.com",  # Your main admin email
    # Add more admin emails here as needed
]

# --- AuthService Class (ENHANCED - Better Beta User Handling) ---
class AuthService:
    def __init__(self):
        self.SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-change-in-prod")
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE = timedelta(minutes=60)
        self.GOOGLE_CLIENT_ID = os.getenv("VITE_GOOGLE_CLIENT_ID")

    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user - FIXED admin detection"""
        print(f"🔍 DEBUG: Creating token for user: {user.email}")

        try:
            user_id = user.id
            email = user.email
            full_name = user.full_name

            # ✅ CRITICAL FIX: The issue is in this logic!
            # First, get the actual database value
            db_admin_value = getattr(user, 'is_admin', False)

            # Also check email list as backup
            email_in_admin_list = email in ADMIN_EMAILS

            # The user should be admin if EITHER condition is true
            is_admin = db_admin_value or email_in_admin_list

            # Debug logging to see exactly what's happening
            print(f"🔍 DEBUG: user.email = {email}")
            print(f"🔍 DEBUG: user object type = {type(user)}")
            print(f"🔍 DEBUG: user.is_admin (from DB) = {db_admin_value}")
            print(f"🔍 DEBUG: email in ADMIN_EMAILS = {email_in_admin_list}")
            print(f"🔍 DEBUG: ADMIN_EMAILS list = {ADMIN_EMAILS}")
            print(f"�� DEBUG: FINAL is_admin value = {is_admin}")

            # Beta access logic (keep as is)
            has_beta_plan = hasattr(user, 'plan_type') and user.plan_type == PlanType.BETA_PRO
            user_is_beta = getattr(user, 'is_beta_user', False)
            is_beta_user = user_is_beta or has_beta_plan or True

            plan_type = user.plan_type.value if hasattr(user, 'plan_type') and user.plan_type else "beta_pro"

        except Exception as e:
            print(f"❌ ERROR: Failed to access user attributes: {e}")
            # Fallback logic
            user_id = getattr(user, 'id', None)
            email = getattr(user, 'email', '')
            full_name = getattr(user, 'full_name', '')
            is_admin = email in ADMIN_EMAILS  # Fallback to email check
            is_beta_user = True
            plan_type = "beta_pro"

        # Create the payload
        payload = {
            "sub": email,
            "user_id": user_id,
            "name": full_name,
            "is_admin": is_admin,  # ✅ This should now be True for your email
            "is_beta_user": is_beta_user,
            "plan_type": plan_type,
            "exp": datetime.utcnow() + self.ACCESS_TOKEN_EXPIRE,
            "iss": "reconbot-auth"
        }

        print(f"🔍 DEBUG: Final token payload = {payload}")
        print(f"🔍 DEBUG: Token payload is_admin = {payload['is_admin']}")

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
            return None

        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def get_or_create_user(self, db: Session, google_info: dict) -> User:
        """
        Gets an existing user by Google ID or email, or creates a new one.
        ✅ ENHANCED: ALL NEW USERS GET BETA ACCESS AUTOMATICALLY
        """
        try:
            email = google_info["email"]
            google_id = google_info["sub"]
            full_name = google_info.get("name", "")

            print(f"DEBUG: Processing user registration/login for: {email}")

            # Try to find user by google_id first
            user = db.query(User).filter(User.google_id == google_id).first()
            if user:
                print(f"DEBUG: Found existing user by Google ID: {user.email}")
                # ✅ ENHANCED: Update existing users to ensure beta access
                updated = False

                should_be_admin = user.email in ADMIN_EMAILS
                if user.is_admin != should_be_admin:
                    print(f"DEBUG: Updating admin status for {user.email}: {user.is_admin} -> {should_be_admin}")
                    user.is_admin = should_be_admin
                    updated = True

                # 🚀 ENSURE BETA ACCESS: During beta phase, ALL users should have beta access
                if not user.is_beta_user:
                    print(f"DEBUG: ✅ Granting beta access to existing user: {user.email}")
                    user.is_beta_user = True
                    updated = True

                if user.plan_type != PlanType.BETA_PRO:
                    print(f"DEBUG: ✅ Setting BETA_PRO plan for existing user: {user.email}")
                    user.plan_type = PlanType.BETA_PRO
                    updated = True

                if updated:
                    db.commit()
                    db.refresh(user)
                    print(f"DEBUG: ✅ Updated existing user: {user.email} (Admin: {user.is_admin}, Beta: {user.is_beta_user})")

                return user

            # If not found by google_id, try by email
            user = db.query(User).filter(User.email == email).first()
            if user:
                print(f"DEBUG: Found existing user by email, updating Google ID: {user.email}")

                # Update existing user with google_id and ensure proper access
                user.google_id = google_id
                user.full_name = full_name or user.full_name

                should_be_admin = email in ADMIN_EMAILS
                if user.is_admin != should_be_admin:
                    print(f"DEBUG: Updating admin status for {email}: {user.is_admin} -> {should_be_admin}")
                    user.is_admin = should_be_admin

                # 🚀 ENSURE BETA ACCESS: During beta phase, ALL users should have beta access
                if not user.is_beta_user:
                    print(f"DEBUG: ✅ Granting beta access to existing email user: {email}")
                    user.is_beta_user = True

                if user.plan_type != PlanType.BETA_PRO:
                    print(f"DEBUG: ✅ Setting BETA_PRO plan for existing email user: {email}")
                    user.plan_type = PlanType.BETA_PRO

                db.commit()
                db.refresh(user)
                print(f"DEBUG: ✅ Updated existing email user: {user.email} (Admin: {user.is_admin}, Beta: {user.is_beta_user})")
                return user

            # 🚀 CREATE NEW USER: If no existing user found, create with FULL BETA ACCESS
            print(f"DEBUG: Creating NEW user for {email}")

            is_admin = email in ADMIN_EMAILS
            print(f"DEBUG: New user admin status: {is_admin} (email in ADMIN_EMAILS: {email in ADMIN_EMAILS})")

            user = User(
                email=email,
                full_name=full_name,
                google_id=google_id,
                is_active=True,
                is_admin=is_admin,
                is_beta_user=True,  # ✅ ALL NEW USERS GET BETA ACCESS
                plan_type=PlanType.BETA_PRO,  # ✅ ALL NEW USERS GET BETA_PRO PLAN
                created_at=datetime.utcnow(),
                usage_reset_date=datetime.utcnow()
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            print(f"DEBUG: ✅ Successfully created NEW user:")
            print(f"       • Email: {user.email}")
            print(f"       • ID: {user.id}")
            print(f"       • Admin: {user.is_admin}")
            print(f"       • Beta User: {user.is_beta_user}")
            print(f"       • Plan Type: {user.plan_type}")

            return user

        except Exception as e:
            print(f"❌ ERROR in get_or_create_user: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error during user creation/retrieval: {str(e)}"
            )
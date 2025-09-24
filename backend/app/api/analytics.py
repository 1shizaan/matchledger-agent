# app/api/analytics.py - SIMPLIFIED VERSION - Accept Any JSON
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
from app.models.user import User
import logging
from datetime import datetime

router = APIRouter()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create HTTPBearer for debugging
security = HTTPBearer(auto_error=False)

# Flexible model that accepts any JSON
class FlexibleAnalyticsEvent(BaseModel):
    class Config:
        extra = "allow"  # Allow any additional fields

    # No required fields - accept anything
    pass

# DEBUG: Optional auth dependency
async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """Get current user but don't raise exception if not authenticated"""
    try:
        logger.info(f"🔍 Analytics Debug - Request URL: {request.url}")
        logger.info(f"🔍 Analytics Debug - Authorization header: {credentials is not None}")

        if credentials:
            logger.info(f"🔍 Analytics Debug - Token scheme: {credentials.scheme}")
            logger.info(f"🔍 Analytics Debug - Token preview: {credentials.credentials[:20] if credentials.credentials else 'None'}...")

            # Try to authenticate using the existing dependencies
            from app.auth.dependencies import get_current_user
            from app.db import get_db

            db_gen = get_db()
            db = next(db_gen)
            try:
                user = await get_current_user(credentials.credentials, db)
                logger.info(f"✅ Analytics Debug - Successfully authenticated: {user.email}")
                return user
            except Exception as auth_error:
                logger.warning(f"⚠️ Analytics Debug - Auth failed: {str(auth_error)}")
                return None
            finally:
                db.close()
        else:
            logger.info("🔍 Analytics Debug - No authorization header provided")
            return None

    except Exception as e:
        logger.error(f"❌ Analytics Debug - Unexpected error: {str(e)}")
        return None

@router.post("/track")
async def track_analytics(
    event: FlexibleAnalyticsEvent,
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Track analytics events - accepts any JSON format"""
    try:
        # Convert Pydantic model to dict to work with any fields
        event_dict = event.dict()

        # Extract event type from various possible fields
        event_type = (
            event_dict.get("event_type") or
            event_dict.get("action") or
            event_dict.get("type") or
            "unknown_event"
        )

        logger.info(f"🎯 Analytics Track - Event: {event_type}")
        logger.info(f"🎯 Analytics Track - User: {user.email if user else 'Anonymous'}")
        logger.info(f"🎯 Analytics Track - Raw Data: {event_dict}")

        # Use provided timestamp or generate new one
        timestamp = event_dict.get("timestamp") or datetime.utcnow().isoformat()

        # Log the complete analytics event
        log_data = {
            "user_id": user.id if user else "anonymous",
            "user_email": user.email if user else "anonymous",
            "event_type": event_type,
            "raw_data": event_dict,  # Store everything the frontend sends
            "timestamp": timestamp,
            "user_agent": request.headers.get("user-agent"),
            "origin": request.headers.get("origin"),
            "referer": request.headers.get("referer"),
            "client_ip": request.client.host if request.client else "unknown"
        }

        logger.info(f"✅ Analytics Event Successfully Tracked: {log_data}")

        return {
            "status": "success",
            "message": "Event tracked successfully",
            "event_type": event_type,
            "tracked_at": timestamp,
            "user_authenticated": user is not None,
            "fields_received": list(event_dict.keys())
        }

    except Exception as e:
        logger.error(f"❌ Analytics Track Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": "Failed to track event but continuing...",
            "error": str(e)
        }

@router.get("/health")
async def analytics_health():
    """Health check for analytics service"""
    logger.info("🏥 Analytics Health Check Called")
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/debug")
async def analytics_debug(request: Request):
    """Debug endpoint to check analytics setup"""
    logger.info(f"🔧 Analytics Debug Called from: {request.client.host}")
    return {
        "message": "Analytics debug endpoint working",
        "timestamp": datetime.utcnow().isoformat(),
        "client_ip": request.client.host,
        "url": str(request.url)
    }
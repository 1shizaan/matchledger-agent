# backend/app/services/usage_tracker.py

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.user import User, PlanType

# Plan limits — customize these if needed
PLAN_LIMITS = {
    PlanType.BETA_PRO: {
        "reconciliations": None,  # Unlimited for now
        "transactions": None,
        "ai_queries": None,
    },
    PlanType.LITE: {
        "reconciliations": 5,
        "transactions": 1000,
        "ai_queries": 10,
    },
    PlanType.PRO: {
        "reconciliations": 50,
        "transactions": 10000,
        "ai_queries": 100,
    },
    PlanType.ENTERPRISE: {
        "reconciliations": None,
        "transactions": None,
        "ai_queries": None,
    }
}

class UsageTracker:
    def __init__(self, db: Session):
        self.db = db

    def _reset_if_needed(self, user: User):
        now = datetime.utcnow()
        if not user.usage_reset_date or user.usage_reset_date + timedelta(days=30) < now:
            user.monthly_reconciliations_count = 0
            user.monthly_transactions_processed = 0
            user.monthly_ai_chat_queries = 0
            user.usage_reset_date = now
            self.db.commit()

    def get_limits(self, user: User):
        return PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS[PlanType.LITE])

    def record_reconciliation_usage(self, user: User, transaction_count: int):
        self._reset_if_needed(user)
        user.monthly_reconciliations_count += 1
        user.monthly_transactions_processed += transaction_count
        self.db.commit()

    def record_ai_chat_usage(self, user: User):
        self._reset_if_needed(user)
        user.monthly_ai_chat_queries += 1
        self.db.commit()

    def check_limits(self, user: User):
        self._reset_if_needed(user)
        limits = self.get_limits(user)

        if limits['reconciliations'] is not None and user.monthly_reconciliations_count >= limits['reconciliations']:
            raise Exception("Monthly reconciliation limit reached.")

        if limits['transactions'] is not None and user.monthly_transactions_processed >= limits['transactions']:
            raise Exception("Monthly transaction limit reached.")

        if limits['ai_queries'] is not None and user.monthly_ai_chat_queries >= limits['ai_queries']:
            raise Exception("Monthly AI query limit reached.")

    def get_usage_stats(self, user: User):
        self._reset_if_needed(user)
        limits = self.get_limits(user)

        return {
            "plan_type": user.plan_type.value,
            "plan_expires_at": user.plan_expires_at.isoformat() if user.plan_expires_at else None,
            "reconciliations": {
                "used": user.monthly_reconciliations_count,
                "limit": limits['reconciliations']
            },
            "transactions": {
                "used": user.monthly_transactions_processed,
                "limit": limits['transactions']
            },
            "ai_queries": {
                "used": user.monthly_ai_chat_queries,
                "limit": limits['ai_queries']
            },
            "beta_feedback_submitted": user.beta_feedback_submitted or 0
        }
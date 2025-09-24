# # reconbot/backend/app/core/config.py

# import os
# from typing import Optional

# class Settings:
#     # Database
#     SQLALCHEMY_DATABASE_URL: str = os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///./reconbot.db")

#     # JWT Settings
#     SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
#     ALGORITHM: str = "HS256"
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

#     # Password Reset Token Settings
#     PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("PASSWORD_RESET_TOKEN_EXPIRE_MINUTES", "60"))

#     # Email Settings
#     SMTP_TLS: bool = os.getenv("SMTP_TLS", "True").lower() == "true"
#     SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
#     SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
#     SMTP_USER: str = os.getenv("SMTP_USER", "")
#     SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
#     EMAILS_FROM_EMAIL: str = os.getenv("EMAILS_FROM_EMAIL", "")
#     EMAILS_FROM_NAME: str = os.getenv("EMAILS_FROM_NAME", "MatchLedger Support")

#     # Frontend URL
#     FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

#     # Google OAuth
#     GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

#     # Environment
#     ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

#     @property
#     def is_production(self) -> bool:
#         return self.ENVIRONMENT.lower() == "production"

# settings = Settings()
import os
from typing import Optional, Literal

class Settings:
    # Database
    SQLALCHEMY_DATABASE_URL: str = os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///./reconbot.db")

    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # AI Provider API Keys - Simplified to OpenAI only
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # Keep for future use
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")  # Keep for future use

    # Provider Configuration - Simplified to OpenAI
    RECONCILIATION_PROVIDER: Literal["openai"] = "openai"
    CHATBOT_PROVIDER: Literal["openai"] = "openai"
    EMBEDDING_PROVIDER: Literal["openai"] = "openai"

    # Feature flags
    ENABLE_PROVIDER_SWITCHING: bool = False  # Disabled for simplicity
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
    COST_OPTIMIZED_MODE: bool = os.getenv("COST_OPTIMIZED_MODE", "true").lower() == "true"

    # Rate limiting and cost controls
    MAX_DAILY_AI_REQUESTS: int = int(os.getenv("MAX_DAILY_AI_REQUESTS", "2000"))  # Increased for cheaper models
    ENABLE_EMBEDDING_CACHE: bool = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

    # Reconciliation specific settings
    MATCH_THRESHOLD_STRONG: float = float(os.getenv("MATCH_THRESHOLD_STRONG", "85.0"))
    MATCH_THRESHOLD_GOOD: float = float(os.getenv("MATCH_THRESHOLD_GOOD", "70.0"))
    MATCH_THRESHOLD_PARTIAL: float = float(os.getenv("MATCH_THRESHOLD_PARTIAL", "45.0"))

    def get_reconciliation_config(self) -> dict:
        """Get reconciliation provider configuration - OpenAI optimized"""
        config = {
            "provider": "openai",
            "config": {
                "primary": "gpt-4o-mini",  # Cost-effective model for reconciliation
                "fallback": "gpt-3.5-turbo",  # Even cheaper fallback
                "max_tokens": 250,  # Slightly increased for better responses
                "temperature": 0.1,  # Low temperature for consistent results
                "embedding_model": "text-embedding-3-small"  # Cost-effective embeddings
            }
        }

        # Use premium models only in demo mode
        if self.DEMO_MODE:
            config["config"]["primary"] = "gpt-4o"  # Still cost-effective but better
            config["config"]["fallback"] = "gpt-4o-mini"

        return config

    def get_chatbot_config(self) -> dict:
        """Get chatbot provider configuration - OpenAI optimized"""
        config = {
            "provider": "openai",
            "config": {
                "primary": "gpt-4o-mini",  # Cost-effective for chat
                "fallback": "gpt-3.5-turbo",  # Backup
                "max_tokens": 800,  # Reasonable for chat responses
                "temperature": 0.3,  # Slightly higher for more natural chat
                "stream": True  # Enable streaming for better UX
            }
        }

        # Enhanced models for demo
        if self.DEMO_MODE:
            config["config"]["primary"] = "gpt-4o"
            config["config"]["max_tokens"] = 1200

        return config

    def get_embedding_config(self) -> dict:
        """Get embedding provider configuration - OpenAI only"""
        return {
            "provider": "openai",
            "model": "text-embedding-3-small",  # Most cost-effective OpenAI embedding
            "dimensions": 1536,
            "batch_size": 100,  # Process in batches for efficiency
            "fallback_enabled": False,  # No fallback needed
            "cache_enabled": self.ENABLE_EMBEDDING_CACHE
        }

    def get_available_providers(self) -> dict:
        """Get list of available providers - OpenAI focused"""
        available = {}

        if self.OPENAI_API_KEY:
            available["openai"] = "✅ OpenAI API configured (Primary)"
        else:
            available["openai"] = "❌ Missing OPENAI_API_KEY (Required)"

        # Show other providers as available for future use
        if self.GROQ_API_KEY:
            available["groq"] = "�� Groq API configured (Available for future)"
        else:
            available["groq"] = "⚪ GROQ_API_KEY not set (Optional)"

        if self.CLAUDE_API_KEY:
            available["claude"] = "💤 Claude API configured (Available for future)"
        else:
            available["claude"] = "⚪ CLAUDE_API_KEY not set (Optional)"

        return available

    def validate_configuration(self) -> tuple[bool, list]:
        """Validate configuration - OpenAI only"""
        issues = []

        # Only check OpenAI since it's our primary provider
        if not self.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is required for all operations")

        # Warn if other keys are missing but don't fail
        if not self.GROQ_API_KEY:
            issues.append("INFO: GROQ_API_KEY not set (future use)")

        if not self.CLAUDE_API_KEY:
            issues.append("INFO: CLAUDE_API_KEY not set (future use)")

        # Only OPENAI_API_KEY is critical
        critical_issues = [issue for issue in issues if "required" in issue.lower()]

        return len(critical_issues) == 0, issues

    def get_cost_estimate(self, transactions: int = 100) -> dict:
        """Enhanced cost estimation for OpenAI models"""

        # Updated OpenAI pricing (as of 2024/2025)
        costs = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-small": 0.00002  # per 1K tokens
        }

        recon_config = self.get_reconciliation_config()
        chat_config = self.get_chatbot_config()
        embed_config = self.get_embedding_config()

        # Token estimates for different operations
        recon_tokens_per_transaction = 120  # Optimized prompts for gpt-4o-mini
        chat_tokens_per_interaction = 400   # Average chat exchange
        embedding_tokens_per_item = 15      # Average text length for embeddings

        # Calculate reconciliation costs (input + output)
        recon_model = recon_config["config"]["primary"]
        recon_input_cost = (transactions * recon_tokens_per_transaction / 1000) * costs[recon_model]["input"]
        recon_output_cost = (transactions * 50 / 1000) * costs[recon_model]["output"]  # ~50 tokens output
        recon_total = recon_input_cost + recon_output_cost

        # Calculate chat costs (assume 20% of transactions generate chat)
        chat_interactions = transactions * 0.2
        chat_model = chat_config["config"]["primary"]
        chat_input_cost = (chat_interactions * chat_tokens_per_interaction / 1000) * costs[chat_model]["input"]
        chat_output_cost = (chat_interactions * 200 / 1000) * costs[chat_model]["output"]  # ~200 tokens response
        chat_total = chat_input_cost + chat_output_cost

        # Calculate embedding costs (2 embeddings per transaction: ledger + bank)
        embed_cost = (transactions * 2 * embedding_tokens_per_item / 1000) * costs["text-embedding-3-small"]

        total_cost = recon_total + chat_total + embed_cost

        # Calculate savings vs premium models
        premium_cost = (transactions * 0.03)  # Rough estimate for GPT-4
        savings = max(0, premium_cost - total_cost)

        return {
            "reconciliation": {
                "cost": round(recon_total, 4),
                "model": recon_model,
                "transactions": transactions
            },
            "chatbot": {
                "cost": round(chat_total, 4),
                "model": chat_model,
                "interactions": int(chat_interactions)
            },
            "embeddings": {
                "cost": round(embed_cost, 4),
                "model": "text-embedding-3-small",
                "items": transactions * 2
            },
            "total_cost": round(total_cost, 4),
            "cost_per_transaction": round(total_cost / transactions, 6),
            "monthly_estimate_1000_txns": round(total_cost * 10, 2),
            "provider": "OpenAI",
            "mode": "Cost-Optimized" if self.COST_OPTIMIZED_MODE else "Standard",
            "savings_vs_premium": round(savings, 4),
            "savings_percentage": round((savings / premium_cost * 100), 1) if premium_cost > 0 else 0
        }

    def get_api_limits(self) -> dict:
        """Get API rate limits and recommendations"""
        return {
            "gpt-4o-mini": {
                "requests_per_minute": 3000,
                "tokens_per_minute": 200000,
                "requests_per_day": 10000,
                "recommended_batch_size": 50
            },
            "text-embedding-3-small": {
                "requests_per_minute": 3000,
                "tokens_per_minute": 1000000,
                "requests_per_day": 10000,
                "recommended_batch_size": 100
            },
            "general": {
                "concurrent_requests": 20,
                "retry_attempts": 3,
                "timeout_seconds": 30
            }
        }

# Create settings instance
settings = Settings()
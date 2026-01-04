"""
Rate Limiting for CosArt API
Implements tier-based rate limiting with Redis backend
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import redis
from config.settings import settings

# Initialize Redis connection
redis_client = redis.Redis(
    host=settings.REDIS_URL.replace("redis://", "").split(":")[0] if ":" in settings.REDIS_URL.replace("redis://", "") else "localhost",
    port=int(settings.REDIS_URL.replace("redis://", "").split(":")[1]) if ":" in settings.REDIS_URL.replace("redis://", "") else 6379,
    decode_responses=True
)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,
    strategy="fixed-window"  # or "moving-window"
)

# Rate limit configurations by tier
RATE_LIMITS = {
    "free": "20/minute",      # 20 requests per minute
    "pro": "100/minute",      # 100 requests per minute
    "studio": "500/minute"    # 500 requests per minute
}

# Special limits for expensive operations
GENERATION_LIMITS = {
    "free": "5/hour",         # 5 generations per hour
    "pro": "50/hour",         # 50 generations per hour
    "studio": "200/hour"      # 200 generations per hour
}

def get_user_tier(request: Request) -> str:
    """Extract user tier from JWT token or default to free"""
    try:
        # This will be implemented when we add auth integration
        # For now, default to free tier
        return "free"
    except:
        return "free"

def get_rate_limit_for_tier(tier: str, endpoint_type: str = "general") -> str:
    """Get appropriate rate limit for user tier and endpoint type"""
    if endpoint_type == "generation":
        return GENERATION_LIMITS.get(tier, GENERATION_LIMITS["free"])
    return RATE_LIMITS.get(tier, RATE_LIMITS["free"])

# Custom rate limit exceeded handler
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.retry_after
        }
    )

# Dynamic rate limiter based on user tier
def dynamic_limiter(request: Request) -> str:
    """Dynamic rate limiting based on user tier"""
    tier = get_user_tier(request)
    return get_rate_limit_for_tier(tier)

# Generation-specific rate limiter
def generation_limiter(request: Request) -> str:
    """Rate limiting specifically for generation endpoints"""
    tier = get_user_tier(request)
    return get_rate_limit_for_tier(tier, "generation")

# Middleware setup
def setup_rate_limiting(app):
    """Add rate limiting middleware to FastAPI app"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    return app
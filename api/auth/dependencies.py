"""
CosArt - Authentication Dependencies
api/auth/dependencies.py
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from .service import verify_token, get_tier_limits
from .schemas import TokenData
from api.database.session import get_db
from api.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    email: str = payload.get("sub")
    user_id: str = payload.get("user_id")

    if email is None and user_id is None:
        raise credentials_exception

    # Get user from database
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
    else:
        user = db.query(User).filter(User.email == email).first()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_tier(min_tier: str):
    """Dependency to require minimum user tier"""
    tier_hierarchy = {'free': 0, 'pro': 1, 'studio': 2}

    def tier_checker(current_user: User = Depends(get_current_active_user)):
        user_tier_level = tier_hierarchy.get(current_user.tier, 0)
        required_level = tier_hierarchy.get(min_tier, 0)

        if user_tier_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {min_tier} tier or higher. Current tier: {current_user.tier}"
            )
        return current_user

    return tier_checker


def require_admin(current_user: User = Depends(get_current_active_user)):
    """Require admin privileges (for future use)"""
    # For now, no admin users - this is for future expansion
    if current_user.tier != "studio":  # Studio users get admin access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def check_daily_quota(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Check and return remaining daily quota"""
    from api.models.usage import DailyQuota

    today = datetime.utcnow().date()
    quota = db.query(DailyQuota).filter(
        DailyQuota.user_id == current_user.id,
        DailyQuota.date == today
    ).first()

    if not quota:
        quota = DailyQuota(
            user_id=current_user.id,
            date=today,
            generations_used=0
        )
        db.add(quota)
        db.commit()

    tier_limits = get_tier_limits(current_user.tier)
    max_daily = tier_limits['daily_generations']

    if max_daily == -1:  # Unlimited
        return {"remaining": -1, "used": quota.generations_used}

    remaining = max(0, max_daily - quota.generations_used)
    return {
        "remaining": remaining,
        "used": quota.generations_used,
        "limit": max_daily
    }


def validate_generation_request(
    resolution: int,
    current_user: User = Depends(get_current_active_user)
):
    """Validate generation request against user tier limits"""
    tier_limits = get_tier_limits(current_user.tier)

    if resolution > tier_limits['max_resolution']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Resolution {resolution}x{resolution} requires {tier_limits['max_resolution']}px max. Upgrade to higher tier."
        )

    return current_user
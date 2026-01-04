"""
CosArt - Authentication Service
api/auth/service.py
"""
from datetime import datetime, timedelta
from typing import Optional
import uuid
from jose import JWTError, jwt
from passlib.context import CryptContext
from config.settings import Settings

settings = Settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.SECRET_KEY or "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def create_verification_token(email: str) -> str:
    """Create email verification token"""
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {"sub": email, "type": "email_verification", "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_password_reset_token(email: str) -> str:
    """Create password reset token"""
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode = {"sub": email, "type": "password_reset", "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_email_token(token: str) -> Optional[str]:
    """Verify email verification token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "email_verification":
            return None
        return payload.get("sub")
    except JWTError:
        return None


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "password_reset":
            return None
        return payload.get("sub")
    except JWTError:
        return None


# Tier limits (as defined in PRD)
TIER_LIMITS = {
    'free': {
        'daily_generations': 20,
        'max_resolution': 512,
        'batch_size': 1,
        'universe_size': 0,
        'api_access': False
    },
    'pro': {
        'daily_generations': -1,  # Unlimited
        'max_resolution': 2048,
        'batch_size': 16,
        'universe_size': 100,
        'api_access': True
    },
    'studio': {
        'daily_generations': -1,
        'max_resolution': 4096,
        'batch_size': 100,
        'universe_size': 500,
        'api_access': True,
        'custom_training': True
    }
}


def get_tier_limits(tier: str) -> dict:
    """Get limits for a user tier"""
    return TIER_LIMITS.get(tier, TIER_LIMITS['free'])
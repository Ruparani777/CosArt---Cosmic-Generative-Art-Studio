"""
CosArt - Authentication Routes
api/auth/routes.py
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import uuid

from .schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    PasswordResetRequest, PasswordReset, EmailVerification
)
from .service import (
    get_password_hash, verify_password, create_access_token,
    create_verification_token, create_password_reset_token,
    verify_email_token, verify_password_reset_token
)
from .dependencies import get_current_active_user
from api.database.session import get_db
from api.models.user import User
from api.repositories.user_repository import UserRepository

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    user_repo = UserRepository(db)

    # Check if user already exists
    existing_user = await user_repo.get_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check username uniqueness if provided
    if user_data.username:
        existing_username = await user_repo.get_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

    # Create user
    hashed_password = get_password_hash(user_data.password)
    verification_token = create_verification_token(user_data.email)

    user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        email_verified=False,
        tier="free",
        is_active=True
    )

    created_user = await user_repo.create(user)

    # TODO: Send verification email
    # background_tasks.add_task(send_verification_email, user.email, verification_token)

    return UserResponse.from_orm(created_user)


@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    user_repo = UserRepository(db)

    user = await user_repo.get_by_email(form_data.username)  # OAuth2 uses username field for email
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is deactivated"
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60,  # 30 minutes in seconds
        user=UserResponse.from_orm(user)
    )


@router.post("/verify-email")
async def verify_email(
    verification: EmailVerification,
    db: Session = Depends(get_db)
):
    """Verify user email with token"""
    user_repo = UserRepository(db)

    email = verify_email_token(verification.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )

    user = await user_repo.get_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.email_verified:
        return {"message": "Email already verified"}

    await user_repo.verify_email(user.id)
    return {"message": "Email verified successfully"}


@router.post("/request-password-reset")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Request password reset"""
    user_repo = UserRepository(db)

    user = await user_repo.get_by_email(reset_request.email)
    if user:
        reset_token = create_password_reset_token(user.email)
        # TODO: Send password reset email
        # background_tasks.add_task(send_password_reset_email, user.email, reset_token)

    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """Reset password with token"""
    user_repo = UserRepository(db)

    email = verify_password_reset_token(reset_data.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    user = await user_repo.get_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    hashed_password = get_password_hash(reset_data.new_password)
    await user_repo.update_password(user.id, hashed_password)

    return {"message": "Password reset successfully"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: dict,  # TODO: Create UserUpdate schema
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    user_repo = UserRepository(db)

    # TODO: Implement profile updates (username, etc.)
    # For now, just return current user
    return UserResponse.from_orm(current_user)
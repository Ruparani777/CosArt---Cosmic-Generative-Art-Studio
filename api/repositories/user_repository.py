"""
CosArt - User Repository
api/repositories/user_repository.py
"""
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import Optional, List
from uuid import UUID

from api.models.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    async def create(self, user: User) -> User:
        """Create a new user"""
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()

    async def update_password(self, user_id: UUID, hashed_password: str):
        """Update user password"""
        user = await self.get_by_id(user_id)
        if user:
            user.hashed_password = hashed_password
            await self.db.commit()

    async def verify_email(self, user_id: UUID):
        """Mark user email as verified"""
        user = await self.get_by_id(user_id)
        if user:
            user.email_verified = True
            await self.db.commit()

    async def update_tier(self, user_id: UUID, tier: str):
        """Update user subscription tier"""
        user = await self.get_by_id(user_id)
        if user:
            user.tier = tier
            await self.db.commit()

    async def update_stripe_customer(self, user_id: UUID, stripe_customer_id: str):
        """Update user's Stripe customer ID"""
        user = await self.get_by_id(user_id)
        if user:
            user.stripe_customer_id = stripe_customer_id
            await self.db.commit()

    async def deactivate_user(self, user_id: UUID):
        """Deactivate user account"""
        user = await self.get_by_id(user_id)
        if user:
            user.is_active = False
            await self.db.commit()

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List all users (admin only)"""
        return self.db.query(User).offset(skip).limit(limit).all()

    async def count(self) -> int:
        """Count total users"""
        return self.db.query(User).count()

    async def search(self, query: str, skip: int = 0, limit: int = 50) -> List[User]:
        """Search users by email or username"""
        search_filter = or_(
            User.email.ilike(f"%{query}%"),
            User.username.ilike(f"%{query}%")
        )
        return self.db.query(User).filter(search_filter).offset(skip).limit(limit).all()
"""
CosArt - Database Session
api/database/session.py
"""
from sqlalchemy.orm import Session
from .connection import SessionLocal


def get_db() -> Session:
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
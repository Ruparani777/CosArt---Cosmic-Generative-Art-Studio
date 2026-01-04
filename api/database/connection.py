"""
CosArt - Database Connection
api/database/connection.py
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import Settings

settings = Settings()

# Database URL
if not settings.DATABASE_URL:
    # Use SQLite for development
    DATABASE_URL = "sqlite:///./cosart.db"
else:
    DATABASE_URL = settings.DATABASE_URL

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables (for testing/reset)"""
    Base.metadata.drop_all(bind=engine)
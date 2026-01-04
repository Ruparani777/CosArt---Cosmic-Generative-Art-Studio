"""
CosArt - Minimal Working Configuration
config/settings.py (SIMPLIFIED VERSION)

This version works without ML dependencies for testing
"""

from typing import Optional


class Settings:
    """
    Simplified settings that work without pydantic-settings
    """
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Model Settings
    MODEL_DIR: str = "models/pretrained"
    CACHE_DIR: str = "cache"
    MAX_MODELS_IN_MEMORY: int = 3

    # Generation Settings
    DEFAULT_RESOLUTION: int = 512
    MAX_RESOLUTION: int = 2048
    MAX_BATCH_SIZE: int = 16
    GENERATION_TIMEOUT: int = 300

    # Universe Mode Settings
    UNIVERSE_SAMPLES: int = 1000  # Reduced for faster testing
    UNIVERSE_CACHE_SIZE: int = 100

    # GPU Settings
    USE_GPU: bool = False  # Default to CPU
    GPU_MEMORY_FRACTION: float = 0.8
    MIXED_PRECISION: bool = False

    # Storage Settings
    GENERATION_CACHE_SIZE: int = 1000
    GENERATION_CACHE_TTL: int = 3600

    # Rate Limiting
    RATE_LIMIT_FREE: int = 20
    RATE_LIMIT_PRO: int = -1

    # Cosmic Features
    ENABLE_UNIVERSE_MODE: bool = True
    ENABLE_PHYSICS_CONTROLS: bool = True

    # Database Settings
    DATABASE_URL: Optional[str] = None  # Will use SQLite for development if not set

    # Authentication Settings
    SECRET_KEY: str = "your-secret-key-change-in-production-please"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Email Settings (for future use)
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    # Debug Settings
    DEBUG: bool = True
    ENABLE_CUSTOM_TRAINING: bool = False

    # External Services
    CLOUD_STORAGE_BUCKET: str = "cosart-models"
    REDIS_URL: str = "redis://localhost:6379"

    # Monitoring
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True

    def __init__(self):
        """Load from environment variables if available"""
        import os

        # Override with environment variables if they exist
        self.USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
        self.API_PORT = int(os.getenv('API_PORT', '8000'))
        self.DEFAULT_RESOLUTION = int(os.getenv('DEFAULT_RESOLUTION', '512'))


# Global settings instance
settings = Settings()
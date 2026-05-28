"""Application configuration."""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Nexlytics API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database - SQLite for dev, PostgreSQL for production
    DATABASE_URL: str = "sqlite:///./nexlytics.db"

    # Auth
    SECRET_KEY: str = Field(default="CHANGE_ME_IN_PRODUCTION_USE_SECRETS_TOKEN_HEX_32")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Storage
    STORAGE_DIR: Path = Path("./storage/uploads")
    MAX_UPLOAD_SIZE_MB: int = 100

    # CORS
    CORS_ORIGINS: list = ["http://localhost:8000", "http://localhost:3000", "*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure storage dir exists
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

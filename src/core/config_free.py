"""Zero-Cost Configuration (Updated for Free Components)."""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings."""
    
    name: str = "TelegramSoccer Zero-Cost"
    version: str = "2.0-free"
    debug: bool = False
    
    class Config:
        env_prefix = ""


class BettingSettings(BaseSettings):
    """Betting strategy settings."""
    
    bankroll_initial: float = Field(default=1000.0, env="BANKROLL_INITIAL")
    target_quote: float = Field(default=1.40, env="TARGET_QUOTE")
    min_probability: float = Field(default=0.68, env="MIN_PROBABILITY")
    max_stake_percentage: float = Field(default=5.0, env="MAX_STAKE_PERCENTAGE")
    stop_loss_percentage: float = Field(default=15.0, env="STOP_LOSS_PERCENTAGE")
    
    class Config:
        env_prefix = ""


class OllamaSettings(BaseSettings):
    """FREE Local LLM settings (Ollama)."""
    
    base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    model: str = Field(default="llama3.2:3b", env="OLLAMA_MODEL")
    # Alternative models: phi4:latest, mistral:7b, qwen2.5:7b
    
    class Config:
        env_prefix = ""


class FreeAPISettings(BaseSettings):
    """FREE API settings."""
    
    # API-Football (100 requests/day FREE)
    api_football_key: Optional[str] = Field(default=None, env="API_FOOTBALL_KEY")
    
    # iSports API (200 requests/day FREE)
    isports_key: Optional[str] = Field(default=None, env="ISPORTS_API_KEY")
    
    class Config:
        env_prefix = ""


class DatabaseSettings(BaseSettings):
    """Database settings (FREE - SQLite)."""
    
    url: str = Field(
        default="sqlite:///data/telegramsoccer.db",
        env="DATABASE_URL"
    )
    
    class Config:
        env_prefix = ""


class TelegramSettings(BaseSettings):
    """Telegram bot settings (FREE)."""
    
    bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    
    class Config:
        env_prefix = ""


class Settings(BaseSettings):
    """Main settings."""
    
    app: AppSettings = AppSettings()
    betting: BettingSettings = BettingSettings()
    ollama: OllamaSettings = OllamaSettings()
    free_apis: FreeAPISettings = FreeAPISettings()
    database: DatabaseSettings = DatabaseSettings()
    telegram: TelegramSettings = TelegramSettings()
    
    # Leagues to track (all free via API-Football)
    leagues: List[str] = [
        "Premier League",
        "Bundesliga",
        "La Liga",
        "Serie A",
        "Ligue 1",
    ]
    
    # Markets to analyze
    markets: List[str] = ["over_1_5", "btts"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

"""Configuration management for TelegramSoccer."""
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration."""
    
    name: str = "TelegramSoccer"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class BettingConfig(BaseSettings):
    """Betting strategy configuration."""
    
    target_quote: float = Field(default=1.40, env="TARGET_QUOTE")
    min_probability: float = Field(default=0.72, env="MIN_PROBABILITY")
    max_stake_percentage: float = Field(default=2.0, env="MAX_STAKE_PERCENTAGE")
    stop_loss_percentage: float = Field(default=15.0, env="STOP_LOSS_PERCENTAGE")
    bankroll_initial: float = Field(default=1000.0, env="BANKROLL_INITIAL")
    
    class Config:
        env_file = ".env"


class LLMConfig(BaseSettings):
    """LLM configuration."""
    
    model: str = Field(default="gpt-4-turbo-preview", env="LLM_MODEL")
    fallback_model: str = "claude-3-sonnet-20240229"
    temperature: float = Field(default=0.2, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    timeout: int = 30
    
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    
    class Config:
        env_file = ".env"


class APIConfig(BaseSettings):
    """External API configuration."""
    
    odds_api_key: str = Field(..., env="ODDS_API_KEY")
    football_data_api_key: str = Field(..., env="FOOTBALL_DATA_API_KEY")
    sportmonks_api_key: str = Field(..., env="SPORTMONKS_API_KEY")
    openweather_api_key: str = Field(..., env="OPENWEATHER_API_KEY")
    
    reddit_client_id: str = Field(default="", env="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="telegramsoccer:v1.0", env="REDDIT_USER_AGENT")
    
    class Config:
        env_file = ".env"


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(..., env="DATABASE_URL")
    test_url: str = Field(default="", env="DATABASE_TEST_URL")
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    
    class Config:
        env_file = ".env"


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""
    
    bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    admin_ids: List[int] = Field(default_factory=list)
    
    class Config:
        env_file = ".env"


class Settings:
    """Master settings container."""
    
    def __init__(self):
        self.app = AppConfig()
        self.betting = BettingConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.telegram = TelegramConfig()
        
        # Load YAML config
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.yaml_config: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.yaml_config = {}
    
    def get_leagues(self, category: str) -> List[str]:
        """Get leagues by category from YAML config."""
        return self.yaml_config.get("leagues", {}).get(category, [])
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """Get market configurations from YAML config."""
        return self.yaml_config.get("betting", {}).get("markets", [])


# Global settings instance
settings = Settings()

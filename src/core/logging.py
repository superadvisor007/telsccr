"""Logging configuration for TelegramSoccer."""
import sys
from pathlib import Path

from loguru import logger

from src.core.config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.app.environment == "development" and "DEBUG" or "INFO",
        colorize=True,
    )
    
    # File handler for all logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "telegramsoccer_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="30 days",
        compression="zip",
    )
    
    # Error file handler
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="100 MB",
        retention="90 days",
        compression="zip",
    )
    
    logger.info("Logging initialized")

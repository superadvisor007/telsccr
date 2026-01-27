"""Main application entry point."""
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bot.telegram_bot import TelegramBot
from src.core.logging import setup_logging
from src.pipeline import Pipeline


async def run_bot_and_pipeline():
    """Run both the bot and daily pipeline."""
    setup_logging()
    logger.info("Starting TelegramSoccer application...")
    
    # Initialize components
    pipeline = Pipeline()
    bot = TelegramBot()
    
    # Run initial pipeline
    logger.info("Running initial pipeline...")
    tips = await pipeline.run_daily_pipeline()
    
    # Get betting statistics
    stats = pipeline.betting_engine.get_statistics()
    
    # Update bot with tips and stats
    await bot.update_bot_data(tips, stats)
    
    # Start bot
    logger.info("Starting Telegram bot...")
    bot.run()


def main():
    """Main entry point."""
    try:
        asyncio.run(run_bot_and_pipeline())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

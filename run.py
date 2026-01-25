#!/usr/bin/env python3
"""
Swiss Soccer Tips Bot - CLI Runner

Usage:
    python run.py bot       # Run Telegram bot
    python run.py webhook   # Run webhook server
    python run.py daily     # Run daily tasks (cron)
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import SoccerTipsBot


async def run_bot():
    """Run the Telegram bot."""
    bot = SoccerTipsBot()
    await bot.initialize()
    
    try:
        await bot.run_bot()
    except KeyboardInterrupt:
        print("\nShutting down bot...")
    finally:
        await bot.shutdown()


async def run_webhook():
    """Run the webhook server."""
    from webhook_server import main as webhook_main
    await webhook_main()


async def run_daily_tasks():
    """Run daily tasks."""
    bot = SoccerTipsBot()
    await bot.run_daily_tasks()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "bot":
        asyncio.run(run_bot())
    elif command == "webhook":
        asyncio.run(run_webhook())
    elif command == "daily":
        asyncio.run(run_daily_tasks())
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

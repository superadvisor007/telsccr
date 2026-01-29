#!/usr/bin/env python3
"""Test Telegram Bot Connection."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from telegram import Bot
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def test_telegram_bot():
    """Test Telegram bot by sending a message."""
    print("\n🤖 Testing Telegram Bot Connection...")
    print("="*60)
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token or bot_token == "your_telegram_bot_token_here":
        print("❌ ERROR: No valid Telegram bot token found in .env")
        print("Please set TELEGRAM_BOT_TOKEN in .env file")
        return False
    
    try:
        # Initialize bot
        bot = Bot(token=bot_token)
        
        # Get bot info
        print(f"\n✓ Bot Token: {bot_token[:20]}...{bot_token[-10:]}")
        
        bot_info = await bot.get_me()
        print(f"✓ Bot Username: @{bot_info.username}")
        print(f"✓ Bot Name: {bot_info.first_name}")
        print(f"✓ Bot ID: {bot_info.id}")
        
        # Get updates to find chat_id
        print("\n📬 Fetching recent messages to find chat ID...")
        updates = await bot.get_updates()
        
        if updates:
            latest_update = updates[-1]
            chat_id = latest_update.message.chat.id if latest_update.message else None
            
            if chat_id:
                print(f"✓ Found Chat ID: {chat_id}")
                
                # Send test message
                test_message = f"""
🎉 *TelegramSoccer Bot - System Test*

✅ Bot ist erfolgreich verbunden!

*Test-Details:*
• Bot: @{bot_info.username}
• Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• System: Zero-Cost Edition
• Status: OPERATIONAL

*Nächste Schritte:*
1. ✓ Bot-Verbindung hergestellt
2. ⏳ API-Schlüssel konfigurieren
3. ⏳ Pipeline starten

Sobald die Pipeline läuft, erhältst du hier tägliche Wett-Tipps! ⚽💰
"""
                
                message = await bot.send_message(
                    chat_id=chat_id,
                    text=test_message,
                    parse_mode='Markdown'
                )
                
                print(f"\n✅ TEST MESSAGE SENT!")
                print(f"   Message ID: {message.message_id}")
                print(f"   Chat ID: {chat_id}")
                print(f"\n📱 Check your Telegram app for the test message!")
                
                return True
            else:
                print("\n⚠️  No chat ID found. Please:")
                print(f"   1. Open Telegram and search for @{bot_info.username}")
                print(f"   2. Send /start to the bot")
                print(f"   3. Run this test again")
                return False
        else:
            print("\n⚠️  No messages found. Please:")
            print(f"   1. Open Telegram and search for @{bot_info.username}")
            print(f"   2. Send /start to the bot")
            print(f"   3. Run this test again")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if bot token is correct")
        print(f"  2. Make sure bot is not blocked")
        print(f"  3. Send /start to bot first")
        return False
    
    print("="*60)


async def main():
    """Run test."""
    success = await test_telegram_bot()
    
    if success:
        print("\n🎉 TELEGRAM BOT TEST PASSED!")
        print("\n✅ Bot is ready to send betting tips")
        return 0
    else:
        print("\n⚠️  TELEGRAM BOT TEST INCOMPLETE")
        print("\nPlease follow the instructions above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

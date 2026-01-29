#!/usr/bin/env python3
"""
Test Telegram Integration
Run: python3 test_telegram.py
"""

import os
import requests
import sys
from datetime import datetime

def test_telegram():
    """Test Telegram bot connection"""
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print("="*60)
    print("🔍 TELEGRAM INTEGRATION TEST")
    print("="*60)
    print(f"\nBot Token: {'✅ SET' if bot_token else '❌ MISSING'}")
    print(f"Chat ID: {'✅ SET' if chat_id else '❌ MISSING'}")
    
    if not bot_token or not chat_id:
        print("\n❌ Missing credentials!\n")
        print("📝 How to fix:")
        print("1. Talk to @BotFather on Telegram:")
        print("   - Send: /newbot")
        print("   - Follow instructions")
        print("   - Copy bot token\n")
        print("2. Get your Chat ID:")
        print("   - Send any message to @userinfobot")
        print("   - Copy your ID\n")
        print("3. Set environment variables:")
        print("   export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("   export TELEGRAM_CHAT_ID='your_chat_id_here'\n")
        print("4. Or add to GitHub Secrets:")
        print("   https://github.com/superadvisor007/telegramsoccer/settings/secrets/actions\n")
        sys.exit(1)
    
    print("\n📤 Sending test message...\n")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message = f"""
🧪 *TELEGRAM TEST*

This is a test message from the telegramsoccer system.

✅ If you see this, Telegram integration is working!

📊 System Status:
• Bot Token: Configured
• Chat ID: Configured
• Connection: Active

_Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}_
"""
    
    data = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ SUCCESS! Check your Telegram for the test message.\n")
            result = response.json()
            print(f"Message ID: {result['result']['message_id']}")
            print(f"Chat: {result['result']['chat']['first_name']}")
            return True
        else:
            print(f"❌ FAILED! Status: {response.status_code}\n")
            print(f"Response: {response.json()}\n")
            
            if response.status_code == 401:
                print("⚠️  Issue: Invalid bot token")
            elif response.status_code == 400:
                print("⚠️  Issue: Invalid chat ID or message format")
            
            return False
            
    except Exception as e:
        print(f"⚠️ ERROR: {e}\n")
        return False

if __name__ == "__main__":
    success = test_telegram()
    sys.exit(0 if success else 1)

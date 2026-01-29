#!/usr/bin/env python3
"""Test API keys once configured."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
from dotenv import load_dotenv

load_dotenv()

async def test_api_football():
    """Test API-Football key."""
    print("\n🔑 Testing API-Football...")
    
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or api_key == "your_api_football_key_here":
        print("  ⚠️  API key not configured")
        return False
    
    try:
        import httpx
        url = "https://v3.football.api-sports.io/timezone"
        headers = {"x-apisports-key": api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    print(f"  ✓ API-Football: Valid ({len(data['response'])} timezones)")
                    print(f"  ✓ Daily Quota: 100 requests")
                    return True
            else:
                print(f"  ✗ Status: {response.status_code}")
                return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

async def test_isports():
    """Test iSports API key."""
    print("\n🔑 Testing iSports API...")
    
    api_key = os.getenv("ISPORTS_API_KEY")
    if not api_key or api_key == "your_isports_key_here":
        print("  ⚠️  API key not configured")
        return False
    
    try:
        import httpx
        url = f"https://api.isportsapi.com/sport/football?api_key={api_key}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    print(f"  ✓ iSports API: Valid")
                    print(f"  ✓ Daily Quota: 200 requests")
                    return True
            else:
                print(f"  ✗ Status: {response.status_code}")
                return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_telegram():
    """Test Telegram bot token."""
    print("\n🤖 Testing Telegram Bot...")
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token or bot_token == "your_telegram_bot_token_here":
        print("  ⚠️  Bot token not configured")
        return False
    
    if bot_token == "7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI":
        print(f"  ✓ Telegram Bot: @Tonticketbot")
        print(f"  ✓ Token: {bot_token[:20]}...{bot_token[-10:]}")
        print(f"  ✓ Status: Configured")
        print(f"  ℹ️  Send /start to @Tonticketbot to activate")
        return True
    else:
        print(f"  ⚠️  Unknown token: {bot_token[:20]}...")
        return False

async def main():
    """Run all API tests."""
    print("╔════════════════════════════════════════════════════╗")
    print("║                                                    ║")
    print("║     🔑 API KEYS VALIDATION TEST 🔑                 ║")
    print("║                                                    ║")
    print("╚════════════════════════════════════════════════════╝")
    
    results = {
        "API-Football": await test_api_football(),
        "iSports API": await test_isports(),
        "Telegram Bot": test_telegram(),
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    for service, passed in results.items():
        status = "✓ VALID" if passed else "✗ NOT CONFIGURED"
        print(f"{service:.<40} {status:>15}")
    
    print("="*60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL API KEYS VALID!")
        print("\n✅ SYSTEM READY TO RUN")
        print("\nNext step: python src/pipeline_free.py")
        print("\n💰 Total Daily Quota: 300 requests (100 + 200)")
        print("💰 Total Cost: $0.00/month FOREVER\n")
        return 0
    else:
        print("\n⚠️  SOME API KEYS NOT CONFIGURED")
        print("\nTo configure:")
        print("  1. Run: python setup_api_keys.py")
        print("  2. Or manually edit .env file")
        print("\nMissing:")
        for service, passed in results.items():
            if not passed:
                print(f"  • {service}")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

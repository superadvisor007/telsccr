#!/usr/bin/env python3
"""
Automated API Key Setup Assistant
Helps obtain and validate free API keys for the zero-cost system.
"""
import os
import sys
import webbrowser
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def open_signup_page(service, url):
    """Open signup page in browser."""
    print(f"🌐 Opening {service} signup page...")
    try:
        webbrowser.open(url)
        print(f"✓ Browser opened for {service}")
        return True
    except Exception as e:
        print(f"✗ Could not open browser: {e}")
        print(f"Please manually visit: {url}")
        return False

def get_api_key_input(service):
    """Get API key from user input."""
    print(f"\n📝 Enter your {service} API key:")
    print(f"   (Or press Enter to skip)")
    key = input(f"{service} Key: ").strip()
    return key if key else None

def update_env_file(keys):
    """Update .env file with API keys."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("✗ .env file not found!")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update keys
    updated_lines = []
    for line in lines:
        if 'API_FOOTBALL_KEY=' in line and keys.get('api_football'):
            updated_lines.append(f"API_FOOTBALL_KEY={keys['api_football']}\n")
        elif 'ISPORTS_API_KEY=' in line and keys.get('isports'):
            updated_lines.append(f"ISPORTS_API_KEY={keys['isports']}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("✓ .env file updated!")
    return True

def main():
    """Main setup assistant."""
    print_header("🔑 API KEY SETUP ASSISTANT - Zero-Cost System")
    
    print("This assistant will help you obtain FREE API keys.\n")
    print("💰 Total Cost: $0.00/month FOREVER\n")
    print("⏱️  Estimated Time: 10 minutes\n")
    
    input("Press Enter to start...\n")
    
    keys = {}
    
    # API-Football
    print_header("1️⃣  API-FOOTBALL (100 requests/day FREE)")
    print("Service: Football data (fixtures, odds, team stats)")
    print("URL: https://www.api-football.com/\n")
    print("Steps:")
    print("  1. Click 'Sign Up' (top right)")
    print("  2. Create account with your email")
    print("  3. Verify email")
    print("  4. Login → Dashboard → Copy API Key")
    
    choice = input("\n📌 Open signup page in browser? (y/n): ").strip().lower()
    if choice == 'y':
        open_signup_page("API-Football", "https://www.api-football.com/")
    
    api_football_key = get_api_key_input("API-Football")
    if api_football_key:
        keys['api_football'] = api_football_key
        print("✓ API-Football key saved")
    
    # iSports API
    print_header("2️⃣  iSPORTS API (200 requests/day FREE)")
    print("Service: Sports odds and betting data")
    print("URL: https://www.isportsapi.com/\n")
    print("Steps:")
    print("  1. Click 'Register' or 'Sign Up'")
    print("  2. Create free account")
    print("  3. Verify email")
    print("  4. Dashboard → API Key → Copy")
    
    choice = input("\n📌 Open signup page in browser? (y/n): ").strip().lower()
    if choice == 'y':
        open_signup_page("iSports API", "https://www.isportsapi.com/")
    
    isports_key = get_api_key_input("iSports API")
    if isports_key:
        keys['isports'] = isports_key
        print("✓ iSports API key saved")
    
    # Telegram Bot
    print_header("3️⃣  TELEGRAM BOT (Already configured!)")
    print("✓ Bot Token: 7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI")
    print("✓ Bot: @Tonticketbot")
    print("✓ Status: Ready")
    print("\n📱 To activate:")
    print("  1. Open Telegram")
    print("  2. Search: @Tonticketbot")
    print("  3. Send: /start")
    
    # Update .env
    if keys:
        print_header("💾 SAVING CONFIGURATION")
        if update_env_file(keys):
            print("✅ API keys saved to .env file!")
        else:
            print("⚠️  Could not update .env file")
            print("\nManually add these to .env:")
            for service, key in keys.items():
                env_var = "API_FOOTBALL_KEY" if service == "api_football" else "ISPORTS_API_KEY"
                print(f"  {env_var}={key}")
    
    # Summary
    print_header("📊 SETUP SUMMARY")
    print(f"API-Football: {'✓ Configured' if 'api_football' in keys else '⏳ Pending'}")
    print(f"iSports API: {'✓ Configured' if 'isports' in keys else '⏳ Pending'}")
    print(f"Telegram Bot: ✓ Configured (@Tonticketbot)")
    
    print("\n🚀 NEXT STEPS:")
    if 'api_football' in keys and 'isports' in keys:
        print("  1. Test API keys: python test_api_keys.py")
        print("  2. Test Telegram: python test_telegram_bot.py")
        print("  3. Run pipeline: python src/pipeline_free.py")
    else:
        print("  1. Complete API key signups (see instructions above)")
        print("  2. Run this script again: python setup_api_keys.py")
        print("  3. Test everything: python test_system.py")
    
    print("\n💰 Total Cost: $0.00/month FOREVER")
    print("\n✅ Setup assistant complete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

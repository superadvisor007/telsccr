#!/usr/bin/env python3
"""Demo Mode - Test system without real API keys."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent))

from telegram import Bot
from dotenv import load_dotenv
import os

load_dotenv()

def generate_mock_fixtures():
    teams = [
        ("Arsenal", "Manchester United"),
        ("Liverpool", "Chelsea"),
        ("Bayern Munich", "Borussia Dortmund"),
        ("Real Madrid", "Barcelona"),
        ("PSG", "Marseille"),
    ]
    
    fixtures = []
    for i, (home, away) in enumerate(teams):
        fixtures.append({
            "home_team": home,
            "away_team": away,
            "home_avg_goals": round(random.uniform(1.5, 2.5), 1),
            "away_avg_goals": round(random.uniform(1.2, 2.2), 1),
            "h2h_over_1_5": round(random.uniform(0.65, 0.90), 2),
            "odds": round(random.uniform(1.18, 1.35), 2),
        })
    return fixtures

async def send_telegram_demo(tips):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token or bot_token == "your_telegram_bot_token_here":
        print("⚠️  Telegram not configured")
        return False
    
    try:
        bot = Bot(token=bot_token)
        updates = await bot.get_updates()
        if not updates:
            print("⚠️  Send /start to @Tonticketbot first")
            return False
        
        chat_id = updates[-1].message.chat.id
        
        message = f"""🎯 *DEMO MODE*

📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}

⚠️ Mock data - Real tips need API keys

*Top 3 Tips:*

"""
        for i, tip in enumerate(tips[:3], 1):
            message += f"*{i}. {tip['home_team']} vs {tip['away_team']}*\n"
            message += f"• Odds: {tip['odds']}\n"
            message += f"• Probability: {tip['probability']:.0%}\n\n"
        
        message += "\nRun: python src/pipeline_free.py"
        
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        print(f"✓ Demo sent to Telegram")
        return True
    except Exception as e:
        print(f"✗ Telegram error: {e}")
        return False

async def main():
    print("\n🎮 DEMO MODE - Testing without API keys\n")
    
    fixtures = generate_mock_fixtures()
    print(f"✓ Generated {len(fixtures)} mock matches")
    
    tips = []
    for f in fixtures:
        avg = f["home_avg_goals"] + f["away_avg_goals"]
        prob = (avg / 4.0) * 0.6 + f["h2h_over_1_5"] * 0.4
        tips.append({**f, "probability": min(0.95, max(0.70, prob))})
    
    tips.sort(key=lambda x: x["probability"], reverse=True)
    
    print("\n📊 Top 3 Demo Tips:")
    for i, tip in enumerate(tips[:3], 1):
        print(f"{i}. {tip['home_team']} vs {tip['away_team']}")
        print(f"   Odds: {tip['odds']} | Probability: {tip['probability']:.0%}\n")
    
    await send_telegram_demo(tips)
    
    print("✅ Demo complete!\n")
    print("Get real tips:")
    print("1. Configure API keys: bash quick_setup.sh")
    print("2. Run pipeline: python src/pipeline_free.py\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

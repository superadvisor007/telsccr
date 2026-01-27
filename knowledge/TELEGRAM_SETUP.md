# Telegram Configuration & Troubleshooting

## 🔐 Setting Up Telegram Bot

### 1. Get Your Bot Token
```bash
# Talk to @BotFather on Telegram
# Send: /newbot
# Follow instructions
# You'll receive: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz

export TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

### 2. Get Your Chat ID
```bash
# Method 1: Use @userinfobot
# Send any message to @userinfobot
# It will reply with your Chat ID

# Method 2: Send message to your bot and check updates
curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates

# Look for "chat":{"id":123456789}
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

### 3. Test Connection
```bash
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -H "Content-Type: application/json" \
  -d "{\"chat_id\":\"${TELEGRAM_CHAT_ID}\",\"text\":\"Test message\"}"
```

---

## 🔧 GitHub Secrets Configuration

### Add Secrets to Repository
```bash
# Go to: https://github.com/superadvisor007/telegramsoccer/settings/secrets/actions

# Add two secrets:
1. TELEGRAM_BOT_TOKEN = your_bot_token
2. TELEGRAM_CHAT_ID = your_chat_id
```

### Check Secrets in Codespace
```bash
# Secrets should be automatically available
echo $TELEGRAM_BOT_TOKEN  # Should show token
echo $TELEGRAM_CHAT_ID    # Should show chat ID
```

---

## 🐛 Debugging Telegram Issues

### Check if credentials exist
```bash
python3 -c "import os; print('Bot Token:', os.getenv('TELEGRAM_BOT_TOKEN', 'MISSING')); print('Chat ID:', os.getenv('TELEGRAM_CHAT_ID', 'MISSING'))"
```

### Test Send Function
```python
import os
import requests

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

if not bot_token or not chat_id:
    print("❌ Credentials missing!")
else:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': '🧪 Test from telegramsoccer system',
        'parse_mode': 'Markdown'
    }
    
    response = requests.post(url, json=data, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
```

### Common Errors

**Error: 401 Unauthorized**
```
Cause: Invalid bot token
Fix: Check TELEGRAM_BOT_TOKEN spelling
```

**Error: 400 Bad Request (chat not found)**
```
Cause: Invalid chat ID
Fix: Check TELEGRAM_CHAT_ID is correct
```

**Error: No response**
```
Cause: Network issues or missing credentials
Fix: Check internet connection and environment variables
```

---

## 📝 Telegram Message Format

### Basic Message
```python
message = "⚽ Match: Bayern vs Dortmund\n" \
          "🎯 Bet: Over 1.5 Goals\n" \
          "📊 Probability: 82%\n" \
          "💰 Odds: 1.25"
```

### Markdown Formatting
```python
message = "*⚽ Bayern Munich vs Borussia Dortmund*\n\n" \
          "🎯 *Bet*: Over 1.5 Goals\n" \
          "📊 *Probability*: 82%\n" \
          "💰 *Odds*: 1.25\n" \
          "✅ *Edge*: +8.5%\n\n" \
          "_Kickoff: 15:30 CET_"
```

### Multiple Matches
```python
message = "🔮 *TOMORROW'S PREDICTIONS*\n" \
          "━━━━━━━━━━━━━━━━━━━━\n\n" \
          "*Match 1*\n" \
          "⚽ Bayern vs Dortmund\n" \
          "🎯 Over 1.5 @ 1.25\n\n" \
          "*Match 2*\n" \
          "⚽ Leipzig vs Leverkusen\n" \
          "🎯 BTTS Yes @ 1.80\n\n" \
          "📈 *Combined Odds*: 2.25"
```

---

## ✅ Recommended Message Template

```python
def format_telegram_message(match, prediction):
    """Format betting recommendation for Telegram"""
    
    # Emojis for different bet types
    bet_emoji = {
        'over_1_5': '⚡',
        'over_2_5': '🔥',
        'btts': '⚔️',
        'under_2_5': '🛡️'
    }
    
    # Status based on edge
    if prediction['edge'] > 15:
        status = '🟢 STRONG VALUE'
    elif prediction['edge'] > 8:
        status = '🟡 GOOD VALUE'
    else:
        status = '⚪ FAIR VALUE'
    
    message = f"""
*{bet_emoji.get(prediction['market'], '⚽')} {match['home']} vs {match['away']}*

🎯 *Market*: {prediction['market_name']}
📊 *Probability*: {prediction['probability']:.1%}
💰 *Odds*: {prediction['odds']:.2f}
✅ *Edge*: +{prediction['edge']:.1f}%

{status}

_Kickoff: {match['kickoff_time']}_
━━━━━━━━━━━━━━━━━━━━
"""
    
    return message.strip()
```

---

## 🔄 Automatic Sending in Workflow

### GitHub Actions Integration
```yaml
- name: Send Predictions to Telegram
  env:
    TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
    TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
  run: |
    python3 src/analysis/tomorrow_matches.py
```

### Local Testing
```bash
# Set env vars for testing
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Run prediction system
python3 src/analysis/tomorrow_matches.py
```

---

## 📊 Logging Telegram Activity

```python
import logging
from datetime import datetime

# Setup logger
logger = logging.getLogger('telegram')
logger.setLevel(logging.INFO)

# File handler
handler = logging.FileHandler('data/logs/telegram_bot.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# In send function
def _send_telegram_message(self, bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Message sent successfully to chat {chat_id}")
            return True
        else:
            logger.error(f"❌ Failed to send: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"⚠️ Telegram send exception: {e}")
        return False
```

---

## 🧪 Complete Test Script

```python
#!/usr/bin/env python3
"""Test Telegram Integration"""

import os
import requests
import sys

def test_telegram():
    """Test Telegram bot connection"""
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print("🔍 Checking Telegram Configuration...")
    print(f"Bot Token: {'✅ SET' if bot_token else '❌ MISSING'}")
    print(f"Chat ID: {'✅ SET' if chat_id else '❌ MISSING'}")
    
    if not bot_token or not chat_id:
        print("\n❌ Missing credentials!")
        print("\nHow to fix:")
        print("1. Talk to @BotFather on Telegram to get bot token")
        print("2. Talk to @userinfobot to get your chat ID")
        print("3. Set environment variables:")
        print(f"   export TELEGRAM_BOT_TOKEN='your_token'")
        print(f"   export TELEGRAM_CHAT_ID='your_chat_id'")
        sys.exit(1)
    
    print("\n📤 Sending test message...")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message = """
🧪 *TELEGRAM TEST*

This is a test message from the telegramsoccer system.

✅ If you see this, Telegram integration is working!

_Test time: {}_
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    data = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ SUCCESS! Check your Telegram for the test message.")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return False
            
    except Exception as e:
        print(f"⚠️ ERROR: {e}")
        return False

if __name__ == "__main__":
    from datetime import datetime
    test_telegram()
```

Save as `test_telegram.py` and run:
```bash
python3 test_telegram.py
```

---

## 📋 Checklist

- [ ] Bot created via @BotFather
- [ ] Bot token saved
- [ ] Chat ID obtained via @userinfobot
- [ ] Secrets added to GitHub repository
- [ ] Test message sent successfully
- [ ] Codespace has access to secrets
- [ ] Logging configured
- [ ] Error handling in place
- [ ] Message formatting looks good

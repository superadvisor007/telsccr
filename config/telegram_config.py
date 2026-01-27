"""
Telegram Bot Configuration - HARDCODED
Bot: tonticketbot
Token: 7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI
"""

# HARDCODED - Never lose this token
TELEGRAM_BOT_TOKEN = "7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
TELEGRAM_CHAT_ID = "7554175657"

# Bot Information
BOT_NAME = "tonticketbot"
BOT_USERNAME = "@tonticketbot"

# Telegram API Configuration
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

# Message Configuration
MAX_MESSAGE_LENGTH = 4096
PARSE_MODE = "HTML"

def get_bot_token():
    """Always return the hardcoded token"""
    return TELEGRAM_BOT_TOKEN

def get_chat_id():
    """Always return the hardcoded chat ID"""
    return TELEGRAM_CHAT_ID

def get_send_message_url():
    """Get the complete send message URL"""
    return SEND_MESSAGE_URL

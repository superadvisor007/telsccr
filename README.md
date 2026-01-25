# Swiss Soccer Tips Telegram Bot 🏆⚽

An automated AI-powered Telegram bot that provides Swiss Super League soccer predictions with a subscription-based system.

## Features

- 🤖 **AI-Powered Predictions**: Uses Groq/Mistral AI for intelligent match analysis
- 💳 **Subscription System**: Free, Basic, and Premium tiers with Stripe payments (CHF/TWINT)
- 📊 **Real-time Data**: Fetches live match data from API-Football
- 🔄 **Automated Daily Updates**: GitHub Actions for daily predictions
- 📈 **Learning System**: Tracks prediction accuracy and improves over time
- 🔐 **Secure Payments**: Stripe integration with webhook support
- 🇨🇭 **Swiss-Focused**: Specialized for Swiss Super League

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Telegram Bot   │────▶│   Database   │     │  API-Football│
│    (aiogram)    │     │   (SQLite)   │     │             │
└────────┬────────┘     └──────────────┘     └──────┬──────┘
         │                                            │
         │              ┌──────────────┐             │
         └─────────────▶│  Prediction  │◀────────────┘
                        │    Engine    │
                        │   (Groq AI)  │
                        └──────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌───────▼──────┐
│ GitHub Actions  │   │ Webhook Server  │   │    Stripe    │
│  (Daily Cron)   │   │    (aiohttp)    │   │   Payments   │
└─────────────────┘   └─────────────────┘   └──────────────┘
```

## Tech Stack

- **Python 3.11+**: Core language
- **aiogram 3.3**: Telegram Bot framework
- **SQLite/aiosqlite**: Database
- **Stripe**: Payment processing (CHF/TWINT)
- **Groq API**: AI predictions (Mixtral model)
- **API-Football**: Match data
- **GitHub Actions**: Automation
- **aiohttp**: Webhook server

## Installation

### Prerequisites

- Python 3.11 or higher
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Stripe Account (with API keys)
- API-Football API Key
- Groq API Key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/superadvisor007/telegramsoccer.git
   cd telegramsoccer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Create required directories**
   ```bash
   mkdir -p data logs
   ```

## Configuration

Edit `.env` file with your credentials:

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=@your_channel

# Stripe
STRIPE_API_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRICE_BASIC=price_xxx
STRIPE_PRICE_PREMIUM=price_xxx

# API Football
API_FOOTBALL_KEY=your_api_key

# Groq AI
GROQ_API_KEY=your_groq_key

# Bot Settings
BOT_ADMIN_IDS=123456789,987654321
```

## Usage

### Run Telegram Bot

```bash
python run.py bot
```

### Run Webhook Server (for payments)

```bash
python run.py webhook
```

### Run Daily Tasks (manually)

```bash
python run.py daily
```

## Subscription Tiers

| Tier | Price | Predictions/Day | Features |
|------|-------|----------------|----------|
| 🆓 **Free** | Free | 1 | Basic predictions |
| 💎 **Basic** | CHF 9.90/month | 5 | + Match statistics |
| ⭐ **Premium** | CHF 19.90/month | 10 | + Detailed AI analysis, H2H data |

## Bot Commands

- `/start` - Initialize bot and see welcome message
- `/help` - Display help information
- `/subscribe` - Subscribe to paid tiers
- `/status` - Check subscription status
- `/cancel` - Cancel subscription
- `/stats` - View bot statistics (admin only)

## GitHub Actions Automation

The bot runs daily predictions automatically via GitHub Actions:

1. **Daily at 6:00 AM UTC** (7:00 AM CET)
2. Fetches upcoming Swiss Super League matches
3. Generates AI predictions
4. Posts to Telegram channel
5. Updates past prediction results
6. Cleans up expired subscriptions

### Setting up GitHub Secrets

Go to Repository Settings → Secrets and add:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHANNEL_ID`
- `STRIPE_API_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_BASIC`
- `STRIPE_PRICE_PREMIUM`
- `API_FOOTBALL_KEY`
- `GROQ_API_KEY`
- `BOT_ADMIN_IDS`

## Webhook Setup

For production deployment with Stripe webhooks:

1. **Deploy webhook server** (e.g., on Oracle Cloud)
   ```bash
   python run.py webhook
   ```

2. **Configure Stripe webhook** endpoint:
   ```
   https://your-domain.com/webhook/stripe
   ```

3. **Select events** to listen for:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`

## Database Schema

The bot uses SQLite with the following tables:

- **users**: User information and subscription status
- **subscriptions**: Payment history
- **predictions**: AI-generated predictions
- **learning_data**: Historical data for model improvement

## Development

### Project Structure

```
telegramsoccer/
├── src/
│   ├── __init__.py
│   ├── main.py              # Main orchestrator
│   ├── bot.py               # Telegram bot handlers
│   ├── database.py          # Database models
│   ├── api_football.py      # API Football client
│   ├── prediction_engine.py # AI prediction engine
│   ├── payment_handler.py   # Stripe integration
│   └── webhook_server.py    # Webhook server
├── .github/
│   └── workflows/
│       └── daily-predictions.yml
├── data/                    # SQLite database
├── logs/                    # Log files
├── requirements.txt
├── .env.example
├── .gitignore
├── run.py                   # CLI runner
└── README.md
```

### Adding New Features

1. Update relevant module in `src/`
2. Test locally with `python run.py bot`
3. Commit and push changes
4. GitHub Actions will handle automation

## API Rate Limits

- **API-Football**: Check your plan limits
- **Groq**: Free tier has rate limits
- **Stripe**: No rate limits for webhooks

## Troubleshooting

### Bot not responding
- Check if bot token is correct
- Verify bot is running: `python run.py bot`

### Payments not working
- Verify Stripe keys are correct
- Check webhook endpoint is accessible
- Review Stripe dashboard for errors

### Predictions not generating
- Check API-Football key and quota
- Verify Groq API key is valid
- Review logs in `logs/bot.log`

## Logging

All logs are stored in `logs/bot.log`. To view:

```bash
tail -f logs/bot.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use for your own projects!

## Support

For issues or questions:
- Open an issue on GitHub
- Contact: [your-contact]

## Credits

- **API-Football**: Match data provider
- **Groq**: AI/LLM provider
- **Stripe**: Payment processing
- **aiogram**: Telegram Bot framework

---

**Made with ❤️ for Swiss Soccer fans**
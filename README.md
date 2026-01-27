# TelegramSoccer ⚽

**AI-Powered Soccer Betting Assistant for Low-Odds Accumulators**

[![CI/CD](https://github.com/superadvisor007/telegramsoccer/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/superadvisor007/telegramsoccer/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A battle-tested Telegram bot that combines LLM contextual analysis with statistical models to identify value in low-odds accumulator markets (~1.40 quote) for Over 1.5 Goals and BTTS.

---

## 🎯 Project Mission

TelegramSoccer aims to be in the **top 1% of soccer betting systems** by:
- **Deep Analysis**: Synthesizing 50+ contextual factors (weather, injuries, psychology, tactics)
- **Ensemble Predictions**: Combining GPT-4/Claude with XGBoost for robust probability estimates
- **Disciplined Staking**: Fixed 1-2% bankroll management with 15% stop-loss
- **Value-First**: Only betting when researched probability > implied probability

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Data APIs] --> B[Feature Engineering]
    B --> C[LLM Analysis]
    B --> D[XGBoost Models]
    C --> E[Ensemble Engine]
    D --> E
    E --> F[Betting Logic]
    F --> G[Telegram Bot]
    H[GitHub Actions] --> B
```

### Key Components

| Module | Purpose | Technologies |
|--------|---------|--------------|
| **Ingestion** | Fetch odds, stats, weather | The-Odds-API, Football-Data, OpenWeather |
| **Features** | Engineer xG, form, H2H metrics | pandas, numpy |
| **LLM** | Contextual match analysis | OpenAI GPT-4, Anthropic Claude |
| **Models** | Statistical predictions | XGBoost, scikit-learn |
| **Betting** | Accumulator building, value detection | Custom algorithms |
| **Bot** | User interface | python-telegram-bot |
| **Automation** | Daily tip generation | GitHub Actions |

---

## 📊 Mathematical Foundation

### Target Quote: 1.40
- **Break-Even Probability**: 71.43% (1 / 1.40)
- **Double Accumulator**: Each leg needs ~1.18 odds (84.5% probability)
- **Value Equation**: Bet only when `your_probability > 1 / odds`

### Expected Value (EV)
```
EV = (Probability × (Odds - 1)) - (1 - Probability)
```
Only bet when **EV > 0**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+
- API Keys (see `.env.example`)

### Installation

```bash
# Clone repository
git clone https://github.com/superadvisor007/telegramsoccer.git
cd telegramsoccer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -c "from src.core.database import init_db; init_db()"

# Run daily pipeline
python src/pipeline.py

# Start Telegram bot
python src/main.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app
```

---

## 📖 Usage

### Telegram Bot Commands

- `/start` - Welcome message and bot info
- `/today` - Get today's betting tips
- `/stats` - View performance statistics
- `/bankroll` - Check current bankroll
- `/help` - Detailed help

### Manual Pipeline Run

```bash
# Generate tips for today
python src/pipeline.py

# Run with specific date
python src/pipeline.py --date 2026-01-28
```

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_betting.py -v

# Run linters
black src tests
isort src tests
flake8 src tests
mypy src
```

---

## 🔧 Configuration

### Key Settings (`config/config.yaml`)

```yaml
betting:
  target_quote: 1.40
  min_probability: 0.72
  max_stake_percentage: 2.0
  stop_loss_percentage: 15.0

llm:
  model: "gpt-4-turbo-preview"
  temperature: 0.2
  max_tokens: 2000
```

### Environment Variables (`.env`)

```env
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=your_openai_key
ODDS_API_KEY=your_odds_api_key
DATABASE_URL=postgresql://user:pass@localhost:5432/telegramsoccer
```

---

## 📈 Performance Tracking

The system logs every bet with:
- Match details and selected market
- Researched vs implied probability
- LLM reasoning and key factors
- Outcome and profit/loss

Access metrics via:
- Telegram bot `/stats` command
- Database `bankroll` table
- Log files in `logs/`

---

## 🛡️ Risk Management

### Core Principles
1. **Fixed Staking**: 1-2% of bankroll per bet
2. **Stop-Loss**: Halt at 15% drawdown
3. **Value-Only**: Never bet without positive EV
4. **Conservative Odds**: Target 1.40 (not 1.70+) for consistency
5. **Logging**: Track every decision for continuous improvement

### Responsible Gambling
⚠️ **This is an analytical tool, not a guarantee of profit.**
- Only bet what you can afford to lose
- Gambling involves substantial risk
- Verify local laws regarding sports betting

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References & Inspiration

- [datarootsio/your-best-bet](https://github.com/datarootsio/your-best-bet) - MLOps pipeline for football predictions
- [smarmau/asknews_mlb](https://github.com/smarmau/asknews_mlb) - Multi-LLM betting bot architecture
- [sports-betting](https://pypi.org/project/sports-betting/) - Python betting utilities

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Contact

- GitHub: [@superadvisor007](https://github.com/superadvisor007)
- Issues: [GitHub Issues](https://github.com/superadvisor007/telegramsoccer/issues)

---

**Built with 💙 for the soccer betting community**

*Remember: In the top 1%, discipline beats intuition.*
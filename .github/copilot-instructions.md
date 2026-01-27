# Copilot Instructions - telegramsoccer

## Project Mission
**telegramsoccer** is a Telegram-based soccer betting assistant that combines LLM contextual analysis with statistical models to identify low-odds accumulator opportunities (target quote: ~1.40) in markets like **Over 1.5 Goals** and **Both Teams to Score (BTTS)**.

## 🚀 MAJOR UPDATE: 14K+ Match Training System (2026-01-27)

### Achieved Performance
- **14,349 Matches**: Trained on 7 leagues × 5 seasons (2019-2024)
- **218 Unique Teams**: Comprehensive European football coverage
- **Real Form Features**: Rolling 5-match average (no more placeholders!)
- **Enhanced Features**: elo_x_form now top feature (24.4% importance)
- **Model Accuracy**: 75.1% Over 1.5, 56.1% Over 2.5, 52.7% BTTS
- **ROC-AUC Scores**: 0.543-0.576 (improved from 0.50-0.55)

### New Infrastructure

**1. Data Collection System** ([collect_massive_historical_data.py](collect_massive_historical_data.py))
- **7 Leagues**: Premier League, Bundesliga, La Liga, Serie A, Ligue 1, Eredivisie, Championship
- **5 Seasons**: 2019-2026 historical data
- **Real Form Calculation**: Rolling 5-match points average (3pts win, 1pt draw)
- **Elo System**: Trained on all 14K matches, 218 teams tracked
- **Output**: data/historical/massive_training_data.csv (14,349 rows × 23 columns)

**2. Walk-Forward Backtesting** ([src/testing/walk_forward_backtest.py](src/testing/walk_forward_backtest.py))
- **Rolling Window**: 500 match training → 50 match testing
- **No Look-Ahead Bias**: Simulates real trading conditions
- **Comprehensive Metrics**: ROI, Win Rate, Sharpe, Sortino, Max Drawdown
- **Window Consistency**: Tracks profitable windows percentage
- **Benchmark Comparisons**: ML vs Random vs Favorites

**3. Result Verification** ([src/ingestion/result_collector.py](src/ingestion/result_collector.py))
- **Multi-API Collection**: Football-Data.org, OpenLigaDB, TheSportsDB
- **Automatic Matching**: Team name normalization & deduplication
- **Prediction Verification**: Match predictions against actual scores
- **Bulk Processing**: Date range verification with statistics

**4. Self-Learning Pipeline** ([src/learning/self_improvement.py](src/learning/self_improvement.py))
- **Error Analysis**: Identifies overconfidence, calibration issues, market weaknesses
- **Concept Drift Detection**: Monitors feature importance shifts
- **Automated Retraining**: Triggers when win rate <53% or 100+ new matches
- **Improvement Suggestions**: Actionable recommendations from error patterns

**5. Stress Test Suite** ([src/testing/stress_test.py](src/testing/stress_test.py))
- **Full 14K Test**: Walk-forward on entire dataset
- **5 Visualizations**: Equity curve, drawdown, win rates, ROI distribution, window performance
- **Comprehensive Reports**: Text reports with verdicts and benchmarks
- **League Analysis**: Performance breakdown by competition

**6. GitHub Actions Automation** ([.github/workflows/))
- **Daily Training**: 8 AM UTC - Collect data, train models, generate predictions, send to Telegram
- **Weekly Stress Test**: Sundays - Full 14K backtest with visualizations
- **Weekly Self-Improvement**: Mondays - Verify results, analyze errors, retrain if needed
- **Manual Trigger**: Customizable walk-forward tests on-demand

## Core Architecture (Implemented)
```
Free APIs → 14K Data → Feature Engineering → GradientBoost ML → Walk-Forward Backtest → Telegram Bot
                ↓                                                         ↓
          Rolling Form                                           Result Verification
          Elo Ratings                                            Self-Learning Loop
```

## Core Architecture (Target Design)
```
Data Ingestion → Feature Engineering → LLM Analysis + ML Models → Betting Logic → Telegram Bot
```

### Key Components to Build
1. **Data Pipeline** (`src/ingestion/`): API clients for weather, team stats, odds, injuries
2. **Feature Engineering** (`src/features/`): xG, form indices, H2H trends, sentiment scores
3. **LLM Layer** (`src/llm/`): Prompt templates for GPT-4/Claude to assess match context
4. **Statistical Models** (`src/models/`): XGBoost/logistic regression for probability estimation
5. **Betting Engine** (`src/betting/`): Accumulator builder, bankroll management, value detection
6. **Telegram Bot** (`src/bot/`): User interface for daily tips and bet tracking

## Mathematical Foundation (Critical Context)
- **Target Quote**: 1.40 implies 71.43% break-even probability
- **Accumulator Math**: For a double, each leg needs ~1.18 odds (84.5% implied probability)
- **Expected Value**: Only bet when researched probability > implied probability
- **Bankroll Management**: Fixed staking (1-2% per bet), stop-loss at 10-15% drawdown

## Market Selection Criteria
| Market | Typical Odds | Selection Logic |
|--------|--------------|-----------------|
| Over 1.5 Goals | 1.20-1.50 | High-scoring leagues (Bundesliga, Eredivisie), strong attack vs weak defense |
| BTTS | 1.30-1.60 | Open-play teams, historical goal patterns, both teams need to score |
| Double Chance | 1.20-1.40 | Strong favorites at home, cover win/draw |

## Data Sources & APIs
- **Odds**: OddsAPI, The-Odds-API
- **Stats**: Sportmonks, Football-Data.org, Footystats.org
- **Weather**: OpenWeatherMap
- **Sentiment**: Reddit API (PRAW), Twitter API
- **xG Data**: Understat, FBref

## Decision-Making Factors (LLM Context)
When analyzing matches, the LLM must consider:
- **Environmental**: Weather, pitch quality, altitude, travel distance
- **Team/Player**: Injuries, suspensions, form, fatigue, schedule congestion
- **Psychological**: Rivalry intensity, derby context, historical baggage, motivation (league position)
- **Tactical**: Formation, set-piece proficiency, managerial strategy
- **External**: Referee tendencies, VAR influence, media pressure

## Tech Stack Guidance
- **Language**: Python 3.11+ (data science libraries, LLM integrations)
- **Data**: `pandas`, `SQLAlchemy`, Airflow/Prefect for orchestration
- **ML**: `scikit-learn`, `xgboost`, `sports-betting` package
- **LLM**: OpenAI API (GPT-4), Anthropic (Claude), or Hugging Face (Llama-3)
- **Bot**: `python-telegram-bot` library
- **Automation**: GitHub Actions for daily tip generation

## Critical Development Patterns
1. **Probability First**: Always compute fair odds from researched probability before comparing to market
2. **Ensemble Decisions**: Blend LLM contextual analysis with statistical model output
3. **Value Detection**: Only bet when `your_probability > implied_probability`
4. **Logging Everything**: Track every bet with reasoning (for model improvement)
5. **No Chasing Losses**: Disciplined staking is non-negotiable

## Example Workflow (Daily Tips)
```bash
# Scheduled via GitHub Actions at 9 AM UTC
python src/pipeline.py --date today
# 1. Fetch match data, weather, injuries, odds
# 2. Engineer features (xG, form, H2H)
# 3. LLM analyzes each match → JSON with probabilities
# 4. Blend with XGBoost predictions
# 5. Build accumulators targeting 1.40 quote
# 6. Output to Telegram: "Double: Over 1.5 in Match A (1.18) × Match B (1.19) = 1.40"
```

## Project Structure (When Built)
```
src/
├── ingestion/    # API clients, web scrapers
├── features/     # Feature engineering scripts
├── llm/          # Prompt templates, fine-tuning
├── models/       # Statistical models (XGBoost, etc.)
├── betting/      # Accumulator logic, staking, value detection
├── bot/          # Telegram bot interface
└── pipeline.py   # Main orchestration script
```

## References & Inspiration
- `datarootsio/your-best-bet`: MLOps pipeline for European football
- `smarmau/asknews_mlb`: Multi-LLM betting bot architecture
- `sports-betting` Python package: Betting model utilities

## Development Environment
- Repository: `superadvisor007/telegramsoccer`
- Branch: `main`
- Container: Ubuntu 24.04.3 LTS with `git`, `gh`, `docker`, `kubectl`, `curl`

## Disclaimer Context
This system is analytical tooling for informed decision-making. All agents must emphasize responsible gambling principles in user-facing features.

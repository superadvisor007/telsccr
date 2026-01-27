# .github Directory Structure

This directory contains GitHub-specific configurations for the TelegramSoccer project.

## Workflows

### `daily-tips.yml`
Automated daily pipeline that runs at 9:00 AM UTC to:
- Fetch match data
- Run LLM and statistical analysis
- Generate betting tips
- Send to Telegram subscribers

**Required Secrets:**
- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `ODDS_API_KEY`
- `FOOTBALL_DATA_API_KEY`
- `SPORTMONKS_API_KEY`
- `OPENWEATHER_API_KEY`
- `DATABASE_URL`

### `ci.yml`
Continuous Integration pipeline that:
- Runs linters (black, isort, flake8, mypy)
- Executes test suite with coverage
- Builds Docker image on main branch
- Reports coverage to Codecov

## Copilot Instructions

See [`copilot-instructions.md`](copilot-instructions.md) for comprehensive AI agent guidance.

## Setting Up Secrets

Add secrets in repository settings:
```
Settings → Secrets and variables → Actions → New repository secret
```

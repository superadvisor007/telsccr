# Data Directory

This directory stores raw and processed data for the TelegramSoccer system.

## Structure

```
data/
├── raw/           # Raw API responses and scraped data
│   ├── odds/      # Odds data from APIs
│   ├── stats/     # Team and match statistics
│   └── weather/   # Weather forecasts
└── processed/     # Cleaned and engineered features
    ├── features/  # Feature matrices for ML models
    └── tips/      # Generated betting tips (JSON)
```

## Usage

- **Raw data**: Preserved for auditing and reprocessing
- **Processed data**: Ready for model input and tip generation
- **Retention**: 30 days (configurable in `.gitignore` patterns)

## Git Ignore

All data files are gitignored to:
- Prevent repository bloat
- Protect API responses
- Keep sensitive information private

Only `.gitkeep` files are tracked to preserve directory structure.

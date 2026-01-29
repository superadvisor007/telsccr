# 🏗️ Projektstruktur - TelegramSoccer

## Übersicht

```
telegramsoccer/
├── 🚀 run_pipeline.py           # Haupt-Einstiegspunkt
├── 📦 src/                       # Quellcode
├── 🧪 tests/                     # Tests (organisiert)
├── 📜 scripts/                   # Hilfsskripte
├── 📚 docs/                      # Dokumentation
├── 📊 data/                      # Daten & Cache
├── 🤖 models/                    # Trainierte Modelle
└── ⚙️ config/                    # Konfiguration
```

## 📦 Quellcode (`src/`)

### Foundation Models (`src/foundation/`)
Das Herz des Systems - LLM-Integration und Caching.

| Modul | Beschreibung |
|-------|--------------|
| `deepseek_engine.py` | Multi-Backend DeepSeek 7B Integration |
| `model_cache.py` | SQLite-basiertes LLM-Output-Caching |

```python
from foundation import DeepSeekEngine, DeepSeekConfig

config = DeepSeekConfig(backend='ollama', model_name='deepseek-llm:7b-chat')
engine = DeepSeekEngine(config)
result = engine.analyze_match("Bayern", "Dortmund", "Bundesliga")
```

### Data Sources (`src/data_sources/`)
Datensammlung aus freien APIs.

| Modul | Beschreibung |
|-------|--------------|
| `statsbomb_client.py` | StatsBomb Open Data Integration |
| `free_football_apis.py` | TheSportsDB, OpenLigaDB, Football-Data.org |

```python
from data_sources import StatsBombClient, FreeFootballAPIs

# StatsBomb für Event-Daten
sb = StatsBombClient()
comps = sb.get_competitions()

# Freie APIs für Live-Daten
apis = FreeFootballAPIs()
matches = apis.get_upcoming_matches("bundesliga")
```

### Feature Engineering (`src/feature_engineering/`)
Fortschrittliche Fußball-Metriken.

| Modul | Beschreibung |
|-------|--------------|
| `spadl_converter.py` | SPADL (Socceraction) Event-Konvertierung |
| `structural_features.py` | xG, PPDA, Pressing, Tactical Features |

```python
from feature_engineering import StructuralFeatureEngine

engine = StructuralFeatureEngine()
features = engine.compute_team_features(
    goals_scored=25,
    shots=180,
    possession=0.55,
    ...
)
```

### Living Agent (`src/living_agent/`)
Der "lebende" Wett-Agent mit 6 Schichten.

| Modul | Beschreibung |
|-------|--------------|
| `orchestrator.py` | Zentrale Steuerung |
| `multi_bet_builder.py` | Akkumulator-Konstruktion |
| `match_analyzer.py` | Spielanalyse |
| `scenario_simulation.py` | Monte-Carlo Szenarien |

### Pipeline (`src/pipeline/`)
Zentrale Orchestrierung.

| Modul | Beschreibung |
|-------|--------------|
| `unified_pipeline.py` | Verbindet alle Komponenten |
| `elite_value_bets.py` | Value-Bet-Erkennung |

```python
from pipeline import UnifiedBettingPipeline

pipeline = UnifiedBettingPipeline()
ticket = pipeline.run_daily_workflow()
```

## 🧪 Tests (`tests/`)

```
tests/
├── conftest.py           # Pytest-Konfiguration & Fixtures
├── unit/                 # Unit-Tests
├── integration/          # Integrationstests
├── stress/               # Belastungstests
└── validation/           # Validierungstests
```

### Wichtige Test-Dateien

| Test | Beschreibung |
|------|--------------|
| `integration/test_unified_pipeline.py` | Pipeline-Integrationstest |
| `stress/stress_test.py` | Walk-Forward-Backtest |
| `validation/` | Modell-Validierung |

### Tests ausführen

```bash
# Alle Tests
pytest tests/

# Nur Unit-Tests
pytest tests/unit/

# Pipeline-Integrationstest
python tests/integration/test_unified_pipeline.py
```

## 📜 Scripts (`scripts/`)

```
scripts/
├── setup/                # Installations-Skripte
│   ├── setup.sh
│   ├── install_free.sh
│   └── quick_setup.sh
├── runners/              # Ausführungs-Skripte
│   └── run_*.sh
└── training/             # Trainings-Skripte
    ├── train_professional_models.py
    └── collect_massive_historical_data.py
```

## 📚 Dokumentation (`docs/`)

```
docs/
├── architecture/         # System-Architektur
│   └── PRODUCTION_ARCHITECTURE.md
└── guides/               # Anleitungen
    ├── APIS_FOREVER_FREE.md
    ├── ZERO_COST_ARCHITECTURE.md
    └── SETUP_SECRETS.md
```

## ⚙️ Konfiguration (`config/`)

| Datei | Beschreibung |
|-------|--------------|
| `config.yaml` | Hauptkonfiguration |
| `telegram_config.py` | Telegram-Einstellungen |

## 🚀 Schnellstart

### 1. Pipeline starten

```bash
# Status prüfen
python run_pipeline.py --status

# Demo-Modus
python run_pipeline.py --demo

# Täglicher Workflow
python run_pipeline.py
```

### 2. Tests ausführen

```bash
python run_pipeline.py --test
```

### 3. DeepSeek aktivieren (optional)

```bash
# Ollama installieren
curl -fsSL https://ollama.com/install.sh | sh

# DeepSeek laden
ollama pull deepseek-llm:7b-chat

# Pipeline mit LLM
python run_pipeline.py
```

## 📊 Datenfluss

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
│  StatsBomb → TheSportsDB → OpenLigaDB → Football-Data.org       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                         │
│  SPADL Conversion → Structural Features → Tactical Analysis     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FOUNDATION MODELS                            │
│  DeepSeek 7B Reasoning → Model Cache → Confidence Scoring       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BETTING LOGIC                             │
│  Value Detection → Multi-Bet Building → Risk Management         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DELIVERY                                 │
│  Telegram Bot → Daily Tips → Performance Tracking               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Umgebungsvariablen

```bash
# Telegram
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# APIs
export FOOTBALL_DATA_API_KEY="your-api-key"  # Optional

# LLM
export OLLAMA_HOST="http://localhost:11434"  # Default
```

## 📈 Performance-Ziele

| Metrik | Ziel | Aktuell |
|--------|------|---------|
| LLM-Kosten | < $20/mo | $0 (Ollama) |
| Inferenz-Zeit | < 5s | ~2s |
| Cache-Hit-Rate | > 70% | 75% |
| Backtest ROI | > 5% | ~7% |

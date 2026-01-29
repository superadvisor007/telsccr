"""
Pytest Configuration & Shared Fixtures
======================================
Central test configuration for all test modules.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory."""
    return PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def models_dir():
    """Return the models directory."""
    return PROJECT_ROOT / "models"


@pytest.fixture(scope="session")
def config_dir():
    """Return the config directory."""
    return PROJECT_ROOT / "config"


@pytest.fixture
def sample_match():
    """Return a sample match for testing."""
    return {
        'home_team': 'Bayern München',
        'away_team': 'Borussia Dortmund',
        'league': 'Bundesliga',
        'date': '2026-01-28',
        'context': {'is_derby': True}
    }


@pytest.fixture
def sample_team_stats():
    """Return sample team statistics."""
    return {
        'Bayern München': {'goals_scored': 2.4, 'goals_conceded': 0.8, 'form_points': 13},
        'Borussia Dortmund': {'goals_scored': 2.1, 'goals_conceded': 1.2, 'form_points': 10},
    }


@pytest.fixture(scope="session")
def telegram_config():
    """Return Telegram configuration (from env or defaults)."""
    return {
        'token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    }


# Skip markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "stress: marks tests as stress tests")
    config.addinivalue_line("markers", "requires_ollama: marks tests that require Ollama")
    config.addinivalue_line("markers", "requires_api: marks tests that require external APIs")

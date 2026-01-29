"""
⚽ Data Sources Package
=======================
Football data collection from various free sources.

Components:
- statsbomb_client.py: StatsBomb Open Data integration
- free_football_apis.py: TheSportsDB, OpenLigaDB, Football-Data.org
- event_processor.py: Raw event processing and normalization
"""

from .statsbomb_client import StatsBombClient
from .free_football_apis import FreeFootballAPIs

__all__ = [
    'StatsBombClient',
    'FreeFootballAPIs',
]

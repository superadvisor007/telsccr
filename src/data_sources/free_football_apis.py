"""
🌐 Free Football APIs
=====================
Integration with free, open football data APIs.

Sources:
1. TheSportsDB - Free team/league/match info
2. OpenLigaDB - German football data (free)
3. Football-Data.org - Free tier (10 req/min)

All APIs are rate-limit friendly and require no payment.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TheSportsDBClient:
    """
    🏆 TheSportsDB API Client
    
    Free API for sports data including:
    - Team information
    - League standings
    - Match schedules
    - Player details
    
    API Key: Not required for free tier
    Rate Limit: None documented (be reasonable)
    
    Docs: https://www.thesportsdb.com/api.php
    """
    
    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
    
    def _fetch(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from API."""
        import requests
        
        cache_key = f"{endpoint}:{json.dumps(params or {})}"
        
        # Check cache
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Cache result
            self._cache[cache_key] = (data, time.time())
            
            return data
            
        except Exception as e:
            logger.warning(f"TheSportsDB API error: {e}")
            return None
    
    def search_team(self, team_name: str) -> Optional[Dict]:
        """Search for a team by name."""
        data = self._fetch("searchteams.php", {"t": team_name})
        if data and data.get('teams'):
            return data['teams'][0]
        return None
    
    def get_team(self, team_id: str) -> Optional[Dict]:
        """Get team details by ID."""
        data = self._fetch("lookupteam.php", {"id": team_id})
        if data and data.get('teams'):
            return data['teams'][0]
        return None
    
    def get_league_table(self, league_id: str, season: str) -> Optional[List[Dict]]:
        """Get league standings."""
        data = self._fetch("lookuptable.php", {"l": league_id, "s": season})
        return data.get('table') if data else None
    
    def get_next_events(self, team_id: str) -> Optional[List[Dict]]:
        """Get upcoming matches for a team."""
        data = self._fetch("eventsnext.php", {"id": team_id})
        return data.get('events') if data else None
    
    def get_last_events(self, team_id: str) -> Optional[List[Dict]]:
        """Get recent matches for a team."""
        data = self._fetch("eventslast.php", {"id": team_id})
        return data.get('results') if data else None
    
    def get_league_events_by_round(
        self,
        league_id: str,
        round_num: int,
        season: str
    ) -> Optional[List[Dict]]:
        """Get matches for a specific round."""
        data = self._fetch("eventsround.php", {
            "id": league_id,
            "r": round_num,
            "s": season
        })
        return data.get('events') if data else None
    
    def search_leagues(self, country: str) -> Optional[List[Dict]]:
        """Search leagues by country."""
        data = self._fetch("search_all_leagues.php", {"c": country})
        return data.get('countries') if data else None


class OpenLigaDBClient:
    """
    🇩🇪 OpenLigaDB API Client
    
    Free API for German football data:
    - Bundesliga, 2. Bundesliga, 3. Liga
    - DFB-Pokal
    - Match results and fixtures
    
    No API key required.
    Docs: https://www.openligadb.de/
    """
    
    BASE_URL = "https://api.openligadb.de"
    
    # League shortcuts
    LEAGUES = {
        'bundesliga': 'bl1',
        'bundesliga_2': 'bl2',
        'bundesliga_3': 'bl3',
        'dfb_pokal': 'dfb',
    }
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 1800  # 30 minutes
    
    def _fetch(self, endpoint: str) -> Optional[Any]:
        """Fetch data from API."""
        import requests
        
        # Check cache
        if endpoint in self._cache:
            cached, timestamp = self._cache[endpoint]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            self._cache[endpoint] = (data, time.time())
            return data
            
        except Exception as e:
            logger.warning(f"OpenLigaDB API error: {e}")
            return None
    
    def get_matches(
        self,
        league: str,
        season: int,
        matchday: int = None
    ) -> Optional[List[Dict]]:
        """
        Get matches for a league/season.
        
        Args:
            league: League key (e.g., 'bundesliga', 'bl1')
            season: Season year (e.g., 2024 for 2024/25)
            matchday: Specific matchday (optional)
        """
        league_code = self.LEAGUES.get(league, league)
        
        if matchday:
            endpoint = f"getmatchdata/{league_code}/{season}/{matchday}"
        else:
            endpoint = f"getmatchdata/{league_code}/{season}"
        
        return self._fetch(endpoint)
    
    def get_current_matchday(self, league: str) -> Optional[int]:
        """Get current matchday number."""
        league_code = self.LEAGUES.get(league, league)
        data = self._fetch(f"getcurrentgroup/{league_code}")
        return data.get('groupOrderID') if isinstance(data, dict) else None
    
    def get_table(self, league: str, season: int) -> Optional[List[Dict]]:
        """Get current league table."""
        league_code = self.LEAGUES.get(league, league)
        return self._fetch(f"getbltable/{league_code}/{season}")
    
    def get_team_matches(self, team_id: int) -> Optional[List[Dict]]:
        """Get matches for a specific team."""
        return self._fetch(f"getmatchdata/{team_id}")
    
    def get_available_leagues(self) -> Optional[List[Dict]]:
        """Get list of available leagues."""
        return self._fetch("getavailableleagues")
    
    def get_goals(self, match_id: int) -> Optional[List[Dict]]:
        """Get goals for a specific match."""
        return self._fetch(f"getgoals/{match_id}")


class FootballDataOrgClient:
    """
    📊 Football-Data.org API Client
    
    Free tier provides:
    - 10 requests per minute
    - Access to major leagues
    - Match fixtures and results
    - Standings and scorers
    
    Requires free API key.
    Docs: https://www.football-data.org/documentation/quickstart
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    # Competition codes
    COMPETITIONS = {
        'premier_league': 'PL',
        'bundesliga': 'BL1',
        'la_liga': 'PD',
        'serie_a': 'SA',
        'ligue_1': 'FL1',
        'eredivisie': 'DED',
        'championship': 'ELC',
        'champions_league': 'CL',
        'europa_league': 'EL',
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FOOTBALL_DATA_API_KEY', '')
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes (respecting rate limits)
        self._last_request = 0
        self._min_interval = 6  # 6 seconds between requests (10/min)
    
    def _fetch(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from API with rate limiting."""
        import requests
        
        if not self.api_key:
            logger.warning("Football-Data.org API key not set. Set FOOTBALL_DATA_API_KEY env var.")
            return None
        
        cache_key = f"{endpoint}:{json.dumps(params or {})}"
        
        # Check cache
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        # Rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"X-Auth-Token": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            self._last_request = time.time()
            
            if response.status_code == 429:
                logger.warning("Rate limit hit. Waiting...")
                time.sleep(60)
                return self._fetch(endpoint, params)
            
            response.raise_for_status()
            data = response.json()
            
            self._cache[cache_key] = (data, time.time())
            return data
            
        except Exception as e:
            logger.warning(f"Football-Data.org API error: {e}")
            return None
    
    def get_competitions(self) -> Optional[List[Dict]]:
        """Get available competitions."""
        data = self._fetch("competitions")
        return data.get('competitions') if data else None
    
    def get_competition(self, competition: str) -> Optional[Dict]:
        """Get competition details."""
        code = self.COMPETITIONS.get(competition, competition)
        return self._fetch(f"competitions/{code}")
    
    def get_matches(
        self,
        competition: str,
        matchday: int = None,
        date_from: str = None,
        date_to: str = None
    ) -> Optional[List[Dict]]:
        """Get matches for a competition."""
        code = self.COMPETITIONS.get(competition, competition)
        
        params = {}
        if matchday:
            params['matchday'] = matchday
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        
        data = self._fetch(f"competitions/{code}/matches", params)
        return data.get('matches') if data else None
    
    def get_standings(self, competition: str) -> Optional[List[Dict]]:
        """Get league standings."""
        code = self.COMPETITIONS.get(competition, competition)
        data = self._fetch(f"competitions/{code}/standings")
        return data.get('standings') if data else None
    
    def get_scorers(self, competition: str) -> Optional[List[Dict]]:
        """Get top scorers."""
        code = self.COMPETITIONS.get(competition, competition)
        data = self._fetch(f"competitions/{code}/scorers")
        return data.get('scorers') if data else None
    
    def get_team(self, team_id: int) -> Optional[Dict]:
        """Get team details."""
        return self._fetch(f"teams/{team_id}")
    
    def get_team_matches(
        self,
        team_id: int,
        status: str = None,
        limit: int = 10
    ) -> Optional[List[Dict]]:
        """Get matches for a team."""
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        data = self._fetch(f"teams/{team_id}/matches", params)
        return data.get('matches') if data else None


class FreeFootballAPIs:
    """
    🌐 Unified Free Football APIs Client
    
    Combines all free sources into a single interface.
    Automatically chooses the best source for each query.
    """
    
    def __init__(self, football_data_key: str = None):
        self.thesportsdb = TheSportsDBClient()
        self.openligadb = OpenLigaDBClient()
        self.footballdata = FootballDataOrgClient(football_data_key)
    
    def get_team_info(self, team_name: str) -> Optional[Dict]:
        """Get team information from best available source."""
        # Try TheSportsDB first (no rate limits)
        result = self.thesportsdb.search_team(team_name)
        if result:
            return {
                'source': 'thesportsdb',
                'id': result.get('idTeam'),
                'name': result.get('strTeam'),
                'country': result.get('strCountry'),
                'league': result.get('strLeague'),
                'stadium': result.get('strStadium'),
                'capacity': result.get('intStadiumCapacity'),
                'founded': result.get('intFormedYear'),
                'logo': result.get('strTeamBadge'),
            }
        return None
    
    def get_upcoming_matches(
        self,
        league: str = None,
        team: str = None,
        days: int = 7
    ) -> List[Dict]:
        """Get upcoming matches."""
        matches = []
        
        # Try OpenLigaDB for German leagues
        if league and 'bundesliga' in league.lower():
            current_day = self.openligadb.get_current_matchday(league)
            if current_day:
                raw = self.openligadb.get_matches(league, 2024, current_day)
                if raw:
                    for m in raw:
                        matches.append({
                            'source': 'openligadb',
                            'home_team': m.get('team1', {}).get('teamName'),
                            'away_team': m.get('team2', {}).get('teamName'),
                            'date': m.get('matchDateTime'),
                            'league': league,
                        })
        
        # Try Football-Data.org
        if not matches and league:
            raw = self.footballdata.get_matches(
                league,
                date_from=datetime.now().strftime('%Y-%m-%d'),
                date_to=(datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
            )
            if raw:
                for m in raw:
                    matches.append({
                        'source': 'football-data.org',
                        'home_team': m.get('homeTeam', {}).get('name'),
                        'away_team': m.get('awayTeam', {}).get('name'),
                        'date': m.get('utcDate'),
                        'league': m.get('competition', {}).get('name'),
                    })
        
        return matches
    
    def get_standings(self, league: str, season: int = None) -> Optional[List[Dict]]:
        """Get league standings."""
        season = season or datetime.now().year
        
        # Try OpenLigaDB for German leagues
        if 'bundesliga' in league.lower():
            raw = self.openligadb.get_table(league, season)
            if raw:
                return [{
                    'position': t.get('position', t.get('rank')),
                    'team': t.get('teamName'),
                    'played': t.get('matches'),
                    'won': t.get('won'),
                    'drawn': t.get('draw'),
                    'lost': t.get('lost'),
                    'goals_for': t.get('goals'),
                    'goals_against': t.get('opponentGoals'),
                    'points': t.get('points'),
                } for t in raw]
        
        # Try Football-Data.org
        raw = self.footballdata.get_standings(league)
        if raw:
            standings = raw[0].get('table', []) if raw else []
            return [{
                'position': t.get('position'),
                'team': t.get('team', {}).get('name'),
                'played': t.get('playedGames'),
                'won': t.get('won'),
                'drawn': t.get('draw'),
                'lost': t.get('lost'),
                'goals_for': t.get('goalsFor'),
                'goals_against': t.get('goalsAgainst'),
                'points': t.get('points'),
            } for t in standings]
        
        return None

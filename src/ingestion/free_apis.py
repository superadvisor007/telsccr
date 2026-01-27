"""Free API clients for soccer data (API-Football, iSports, Sportmonks)."""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class APIFootballClient:
    """
    API-Football Free Tier client.
    
    Limits: 100 requests/day
    Coverage: Major leagues, live scores, fixtures, statistics
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }
        self.requests_today = 0
        self.daily_limit = 100
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with retry logic."""
        if self.requests_today >= self.daily_limit:
            raise Exception("API-Football daily quota exceeded")
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            self.requests_today += 1
            logger.debug(f"API-Football request {self.requests_today}/{self.daily_limit}: {endpoint}")
            
            return response.json()
    
    async def get_fixtures(
        self,
        league_id: int,
        season: int = 2026,
        date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get fixtures for a league.
        
        Args:
            league_id: API-Football league ID (e.g., 39 = Premier League)
            season: Year (e.g., 2026)
            date: Optional date filter (YYYY-MM-DD)
        """
        params = {
            "league": league_id,
            "season": season,
        }
        
        if date:
            params["date"] = date
        else:
            # Default to next 3 days
            params["from"] = datetime.now().strftime("%Y-%m-%d")
            params["to"] = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        
        try:
            data = await self._get("fixtures", params=params)
            return data.get("response", [])
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {e}")
            return []
    
    async def get_odds(
        self,
        fixture_id: int,
        bookmaker_id: int = 8,  # 8 = Bet365
    ) -> Dict:
        """Get betting odds for a fixture."""
        params = {
            "fixture": fixture_id,
            "bookmaker": bookmaker_id,
        }
        
        try:
            data = await self._get("odds", params=params)
            return data.get("response", [{}])[0]
        except Exception as e:
            logger.warning(f"Failed to fetch odds for fixture {fixture_id}: {e}")
            return {}
    
    async def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int = 2026,
    ) -> Dict:
        """Get team statistics for current season."""
        params = {
            "team": team_id,
            "league": league_id,
            "season": season,
        }
        
        try:
            data = await self._get("teams/statistics", params=params)
            return data.get("response", {})
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return {}


class iSportsAPIClient:
    """
    iSports API Free Tier client.
    
    Limits: 200 requests/day
    Focus: Betting odds data
    """
    
    BASE_URL = "https://api.isportsapi.com/sport/football"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_today = 0
        self.daily_limit = 200
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request."""
        if self.requests_today >= self.daily_limit:
            raise Exception("iSports daily quota exceeded")
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        params = params or {}
        params["api_key"] = self.api_key
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            self.requests_today += 1
            logger.debug(f"iSports request {self.requests_today}/{self.daily_limit}: {endpoint}")
            
            return response.json()
    
    async def get_odds(
        self,
        match_id: int,
        market: str = "1x2",
    ) -> List[Dict]:
        """Get odds for a match."""
        try:
            data = await self._get(f"odds/{match_id}", params={"market": market})
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return []


class QuotaManager:
    """
    Intelligent quota manager that pools requests across multiple free APIs.
    
    Strategy:
    - Track daily usage per API
    - Rotate requests to maximize total daily quota
    - Fallback to alternative APIs when quotas exhausted
    """
    
    def __init__(
        self,
        api_football_key: Optional[str] = None,
        isports_key: Optional[str] = None,
    ):
        self.apis = {}
        
        if api_football_key:
            self.apis['api_football'] = APIFootballClient(api_football_key)
        
        if isports_key:
            self.apis['isports'] = iSportsAPIClient(isports_key)
        
        self.last_reset = datetime.now().date()
        
        logger.info(f"QuotaManager initialized with {len(self.apis)} APIs")
    
    def reset_if_new_day(self):
        """Reset quotas if it's a new day."""
        today = datetime.now().date()
        
        if today > self.last_reset:
            for api in self.apis.values():
                api.requests_today = 0
            
            self.last_reset = today
            logger.info("Daily API quotas reset")
    
    async def get_fixtures(
        self,
        league_id: int,
        date: Optional[str] = None,
    ) -> List[Dict]:
        """Get fixtures using available API."""
        self.reset_if_new_day()
        
        if 'api_football' in self.apis:
            api = self.apis['api_football']
            
            if api.requests_today < api.daily_limit - 10:  # Keep buffer
                return await api.get_fixtures(league_id, date=date)
        
        logger.warning("No API quota available for fixtures")
        return []
    
    async def get_odds(
        self,
        fixture_id: Optional[int] = None,
        match_id: Optional[int] = None,
    ) -> Dict:
        """Get odds using best available API."""
        self.reset_if_new_day()
        
        # Try iSports first (dedicated odds API, higher quota)
        if 'isports' in self.apis and match_id:
            api = self.apis['isports']
            
            if api.requests_today < api.daily_limit - 20:
                odds_data = await api.get_odds(match_id)
                if odds_data:
                    return self._parse_isports_odds(odds_data)
        
        # Fallback to API-Football
        if 'api_football' in self.apis and fixture_id:
            api = self.apis['api_football']
            
            if api.requests_today < api.daily_limit - 5:
                return await api.get_odds(fixture_id)
        
        logger.warning("No API quota available for odds")
        return {}
    
    def _parse_isports_odds(self, odds_data: List[Dict]) -> Dict:
        """Parse iSports odds format to unified format."""
        # iSports has different format, normalize it
        parsed = {}
        
        for bookmaker in odds_data:
            for market in bookmaker.get('markets', []):
                if market.get('name') == 'Over/Under 1.5':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == 'Over 1.5':
                            parsed['over_1_5_odds'] = float(outcome.get('odds', 1.5))
                
                elif market.get('name') == 'Both Teams To Score':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == 'Yes':
                            parsed['btts_odds'] = float(outcome.get('odds', 1.8))
        
        return parsed
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current quota usage."""
        return {
            api_name: {
                'used': api.requests_today,
                'limit': api.daily_limit,
                'remaining': api.daily_limit - api.requests_today,
                'percentage': (api.requests_today / api.daily_limit * 100),
            }
            for api_name, api in self.apis.items()
        }

"""API Football client for fetching match data."""
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class APIFootballClient:
    """Client for API-Football API."""

    def __init__(self, api_key: str, base_url: str = "https://v3.football.api-sports.io"):
        """Initialize API Football client.
        
        Args:
            api_key: API-Football API key
            base_url: Base URL for API-Football
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "x-apisports-key": api_key
        }

    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data
        """
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
            except Exception as e:
                logger.error(f"API request failed: {e}")
                raise

    async def get_fixtures_by_league(
        self,
        league_id: int,
        season: int,
        date: Optional[str] = None
    ) -> List[Dict]:
        """Get fixtures for a specific league and season.
        
        Args:
            league_id: League ID (e.g., 207 for Swiss Super League)
            season: Season year
            date: Optional date filter (YYYY-MM-DD)
            
        Returns:
            List of fixtures
        """
        params = {
            "league": league_id,
            "season": season
        }
        
        if date:
            params["date"] = date
            
        try:
            data = await self._make_request("fixtures", params)
            fixtures = data.get("response", [])
            logger.info(f"Fetched {len(fixtures)} fixtures for league {league_id}")
            return fixtures
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {e}")
            return []

    async def get_upcoming_matches(
        self,
        league_id: int,
        season: int,
        days_ahead: int = 7
    ) -> List[Dict]:
        """Get upcoming matches for the next N days.
        
        Args:
            league_id: League ID
            season: Season year
            days_ahead: Number of days to look ahead
            
        Returns:
            List of upcoming fixtures
        """
        all_fixtures = []
        today = datetime.now()
        
        # Batch requests with rate limiting
        tasks = []
        for i in range(days_ahead):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            tasks.append(self.get_fixtures_by_league(league_id, season, date))
            
            # Process in batches of 3 to avoid rate limits
            if len(tasks) >= 3:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        all_fixtures.extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Failed to fetch fixtures: {result}")
                tasks = []
                # Small delay between batches
                await asyncio.sleep(0.5)
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_fixtures.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Failed to fetch fixtures: {result}")
            
        return all_fixtures

    async def get_match_statistics(self, fixture_id: int) -> Dict:
        """Get detailed statistics for a match.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Match statistics
        """
        params = {"fixture": fixture_id}
        
        try:
            data = await self._make_request("fixtures/statistics", params)
            return data.get("response", {})
        except Exception as e:
            logger.error(f"Failed to fetch match statistics: {e}")
            return {}

    async def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int
    ) -> Dict:
        """Get team statistics for a season.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season: Season year
            
        Returns:
            Team statistics
        """
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        try:
            data = await self._make_request("teams/statistics", params)
            return data.get("response", {})
        except Exception as e:
            logger.error(f"Failed to fetch team statistics: {e}")
            return {}

    async def get_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        last: int = 10
    ) -> List[Dict]:
        """Get head-to-head matches between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last: Number of recent matches
            
        Returns:
            List of H2H matches
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        }
        
        try:
            data = await self._make_request("fixtures/headtohead", params)
            return data.get("response", [])
        except Exception as e:
            logger.error(f"Failed to fetch H2H matches: {e}")
            return []

    def parse_fixture(self, fixture: Dict) -> Dict:
        """Parse fixture data into simplified format.
        
        Args:
            fixture: Raw fixture data from API
            
        Returns:
            Parsed fixture data
        """
        return {
            "match_id": fixture["fixture"]["id"],
            "league_id": fixture["league"]["id"],
            "league_name": fixture["league"]["name"],
            "home_team": fixture["teams"]["home"]["name"],
            "home_team_id": fixture["teams"]["home"]["id"],
            "away_team": fixture["teams"]["away"]["name"],
            "away_team_id": fixture["teams"]["away"]["id"],
            "match_date": fixture["fixture"]["date"],
            "venue": fixture["fixture"]["venue"]["name"],
            "status": fixture["fixture"]["status"]["short"],
            "home_score": fixture["goals"]["home"],
            "away_score": fixture["goals"]["away"]
        }

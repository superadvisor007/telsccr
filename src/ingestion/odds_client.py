"""Odds API client for fetching betting odds."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.config import settings
from src.ingestion.base_client import BaseAPIClient


class OddsAPIClient(BaseAPIClient):
    """Client for The Odds API (https://the-odds-api.com/)."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.the-odds-api.com/v4",
            api_key=settings.api.odds_api_key,
        )
    
    async def get_soccer_leagues(self) -> List[Dict[str, Any]]:
        """Get available soccer leagues."""
        try:
            data = await self._get("sports")
            soccer_leagues = [
                sport for sport in data 
                if sport.get("group") == "soccer" and sport.get("active")
            ]
            logger.info(f"Found {len(soccer_leagues)} active soccer leagues")
            return soccer_leagues
        except Exception as e:
            logger.error(f"Failed to fetch soccer leagues: {e}")
            return []
    
    async def get_odds(
        self,
        sport_key: str,
        markets: List[str] = ["totals", "btts"],
        regions: str = "eu",
        odds_format: str = "decimal"
    ) -> List[Dict[str, Any]]:
        """
        Get odds for a specific sport.
        
        Args:
            sport_key: Sport identifier (e.g., "soccer_epl", "soccer_germany_bundesliga")
            markets: List of markets to fetch (totals for Over/Under, btts for BTTS)
            regions: Region for odds (eu, us, uk, au)
            odds_format: Odds format (decimal, american)
        """
        try:
            params = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            }
            
            data = await self._get(f"sports/{sport_key}/odds", params=params)
            logger.info(f"Fetched odds for {len(data)} matches in {sport_key}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch odds for {sport_key}: {e}")
            return []
    
    async def get_upcoming_matches(
        self,
        leagues: List[str],
        hours_ahead: int = 72
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming matches with odds from multiple leagues.
        
        Args:
            leagues: List of league keys (e.g., ["soccer_epl", "soccer_germany_bundesliga"])
            hours_ahead: Fetch matches within next N hours
        """
        all_matches = []
        
        for league in leagues:
            try:
                matches = await self.get_odds(league)
                
                # Filter matches within time window
                now = datetime.utcnow()
                for match in matches:
                    commence_time = datetime.fromisoformat(
                        match["commence_time"].replace("Z", "+00:00")
                    )
                    
                    hours_until_match = (commence_time - now).total_seconds() / 3600
                    if 0 < hours_until_match <= hours_ahead:
                        all_matches.append(match)
                
            except Exception as e:
                logger.error(f"Failed to fetch matches for {league}: {e}")
                continue
        
        logger.info(f"Found {len(all_matches)} upcoming matches")
        return all_matches
    
    def parse_odds(self, match_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Parse odds from match data.
        
        Returns:
            Dictionary with over_1_5, btts, and other relevant odds
        """
        odds_dict = {
            "over_1_5": None,
            "under_1_5": None,
            "over_2_5": None,
            "under_2_5": None,
            "btts_yes": None,
            "btts_no": None,
        }
        
        bookmakers = match_data.get("bookmakers", [])
        if not bookmakers:
            return odds_dict
        
        # Use first available bookmaker (usually best odds)
        bookmaker = bookmakers[0]
        markets = bookmaker.get("markets", [])
        
        for market in markets:
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])
            
            if market_key == "totals":
                for outcome in outcomes:
                    point = outcome.get("point")
                    name = outcome.get("name")
                    price = outcome.get("price")
                    
                    if point == 1.5:
                        if name == "Over":
                            odds_dict["over_1_5"] = price
                        elif name == "Under":
                            odds_dict["under_1_5"] = price
                    elif point == 2.5:
                        if name == "Over":
                            odds_dict["over_2_5"] = price
                        elif name == "Under":
                            odds_dict["under_2_5"] = price
            
            elif market_key == "btts":
                for outcome in outcomes:
                    name = outcome.get("name")
                    price = outcome.get("price")
                    
                    if name == "Yes":
                        odds_dict["btts_yes"] = price
                    elif name == "No":
                        odds_dict["btts_no"] = price
        
        return odds_dict

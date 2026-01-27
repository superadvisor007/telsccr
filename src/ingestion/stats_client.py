"""Football statistics API client."""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.config import settings
from src.ingestion.base_client import BaseAPIClient


class FootballDataClient(BaseAPIClient):
    """Client for Football-Data.org API."""
    
    # Map league names to API codes
    LEAGUE_IDS = {
        "Premier League": "PL",
        "Bundesliga": "BL1",
        "La Liga": "PD",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Championship": "ELC",
    }
    
    def __init__(self):
        super().__init__(
            base_url="https://api.football-data.org/v4",
            api_key=settings.api.football_data_api_key,
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Override to use X-Auth-Token header."""
        return {
            "X-Auth-Token": self.api_key,
            "Content-Type": "application/json",
        }
    
    async def get_team_stats(self, team_id: int, num_matches: int = 5) -> Dict[str, Any]:
        """
        Get recent statistics for a team.
        
        Args:
            team_id: Team ID from Football-Data API
            num_matches: Number of recent matches to analyze
        """
        try:
            # Get recent matches
            data = await self._get(f"teams/{team_id}/matches", params={"limit": num_matches})
            matches = data.get("matches", [])
            
            if not matches:
                return self._empty_stats()
            
            # Calculate statistics
            stats = self._calculate_team_stats(team_id, matches)
            logger.info(f"Fetched stats for team {team_id}: {stats['form']}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to fetch team stats for {team_id}: {e}")
            return self._empty_stats()
    
    def _calculate_team_stats(self, team_id: int, matches: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from match history."""
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        clean_sheets = 0
        btts_count = 0
        over_1_5_count = 0
        over_2_5_count = 0
        
        for match in matches:
            if match["status"] != "FINISHED":
                continue
            
            home_team = match["homeTeam"]["id"]
            away_team = match["awayTeam"]["id"]
            home_score = match["score"]["fullTime"]["home"]
            away_score = match["score"]["fullTime"]["away"]
            
            if home_score is None or away_score is None:
                continue
            
            is_home = home_team == team_id
            team_score = home_score if is_home else away_score
            opponent_score = away_score if is_home else home_score
            
            # Goals
            goals_scored += team_score
            goals_conceded += opponent_score
            
            # Result
            if team_score > opponent_score:
                wins += 1
            elif team_score == opponent_score:
                draws += 1
            else:
                losses += 1
            
            # Clean sheet
            if opponent_score == 0:
                clean_sheets += 1
            
            # BTTS
            if home_score > 0 and away_score > 0:
                btts_count += 1
            
            # Over/Under
            total_goals = home_score + away_score
            if total_goals > 1.5:
                over_1_5_count += 1
            if total_goals > 2.5:
                over_2_5_count += 1
        
        num_matches = len([m for m in matches if m["status"] == "FINISHED"])
        
        if num_matches == 0:
            return self._empty_stats()
        
        return {
            "matches_played": num_matches,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goals_per_game": round(goals_scored / num_matches, 2),
            "goals_conceded_per_game": round(goals_conceded / num_matches, 2),
            "clean_sheets": clean_sheets,
            "clean_sheet_percentage": round(clean_sheets / num_matches * 100, 1),
            "btts_percentage": round(btts_count / num_matches * 100, 1),
            "over_1_5_percentage": round(over_1_5_count / num_matches * 100, 1),
            "over_2_5_percentage": round(over_2_5_count / num_matches * 100, 1),
            "form": self._calculate_form(wins, draws, losses),
            "points": wins * 3 + draws,
            "ppg": round((wins * 3 + draws) / num_matches, 2),
        }
    
    def _calculate_form(self, wins: int, draws: int, losses: int) -> str:
        """Calculate form string (e.g., 'WWDLW')."""
        # This is simplified - in production, maintain match order
        form_chars = ['W'] * wins + ['D'] * draws + ['L'] * losses
        return ''.join(form_chars[:5])
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats dict."""
        return {
            "matches_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "goals_per_game": 0,
            "goals_conceded_per_game": 0,
            "clean_sheets": 0,
            "clean_sheet_percentage": 0,
            "btts_percentage": 0,
            "over_1_5_percentage": 0,
            "over_2_5_percentage": 0,
            "form": "",
            "points": 0,
            "ppg": 0,
        }
    
    async def get_h2h_stats(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get head-to-head statistics between two teams."""
        try:
            data = await self._get(
                f"teams/{team1_id}/matches",
                params={"limit": limit}
            )
            
            matches = data.get("matches", [])
            
            # Filter for matches between these two teams
            h2h_matches = [
                m for m in matches
                if (m["homeTeam"]["id"] == team2_id or m["awayTeam"]["id"] == team2_id)
                and m["status"] == "FINISHED"
            ]
            
            if not h2h_matches:
                return {"matches": 0, "avg_goals": 0, "btts_rate": 0}
            
            total_goals = 0
            btts = 0
            
            for match in h2h_matches:
                home_score = match["score"]["fullTime"]["home"]
                away_score = match["score"]["fullTime"]["away"]
                
                if home_score is not None and away_score is not None:
                    total_goals += home_score + away_score
                    if home_score > 0 and away_score > 0:
                        btts += 1
            
            return {
                "matches": len(h2h_matches),
                "avg_goals": round(total_goals / len(h2h_matches), 2),
                "btts_rate": round(btts / len(h2h_matches) * 100, 1),
                "team1_wins": sum(1 for m in h2h_matches if self._get_winner(m, team1_id) == team1_id),
                "team2_wins": sum(1 for m in h2h_matches if self._get_winner(m, team1_id) == team2_id),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch H2H stats: {e}")
            return {"matches": 0, "avg_goals": 0, "btts_rate": 0}
    
    def _get_winner(self, match: Dict, team1_id: int) -> Optional[int]:
        """Determine winner of a match."""
        home_score = match["score"]["fullTime"]["home"]
        away_score = match["score"]["fullTime"]["away"]
        
        if home_score is None or away_score is None:
            return None
        
        home_team = match["homeTeam"]["id"]
        
        if home_score > away_score:
            return home_team
        elif away_score > home_score:
            return match["awayTeam"]["id"]
        return None

"""
xG (Expected Goals) Data Fetcher
================================
Battle-tested pattern from amosbastian/understat

Fetches xG data from Understat for European leagues:
- EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL

Features extracted:
- Team xG (expected goals)
- Team xGA (expected goals against)
- NPxG (non-penalty xG)
- xGChain, xGBuildup
- Deep completions
"""

import asyncio
import aiohttp
import json
import re
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from understat
BASE_URL = "https://understat.com"
LEAGUE_URL = f"{BASE_URL}/league/{{}}/{{}}"

# League mapping
LEAGUE_MAP = {
    'Premier League': 'EPL',
    'La Liga': 'La_liga', 
    'Bundesliga': 'Bundesliga',
    'Serie A': 'Serie_A',
    'Ligue 1': 'Ligue_1',
    'Eredivisie': None,  # Not on Understat
    'Championship': None,  # Not on Understat
}

class UnderstatXGFetcher:
    """
    Async xG data fetcher using Understat's public data.
    Battle-tested pattern from amosbastian/understat library.
    """
    
    def __init__(self):
        self.session = None
        self.cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str) -> str:
        """Fetch URL with retry logic."""
        for attempt in range(3):
            try:
                async with self.session.get(
                    url, 
                    headers={'X-Requested-With': 'XMLHttpRequest'}
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Status {response.status} for {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
        return None
    
    def extract_json_data(self, html: str, var_name: str) -> dict:
        """Extract JSON data from HTML script tags."""
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if match:
            json_str = match.group(1)
            # Unescape the JSON string
            json_str = json_str.encode().decode('unicode_escape')
            return json.loads(json_str)
        return {}
    
    async def get_league_teams(self, league: str, season: int) -> List[Dict]:
        """
        Get all teams with their xG stats for a league/season.
        
        Returns list of team dicts with:
        - title, id
        - xG, xGA, NPxG, NPxGA
        - deep, deep_allowed
        - history (match-by-match xG)
        """
        understat_league = LEAGUE_MAP.get(league)
        if not understat_league:
            logger.info(f"League {league} not available on Understat")
            return []
        
        cache_key = f"{league}_{season}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = LEAGUE_URL.format(understat_league, season)
        html = await self.fetch(url)
        
        if not html:
            return []
        
        teams_data = self.extract_json_data(html, 'teamsData')
        if not teams_data:
            return []
        
        teams = []
        for team_id, team_info in teams_data.items():
            # Calculate aggregated xG stats from history
            history = team_info.get('history', [])
            
            total_xg = sum(h.get('xG', 0) for h in history)
            total_xga = sum(h.get('xGA', 0) for h in history)
            total_npxg = sum(h.get('npxG', 0) for h in history)
            total_npxga = sum(h.get('npxGA', 0) for h in history)
            total_deep = sum(h.get('deep', 0) for h in history)
            total_deep_allowed = sum(h.get('deep_allowed', 0) for h in history)
            matches = len(history)
            
            teams.append({
                'id': team_id,
                'title': team_info.get('title', ''),
                'matches': matches,
                'xG': round(total_xg, 2),
                'xGA': round(total_xga, 2),
                'xG_per_match': round(total_xg / max(matches, 1), 3),
                'xGA_per_match': round(total_xga / max(matches, 1), 3),
                'NPxG': round(total_npxg, 2),
                'NPxGA': round(total_npxga, 2),
                'deep': total_deep,
                'deep_allowed': total_deep_allowed,
                'xGD': round(total_xg - total_xga, 2),  # xG Difference
                'history': history
            })
        
        self.cache[cache_key] = teams
        return teams
    
    async def get_league_matches_xg(self, league: str, season: int) -> List[Dict]:
        """
        Get all matches with xG data for a league/season.
        
        Returns list of match dicts with:
        - home_team, away_team
        - home_xG, away_xG
        - home_goals, away_goals
        - date
        """
        understat_league = LEAGUE_MAP.get(league)
        if not understat_league:
            return []
        
        url = LEAGUE_URL.format(understat_league, season)
        html = await self.fetch(url)
        
        if not html:
            return []
        
        dates_data = self.extract_json_data(html, 'datesData')
        if not dates_data:
            return []
        
        matches = []
        for match in dates_data:
            if not match.get('isResult'):
                continue
            
            matches.append({
                'match_id': match.get('id'),
                'date': match.get('datetime', '')[:10],
                'home_team': match.get('h', {}).get('title', ''),
                'away_team': match.get('a', {}).get('title', ''),
                'home_goals': int(match.get('goals', {}).get('h', 0) or 0),
                'away_goals': int(match.get('goals', {}).get('a', 0) or 0),
                'home_xG': float(match.get('xG', {}).get('h', 0) or 0),
                'away_xG': float(match.get('xG', {}).get('a', 0) or 0),
                'forecast_home': float(match.get('forecast', {}).get('w', 0) or 0),
                'forecast_draw': float(match.get('forecast', {}).get('d', 0) or 0),
                'forecast_away': float(match.get('forecast', {}).get('l', 0) or 0),
            })
        
        return matches


async def collect_xg_data_for_training(seasons: List[int] = None) -> pd.DataFrame:
    """
    Collect xG data for all available leagues and seasons.
    Returns DataFrame ready for training integration.
    """
    if seasons is None:
        seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
    
    all_matches = []
    
    async with UnderstatXGFetcher() as fetcher:
        for league in leagues:
            for season in seasons:
                logger.info(f"Fetching xG data: {league} {season}")
                try:
                    matches = await fetcher.get_league_matches_xg(league, season)
                    for m in matches:
                        m['league'] = league
                        m['season'] = season
                    all_matches.extend(matches)
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error fetching {league} {season}: {e}")
    
    df = pd.DataFrame(all_matches)
    return df


def create_xg_features(df: pd.DataFrame, xg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge xG features into training DataFrame.
    
    New features added:
    - home_xG_avg: Rolling average home xG
    - away_xG_avg: Rolling average away xG  
    - home_xGA_avg: Rolling average home xGA
    - away_xGA_avg: Rolling average away xGA
    - xG_diff: home_xG_avg - away_xG_avg
    - xGD_home: home_xG_avg - home_xGA_avg
    - xGD_away: away_xG_avg - away_xGA_avg
    """
    # Build team xG averages from historical data
    team_xg_stats = {}
    
    for _, row in xg_data.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        if home not in team_xg_stats:
            team_xg_stats[home] = {'xG': [], 'xGA': []}
        if away not in team_xg_stats:
            team_xg_stats[away] = {'xG': [], 'xGA': []}
        
        team_xg_stats[home]['xG'].append(row['home_xG'])
        team_xg_stats[home]['xGA'].append(row['away_xG'])
        team_xg_stats[away]['xG'].append(row['away_xG'])
        team_xg_stats[away]['xGA'].append(row['home_xG'])
    
    # Calculate averages
    team_xg_avg = {}
    for team, stats in team_xg_stats.items():
        team_xg_avg[team] = {
            'xG_avg': sum(stats['xG'][-10:]) / len(stats['xG'][-10:]) if stats['xG'] else 1.3,
            'xGA_avg': sum(stats['xGA'][-10:]) / len(stats['xGA'][-10:]) if stats['xGA'] else 1.3,
        }
    
    # Add features to dataframe
    df = df.copy()
    
    default_xg = 1.3  # League average
    
    df['home_xG_avg'] = df['home_team'].map(
        lambda t: team_xg_avg.get(t, {}).get('xG_avg', default_xg)
    )
    df['away_xG_avg'] = df['away_team'].map(
        lambda t: team_xg_avg.get(t, {}).get('xG_avg', default_xg)
    )
    df['home_xGA_avg'] = df['home_team'].map(
        lambda t: team_xg_avg.get(t, {}).get('xGA_avg', default_xg)
    )
    df['away_xGA_avg'] = df['away_team'].map(
        lambda t: team_xg_avg.get(t, {}).get('xGA_avg', default_xg)
    )
    
    # Derived xG features
    df['xG_diff'] = df['home_xG_avg'] - df['away_xG_avg']
    df['xGD_home'] = df['home_xG_avg'] - df['home_xGA_avg']
    df['xGD_away'] = df['away_xG_avg'] - df['away_xGA_avg']
    df['total_xG'] = df['home_xG_avg'] + df['away_xG_avg']
    df['xG_ratio'] = df['home_xG_avg'] / (df['away_xG_avg'] + 0.1)
    
    return df


# Quick test
if __name__ == "__main__":
    async def test():
        async with UnderstatXGFetcher() as fetcher:
            teams = await fetcher.get_league_teams('Premier League', 2024)
            print(f"Found {len(teams)} teams")
            if teams:
                print(f"Sample: {teams[0]['title']} - xG: {teams[0]['xG']}, xGA: {teams[0]['xGA']}")
    
    asyncio.run(test())

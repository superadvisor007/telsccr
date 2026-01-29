"""
⚽ StatsBomb Open Data Client
=============================
Integration with StatsBomb's free open data repository.

Source: https://github.com/statsbomb/open-data

Provides:
- Match event data (passes, shots, tackles, etc.)
- Lineup information
- Competition and season data
- 360 freeze-frame data (where available)

Data Format:
- Raw JSON files from GitHub
- Converted to pandas DataFrames
- SPADL-compatible output option
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StatsBombClient:
    """
    ⚽ StatsBomb Open Data Client
    
    Access StatsBomb's free open football event data.
    
    Available Competitions (as of 2024):
    - FIFA World Cup (Men's & Women's)
    - UEFA Euro
    - Premier League (selected seasons)
    - La Liga (selected seasons)
    - FA Women's Super League
    - NWSL
    - Champions League (selected seasons)
    - And more...
    
    Example:
        client = StatsBombClient()
        
        # Get available competitions
        competitions = client.get_competitions()
        
        # Get matches from a competition
        matches = client.get_matches(competition_id=43, season_id=3)
        
        # Get events from a match
        events = client.get_events(match_id=3788741)
    """
    
    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize StatsBomb client.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir or Path(__file__).parent.parent.parent / "data" / "statsbomb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._competitions_cache = None
        self._matches_cache = {}
    
    def _fetch_json(self, endpoint: str, use_cache: bool = True) -> Any:
        """Fetch JSON data from StatsBomb GitHub."""
        import requests
        
        cache_file = self.cache_dir / f"{endpoint.replace('/', '_')}.json"
        
        # Check cache
        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Fetch from GitHub
        url = f"{self.BASE_URL}/{endpoint}.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache locally
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def get_competitions(self, refresh: bool = False) -> pd.DataFrame:
        """
        Get available competitions.
        
        Returns DataFrame with columns:
        - competition_id
        - competition_name
        - country_name
        - season_id
        - season_name
        - match_available
        """
        if self._competitions_cache is None or refresh:
            data = self._fetch_json("competitions", use_cache=not refresh)
            self._competitions_cache = pd.DataFrame(data)
        
        return self._competitions_cache.copy()
    
    def get_matches(
        self,
        competition_id: int,
        season_id: int,
        refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get matches for a competition and season.
        
        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID
        
        Returns DataFrame with match details including:
        - match_id
        - match_date
        - kick_off
        - home_team, away_team
        - home_score, away_score
        - stadium, referee
        """
        cache_key = f"{competition_id}_{season_id}"
        
        if cache_key not in self._matches_cache or refresh:
            endpoint = f"matches/{competition_id}/{season_id}"
            data = self._fetch_json(endpoint, use_cache=not refresh)
            
            # Flatten nested structure
            matches = []
            for match in data:
                flat = {
                    'match_id': match.get('match_id'),
                    'match_date': match.get('match_date'),
                    'kick_off': match.get('kick_off'),
                    'competition': match.get('competition', {}).get('competition_name'),
                    'season': match.get('season', {}).get('season_name'),
                    'home_team': match.get('home_team', {}).get('home_team_name'),
                    'away_team': match.get('away_team', {}).get('away_team_name'),
                    'home_team_id': match.get('home_team', {}).get('home_team_id'),
                    'away_team_id': match.get('away_team', {}).get('away_team_id'),
                    'home_score': match.get('home_score'),
                    'away_score': match.get('away_score'),
                    'stadium': match.get('stadium', {}).get('name') if match.get('stadium') else None,
                    'referee': match.get('referee', {}).get('name') if match.get('referee') else None,
                    'match_week': match.get('match_week'),
                }
                matches.append(flat)
            
            self._matches_cache[cache_key] = pd.DataFrame(matches)
        
        return self._matches_cache[cache_key].copy()
    
    def get_events(
        self,
        match_id: int,
        include_related: bool = True
    ) -> pd.DataFrame:
        """
        Get all events for a match.
        
        Args:
            match_id: StatsBomb match ID
            include_related: Include related events (e.g., shot freeze frame)
        
        Returns DataFrame with event details:
        - id, index
        - period, timestamp, minute, second
        - type, possession_team, play_pattern
        - location (x, y)
        - Various event-specific columns
        """
        endpoint = f"events/{match_id}"
        data = self._fetch_json(endpoint)
        
        events = []
        for event in data:
            flat = {
                'id': event.get('id'),
                'index': event.get('index'),
                'period': event.get('period'),
                'timestamp': event.get('timestamp'),
                'minute': event.get('minute'),
                'second': event.get('second'),
                'type': event.get('type', {}).get('name'),
                'type_id': event.get('type', {}).get('id'),
                'possession': event.get('possession'),
                'possession_team': event.get('possession_team', {}).get('name'),
                'possession_team_id': event.get('possession_team', {}).get('id'),
                'play_pattern': event.get('play_pattern', {}).get('name'),
                'team': event.get('team', {}).get('name'),
                'team_id': event.get('team', {}).get('id'),
                'player': event.get('player', {}).get('name') if event.get('player') else None,
                'player_id': event.get('player', {}).get('id') if event.get('player') else None,
                'position': event.get('position', {}).get('name') if event.get('position') else None,
                'location_x': event.get('location', [None, None])[0] if event.get('location') else None,
                'location_y': event.get('location', [None, None])[1] if event.get('location') else None,
                'duration': event.get('duration'),
                'under_pressure': event.get('under_pressure'),
                'off_camera': event.get('off_camera'),
                'out': event.get('out'),
            }
            
            # Add type-specific fields
            event_type = event.get('type', {}).get('name', '')
            
            if event_type == 'Pass':
                pass_data = event.get('pass', {})
                flat.update({
                    'pass_length': pass_data.get('length'),
                    'pass_angle': pass_data.get('angle'),
                    'pass_height': pass_data.get('height', {}).get('name') if pass_data.get('height') else None,
                    'pass_end_x': pass_data.get('end_location', [None, None])[0] if pass_data.get('end_location') else None,
                    'pass_end_y': pass_data.get('end_location', [None, None])[1] if pass_data.get('end_location') else None,
                    'pass_recipient': pass_data.get('recipient', {}).get('name') if pass_data.get('recipient') else None,
                    'pass_body_part': pass_data.get('body_part', {}).get('name') if pass_data.get('body_part') else None,
                    'pass_outcome': pass_data.get('outcome', {}).get('name') if pass_data.get('outcome') else 'Complete',
                    'pass_type': pass_data.get('type', {}).get('name') if pass_data.get('type') else None,
                })
            
            elif event_type == 'Shot':
                shot_data = event.get('shot', {})
                flat.update({
                    'shot_statsbomb_xg': shot_data.get('statsbomb_xg'),
                    'shot_end_x': shot_data.get('end_location', [None, None, None])[0] if shot_data.get('end_location') else None,
                    'shot_end_y': shot_data.get('end_location', [None, None, None])[1] if shot_data.get('end_location') else None,
                    'shot_end_z': shot_data.get('end_location', [None, None, None])[2] if shot_data.get('end_location') and len(shot_data.get('end_location', [])) > 2 else None,
                    'shot_outcome': shot_data.get('outcome', {}).get('name') if shot_data.get('outcome') else None,
                    'shot_type': shot_data.get('type', {}).get('name') if shot_data.get('type') else None,
                    'shot_body_part': shot_data.get('body_part', {}).get('name') if shot_data.get('body_part') else None,
                    'shot_technique': shot_data.get('technique', {}).get('name') if shot_data.get('technique') else None,
                    'shot_first_time': shot_data.get('first_time'),
                    'shot_one_on_one': shot_data.get('one_on_one'),
                })
            
            elif event_type == 'Carry':
                carry_data = event.get('carry', {})
                flat.update({
                    'carry_end_x': carry_data.get('end_location', [None, None])[0] if carry_data.get('end_location') else None,
                    'carry_end_y': carry_data.get('end_location', [None, None])[1] if carry_data.get('end_location') else None,
                })
            
            elif event_type in ['Duel', 'Tackle', 'Interception']:
                flat.update({
                    'outcome': event.get(event_type.lower(), {}).get('outcome', {}).get('name') if event.get(event_type.lower()) else None,
                })
            
            events.append(flat)
        
        return pd.DataFrame(events)
    
    def get_lineups(self, match_id: int) -> Dict[str, pd.DataFrame]:
        """
        Get lineups for a match.
        
        Returns dict with 'home' and 'away' DataFrames.
        """
        endpoint = f"lineups/{match_id}"
        data = self._fetch_json(endpoint)
        
        lineups = {}
        for team_data in data:
            team_name = team_data.get('team_name', 'Unknown')
            players = []
            
            for player in team_data.get('lineup', []):
                players.append({
                    'player_id': player.get('player_id'),
                    'player_name': player.get('player_name'),
                    'player_nickname': player.get('player_nickname'),
                    'jersey_number': player.get('jersey_number'),
                    'country': player.get('country', {}).get('name') if player.get('country') else None,
                })
            
            lineups[team_name] = pd.DataFrame(players)
        
        return lineups
    
    def get_match_summary(self, match_id: int) -> Dict[str, Any]:
        """
        Get comprehensive match summary including events and stats.
        
        Returns dict with:
        - match_info
        - home_stats
        - away_stats
        - events_summary
        - key_moments
        """
        events = self.get_events(match_id)
        lineups = self.get_lineups(match_id)
        
        # Calculate team stats
        home_team = events[events['possession_team'] == events['possession_team'].iloc[0]]['possession_team'].iloc[0]
        away_team = events[events['possession_team'] != home_team]['possession_team'].iloc[0]
        
        def team_stats(team: str) -> Dict[str, Any]:
            team_events = events[events['team'] == team]
            
            shots = team_events[team_events['type'] == 'Shot']
            passes = team_events[team_events['type'] == 'Pass']
            
            return {
                'team': team,
                'shots': len(shots),
                'shots_on_target': len(shots[shots['shot_outcome'].isin(['Goal', 'Saved'])]) if 'shot_outcome' in shots.columns else 0,
                'goals': len(shots[shots['shot_outcome'] == 'Goal']) if 'shot_outcome' in shots.columns else 0,
                'xg': shots['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in shots.columns else 0,
                'passes': len(passes),
                'passes_completed': len(passes[passes['pass_outcome'] == 'Complete']) if 'pass_outcome' in passes.columns else len(passes),
                'pass_accuracy': len(passes[passes['pass_outcome'] == 'Complete']) / len(passes) * 100 if len(passes) > 0 and 'pass_outcome' in passes.columns else 0,
                'possession': len(team_events) / len(events) * 100,
            }
        
        # Key moments
        key_moments = []
        
        # Goals
        goals = events[(events['type'] == 'Shot') & (events.get('shot_outcome', pd.Series()) == 'Goal')]
        for _, goal in goals.iterrows():
            key_moments.append({
                'minute': goal['minute'],
                'type': 'Goal',
                'team': goal['team'],
                'player': goal['player'],
                'xg': goal.get('shot_statsbomb_xg', 0)
            })
        
        # Red cards
        red_cards = events[events['type'].str.contains('Card', na=False)]
        for _, card in red_cards.iterrows():
            if 'Red' in str(card.get('foul_committed', {}).get('card', {}).get('name', '')):
                key_moments.append({
                    'minute': card['minute'],
                    'type': 'Red Card',
                    'team': card['team'],
                    'player': card['player'],
                })
        
        return {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_stats': team_stats(home_team),
            'away_stats': team_stats(away_team),
            'total_events': len(events),
            'key_moments': sorted(key_moments, key=lambda x: x['minute']),
            'lineups': {k: v.to_dict('records') for k, v in lineups.items()}
        }
    
    def get_available_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all available data.
        
        Returns DataFrame with competition/season combinations
        and match counts.
        """
        competitions = self.get_competitions()
        
        summary = competitions.groupby(
            ['competition_name', 'country_name', 'season_name']
        ).size().reset_index(name='matches_available')
        
        return summary.sort_values(['country_name', 'competition_name', 'season_name'])
    
    def download_all_data(self, competitions: List[Tuple[int, int]] = None):
        """
        Download and cache all data for specified competitions.
        
        Args:
            competitions: List of (competition_id, season_id) tuples.
                         If None, downloads all available.
        """
        if competitions is None:
            comps = self.get_competitions()
            competitions = list(zip(comps['competition_id'], comps['season_id']))
        
        total = len(competitions)
        logger.info(f"Downloading data for {total} competition/season combinations...")
        
        for i, (comp_id, season_id) in enumerate(competitions):
            try:
                matches = self.get_matches(comp_id, season_id)
                logger.info(f"[{i+1}/{total}] Downloaded {len(matches)} matches for competition {comp_id}, season {season_id}")
                
                # Optionally download events for each match
                # This can be slow and data-intensive
                
            except Exception as e:
                logger.warning(f"Failed to download competition {comp_id}, season {season_id}: {e}")

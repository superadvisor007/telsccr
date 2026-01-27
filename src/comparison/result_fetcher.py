"""
🔄 Multi-Source Result Fetcher
==============================

Collects match results from multiple free APIs for redundancy:
- Football-Data.org (best quality, rate limited)
- OpenLigaDB (Bundesliga focus, no key)
- TheSportsDB (free tier, global coverage)

Uses fuzzy team name matching and deduplication.
"""

import requests
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Verified match result from API"""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    league: str
    source: str
    match_id: Optional[str] = None
    
    # Computed outcomes
    total_goals: int = field(init=False)
    over_1_5: bool = field(init=False)
    over_2_5: bool = field(init=False)
    over_3_5: bool = field(init=False)
    btts: bool = field(init=False)
    home_win: bool = field(init=False)
    away_win: bool = field(init=False)
    draw: bool = field(init=False)
    
    def __post_init__(self):
        self.total_goals = self.home_goals + self.away_goals
        self.over_1_5 = self.total_goals > 1.5
        self.over_2_5 = self.total_goals > 2.5
        self.over_3_5 = self.total_goals > 3.5
        self.btts = self.home_goals > 0 and self.away_goals > 0
        self.home_win = self.home_goals > self.away_goals
        self.away_win = self.away_goals > self.home_goals
        self.draw = self.home_goals == self.away_goals
    
    def get_outcome(self, market: str) -> bool:
        """Get outcome for a specific market"""
        market_map = {
            'over_1_5': self.over_1_5,
            'over_2_5': self.over_2_5,
            'over_3_5': self.over_3_5,
            'btts': self.btts,
            'home_win': self.home_win,
            'away_win': self.away_win,
            'draw': self.draw,
            '1x2_home': self.home_win,
            '1x2_away': self.away_win,
            '1x2_draw': self.draw,
        }
        return market_map.get(market.lower().replace(' ', '_'), False)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TeamNameNormalizer:
    """Normalize team names for fuzzy matching across sources"""
    
    # Common variations mapping
    TEAM_ALIASES = {
        # English
        'manchester united': ['man utd', 'man united', 'mufc'],
        'manchester city': ['man city', 'mcfc'],
        'tottenham hotspur': ['tottenham', 'spurs', 'thfc'],
        'wolverhampton wanderers': ['wolves', 'wolverhampton'],
        'brighton hove albion': ['brighton', 'bha'],
        'west ham united': ['west ham', 'whu'],
        'newcastle united': ['newcastle', 'nufc'],
        'nottingham forest': ['forest', 'nffc'],
        
        # German
        'bayern munich': ['bayern münchen', 'fc bayern', 'bayern'],
        'borussia dortmund': ['dortmund', 'bvb'],
        'bayer leverkusen': ['leverkusen', 'bayer 04'],
        'rb leipzig': ['rasenballsport leipzig', 'leipzig'],
        'eintracht frankfurt': ['frankfurt', 'sge'],
        'borussia monchengladbach': ['gladbach', 'mönchengladbach', 'bmg'],
        'vfb stuttgart': ['stuttgart', 'vfb'],
        'fc koln': ['köln', 'cologne', 'fc cologne'],
        
        # Spanish
        'real madrid': ['real madrid cf', 'rmcf'],
        'barcelona': ['fc barcelona', 'barca', 'fcb'],
        'atletico madrid': ['atlético madrid', 'atletico', 'atm'],
        'real betis': ['betis', 'real betis balompie'],
        'athletic bilbao': ['athletic club', 'bilbao'],
        'real sociedad': ['sociedad', 'la real'],
        
        # Italian
        'inter milan': ['inter', 'internazionale', 'fc internazionale'],
        'ac milan': ['milan', 'acm'],
        'juventus': ['juve', 'juventus fc'],
        'napoli': ['ssc napoli', 'ssc'],
        'as roma': ['roma', 'asr'],
        'lazio': ['ss lazio', 'ssl'],
        
        # French
        'paris saint-germain': ['psg', 'paris sg', 'paris'],
        'olympique marseille': ['marseille', 'om'],
        'olympique lyon': ['lyon', 'ol'],
        'as monaco': ['monaco', 'asm'],
        
        # Dutch
        'ajax amsterdam': ['ajax', 'afc ajax'],
        'psv eindhoven': ['psv'],
        'feyenoord rotterdam': ['feyenoord'],
    }
    
    @classmethod
    def normalize(cls, name: str) -> str:
        """Normalize team name for matching"""
        if not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(fc|cf|sc|ac|as|ss|afc|ssc|1\.|)\b', '', name)
        name = re.sub(r'\b(united|city|town|rovers|wanderers|athletic|sporting)\b', '', name)
        
        # Remove special characters
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    @classmethod
    def match_teams(cls, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Check if two team names refer to the same team"""
        norm1 = cls.normalize(name1)
        norm2 = cls.normalize(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Check aliases
        for canonical, aliases in cls.TEAM_ALIASES.items():
            canonical_norm = cls.normalize(canonical)
            all_names = [canonical_norm] + [cls.normalize(a) for a in aliases]
            
            if norm1 in all_names and norm2 in all_names:
                return True
            
            # Partial match
            for n in all_names:
                if n in norm1 or norm1 in n:
                    for m in all_names:
                        if m in norm2 or norm2 in m:
                            return True
        
        # Fuzzy match
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold


class MultiSourceResultFetcher:
    """
    Fetch match results from multiple APIs for redundancy.
    
    Priority order:
    1. Football-Data.org (most reliable)
    2. OpenLigaDB (Bundesliga)
    3. TheSportsDB (backup)
    """
    
    def __init__(self, football_data_key: Optional[str] = None):
        self.football_data_key = football_data_key or os.getenv('FOOTBALL_DATA_API_KEY')
        self.normalizer = TeamNameNormalizer()
        
        # API endpoints
        self.endpoints = {
            'football_data': 'https://api.football-data.org/v4',
            'openligadb': 'https://api.openligadb.de',
            'thesportsdb': 'https://www.thesportsdb.com/api/v1/json/3'
        }
        
        # League mappings
        self.league_ids = {
            'football_data': {
                'Premier League': 'PL',
                'Bundesliga': 'BL1',
                'La Liga': 'PD',
                'Serie A': 'SA',
                'Ligue 1': 'FL1',
                'Eredivisie': 'DED',
                'Championship': 'ELC'
            },
            'openligadb': {
                'Bundesliga': 'bl1',
                'Bundesliga 2': 'bl2'
            },
            'thesportsdb': {
                'Premier League': 4328,
                'Bundesliga': 4331,
                'La Liga': 4335,
                'Serie A': 4332,
                'Ligue 1': 4334,
                'Eredivisie': 4337
            }
        }
    
    def fetch_results(
        self,
        date: str,
        leagues: Optional[List[str]] = None
    ) -> List[MatchResult]:
        """
        Fetch results from all available sources and deduplicate.
        
        Args:
            date: Date string in YYYY-MM-DD format
            leagues: Optional list of league names to filter
        
        Returns:
            Deduplicated list of MatchResult objects
        """
        all_results = []
        
        # 1. Football-Data.org (best quality)
        if self.football_data_key:
            logger.info("Fetching from Football-Data.org...")
            results = self._fetch_football_data(date, leagues)
            logger.info(f"  Found {len(results)} matches")
            all_results.extend(results)
        
        # 2. OpenLigaDB (Bundesliga)
        logger.info("Fetching from OpenLigaDB...")
        results = self._fetch_openligadb(date)
        logger.info(f"  Found {len(results)} matches")
        all_results.extend(results)
        
        # 3. TheSportsDB (backup)
        logger.info("Fetching from TheSportsDB...")
        results = self._fetch_thesportsdb(date, leagues)
        logger.info(f"  Found {len(results)} matches")
        all_results.extend(results)
        
        # Deduplicate
        unique_results = self._deduplicate(all_results)
        logger.info(f"Total unique matches: {len(unique_results)}")
        
        return unique_results
    
    def fetch_results_range(
        self,
        start_date: str,
        end_date: str,
        leagues: Optional[List[str]] = None
    ) -> List[MatchResult]:
        """Fetch results for a date range"""
        all_results = []
        
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            results = self.fetch_results(date_str, leagues)
            all_results.extend(results)
            current += timedelta(days=1)
        
        return all_results
    
    def _fetch_football_data(self, date: str, leagues: Optional[List[str]]) -> List[MatchResult]:
        """Fetch from Football-Data.org"""
        results = []
        
        try:
            response = requests.get(
                f"{self.endpoints['football_data']}/matches",
                headers={'X-Auth-Token': self.football_data_key},
                params={
                    'dateFrom': date,
                    'dateTo': date,
                    'status': 'FINISHED'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                for m in data.get('matches', []):
                    # Filter by league if specified
                    league_name = m['competition']['name']
                    if leagues and not any(l.lower() in league_name.lower() for l in leagues):
                        continue
                    
                    score = m.get('score', {}).get('fullTime', {})
                    if score.get('home') is not None and score.get('away') is not None:
                        results.append(MatchResult(
                            date=m['utcDate'][:10],
                            home_team=m['homeTeam']['name'],
                            away_team=m['awayTeam']['name'],
                            home_goals=score['home'],
                            away_goals=score['away'],
                            league=league_name,
                            source='football-data.org',
                            match_id=str(m['id'])
                        ))
            elif response.status_code == 429:
                logger.warning("Football-Data.org rate limit reached")
            else:
                logger.warning(f"Football-Data.org error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Football-Data.org exception: {e}")
        
        return results
    
    def _fetch_openligadb(self, date: str) -> List[MatchResult]:
        """Fetch from OpenLigaDB (Bundesliga)"""
        results = []
        
        try:
            # Determine season
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            year = date_obj.year if date_obj.month >= 7 else date_obj.year - 1
            
            for league_code in ['bl1', 'bl2']:
                response = requests.get(
                    f"{self.endpoints['openligadb']}/getmatchdata/{league_code}/{year}",
                    timeout=15
                )
                
                if response.status_code == 200:
                    matches = response.json()
                    for m in matches:
                        match_date = m.get('matchDateTime', '')[:10]
                        if match_date != date:
                            continue
                        
                        if not m.get('matchIsFinished'):
                            continue
                        
                        match_results = m.get('matchResults', [])
                        if not match_results:
                            continue
                        
                        # Get final result (last in list)
                        final = match_results[-1]
                        
                        league_name = 'Bundesliga' if league_code == 'bl1' else 'Bundesliga 2'
                        
                        results.append(MatchResult(
                            date=match_date,
                            home_team=m['team1']['teamName'],
                            away_team=m['team2']['teamName'],
                            home_goals=final['pointsTeam1'],
                            away_goals=final['pointsTeam2'],
                            league=league_name,
                            source='openligadb',
                            match_id=str(m.get('matchID'))
                        ))
                        
        except Exception as e:
            logger.error(f"OpenLigaDB exception: {e}")
        
        return results
    
    def _fetch_thesportsdb(self, date: str, leagues: Optional[List[str]]) -> List[MatchResult]:
        """Fetch from TheSportsDB"""
        results = []
        
        try:
            response = requests.get(
                f"{self.endpoints['thesportsdb']}/eventsday.php",
                params={'d': date, 's': 'Soccer'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events') or []
                
                for e in events:
                    # Filter by league if specified
                    league_name = e.get('strLeague', '')
                    if leagues and not any(l.lower() in league_name.lower() for l in leagues):
                        continue
                    
                    home_score = e.get('intHomeScore')
                    away_score = e.get('intAwayScore')
                    
                    # Only include finished matches
                    if home_score is None or away_score is None:
                        continue
                    
                    try:
                        results.append(MatchResult(
                            date=e['dateEvent'],
                            home_team=e['strHomeTeam'],
                            away_team=e['strAwayTeam'],
                            home_goals=int(home_score),
                            away_goals=int(away_score),
                            league=league_name,
                            source='thesportsdb',
                            match_id=e.get('idEvent')
                        ))
                    except (ValueError, TypeError):
                        continue
                        
        except Exception as e:
            logger.error(f"TheSportsDB exception: {e}")
        
        return results
    
    def _deduplicate(self, results: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate matches using fuzzy team name matching"""
        seen = set()
        unique = []
        
        # Sort by source priority (football-data > openligadb > thesportsdb)
        source_priority = {'football-data.org': 0, 'openligadb': 1, 'thesportsdb': 2}
        results_sorted = sorted(results, key=lambda r: source_priority.get(r.source, 3))
        
        for r in results_sorted:
            home_norm = self.normalizer.normalize(r.home_team)
            away_norm = self.normalizer.normalize(r.away_team)
            
            # Create a key for deduplication
            key = (home_norm[:10], away_norm[:10], r.date)
            
            # Check for similar existing matches
            is_duplicate = False
            for existing_key in seen:
                if (r.date == existing_key[2] and
                    self.normalizer.match_teams(home_norm, existing_key[0]) and
                    self.normalizer.match_teams(away_norm, existing_key[1])):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(key)
                unique.append(r)
        
        return unique
    
    def find_result(
        self,
        home_team: str,
        away_team: str,
        date: str,
        results: Optional[List[MatchResult]] = None
    ) -> Optional[MatchResult]:
        """
        Find a specific match result using fuzzy matching.
        
        Args:
            home_team: Home team name (can be approximate)
            away_team: Away team name (can be approximate)
            date: Match date (YYYY-MM-DD)
            results: Optional pre-fetched results list
        
        Returns:
            MatchResult if found, None otherwise
        """
        if results is None:
            results = self.fetch_results(date)
        
        for r in results:
            if r.date != date:
                continue
            
            if (self.normalizer.match_teams(r.home_team, home_team) and
                self.normalizer.match_teams(r.away_team, away_team)):
                return r
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test the result fetcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch match results')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='Date to fetch (YYYY-MM-DD)')
    parser.add_argument('--leagues', type=str, nargs='+', help='Leagues to filter')
    args = parser.parse_args()
    
    fetcher = MultiSourceResultFetcher()
    results = fetcher.fetch_results(args.date, args.leagues)
    
    print(f"\n📊 Results for {args.date}")
    print("=" * 60)
    
    for r in results:
        goals_emoji = "⚽" * min(r.total_goals, 6)
        btts_emoji = "✅" if r.btts else "❌"
        print(f"{r.home_team} {r.home_goals}-{r.away_goals} {r.away_team}")
        print(f"   {goals_emoji} | BTTS: {btts_emoji} | {r.league} [{r.source}]")
    
    print(f"\nTotal: {len(results)} matches")


if __name__ == '__main__':
    main()

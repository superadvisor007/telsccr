#!/usr/bin/env python3
"""
🔄 AUTOMATIC RESULT COLLECTOR
=============================
Collects match results automatically for self-training:

1. Fetches results from free APIs (OpenLigaDB, TheSportsDB)
2. Matches with stored predictions
3. Verifies predictions automatically
4. Updates training metrics

This enables TRUE 24/7 learning!
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AutoResultCollector:
    """
    🔄 Automatic Result Collection
    
    Workflow:
    1. Check for matches that ended yesterday
    2. Fetch results from multiple APIs
    3. Match with stored predictions
    4. Calculate win/loss for each prediction
    """
    
    def __init__(self):
        self.cache_dir = Path("data/results_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints (all FREE)
        self.openliga_url = "https://api.openligadb.de"
        self.sportsdb_url = "https://www.thesportsdb.com/api/v1/json/3"
        
        # League mapping
        self.openliga_leagues = {
            'Bundesliga': 'bl1',
            '2. Bundesliga': 'bl2',
        }
        
        self.sportsdb_leagues = {
            'Premier League': '4328',
            'La Liga': '4335',
            'Serie A': '4332',
            'Ligue 1': '4334',
            'Eredivisie': '4337',
            'Championship': '4396',
        }
    
    def fetch_yesterdays_results(self) -> List[Dict]:
        """Fetch all results from yesterday"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        all_results = []
        
        # Fetch Bundesliga from OpenLigaDB
        for league_name, league_code in self.openliga_leagues.items():
            results = self._fetch_openliga_results(league_code, yesterday)
            for r in results:
                r['league'] = league_name
            all_results.extend(results)
        
        # Fetch other leagues from TheSportsDB
        for league_name, league_id in self.sportsdb_leagues.items():
            results = self._fetch_sportsdb_results(league_id, yesterday)
            for r in results:
                r['league'] = league_name
            all_results.extend(results)
        
        return all_results
    
    def _fetch_openliga_results(self, league_code: str, date: str) -> List[Dict]:
        """Fetch results from OpenLigaDB"""
        results = []
        
        try:
            # Get current season matches
            season = datetime.now().year
            if datetime.now().month < 7:
                season -= 1
            
            url = f"{self.openliga_url}/getmatchdata/{league_code}/{season}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                matches = resp.json()
                
                for match in matches:
                    # Check if match was on target date
                    match_date = match.get('matchDateTime', '')[:10]
                    if match_date != date:
                        continue
                    
                    # Check if match is finished
                    if not match.get('matchIsFinished', False):
                        continue
                    
                    # Get final result
                    final_result = None
                    for result in match.get('matchResults', []):
                        if result.get('resultTypeID') == 2:  # Final result
                            final_result = result
                            break
                    
                    if final_result:
                        results.append({
                            'home_team': match.get('team1', {}).get('teamName', ''),
                            'away_team': match.get('team2', {}).get('teamName', ''),
                            'home_goals': final_result.get('pointsTeam1', 0),
                            'away_goals': final_result.get('pointsTeam2', 0),
                            'date': match_date,
                            'source': 'OpenLigaDB'
                        })
        
        except Exception as e:
            print(f"   ⚠️  OpenLigaDB error: {e}")
        
        return results
    
    def _fetch_sportsdb_results(self, league_id: str, date: str) -> List[Dict]:
        """Fetch results from TheSportsDB"""
        results = []
        
        try:
            url = f"{self.sportsdb_url}/eventspastleague.php?id={league_id}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                events = data.get('events', []) or []
                
                for event in events:
                    event_date = event.get('dateEvent', '')
                    if event_date != date:
                        continue
                    
                    home_score = event.get('intHomeScore')
                    away_score = event.get('intAwayScore')
                    
                    if home_score is not None and away_score is not None:
                        results.append({
                            'home_team': event.get('strHomeTeam', ''),
                            'away_team': event.get('strAwayTeam', ''),
                            'home_goals': int(home_score),
                            'away_goals': int(away_score),
                            'date': event_date,
                            'source': 'TheSportsDB'
                        })
        
        except Exception as e:
            print(f"   ⚠️  TheSportsDB error: {e}")
        
        return results
    
    def verify_predictions(self, predictions_file: str = "data/self_training/predictions.json") -> Dict:
        """
        Verify stored predictions against actual results.
        
        Returns summary of verification.
        """
        predictions_path = Path(predictions_file)
        if not predictions_path.exists():
            return {'status': 'no_predictions', 'verified': 0}
        
        # Load predictions
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        
        # Fetch results
        results = self.fetch_yesterdays_results()
        
        print(f"\n📊 Found {len(results)} match results")
        
        verified_count = 0
        correct_count = 0
        profit = 0.0
        
        for pred_id, pred in predictions.items():
            # Skip already verified
            if pred.get('verified_at'):
                continue
            
            # Try to find matching result
            for result in results:
                if (self._teams_match(pred['home_team'], result['home_team']) and
                    self._teams_match(pred['away_team'], result['away_team'])):
                    
                    # Verify prediction
                    home_goals = result['home_goals']
                    away_goals = result['away_goals']
                    
                    outcome = self._determine_outcome(
                        pred['market'], home_goals, away_goals
                    )
                    
                    # Update prediction
                    pred['actual_home_goals'] = home_goals
                    pred['actual_away_goals'] = away_goals
                    pred['actual_outcome'] = outcome
                    pred['was_correct'] = outcome
                    pred['verified_at'] = datetime.now().isoformat()
                    
                    # Calculate profit/loss
                    if outcome:
                        pred['profit_loss'] = (pred['market_odds'] - 1) * pred['kelly_stake']
                        correct_count += 1
                    else:
                        pred['profit_loss'] = -pred['kelly_stake']
                    
                    profit += pred['profit_loss']
                    verified_count += 1
                    
                    status = "✅" if outcome else "❌"
                    print(f"   {status} {pred['home_team']} vs {pred['away_team']} ({pred['market']})")
                    print(f"      Score: {home_goals}-{away_goals} | P/L: {pred['profit_loss']:+.2%}")
                    
                    break
        
        # Save updated predictions
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        win_rate = correct_count / verified_count if verified_count > 0 else 0
        
        return {
            'status': 'success',
            'verified': verified_count,
            'correct': correct_count,
            'win_rate': win_rate,
            'profit': profit
        }
    
    def _teams_match(self, name1: str, name2: str) -> bool:
        """Fuzzy match team names"""
        n1 = name1.lower().replace(' fc', '').replace(' cf', '').strip()
        n2 = name2.lower().replace(' fc', '').replace(' cf', '').strip()
        
        # Direct match
        if n1 == n2:
            return True
        
        # Partial match
        if n1 in n2 or n2 in n1:
            return True
        
        # Common abbreviations
        abbrevs = {
            'bayern münchen': ['bayern', 'fc bayern'],
            'borussia dortmund': ['dortmund', 'bvb'],
            'manchester united': ['man utd', 'man united'],
            'manchester city': ['man city'],
            'tottenham': ['spurs', 'tottenham hotspur'],
            'psg': ['paris saint-germain', 'paris sg'],
        }
        
        for full, abbrs in abbrevs.items():
            if n1 == full or n1 in abbrs:
                if n2 == full or n2 in abbrs:
                    return True
        
        return False
    
    def _determine_outcome(self, market: str, home_goals: int, away_goals: int) -> bool:
        """Determine if prediction was correct"""
        total = home_goals + away_goals
        both_scored = home_goals > 0 and away_goals > 0
        
        outcomes = {
            'over_0_5': total > 0,
            'over_1_5': total > 1,
            'over_2_5': total > 2,
            'over_3_5': total > 3,
            'under_0_5': total == 0,
            'under_1_5': total < 2,
            'under_2_5': total < 3,
            'under_3_5': total < 4,
            'btts_yes': both_scored,
            'btts_no': not both_scored,
            'home_win': home_goals > away_goals,
            'draw': home_goals == away_goals,
            'away_win': home_goals < away_goals,
            'home_or_draw': home_goals >= away_goals,
            'away_or_draw': home_goals <= away_goals,
        }
        
        return outcomes.get(market, False)
    
    def run_daily_verification(self) -> str:
        """Run daily verification and return report"""
        print("\n" + "=" * 60)
        print("🔄 DAILY RESULT VERIFICATION")
        print("=" * 60)
        
        result = self.verify_predictions()
        
        report = []
        report.append(f"\n📊 VERIFICATION REPORT")
        report.append(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"   Status: {result['status']}")
        
        if result['status'] == 'success':
            report.append(f"   Verified: {result['verified']} predictions")
            report.append(f"   Correct: {result['correct']}")
            report.append(f"   Win Rate: {result['win_rate']:.1%}")
            report.append(f"   Profit: {result['profit']:+.2%}")
        
        return '\n'.join(report)


def test_result_collector():
    """Test the result collector"""
    print("=" * 60)
    print("🔄 AUTO RESULT COLLECTOR TEST")
    print("=" * 60)
    
    collector = AutoResultCollector()
    
    # Fetch yesterday's results
    results = collector.fetch_yesterdays_results()
    
    print(f"\n📊 Found {len(results)} results from yesterday:")
    for r in results[:10]:  # Show first 10
        print(f"   {r['home_team']} {r['home_goals']}-{r['away_goals']} {r['away_team']} ({r['league']})")
    
    if len(results) > 10:
        print(f"   ... and {len(results) - 10} more")
    
    # Run verification
    print(collector.run_daily_verification())
    
    print("\n✅ Result collector working!")


if __name__ == "__main__":
    test_result_collector()

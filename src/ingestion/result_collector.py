"""
Automated Result Verification System
Collects match results from multiple free APIs and verifies predictions
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json

class ResultCollector:
    """
    Multi-API result collection system
    Sources: Football-Data.org, OpenLigaDB, TheSportsDB
    """
    
    def __init__(self, football_data_key: Optional[str] = None):
        self.football_data_key = football_data_key
        self.results_cache = {}
    
    def collect_results_for_date(self, date: str, leagues: List[str] = None) -> List[Dict]:
        """
        Collect all match results for a specific date
        
        Args:
            date: Date string 'YYYY-MM-DD'
            leagues: List of league names to filter (None = all)
        
        Returns:
            List of match results with scores
        """
        print(f"\n🔍 Collecting results for {date}...")
        
        all_results = []
        
        # Try multiple sources
        fd_results = self._collect_from_football_data(date, leagues)
        all_results.extend(fd_results)
        
        openliga_results = self._collect_from_openligadb(date)
        all_results.extend(openliga_results)
        
        thesportsdb_results = self._collect_from_thesportsdb(date, leagues)
        all_results.extend(thesportsdb_results)
        
        # Deduplicate
        unique_results = self._deduplicate_results(all_results)
        
        print(f"✅ Found {len(unique_results)} unique match results")
        return unique_results
    
    def _collect_from_football_data(self, date: str, leagues: List[str]) -> List[Dict]:
        """Collect from Football-Data.org API"""
        if not self.football_data_key:
            return []
        
        print("  📡 Querying Football-Data.org...")
        
        results = []
        url = "https://api.football-data.org/v4/matches"
        
        headers = {'X-Auth-Token': self.football_data_key}
        params = {
            'dateFrom': date,
            'dateTo': date,
            'status': 'FINISHED'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for match in data.get('matches', []):
                    league_name = match['competition']['name']
                    
                    # Filter by league if specified
                    if leagues and league_name not in leagues:
                        continue
                    
                    result = {
                        'date': match['utcDate'][:10],
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'home_score': match['score']['fullTime']['home'],
                        'away_score': match['score']['fullTime']['away'],
                        'league': league_name,
                        'source': 'football-data.org'
                    }
                    results.append(result)
                
                print(f"    ✅ Found {len(results)} matches")
            
            elif response.status_code == 429:
                print(f"    ⚠️  Rate limit exceeded")
            else:
                print(f"    ⚠️  Status {response.status_code}")
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        return results
    
    def _collect_from_openligadb(self, date: str) -> List[Dict]:
        """Collect from OpenLigaDB (Bundesliga only)"""
        print("  📡 Querying OpenLigaDB...")
        
        results = []
        
        # Determine season year from date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        year = date_obj.year if date_obj.month >= 7 else date_obj.year - 1
        
        url = f"https://api.openligadb.de/getmatchdata/bl1/{year}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                matches = response.json()
                
                for match in matches:
                    match_date = match['matchDateTime'][:10]
                    
                    if match_date == date and match.get('matchIsFinished'):
                        result = {
                            'date': match_date,
                            'home_team': match['team1']['teamName'],
                            'away_team': match['team2']['teamName'],
                            'home_score': match['matchResults'][-1]['pointsTeam1'],
                            'away_score': match['matchResults'][-1]['pointsTeam2'],
                            'league': 'Bundesliga',
                            'source': 'openligadb'
                        }
                        results.append(result)
                
                print(f"    ✅ Found {len(results)} Bundesliga matches")
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        return results
    
    def _collect_from_thesportsdb(self, date: str, leagues: List[str]) -> List[Dict]:
        """Collect from TheSportsDB (limited but free)"""
        print("  📡 Querying TheSportsDB...")
        
        results = []
        
        # TheSportsDB league IDs
        league_ids = {
            'Premier League': '4328',
            'Bundesliga': '4331',
            'La Liga': '4335',
            'Serie A': '4332',
            'Ligue 1': '4334'
        }
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        season = f"{date_obj.year}-{date_obj.year + 1}" if date_obj.month >= 7 else f"{date_obj.year - 1}-{date_obj.year}"
        
        for league_name, league_id in league_ids.items():
            if leagues and league_name not in leagues:
                continue
            
            url = f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id={league_id}&s={season}"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('events'):
                        for event in data['events']:
                            event_date = event.get('dateEvent')
                            
                            if event_date == date and event.get('intHomeScore') and event.get('intAwayScore'):
                                result = {
                                    'date': event_date,
                                    'home_team': event['strHomeTeam'],
                                    'away_team': event['strAwayTeam'],
                                    'home_score': int(event['intHomeScore']),
                                    'away_score': int(event['intAwayScore']),
                                    'league': league_name,
                                    'source': 'thesportsdb'
                                }
                                results.append(result)
                
                time.sleep(1)  # Rate limiting
            
            except Exception as e:
                print(f"    ⚠️  {league_name}: {e}")
                continue
        
        print(f"    ✅ Found {len(results)} matches")
        return results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate matches from different sources"""
        seen = set()
        unique = []
        
        for result in results:
            # Create unique key
            key = (
                result['date'],
                result['home_team'].lower().strip(),
                result['away_team'].lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(result)
        
        return unique
    
    def verify_prediction(self, prediction: Dict, results: List[Dict]) -> Optional[Dict]:
        """
        Verify a prediction against actual results
        
        Args:
            prediction: Dict with home_team, away_team, market, probability
            results: List of match results
        
        Returns:
            Dict with verification details or None if no match found
        """
        # Find matching result
        for result in results:
            if (
                self._normalize_team_name(prediction['home_team']) == self._normalize_team_name(result['home_team']) and
                self._normalize_team_name(prediction['away_team']) == self._normalize_team_name(result['away_team'])
            ):
                # Evaluate market outcome
                home_score = result['home_score']
                away_score = result['away_score']
                total = home_score + away_score
                
                market = prediction['market']
                
                if market == 'over_1_5':
                    actual_outcome = total > 1.5
                elif market == 'over_2_5':
                    actual_outcome = total > 2.5
                elif market == 'btts':
                    actual_outcome = (home_score > 0 and away_score > 0)
                elif market == 'under_1_5':
                    actual_outcome = total < 1.5
                elif market == 'under_2_5':
                    actual_outcome = total < 2.5
                else:
                    actual_outcome = None
                
                won = actual_outcome == True if actual_outcome is not None else None
                
                return {
                    'prediction': prediction,
                    'result': result,
                    'outcome': won,
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_goals': total
                }
        
        return None
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching"""
        # Remove common variations
        replacements = {
            'fc ': '',
            'f.c. ': '',
            '1. ': '',
            'borussia ': 'bor. ',
            'bayern ': 'bay. ',
            'manchester ': 'man ',
        }
        
        normalized = name.lower().strip()
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def bulk_verify_predictions(self, predictions: List[Dict], start_date: str, end_date: str) -> Dict:
        """
        Verify multiple predictions in date range
        
        Returns:
            Dict with statistics and individual results
        """
        print(f"\n🔍 Bulk Verification: {start_date} to {end_date}")
        print(f"📋 Total Predictions: {len(predictions)}")
        
        # Collect all results in date range
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_results = []
        current = date_obj
        
        while current <= end_obj:
            date_str = current.strftime('%Y-%m-%d')
            daily_results = self.collect_results_for_date(date_str)
            all_results.extend(daily_results)
            current += timedelta(days=1)
            time.sleep(2)  # Rate limiting
        
        print(f"\n✅ Collected {len(all_results)} total results")
        
        # Verify each prediction
        verified = []
        unmatched = []
        
        for pred in predictions:
            verification = self.verify_prediction(pred, all_results)
            
            if verification:
                verified.append(verification)
            else:
                unmatched.append(pred)
        
        # Calculate statistics
        if verified:
            total = len(verified)
            wins = sum(1 for v in verified if v['outcome'] == True)
            losses = sum(1 for v in verified if v['outcome'] == False)
            win_rate = wins / total
            
            stats = {
                'total_verified': total,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'unmatched': len(unmatched),
                'verified_predictions': verified,
                'unmatched_predictions': unmatched
            }
        else:
            stats = {
                'total_verified': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'unmatched': len(unmatched),
                'verified_predictions': [],
                'unmatched_predictions': unmatched
            }
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"   Verified:  {stats['total_verified']}/{len(predictions)}")
        print(f"   Wins:      {stats['wins']} ({stats['win_rate']*100:.1f}%)")
        print(f"   Losses:    {stats['losses']}")
        print(f"   Unmatched: {stats['unmatched']}")
        
        return stats


if __name__ == "__main__":
    # Test result collection
    collector = ResultCollector(football_data_key="your_optional_key")
    
    # Test for yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    results = collector.collect_results_for_date(yesterday)
    
    print(f"\n📋 Sample Results:")
    for result in results[:5]:
        print(f"   {result['home_team']} {result['home_score']}-{result['away_score']} {result['away_team']} ({result['league']})")
    
    # Test prediction verification
    if results:
        test_prediction = {
            'home_team': results[0]['home_team'],
            'away_team': results[0]['away_team'],
            'market': 'over_2_5',
            'probability': 0.65
        }
        
        verification = collector.verify_prediction(test_prediction, results)
        
        if verification:
            print(f"\n✅ Verification Test:")
            print(f"   Match: {verification['result']['home_team']} vs {verification['result']['away_team']}")
            print(f"   Score: {verification['home_score']}-{verification['away_score']}")
            print(f"   Market: {test_prediction['market']}")
            print(f"   Outcome: {'WON' if verification['outcome'] else 'LOST'}")

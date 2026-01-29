# Historical Data Collection - Maximum Training Pipeline

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.advanced_features import EloRatingSystem, AdvancedFeatureEngineer

class MassiveHistoricalCollector:
    """Collect 900+ matches from multiple free APIs for maximum ML training"""
    
    def __init__(self):
        self.elo_system = EloRatingSystem()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.collected_matches = []
        
    def collect_openligadb_bundesliga(self, seasons: int = 5) -> List[Dict]:
        """Collect Bundesliga matches from OpenLigaDB (free, no API key)"""
        print(f"🇩🇪 Collecting {seasons} seasons of Bundesliga data...")
        matches = []
        
        current_year = 2024
        for i in range(seasons):
            season_year = current_year - i
            url = f"https://api.openligadb.de/getmatchdata/bl1/{season_year}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    season_matches = response.json()
                    for match in season_matches:
                        if match.get('matchIsFinished'):
                            matches.append({
                                'date': match['matchDateTime'][:10],
                                'home_team': match['team1']['teamName'],
                                'away_team': match['team2']['teamName'],
                                'home_goals': match['matchResults'][-1]['pointsTeam1'],
                                'away_goals': match['matchResults'][-1]['pointsTeam2'],
                                'league': 'Bundesliga',
                                'season': f'{season_year}-{season_year+1}'
                            })
                    print(f"  ✅ Season {season_year}: {len(season_matches)} matches")
                    time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"  ❌ Error fetching season {season_year}: {e}")
        
        print(f"  📊 Total Bundesliga: {len(matches)} matches")
        return matches
    
    def collect_thesportsdb_league(self, league_id: str, league_name: str, seasons: int = 2) -> List[Dict]:
        """Collect matches from TheSportsDB (free, limited but useful)"""
        print(f"🏆 Collecting {league_name}...")
        matches = []
        
        current_season = "2024-2025"
        # TheSportsDB free tier: last season results
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id={league_id}&s={current_season}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('events'):
                    for event in data['events']:
                        if event.get('intHomeScore') and event.get('intAwayScore'):
                            matches.append({
                                'date': event['dateEvent'],
                                'home_team': event['strHomeTeam'],
                                'away_team': event['strAwayTeam'],
                                'home_goals': int(event['intHomeScore']),
                                'away_goals': int(event['intAwayScore']),
                                'league': league_name,
                                'season': current_season
                            })
                    print(f"  ✅ {league_name}: {len(matches)} matches")
            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"  ❌ Error fetching {league_name}: {e}")
        
        return matches
    
    def collect_football_data_org(self) -> List[Dict]:
        """Collect from football-data.org CSV files (free, comprehensive)"""
        print("📈 Collecting from football-data.org...")
        matches = []
        
        leagues = {
            'E0': 'Premier League',
            'D1': 'Bundesliga',
            'SP1': 'La Liga',
            'I1': 'Serie A',
            'F1': 'Ligue 1',
            'N1': 'Eredivisie',
            'E1': 'Championship'
        }
        
        seasons = ['2324', '2223', '2122', '2021', '1920']  # Last 5 seasons
        
        for season_code in seasons:
            for league_code, league_name in leagues.items():
                url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
                
                try:
                    df = pd.read_csv(url, encoding='latin-1')
                    for _, row in df.iterrows():
                        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                            # Convert date
                            date_str = row['Date']
                            try:
                                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                            except:
                                try:
                                    date_obj = datetime.strptime(date_str, '%d/%m/%y')
                                except:
                                    continue
                            
                            matches.append({
                                'date': date_obj.strftime('%Y-%m-%d'),
                                'home_team': row['HomeTeam'],
                                'away_team': row['AwayTeam'],
                                'home_goals': int(row['FTHG']),
                                'away_goals': int(row['FTAG']),
                                'league': league_name,
                                'season': f'20{season_code[:2]}-20{season_code[2:]}'
                            })
                    
                    print(f"  ✅ {league_name} ({season_code}): {len(df)} matches")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠️  {league_name} ({season_code}): {e}")
        
        print(f"  📊 Total football-data.org: {len(matches)} matches")
        return matches
    
    def collect_synthetic_data(self, num_matches: int = 0) -> List[Dict]:
        """Generate synthetic data for additional training (based on realistic patterns)"""
        if num_matches == 0:
            print("⏭️  Skipping synthetic data generation")
            return []
        print(f"🤖 Generating {num_matches} synthetic matches...")
        """Generate synthetic data for additional training (based on realistic patterns)"""
        print(f"🤖 Generating {num_matches} synthetic matches...")
        import random
        
        teams = [
            'Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Man United',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
            'Inter Milan', 'AC Milan', 'Juventus', 'Napoli',
            'PSG', 'Lyon', 'Monaco', 'Marseille'
        ]
        
        matches = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(num_matches):
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            # Realistic goal distributions (Poisson-like)
            home_goals = random.choices([0,1,2,3,4,5], weights=[15,30,30,15,7,3])[0]
            away_goals = random.choices([0,1,2,3,4], weights=[20,35,25,15,5])[0]
            
            match_date = base_date + timedelta(days=i*2)
            
            matches.append({
                'date': match_date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'league': 'Mixed',
                'season': '2023-2024'
            })
        
        print(f"  ✅ Generated {num_matches} synthetic matches")
        return matches
    
    def enrich_with_features(self, matches: List[Dict]) -> pd.DataFrame:
        """Add advanced features to raw match data"""
        print("\n🔬 Enriching matches with advanced features...")
        
        df = pd.DataFrame(matches)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Train Elo system on chronological data
        print("  ⚡ Training Elo rating system...")
        for _, row in df.iterrows():
            self.elo_system.update_ratings(
                row['home_team'],
                row['away_team'],
                row['home_goals'],
                row['away_goals']
            )
        
        # Calculate rolling form (last 5 matches per team)
        team_recent_results = {}  # {team: [result1, result2, ...]}
        
        # Add features
        enriched_rows = []
        for idx, row in df.iterrows():
            # Get current Elo ratings
            home_elo = self.elo_system.get_rating(row['home_team'])
            away_elo = self.elo_system.get_rating(row['away_team'])
            elo_diff = home_elo - away_elo
            
            # Simple probability estimates based on Elo difference
            # Higher Elo difference = more goals expected
            elo_factor = (elo_diff / 400)  # Normalized
            expected_home_goals = 1.5 + max(-0.5, min(1.0, elo_factor * 0.5))
            expected_away_goals = 1.3 + max(-0.5, min(1.0, -elo_factor * 0.5))
            expected_total = expected_home_goals + expected_away_goals
            
            # Simple Over/BTTS probabilities
            over_1_5_prob = min(0.95, 0.70 + (expected_total - 2.8) * 0.1)
            over_2_5_prob = min(0.90, 0.50 + (expected_total - 2.8) * 0.15)
            btts_prob = 0.60 if (expected_home_goals > 0.8 and expected_away_goals > 0.8) else 0.40
            
            # Calculate real form (last 5 matches average points)
            home_team = row['home_team']
            away_team = row['away_team']
            
            if home_team not in team_recent_results:
                team_recent_results[home_team] = []
            if away_team not in team_recent_results:
                team_recent_results[away_team] = []
            
            # Get form before this match (last 5 results)
            home_recent = team_recent_results[home_team][-5:] if len(team_recent_results[home_team]) > 0 else []
            away_recent = team_recent_results[away_team][-5:] if len(team_recent_results[away_team]) > 0 else []
            
            # Calculate form index (0-100): 3pts for win, 1pt for draw, 0 for loss
            home_form = (sum(home_recent) / len(home_recent) * 100 / 3) if home_recent else 50.0
            away_form = (sum(away_recent) / len(away_recent) * 100 / 3) if away_recent else 50.0
            
            # Record this match result for future form calculations
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            if home_goals > away_goals:
                home_points, away_points = 3, 0
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1
            
            team_recent_results[home_team].append(home_points)
            team_recent_results[away_team].append(away_points)
            
            enriched_row = {
                **row,
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'predicted_home_goals': expected_home_goals,
                'predicted_away_goals': expected_away_goals,
                'predicted_total_goals': expected_total,
                'over_1_5_prob': over_1_5_prob,
                'over_2_5_prob': over_2_5_prob,
                'btts_prob': btts_prob,
                'home_form': home_form,
                'away_form': away_form,
                'total_goals': row['home_goals'] + row['away_goals'],
                'over_1_5': 1 if (row['home_goals'] + row['away_goals']) > 1.5 else 0,
                'over_2_5': 1 if (row['home_goals'] + row['away_goals']) > 2.5 else 0,
                'btts': 1 if (row['home_goals'] > 0 and row['away_goals'] > 0) else 0,
                'under_1_5': 1 if (row['home_goals'] + row['away_goals']) < 1.5 else 0,
            }
            
            enriched_rows.append(enriched_row)
        
        df_enriched = pd.DataFrame(enriched_rows)
        print(f"  ✅ Added {len(df_enriched.columns) - len(df.columns)} new features")
        
        return df_enriched
    
    def collect_all(self) -> pd.DataFrame:
        """Master collection function - gather all available data"""
        print("=" * 80)
        print("🚀 MASSIVE HISTORICAL DATA COLLECTION - MAXIMUM ML TRAINING")
        print("=" * 80)
        
        all_matches = []
        
        # Source 1: OpenLigaDB (Bundesliga - 3 seasons)
        all_matches.extend(self.collect_openligadb_bundesliga(seasons=3))
        
        # Source 2: football-data.org (5 leagues × 3 seasons)
        all_matches.extend(self.collect_football_data_org())
        
        # Source 3: TheSportsDB (additional leagues)
        thesportsdb_leagues = {
            '4328': 'Premier League',
            '4331': 'La Liga',
            '4332': 'Serie A',
            '4334': 'Ligue 1',
        }
        
        for league_id, league_name in thesportsdb_leagues.items():
            all_matches.extend(self.collect_thesportsdb_league(league_id, league_name))
        
        # Source 4: Synthetic data (for augmentation)
        all_matches.extend(self.collect_synthetic_data(num_matches=200))
        
        print("\n" + "=" * 80)
        print(f"📊 TOTAL COLLECTED: {len(all_matches)} matches")
        print("=" * 80)
        
        # Enrich with features
        df_enriched = self.enrich_with_features(all_matches)
        
        # Save to CSV
        output_path = 'data/historical/massive_training_data.csv'
        os.makedirs('data/historical', exist_ok=True)
        df_enriched.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved to: {output_path}")
        print(f"📊 Final dataset: {len(df_enriched)} rows × {len(df_enriched.columns)} columns")
        
        # Statistics
        print("\n📈 DATASET STATISTICS:")
        print(f"  • Leagues: {df_enriched['league'].nunique()}")
        print(f"  • Seasons: {df_enriched['season'].nunique()}")
        print(f"  • Teams: {len(set(df_enriched['home_team'].unique()) | set(df_enriched['away_team'].unique()))}")
        print(f"  • Date Range: {df_enriched['date'].min()} to {df_enriched['date'].max()}")
        print(f"  • Average Goals: {df_enriched['total_goals'].mean():.2f}")
        print(f"  • Over 2.5 Rate: {df_enriched['over_2_5'].mean()*100:.1f}%")
        print(f"  • BTTS Rate: {df_enriched['btts'].mean()*100:.1f}%")
        
        return df_enriched

if __name__ == "__main__":
    collector = MassiveHistoricalCollector()
    df = collector.collect_all()
    
    print("\n" + "=" * 80)
    print("✅ MASSIVE DATA COLLECTION COMPLETE")
    print("🚀 Ready for maximum ML training")
    print("=" * 80)

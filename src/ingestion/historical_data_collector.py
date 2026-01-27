"""
Historical Data Collection Pipeline
Collects past match results for model training
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from pathlib import Path
import json
import sys
sys.path.append('/workspaces/telegramsoccer')

from src.features.advanced_features import (
    EloRatingSystem, AdvancedFeatureEngineer, EloConfig
)


class HistoricalDataCollector:
    """
    Collects historical match data from free APIs
    Generates features for ML training
    """
    
    def __init__(self):
        self.elo_system = EloRatingSystem(EloConfig())
        self.feature_engineer = AdvancedFeatureEngineer(self.elo_system)
        
        # Free APIs
        self.apis = {
            'football_data': 'https://api.football-data.org/v4',
            'thesportsdb': 'https://www.thesportsdb.com/api/v1/json/3',
            'openligadb': 'https://api.openligadb.de'
        }
    
    def collect_bundesliga_history(
        self,
        seasons: List[str] = ['2023', '2022', '2021'],
        league: str = 'bl1'
    ) -> pd.DataFrame:
        """
        Collect Bundesliga historical data from OpenLigaDB (FREE, no key needed)
        
        Args:
            seasons: List of season years ['2023', '2022']
            league: 'bl1' (Bundesliga) or 'bl2' (2. Bundesliga)
        
        Returns:
            DataFrame with historical matches
        """
        print(f"\n🔍 Collecting {len(seasons)} seasons of {league.upper()} data...")
        
        all_matches = []
        
        for season in seasons:
            print(f"   Fetching {season} season...")
            
            url = f"{self.apis['openligadb']}/getmatchdata/{league}/{season}"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    matches = response.json()
                    
                    for match in matches:
                        # Extract match data
                        if match.get('matchIsFinished'):
                            home_team = match['team1']['teamName']
                            away_team = match['team2']['teamName']
                            
                            # Get goals
                            final_result = None
                            for result in match.get('matchResults', []):
                                if result['resultTypeID'] == 2:  # Final result
                                    final_result = result
                                    break
                            
                            if final_result:
                                home_goals = final_result['pointsTeam1']
                                away_goals = final_result['pointsTeam2']
                                
                                match_data = {
                                    'match_id': match['matchID'],
                                    'date': match['matchDateTime'],
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_goals': home_goals,
                                    'away_goals': away_goals,
                                    'total_goals': home_goals + away_goals,
                                    'season': season,
                                    'league': league
                                }
                                
                                all_matches.append(match_data)
                    
                    print(f"      ✅ {len(matches)} matches collected")
                    time.sleep(1)  # Rate limiting
                
                else:
                    print(f"      ❌ Failed: {response.status_code}")
            
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        df = pd.DataFrame(all_matches)
        print(f"\n✅ Total matches collected: {len(df)}\n")
        
        return df
    
    def enrich_with_features(
        self,
        matches_df: pd.DataFrame,
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Enrich historical matches with advanced features
        
        Args:
            matches_df: DataFrame with basic match data
            window_size: Number of past matches to consider for rolling stats
        
        Returns:
            DataFrame with 50+ features per match
        """
        print(f"🔧 Generating advanced features for {len(matches_df)} matches...")
        
        # Sort by date
        matches_df = matches_df.sort_values('date').reset_index(drop=True)
        
        enriched_matches = []
        
        # Track team histories for feature calculation
        team_histories = {}
        
        for idx, row in matches_df.iterrows():
            if idx % 100 == 0:
                print(f"   Processing match {idx}/{len(matches_df)}...")
            
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            
            # Initialize team histories if needed
            if home_team not in team_histories:
                team_histories[home_team] = {'matches': [], 'goals_for': [], 'goals_against': []}
            if away_team not in team_histories:
                team_histories[away_team] = {'matches': [], 'goals_for': [], 'goals_against': []}
            
            # Get recent form for both teams (from past matches)
            home_history = team_histories[home_team]
            away_history = team_histories[away_team]
            
            # Only generate features if we have enough history
            if len(home_history['matches']) >= 5 and len(away_history['matches']) >= 5:
                try:
                    # Build features
                    features = {}
                    
                    # 1. Elo ratings & predictions
                    home_elo = self.elo_system.get_rating(home_team)
                    away_elo = self.elo_system.get_rating(away_team)
                    elo_pred = self.elo_system.predict_match_outcome_probability(home_team, away_team)
                    
                    features['elo_home'] = home_elo
                    features['elo_away'] = away_elo
                    features['elo_diff'] = home_elo - away_elo
                    features['elo_pred_over_1_5'] = elo_pred['over_1_5']
                    features['elo_pred_over_2_5'] = elo_pred['over_2_5']
                    features['elo_pred_btts'] = elo_pred['btts']
                    
                    # 2. Form indices
                    home_form = self.feature_engineer.calculate_form_index(
                        home_history['matches'][-window_size:],
                        window=window_size
                    )
                    away_form = self.feature_engineer.calculate_form_index(
                        away_history['matches'][-window_size:],
                        window=window_size
                    )
                    
                    features['form_home'] = home_form
                    features['form_away'] = away_form
                    features['form_diff'] = home_form - away_form
                    
                    # 3. Goal statistics
                    home_goals_for_avg = sum(home_history['goals_for'][-window_size:]) / min(len(home_history['goals_for']), window_size)
                    home_goals_against_avg = sum(home_history['goals_against'][-window_size:]) / min(len(home_history['goals_against']), window_size)
                    away_goals_for_avg = sum(away_history['goals_for'][-window_size:]) / min(len(away_history['goals_for']), window_size)
                    away_goals_against_avg = sum(away_history['goals_against'][-window_size:]) / min(len(away_history['goals_against']), window_size)
                    
                    features['home_goals_for_avg'] = home_goals_for_avg
                    features['home_goals_against_avg'] = home_goals_against_avg
                    features['away_goals_for_avg'] = away_goals_for_avg
                    features['away_goals_against_avg'] = away_goals_against_avg
                    features['total_goals_expected'] = home_goals_for_avg + away_goals_for_avg
                    
                    # 4. Derived stats
                    features['home_attack_strength'] = home_goals_for_avg / (away_goals_against_avg + 0.1)
                    features['away_attack_strength'] = away_goals_for_avg / (home_goals_against_avg + 0.1)
                    features['home_defense_weakness'] = home_goals_against_avg
                    features['away_defense_weakness'] = away_goals_against_avg
                    
                    # Add basic match info
                    match_data = {
                        'match_id': row['match_id'],
                        'date': date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': row['home_goals'],
                        'away_goals': row['away_goals'],
                        'total_goals': row['total_goals']
                    }
                    
                    # Target variables
                    match_data['over_1_5'] = 1 if row['total_goals'] > 1.5 else 0
                    match_data['over_2_5'] = 1 if row['total_goals'] > 2.5 else 0
                    match_data['under_1_5'] = 1 if row['total_goals'] <= 1.5 else 0
                    match_data['btts'] = 1 if (row['home_goals'] > 0 and row['away_goals'] > 0) else 0
                    
                    # Combine
                    match_data.update(features)
                    enriched_matches.append(match_data)
                
                except Exception as e:
                    print(f"      ⚠️  Error processing match {idx}: {e}")
            
            # Update histories (after generating features)
            home_result = 'W' if row['home_goals'] > row['away_goals'] else ('D' if row['home_goals'] == row['away_goals'] else 'L')
            away_result = 'W' if row['away_goals'] > row['home_goals'] else ('D' if row['away_goals'] == row['home_goals'] else 'L')
            
            team_histories[home_team]['matches'].append(home_result)
            team_histories[home_team]['goals_for'].append(row['home_goals'])
            team_histories[home_team]['goals_against'].append(row['away_goals'])
            
            team_histories[away_team]['matches'].append(away_result)
            team_histories[away_team]['goals_for'].append(row['away_goals'])
            team_histories[away_team]['goals_against'].append(row['home_goals'])
            
            # Update Elo ratings
            self.elo_system.update_ratings(
                home_team, away_team,
                row['home_goals'], row['away_goals'],
                date
            )
        
        enriched_df = pd.DataFrame(enriched_matches)
        print(f"\n✅ Generated {len(enriched_df.columns) - 7} features for {len(enriched_df)} matches\n")
        
        return enriched_df
    
    def save_training_data(
        self,
        df: pd.DataFrame,
        output_path: Path = Path('data/training_data.csv')
    ):
        """Save training data to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✅ Training data saved to {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print()


def main():
    """Collect and prepare training data"""
    collector = HistoricalDataCollector()
    
    # Collect Bundesliga history (3 seasons = ~900 matches)
    historical_matches = collector.collect_bundesliga_history(
        seasons=['2023', '2022', '2021'],
        league='bl1'
    )
    
    if len(historical_matches) > 0:
        # Generate features
        enriched_data = collector.enrich_with_features(historical_matches)
        
        # Save training data
        collector.save_training_data(enriched_data)
        
        print("🎯 TRAINING DATA READY!")
        print(f"   Total matches: {len(enriched_data)}")
        print(f"   Over 1.5 rate: {enriched_data['over_1_5'].mean():.2%}")
        print(f"   Over 2.5 rate: {enriched_data['over_2_5'].mean():.2%}")
        print(f"   BTTS rate: {enriched_data['btts'].mean():.2%}")
        print(f"   Under 1.5 rate: {enriched_data['under_1_5'].mean():.2%}")
    else:
        print("❌ No historical data collected")


if __name__ == '__main__':
    main()

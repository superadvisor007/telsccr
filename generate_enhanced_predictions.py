# Enhanced Prediction System - ML + Knowledge for Top 1%

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.advanced_features import EloRatingSystem, ValueBettingCalculator

class Top1PercentPredictionSystem:
    """
    Final Enhanced System combining:
    - ML Models (trained on 6582 matches)
    - Domain Knowledge (10+ expert files)
    - Elo Ratings
    - Value Betting Logic
    """
    
    def __init__(self):
        self.elo_system = EloRatingSystem()
        self.load_models()
        self.load_elo_ratings()
        
        # Knowledge base
        self.knowledge = {
            'league_patterns': {
                'Bundesliga': {'avg_goals': 3.1, 'over_2_5_rate': 0.58, 'btts_rate': 0.52, 'scoring_tendency': 'high'},
                'Premier League': {'avg_goals': 2.8, 'over_2_5_rate': 0.53, 'btts_rate': 0.48, 'scoring_tendency': 'medium-high'},
                'La Liga': {'avg_goals': 2.6, 'over_2_5_rate': 0.48, 'btts_rate': 0.45, 'scoring_tendency': 'medium'},
                'Serie A': {'avg_goals': 2.5, 'over_2_5_rate': 0.43, 'btts_rate': 0.42, 'scoring_tendency': 'low'},
                'Ligue 1': {'avg_goals': 2.7, 'over_2_5_rate': 0.50, 'btts_rate': 0.46, 'scoring_tendency': 'medium'},
            },
            'minimum_edge': 0.08,  # 8% minimum for top 1% (increased from 5%)
            'excellent_edge': 0.12,  # 12%+ = exceptional
            'confidence_thresholds': {
                'high': 0.75,  # 75%+ probability = high confidence
                'medium': 0.65,  # 65-75% = medium confidence
                'low': 0.55  # 55-65% = low confidence
            }
        }
    
    def load_models(self):
        """Load trained ML models"""
        model_dir = 'models/knowledge_enhanced'
        self.models = {}
        self.scalers = {}
        
        for market in ['over_1_5', 'over_2_5', 'btts', 'under_1_5']:
            model_path = f"{model_dir}/{market}_model.pkl"
            scaler_path = f"{model_dir}/{market}_scaler.pkl"
            
            if Path(model_path).exists():
                self.models[market] = joblib.load(model_path)
                self.scalers[market] = joblib.load(scaler_path)
    
    def load_elo_ratings(self):
        """Load Elo ratings from training data"""
        try:
            df = pd.read_csv('data/historical/massive_training_data.csv')
            
            # Rebuild Elo system from historical data
            df_sorted = df.sort_values('date')
            for _, row in df_sorted.iterrows():
                self.elo_system.update_ratings(
                    row['home_team'],
                    row['away_team'],
                    row['home_goals'],
                    row['away_goals']
                )
            
            print(f"  ✅ Loaded Elo ratings for {len(self.elo_system.ratings)} teams")
        except Exception as e:
            print(f"  ⚠️  Could not load Elo ratings: {e}")
    
    def calculate_features(self, home_team: str, away_team: str, league: str) -> Dict:
        """Calculate all features for prediction"""
        
        # Elo features
        home_elo = self.elo_system.get_rating(home_team)
        away_elo = self.elo_system.get_rating(away_team)
        elo_diff = home_elo - away_elo
        
        # Expected goals (simplified Poisson-based model)
        elo_factor = elo_diff / 400
        predicted_home_goals = 1.5 + max(-0.5, min(1.0, elo_factor * 0.5))
        predicted_away_goals = 1.3 + max(-0.5, min(1.0, -elo_factor * 0.5))
        predicted_total = predicted_home_goals + predicted_away_goals
        
        # League adjustment
        league_info = self.knowledge['league_patterns'].get(league, {'avg_goals': 2.8})
        league_factor = league_info['avg_goals'] / 2.8
        predicted_total *= league_factor
        
        # Form (placeholder - would calculate from recent matches in production)
        home_form = 75.0
        away_form = 70.0
        
        features = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_advantage': elo_diff / 400,
            'elo_home_strength': home_elo / 1500,
            'elo_away_strength': away_elo / 1500,
            'predicted_home_goals': predicted_home_goals,
            'predicted_away_goals': predicted_away_goals,
            'predicted_total_goals': predicted_total,
            'total_goal_expectation': predicted_total,
            'goal_differential_expectation': predicted_home_goals - predicted_away_goals,
            'home_form': home_form,
            'away_form': away_form,
            'form_advantage': home_form - away_form,
            'league_avg_goals': league_info['avg_goals'],
            'elo_x_form': (elo_diff / 400) * (home_form - away_form),
            'goals_x_league': predicted_total * (league_info['avg_goals'] / 2.8)
        }
        
        return features
    
    def ml_predict(self, features: Dict, market: str) -> float:
        """Get ML model prediction"""
        if market not in self.models:
            return 0.5  # Default
        
        feature_list = [
            'home_elo', 'away_elo', 'elo_diff', 'elo_advantage',
            'elo_home_strength', 'elo_away_strength',
            'predicted_home_goals', 'predicted_away_goals', 'predicted_total_goals',
            'total_goal_expectation', 'goal_differential_expectation',
            'home_form', 'away_form', 'form_advantage',
            'league_avg_goals', 'elo_x_form', 'goals_x_league'
        ]
        
        X = np.array([[features[f] for f in feature_list]])
        X_scaled = self.scalers[market].transform(X)
        
        probability = self.models[market].predict_proba(X_scaled)[0, 1]
        return probability
    
    def knowledge_adjust(self, ml_prob: float, features: Dict, market: str, league: str) -> float:
        """Apply domain knowledge adjustments"""
        adjusted = ml_prob
        
        # League-specific adjustment
        league_info = self.knowledge['league_patterns'].get(league, {})
        
        if 'over' in market:
            # High-scoring leagues → boost Over probability
            if league_info.get('scoring_tendency') == 'high':
                adjusted *= 1.03
            elif league_info.get('scoring_tendency') == 'low':
                adjusted *= 0.97
        
        # Elo dominance adjustment
        elo_diff = abs(features['elo_diff'])
        if elo_diff > 200:  # Dominant team
            if 'over' in market:
                adjusted *= 1.02  # Dominant teams score more
            if market == 'btts':
                adjusted *= 0.98  # Underdog less likely to score
        
        # Expected goals adjustment
        if features['predicted_total_goals'] > 3.5:  # High-scoring expected
            if market == 'over_2_5':
                adjusted *= 1.05
            if market == 'under_1_5':
                adjusted *= 0.90
        
        return min(0.95, max(0.05, adjusted))  # Keep in bounds
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        market_odds: Dict[str, float] = None
    ) -> Dict:
        """
        Complete match prediction with value detection
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            market_odds: Dict of market odds {market: decimal_odds}
        
        Returns:
            Dict with predictions and value bets
        """
        
        # Calculate features
        features = self.calculate_features(home_team, away_team, league)
        
        # ML + Knowledge predictions
        predictions = {}
        for market in ['over_1_5', 'over_2_5', 'btts', 'under_1_5']:
            ml_prob = self.ml_predict(features, market)
            final_prob = self.knowledge_adjust(ml_prob, features, market, league)
            
            predictions[market] = {
                'ml_probability': ml_prob,
                'knowledge_adjusted': final_prob,
                'final_probability': final_prob
            }
        
        # Value detection
        value_bets = []
        if market_odds:
            for market, odds in market_odds.items():
                if market in predictions:
                    model_prob = predictions[market]['final_probability']
                    market_prob = 1 / odds
                    edge = model_prob - market_prob
                    
                    # Top 1% standard: minimum 8% edge
                    if edge >= self.knowledge['minimum_edge']:
                        ev = ValueBettingCalculator.calculate_expected_value(
                            model_prob, odds, stake=10
                        )
                        
                        confidence = 'HIGH' if model_prob >= self.knowledge['confidence_thresholds']['high'] else \
                                   'MEDIUM' if model_prob >= self.knowledge['confidence_thresholds']['medium'] else 'LOW'
                        
                        kelly_stake = ValueBettingCalculator.calculate_kelly_stake(
                            model_prob, odds, bankroll=1000, kelly_fraction=0.25
                        )
                        
                        value_bets.append({
                            'market': market,
                            'odds': odds,
                            'model_probability': model_prob,
                            'market_probability': market_prob,
                            'edge': edge,
                            'expected_value': ev,
                            'confidence': confidence,
                            'kelly_stake': kelly_stake,
                            'kelly_percent': (kelly_stake / 1000) * 100
                        })
        
        # Sort by edge (highest first)
        value_bets = sorted(value_bets, key=lambda x: x['edge'], reverse=True)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'features': features,
            'predictions': predictions,
            'value_bets': value_bets,
            'top_1_percent_quality': len([b for b in value_bets if b['edge'] >= 0.12]) > 0
        }
    
    def generate_tomorrow_predictions(self) -> List[Dict]:
        """Generate predictions for tomorrow's matches"""
        
        print("\n🔮 GENERATING ENHANCED PREDICTIONS FOR TOMORROW")
        print("=" * 80)
        
        # Tomorrow's matches (would come from API in production)
        tomorrow_matches = [
            {
                'home_team': 'Bayern München',
                'away_team': 'Borussia Dortmund',
                'league': 'Bundesliga',
                'date': '2026-01-28',
                'time': '18:30',
                'market_odds': {
                    'over_1_5': 1.15,
                    'over_2_5': 1.60,
                    'btts': 1.55,
                    'under_1_5': 6.50
                }
            },
            {
                'home_team': 'RB Leipzig',
                'away_team': 'Bayer Leverkusen',
                'league': 'Bundesliga',
                'date': '2026-01-28',
                'time': '15:30',
                'market_odds': {
                    'over_1_5': 1.20,
                    'over_2_5': 1.70,
                    'btts': 1.60,
                    'under_1_5': 5.50
                }
            },
            {
                'home_team': 'Manchester City',
                'away_team': 'Arsenal',
                'league': 'Premier League',
                'date': '2026-01-28',
                'time': '16:00',
                'market_odds': {
                    'over_1_5': 1.22,
                    'over_2_5': 1.65,
                    'btts': 1.70,
                    'under_1_5': 5.00
                }
            },
            {
                'home_team': 'Liverpool',
                'away_team': 'Chelsea',
                'league': 'Premier League',
                'date': '2026-01-28',
                'time': '13:30',
                'market_odds': {
                    'over_1_5': 1.18,
                    'over_2_5': 1.55,
                    'btts': 1.65,
                    'under_1_5': 6.00
                }
            },
            {
                'home_team': 'VfB Stuttgart',
                'away_team': 'Eintracht Frankfurt',
                'league': 'Bundesliga',
                'date': '2026-01-28',
                'time': '15:30',
                'market_odds': {
                    'over_1_5': 1.25,
                    'over_2_5': 1.75,
                    'btts': 1.62,
                    'under_1_5': 4.50
                }
            }
        ]
        
        all_predictions = []
        all_value_bets = []
        
        for match in tomorrow_matches:
            print(f"\n📊 {match['home_team']} vs {match['away_team']} ({match['league']})")
            
            prediction = self.predict_match(
                match['home_team'],
                match['away_team'],
                match['league'],
                match['market_odds']
            )
            
            prediction['date'] = match['date']
            prediction['time'] = match['time']
            all_predictions.append(prediction)
            
            # Display value bets
            if prediction['value_bets']:
                for bet in prediction['value_bets']:
                    print(f"  ✅ {bet['market'].upper()}: {bet['edge']:.1%} edge | {bet['confidence']} confidence")
                    all_value_bets.append({**bet, **match})
            else:
                print("  ❌ No value bets found")
        
        # Summary
        print("\n" + "=" * 80)
        print("📈 PREDICTION SUMMARY")
        print("=" * 80)
        print(f"Total Matches Analyzed: {len(all_predictions)}")
        print(f"Total Value Bets Found: {len(all_value_bets)}")
        print(f"Average Edge: {np.mean([b['edge'] for b in all_value_bets]):.1%}" if all_value_bets else "Average Edge: N/A")
        print(f"Top 1% Bets (12%+ edge): {len([b for b in all_value_bets if b['edge'] >= 0.12])}")
        print(f"High Confidence Bets: {len([b for b in all_value_bets if b['confidence'] == 'HIGH'])}")
        
        return all_predictions, all_value_bets


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ENHANCED PREDICTION SYSTEM - ML + KNOWLEDGE FOR TOP 1%")
    print("=" * 80)
    
    # Initialize system
    print("\n⚡ Initializing Enhanced System...")
    system = Top1PercentPredictionSystem()
    
    # Generate predictions
    predictions, value_bets = system.generate_tomorrow_predictions()
    
    # Display top bets
    if value_bets:
        print("\n" + "=" * 80)
        print("⭐ TOP VALUE BETS FOR TOMORROW")
        print("=" * 80)
        
        # Sort by edge
        top_bets = sorted(value_bets, key=lambda x: x['edge'], reverse=True)[:10]
        
        for i, bet in enumerate(top_bets, 1):
            print(f"\n{i}. {bet['home_team']} vs {bet['away_team']} | {bet['time']}")
            print(f"   Market: {bet['market'].upper()}")
            print(f"   Odds: {bet['odds']:.2f}")
            print(f"   Model Probability: {bet['model_probability']:.1%}")
            print(f"   Market Probability: {bet['market_probability']:.1%}")
            print(f"   Edge: {bet['edge']:.1%} {'🔥' if bet['edge'] >= 0.12 else '✅'}")
            print(f"   Expected Value: ${bet['expected_value']:.2f}")
            print(f"   Confidence: {bet['confidence']}")
            print(f"   Kelly Stake: {bet['kelly_percent']:.1f}% of bankroll (${bet['kelly_stake']:.2f})")
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED PREDICTION SYSTEM COMPLETE")
    print("🎯 ML Models + Knowledge Base + Value Detection")
    print("🚀 Top 1% Performance Standards Applied")
    print("=" * 80)

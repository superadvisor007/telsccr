"""
INTEGRATED PREDICTION PIPELINE WITH LIVE BET TRACKING
====================================================

Combines:
1. Advanced ML predictions (calibrated models)
2. Live bet tracking (SQLite logging)
3. Value bet detection (Kelly Criterion)
4. Telegram notifications

Flow:
1. Fetch today's matches
2. Generate predictions with calibrated models
3. Calculate expected value (EV)
4. Filter high-confidence bets (EV >5%, confidence >60%)
5. Log bets to tracker
6. Send to Telegram
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, date
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tracking.live_bet_tracker import LiveBetTracker
from src.features.advanced_features import ValueBettingCalculator, EloRatingSystem


class IntegratedPredictionPipeline:
    """
    Production prediction pipeline with live tracking integration
    """
    
    def __init__(
        self,
        model_dir: str = "models/knowledge_enhanced",
        min_ev: float = 0.05,  # Minimum 5% expected value
        min_confidence: float = 0.60,  # Minimum 60% predicted probability
        max_odds: float = 2.0,  # Maximum odds (avoid long shots)
        use_calibrated: bool = True
    ):
        self.model_dir = Path(model_dir)
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.max_odds = max_odds
        self.use_calibrated = use_calibrated
        
        # Initialize tracker
        self.tracker = LiveBetTracker()
        
        # Load models
        self.models = self._load_models()
        self.scalers = self._load_scalers()
        
        # Initialize calculators
        self.value_calculator = ValueBettingCalculator()
        self.elo_system = EloRatingSystem()
        
        print(f"✅ Pipeline initialized with {len(self.models)} markets")
    
    def _load_models(self) -> Dict:
        """Load trained models (calibrated if available)"""
        models = {}
        
        for market in ['over_1_5', 'over_2_5', 'btts']:
            # Try calibrated model first
            if self.use_calibrated:
                calibrated_path = self.model_dir / f"{market}_calibrated_model.pkl"
                if calibrated_path.exists():
                    models[market] = joblib.load(calibrated_path)
                    print(f"  ✅ Loaded calibrated {market} model")
                    continue
            
            # Fallback to regular model
            model_path = self.model_dir / f"{market}_model.pkl"
            if model_path.exists():
                models[market] = joblib.load(model_path)
                print(f"  ✅ Loaded {market} model")
            else:
                print(f"  ⚠️  Model not found: {market}")
        
        return models
    
    def _load_scalers(self) -> Dict:
        """Load feature scalers"""
        scalers = {}
        
        for market in ['over_1_5', 'over_2_5', 'btts']:
            scaler_path = self.model_dir / f"{market}_scaler.pkl"
            if scaler_path.exists():
                scalers[market] = joblib.load(scaler_path)
        
        return scalers
    
    def engineer_features(self, match: Dict) -> Dict:
        """
        Engineer features for prediction (matching training features)
        
        Expected match dict:
        {
            'home_team': str,
            'away_team': str,
            'league': str,
            'home_elo': float,
            'away_elo': float,
            'home_form': float,
            'away_form': float,
            'predicted_home_goals': float,
            'predicted_away_goals': float,
        }
        """
        features = {}
        
        # Core Elo features
        features['elo_diff'] = match['home_elo'] - match['away_elo']
        features['elo_advantage'] = features['elo_diff'] / 400
        features['elo_home_strength'] = match['home_elo'] / 1500
        features['elo_away_strength'] = match['away_elo'] / 1500
        features['home_elo'] = match['home_elo']
        features['away_elo'] = match['away_elo']
        
        # xG features
        features['xg_differential'] = match['predicted_home_goals'] - match['predicted_away_goals']
        features['xg_total'] = match['predicted_home_goals'] + match['predicted_away_goals']
        features['xg_home_dominance'] = match['predicted_home_goals'] / (features['xg_total'] + 0.1)
        features['predicted_home_goals'] = match['predicted_home_goals']
        features['predicted_away_goals'] = match['predicted_away_goals']
        features['predicted_total_goals'] = features['xg_total']
        
        # League calibration
        league_scoring_rates = {
            'Bundesliga': 1.11,
            'Eredivisie': 1.08,
            'Premier League': 1.00,
            'Ligue 1': 0.96,
            'La Liga': 0.93,
            'Serie A': 0.89,
            'Championship': 1.02,
            'Mixed': 1.00
        }
        features['league_scoring_factor'] = league_scoring_rates.get(match['league'], 1.0)
        features['xg_adjusted'] = features['xg_total'] * features['league_scoring_factor']
        
        # Form features
        features['home_form'] = match.get('home_form', 0.0)
        features['away_form'] = match.get('away_form', 0.0)
        features['form_advantage'] = features['home_form'] - features['away_form']
        features['form_momentum'] = (features['home_form'] + features['away_form']) / 2
        
        # Interaction features
        features['elo_x_league'] = features['elo_advantage'] * features['league_scoring_factor']
        features['elo_x_form'] = features['elo_advantage'] * features['form_advantage']
        features['xg_x_elo'] = features['xg_differential'] * features['elo_advantage']
        
        # Composite strength
        features['home_composite_strength'] = (
            features['elo_home_strength'] * 0.4 +
            features['home_form'] / 3.0 * 0.3 +
            features['predicted_home_goals'] / 3.0 * 0.3
        )
        features['away_composite_strength'] = (
            features['elo_away_strength'] * 0.4 +
            features['away_form'] / 3.0 * 0.3 +
            features['predicted_away_goals'] / 3.0 * 0.3
        )
        features['strength_imbalance'] = features['home_composite_strength'] - features['away_composite_strength']
        
        # BTTS-specific features
        features['joint_attack_strength'] = min(features['predicted_home_goals'], features['predicted_away_goals'])
        features['joint_offensive_potential'] = features['predicted_home_goals'] * features['predicted_away_goals']
        features['strength_balance'] = 1.0 - abs(features['home_composite_strength'] - features['away_composite_strength'])
        features['offensive_match'] = float((features['home_form'] > 1.5) and (features['away_form'] > 1.5))
        
        return features
    
    def predict_match(self, match: Dict) -> Dict:
        """
        Generate predictions for a single match across all markets
        
        Returns:
            {
                'match_id': str,
                'home_team': str,
                'away_team': str,
                'league': str,
                'predictions': {
                    'over_1_5': {'probability': float, 'confidence': str},
                    'over_2_5': {...},
                    'btts': {...}
                }
            }
        """
        # Engineer features
        features = self.engineer_features(match)
        
        # Create feature vector (matching training order)
        feature_cols = [
            'elo_advantage', 'elo_home_strength', 'elo_away_strength', 'elo_diff',
            'home_elo', 'away_elo',
            'xg_differential', 'xg_total', 'xg_home_dominance', 'xg_adjusted',
            'predicted_home_goals', 'predicted_away_goals', 'predicted_total_goals',
            'home_form', 'away_form', 'form_advantage', 'form_momentum',
            'league_scoring_factor',
            'elo_x_league', 'elo_x_form', 'xg_x_elo',
            'home_composite_strength', 'away_composite_strength', 'strength_imbalance',
            'joint_attack_strength', 'joint_offensive_potential', 'strength_balance', 'offensive_match',
        ]
        
        X = np.array([[features[col] for col in feature_cols]])
        
        # Generate predictions
        predictions = {}
        
        for market, model in self.models.items():
            # Scale features
            if market in self.scalers:
                X_scaled = self.scalers[market].transform(X)
            else:
                X_scaled = X
            
            # Predict probability
            prob = model.predict_proba(X_scaled)[0, 1]
            
            # Determine confidence
            if prob >= 0.70:
                confidence = 'high'
            elif prob >= 0.60:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            predictions[market] = {
                'probability': prob,
                'confidence': confidence
            }
        
        return {
            'match_id': match.get('match_id', f"{match['home_team']}_vs_{match['away_team']}_{match.get('date', datetime.now().strftime('%Y%m%d'))}"),
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'league': match['league'],
            'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
            'predictions': predictions
        }
    
    def find_value_bets(
        self,
        predictions: Dict,
        odds: Dict[str, float]
    ) -> List[Dict]:
        """
        Find value bets based on predicted probabilities and market odds
        
        Args:
            predictions: Output from predict_match()
            odds: Market odds {'over_1_5': 1.50, 'over_2_5': 2.10, ...}
        
        Returns:
            List of value bets with EV, stake recommendations, etc.
        """
        value_bets = []
        
        for market, pred in predictions['predictions'].items():
            if market not in odds:
                continue
            
            market_odds = odds[market]
            predicted_prob = pred['probability']
            
            # Skip if below thresholds
            if predicted_prob < self.min_confidence:
                continue
            
            if market_odds > self.max_odds:
                continue
            
            # Calculate expected value
            implied_prob = 1.0 / market_odds
            ev = (predicted_prob * market_odds - 1.0) * 100  # EV as percentage
            
            if ev >= self.min_ev * 100:
                # Calculate Kelly stake (conservative: 25% of full Kelly)
                kelly_fraction = 0.25
                edge = predicted_prob - implied_prob
                kelly_stake = kelly_fraction * (edge / (market_odds - 1.0)) if market_odds > 1.0 else 0.01
                
                # Clamp between 1-5% of bankroll
                kelly_stake = max(0.01, min(0.05, kelly_stake))
                
                value_bets.append({
                    'match_id': predictions['match_id'],
                    'home_team': predictions['home_team'],
                    'away_team': predictions['away_team'],
                    'league': predictions['league'],
                    'date': predictions['date'],
                    'market': market,
                    'predicted_prob': predicted_prob,
                    'odds': market_odds,
                    'implied_prob': implied_prob,
                    'ev': ev,
                    'kelly_stake': kelly_stake,
                    'confidence': pred['confidence'],
                })
        
        return value_bets
    
    def log_bets_to_tracker(
        self,
        value_bets: List[Dict],
        bankroll: float = 1000.0
    ):
        """
        Log value bets to live tracker
        
        Args:
            value_bets: List of value bets from find_value_bets()
            bankroll: Current bankroll for stake calculation
        """
        for bet in value_bets:
            stake = bet['kelly_stake'] * bankroll
            
            bet_id = self.tracker.log_bet(
                match_id=bet['match_id'],
                match_date=bet['date'],
                home_team=bet['home_team'],
                away_team=bet['away_team'],
                league=bet['league'],
                market=bet['market'],
                predicted_prob=bet['predicted_prob'],
                odds=bet['odds'],
                stake=stake,
                confidence=bet['confidence']
            )
            
            if bet_id:
                print(f"✅ Logged bet: {bet['home_team']} vs {bet['away_team']} | {bet['market']} @ {bet['odds']:.2f} (EV: {bet['ev']:+.1f}%)")
    
    def generate_daily_predictions(
        self,
        matches: List[Dict],
        odds_dict: Dict[str, Dict[str, float]],
        bankroll: float = 1000.0
    ) -> Dict:
        """
        Complete daily prediction workflow
        
        Args:
            matches: List of match dicts
            odds_dict: {match_id: {'over_1_5': 1.50, ...}}
            bankroll: Current bankroll
        
        Returns:
            {
                'total_matches': int,
                'value_bets_found': int,
                'total_ev': float,
                'bets': List[Dict]
            }
        """
        print(f"\n{'='*70}")
        print(f"📊 DAILY PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*70}\n")
        
        all_value_bets = []
        
        for match in matches:
            # Generate predictions
            predictions = self.predict_match(match)
            
            # Get odds
            match_id = predictions['match_id']
            odds = odds_dict.get(match_id, {})
            
            if not odds:
                print(f"⚠️  No odds for {predictions['home_team']} vs {predictions['away_team']}")
                continue
            
            # Find value bets
            value_bets = self.find_value_bets(predictions, odds)
            all_value_bets.extend(value_bets)
        
        # Log to tracker
        if all_value_bets:
            self.log_bets_to_tracker(all_value_bets, bankroll)
        
        # Summary
        total_ev = sum(bet['ev'] for bet in all_value_bets)
        
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        print(f"Total matches analyzed: {len(matches)}")
        print(f"Value bets found: {len(all_value_bets)}")
        print(f"Total expected value: {total_ev:+.1f}%")
        print(f"{'='*70}\n")
        
        return {
            'total_matches': len(matches),
            'value_bets_found': len(all_value_bets),
            'total_ev': total_ev,
            'bets': all_value_bets
        }


if __name__ == "__main__":
    # Demo usage
    pipeline = IntegratedPredictionPipeline()
    
    # Example match
    demo_match = {
        'match_id': 'demo_001',
        'home_team': 'Bayern München',
        'away_team': 'Borussia Dortmund',
        'league': 'Bundesliga',
        'date': '2026-01-30',
        'home_elo': 1850,
        'away_elo': 1720,
        'home_form': 2.5,
        'away_form': 1.8,
        'predicted_home_goals': 2.1,
        'predicted_away_goals': 1.4,
    }
    
    # Generate predictions
    predictions = pipeline.predict_match(demo_match)
    
    print("\n📊 PREDICTIONS:")
    for market, pred in predictions['predictions'].items():
        print(f"  {market.upper()}: {pred['probability']:.1%} ({pred['confidence']} confidence)")
    
    # Find value bets (example odds)
    odds = {
        'over_1_5': 1.25,
        'over_2_5': 1.65,
        'btts': 1.70
    }
    
    value_bets = pipeline.find_value_bets(predictions, odds)
    
    if value_bets:
        print("\n💰 VALUE BETS FOUND:")
        for bet in value_bets:
            print(f"  {bet['market'].upper()} @ {bet['odds']:.2f} | EV: {bet['ev']:+.1f}% | Stake: {bet['kelly_stake']:.1%}")
    else:
        print("\n⚠️  No value bets found (all below EV threshold)")

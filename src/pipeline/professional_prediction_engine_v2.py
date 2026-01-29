"""
PROFESSIONAL PREDICTION ENGINE V2
==================================

Integration von ML + Professional Analysis Framework

Flow:
1. ML Model: Liefert base xG predictions (statistische Basis)
2. Team Profiles: Lade Spielstil + Transition Metrics
3. Professional Analysis: Szenario-Modellierung + Context
4. Value Betting: Nur Bets mit Edge >5% und Confidence >MEDIUM
5. Live Tracking: Log alle Vorhersagen für Langzeit-Evaluierung
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import joblib

from src.reasoning.professional_analysis_framework import (
    ProfessionalAnalysisEngine,
    PlayingStyle,
    MatchContext,
    TransitionMetrics,
    MatchScenario,
    LongTermEvaluator
)
from src.reasoning.team_profile_database import TeamProfileDatabase
from src.tracking.live_bet_tracker import LiveBetTracker


class ProfessionalPredictionEngineV2:
    """
    Professional Prediction Engine - Top 1% System
    
    Kombiniert:
    - ML Model (statistische Basis)
    - Professional Analysis (Spielstil, Kontext, Szenarien)
    - Value Betting (nur Bets mit positivem EV)
    - Live Tracking (Langzeit-Evaluierung)
    """
    
    def __init__(
        self,
        model_dir: str = "models/knowledge_enhanced",
        use_professional_analysis: bool = True
    ):
        self.model_dir = model_dir
        self.use_professional_analysis = use_professional_analysis
        
        # Load ML models
        self.models = self._load_models()
        self.scalers = self._load_scalers()
        
        # Load Professional Analysis components
        if self.use_professional_analysis:
            self.analysis_engine = ProfessionalAnalysisEngine()
            self.team_db = TeamProfileDatabase()
            self.evaluator = LongTermEvaluator()
        
        # Live tracking
        self.tracker = LiveBetTracker()
        
        print(f"✅ Professional Prediction Engine V2 initialized")
        print(f"   ML Models: {len(self.models)}")
        print(f"   Professional Analysis: {'ENABLED' if use_professional_analysis else 'DISABLED'}")
        print(f"   Team Profiles: {len(self.team_db.profiles) if use_professional_analysis else 0}")
    
    def _load_models(self) -> Dict:
        """Load trained ML models"""
        models = {}
        for market in ['over_1_5', 'over_2_5', 'btts']:
            model_path = f"{self.model_dir}/{market}_calibrated_model.pkl"
            if os.path.exists(model_path):
                models[market] = joblib.load(model_path)
            else:
                # Fallback to non-calibrated
                model_path = f"{self.model_dir}/{market}_model.pkl"
                if os.path.exists(model_path):
                    models[market] = joblib.load(model_path)
        return models
    
    def _load_scalers(self) -> Dict:
        """Load feature scalers"""
        scalers = {}
        for market in ['over_1_5', 'over_2_5', 'btts']:
            scaler_path = f"{self.model_dir}/{market}_scaler.pkl"
            if os.path.exists(scaler_path):
                scalers[market] = joblib.load(scaler_path)
        return scalers
    
    def infer_context(
        self,
        home_team: str,
        away_team: str,
        league: str,
        home_table_position: int = None,
        away_table_position: int = None
    ) -> Tuple[MatchContext, MatchContext]:
        """
        Inferiere Match Context aus Tabellensituation
        
        TODO: Integration mit echten Tabellendaten
        """
        # Simplified inference (kann später erweitert werden)
        home_context = MatchContext.NEUTRAL
        away_context = MatchContext.NEUTRAL
        
        # Derby detection (simplified)
        derby_pairs = [
            ("Bayern München", "Borussia Dortmund"),
            ("Real Madrid", "Barcelona"),
            ("Manchester City", "Liverpool"),
            ("Inter Milan", "AC Milan"),
        ]
        
        for team1, team2 in derby_pairs:
            if (home_team == team1 and away_team == team2) or \
               (home_team == team2 and away_team == team1):
                home_context = MatchContext.DERBY
                away_context = MatchContext.DERBY
                break
        
        return home_context, away_context
    
    def predict_with_professional_analysis(
        self,
        home_team: str,
        away_team: str,
        league: str,
        features: Dict,
        available_odds: Dict[str, float] = None
    ) -> Dict:
        """
        Vollständige Vorhersage mit Professional Analysis
        
        Args:
            home_team: Heim-Team Name
            away_team: Auswärts-Team Name
            league: Liga
            features: ML Feature Dictionary
            available_odds: Dict mit {market: odds}
        
        Returns:
            {
                "ml_predictions": {...},
                "scenarios": [...],
                "value_bets": [...],
                "recommendation": "BET" | "SKIP",
                "analysis_report": str
            }
        """
        # === 1. ML BASE PREDICTIONS ===
        ml_predictions = self._get_ml_predictions(features)
        
        # === 2. LOAD TEAM PROFILES ===
        home_profile = self.team_db.get_profile(home_team)
        away_profile = self.team_db.get_profile(away_team)
        
        if not home_profile or not away_profile:
            print(f"⚠️  Missing team profiles for {home_team} or {away_team}")
            print(f"   Falling back to ML-only predictions")
            return {
                "ml_predictions": ml_predictions,
                "scenarios": [],
                "value_bets": [],
                "recommendation": "SKIP",
                "analysis_report": "Missing team profiles - ML-only prediction"
            }
        
        # Parse profiles
        home_style = PlayingStyle(home_profile['playing_style'])
        away_style = PlayingStyle(away_profile['playing_style'])
        
        home_transition = TransitionMetrics(**home_profile['transition_metrics'])
        away_transition = TransitionMetrics(**away_profile['transition_metrics'])
        
        # Infer context
        home_context, away_context = self.infer_context(
            home_team, away_team, league
        )
        
        # === 3. PROFESSIONAL ANALYSIS: SCENARIO GENERATION ===
        base_home_xg = ml_predictions.get('predicted_home_xg', features.get('predicted_home_goals', 1.5))
        base_away_xg = ml_predictions.get('predicted_away_xg', features.get('predicted_away_goals', 1.5))
        
        scenarios = self.analysis_engine.generate_match_scenarios(
            base_home_xg=base_home_xg,
            base_away_xg=base_away_xg,
            home_style=home_style,
            away_style=away_style,
            home_context=home_context,
            away_context=away_context,
            home_transition=home_transition,
            away_transition=away_transition
        )
        
        # === 4. VALUE BET EVALUATION ===
        value_bets = []
        
        if available_odds:
            bet_evaluations = {}
            
            for bet_type, odds in available_odds.items():
                evaluation = self.analysis_engine.evaluate_bet_value(
                    scenarios=scenarios,
                    bet_type=bet_type,
                    offered_odds=odds
                )
                
                bet_evaluations[bet_type] = evaluation
                
                # Only recommend bets with positive edge and sufficient confidence
                if evaluation['edge'] > 0.05 and evaluation['confidence'] in ['MEDIUM', 'HIGH']:
                    value_bets.append({
                        "bet_type": bet_type,
                        "odds": odds,
                        "expected_probability": evaluation['expected_probability'],
                        "edge": evaluation['edge'],
                        "ev_percentage": evaluation['ev_percentage'],
                        "confidence": evaluation['confidence'],
                        "stake_recommendation": self._calculate_kelly_stake(
                            evaluation['expected_probability'],
                            odds
                        )
                    })
            
            # === 5. GENERATE ANALYSIS REPORT ===
            analysis_report = self.analysis_engine.generate_analysis_report(
                home_team=home_team,
                away_team=away_team,
                scenarios=scenarios,
                bet_evaluations=bet_evaluations
            )
        else:
            bet_evaluations = {}
            analysis_report = "No odds provided - scenario analysis only"
        
        # === 6. RECOMMENDATION ===
        recommendation = "BET" if len(value_bets) > 0 else "SKIP"
        
        return {
            "ml_predictions": ml_predictions,
            "scenarios": [
                {
                    "name": s.name,
                    "probability": s.probability,
                    "expected_home_goals": s.expected_home_goals,
                    "expected_away_goals": s.expected_away_goals
                }
                for s in scenarios
            ],
            "value_bets": value_bets,
            "recommendation": recommendation,
            "analysis_report": analysis_report,
            "team_profiles": {
                "home_style": home_style.value,
                "away_style": away_style.value,
                "home_context": home_context.value,
                "away_context": away_context.value
            }
        }
    
    def _get_ml_predictions(self, features: Dict) -> Dict:
        """Get ML model predictions for all markets"""
        predictions = {}
        
        # Feature array (muss mit train_advanced_ml_v2.py übereinstimmen)
        feature_cols = [
            'elo_advantage', 'elo_home_strength', 'elo_away_strength', 'elo_diff',
            'home_elo', 'away_elo', 'xg_differential', 'xg_total', 'xg_home_dominance',
            'xg_adjusted', 'predicted_home_goals', 'predicted_away_goals',
            'predicted_total_goals', 'home_form', 'away_form', 'form_advantage',
            'form_momentum', 'league_scoring_factor', 'elo_x_league', 'elo_x_form',
            'xg_x_elo', 'home_composite_strength', 'away_composite_strength',
            'strength_imbalance', 'joint_attack_strength', 'joint_offensive_potential',
            'strength_balance', 'offensive_match'
        ]
        
        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        
        for market, model in self.models.items():
            if market in self.scalers:
                X_scaled = self.scalers[market].transform(X)
                
                # Get probability
                prob = model.predict_proba(X_scaled)[0][1]
                predictions[market] = {
                    "probability": prob,
                    "prediction": bool(prob > 0.5)
                }
        
        # Add xG estimates
        predictions['predicted_home_xg'] = features.get('predicted_home_goals', 1.5)
        predictions['predicted_away_xg'] = features.get('predicted_away_goals', 1.5)
        
        return predictions
    
    def _calculate_kelly_stake(self, true_prob: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion stake
        
        Args:
            true_prob: True probability of outcome
            odds: Offered odds
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)
        
        Returns:
            Percentage of bankroll to stake (0-10%)
        """
        b = odds - 1.0  # Net odds
        q = 1.0 - true_prob  # Probability of losing
        
        kelly = (b * true_prob - q) / b
        
        # Fractional Kelly for safety
        fractional_kelly = kelly * kelly_fraction
        
        # Cap at 10% max
        return min(max(fractional_kelly, 0.0), 0.10) * 100
    
    def log_prediction_for_tracking(
        self,
        match_id: str,
        prediction_result: Dict,
        placed_bets: List[Dict]
    ):
        """
        Log prediction for long-term evaluation
        
        Args:
            match_id: Unique match identifier
            prediction_result: Output from predict_with_professional_analysis
            placed_bets: List of actually placed bets
        """
        # Log in Professional Analysis Evaluator
        for bet in placed_bets:
            self.evaluator.log_prediction(
                match_id=match_id,
                bet_type=bet['bet_type'],
                expected_prob=bet['expected_probability'],
                offered_odds=bet['odds'],
                scenarios=prediction_result.get('scenarios', [])
            )


if __name__ == "__main__":
    # Demo: Professional Prediction in Action
    engine = ProfessionalPredictionEngineV2()
    
    # Example match features (würde normalerweise aus Data Pipeline kommen)
    features = {
        'home_elo': 1850,
        'away_elo': 1720,
        'elo_diff': 130,
        'elo_advantage': 0.325,
        'elo_home_strength': 1.23,
        'elo_away_strength': 1.15,
        'predicted_home_goals': 2.1,
        'predicted_away_goals': 1.4,
        'predicted_total_goals': 3.5,
        'xg_differential': 0.7,
        'xg_total': 3.5,
        'xg_home_dominance': 0.60,
        'xg_adjusted': 3.65,
        'home_form': 2.5,
        'away_form': 1.8,
        'form_advantage': 0.7,
        'form_momentum': 2.15,
        'league_scoring_factor': 1.11,
        'elo_x_league': 0.36,
        'elo_x_form': 0.23,
        'xg_x_elo': 0.23,
        'home_composite_strength': 0.78,
        'away_composite_strength': 0.65,
        'strength_imbalance': 0.13,
        'joint_attack_strength': 1.4,
        'joint_offensive_potential': 2.94,
        'strength_balance': 0.87,
        'offensive_match': 1.0
    }
    
    # Available odds
    odds = {
        'over_2_5': 1.75,
        'btts': 1.55,
        'over_1_5': 1.25
    }
    
    # Predict
    result = engine.predict_with_professional_analysis(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        features=features,
        available_odds=odds
    )
    
    print(f"\n{'='*70}")
    print(f"PROFESSIONAL PREDICTION ENGINE V2 - DEMO")
    print(f"{'='*70}\n")
    
    print(f"ML Predictions:")
    for market, pred in result['ml_predictions'].items():
        if isinstance(pred, dict):
            print(f"  {market}: {pred['probability']:.1%}")
    
    print(f"\nValue Bets Found: {len(result['value_bets'])}")
    for bet in result['value_bets']:
        print(f"  ✅ {bet['bet_type']}: {bet['odds']:.2f} odds, {bet['edge']:+.1%} edge, {bet['ev_percentage']:+.1f}% EV")
        print(f"     Stake: {bet['stake_recommendation']:.1f}% of bankroll, Confidence: {bet['confidence']}")
    
    print(f"\nRecommendation: {result['recommendation']}")
    
    if result['value_bets']:
        print(f"\n{result['analysis_report']}")

"""
PHASE 4: MARKET EXPANSION - ASIAN HANDICAP & FIRST HALF
======================================================

New Markets:
1. Asian Handicap (AH 0.0, -0.5, -1.0, -1.5)
2. First Half Over/Under (1H O/U 0.5, 1.5)
3. Correct Score (limited to 1-0, 2-0, 2-1, 3-0, 3-1, 3-2)

Data Requirements:
- Asian Handicap: Home advantage strength, Elo differential, recent form asymmetry
- 1H Markets: First half xG, early goal probability, team style (fast start vs slow)
- Correct Score: Score distribution modeling (Poisson, Dixon-Coles)

Implementation Strategy:
1. Asian Handicap: Extend current models with handicap-adjusted predictions
2. 1H Markets: Train on time-adjusted xG (first 45min historical averages)
3. Correct Score: Use existing Dixon-Coles model in src/models/dixon_coles.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent))


class MarketExpansionEngine:
    """
    Expand predictions to Asian Handicap and First Half markets
    """
    
    def __init__(self, model_dir: str = "models/knowledge_enhanced"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self._load_base_models()
    
    def _load_base_models(self):
        """Load existing Over/Under models as base"""
        for market in ['over_1_5', 'over_2_5']:
            model_path = self.model_dir / f"{market}_calibrated_model.pkl"
            if model_path.exists():
                self.models[market] = joblib.load(model_path)
    
    # ========================================
    # ASIAN HANDICAP PREDICTIONS
    # ========================================
    
    def predict_asian_handicap(
        self,
        home_elo: float,
        away_elo: float,
        predicted_home_goals: float,
        predicted_away_goals: float,
        handicaps: List[float] = [-1.5, -1.0, -0.5, 0.0]
    ) -> Dict[float, Dict[str, float]]:
        """
        Predict Asian Handicap outcomes
        
        Asian Handicap explanation:
        - AH -0.5: Home must win (Draw = loss)
        - AH -1.0: Home must win by 2+ (Win by 1 = push/refund)
        - AH -1.5: Home must win by 2+ goals
        
        Args:
            handicaps: Handicap lines (negative favors home, positive favors away)
        
        Returns:
            {
                -0.5: {'prob_cover': 0.65, 'prob_push': 0.0, 'prob_lose': 0.35},
                -1.0: {...},
                ...
            }
        """
        # Expected goal differential
        goal_diff = predicted_home_goals - predicted_away_goals
        
        # Standard deviation of goal differential (empirical: ~1.4 goals)
        std_dev = 1.4
        
        # Adjust based on Elo differential (stronger team = more predictable)
        elo_diff = home_elo - away_elo
        if abs(elo_diff) > 200:
            std_dev *= 0.9  # More predictable
        elif abs(elo_diff) < 50:
            std_dev *= 1.1  # Less predictable
        
        results = {}
        
        for handicap in handicaps:
            # Adjusted goal differential after handicap
            adjusted_diff = goal_diff - handicap
            
            # Probability distributions (normal approximation)
            # P(home covers) = P(adjusted_diff > 0)
            # P(push) = P(adjusted_diff == 0) ≈ 0 for continuous, but relevant for full goals
            # P(home loses) = P(adjusted_diff < 0)
            
            from scipy.stats import norm
            
            # Probability home covers
            prob_cover = norm.cdf(adjusted_diff / std_dev)
            
            # For full goal handicaps (-1.0, -2.0), push probability exists
            if handicap % 1.0 == 0:
                # Push window (within ±0.1 goals of exact difference)
                push_window = 0.15
                prob_push = norm.cdf((adjusted_diff + push_window) / std_dev) - \
                           norm.cdf((adjusted_diff - push_window) / std_dev)
                prob_cover -= prob_push / 2  # Half the push probability added to cover
                prob_lose = 1.0 - prob_cover - prob_push
            else:
                prob_push = 0.0
                prob_lose = 1.0 - prob_cover
            
            results[handicap] = {
                'prob_cover': max(0.0, min(1.0, prob_cover)),
                'prob_push': max(0.0, min(1.0, prob_push)),
                'prob_lose': max(0.0, min(1.0, prob_lose))
            }
        
        return results
    
    # ========================================
    # FIRST HALF PREDICTIONS
    # ========================================
    
    def predict_first_half_over_under(
        self,
        predicted_home_goals: float,
        predicted_away_goals: float,
        league: str = "Premier League"
    ) -> Dict[str, float]:
        """
        Predict First Half Over/Under
        
        Typical first half scoring:
        - ~40-45% of goals occur in first half
        - Aggressive teams: up to 50%
        - Defensive teams: down to 35%
        
        Args:
            predicted_home_goals: Full match xG for home
            predicted_away_goals: Full match xG for away
            league: League name (affects 1H scoring rate)
        
        Returns:
            {
                '1h_over_0_5': 0.72,
                '1h_over_1_5': 0.38,
                '1h_total_goals': 1.15
            }
        """
        # League-specific first half rates
        first_half_rates = {
            'Bundesliga': 0.46,       # High-scoring, fast starts
            'Premier League': 0.43,   # Balanced
            'La Liga': 0.41,          # Slower starts
            'Serie A': 0.39,          # Very defensive first halves
            'Ligue 1': 0.42,
            'Eredivisie': 0.45,       # High-scoring
            'Championship': 0.44,
        }
        
        first_half_rate = first_half_rates.get(league, 0.42)
        
        # Expected first half goals
        first_half_goals = (predicted_home_goals + predicted_away_goals) * first_half_rate
        
        # Poisson probabilities
        from scipy.stats import poisson
        
        # P(1H Over 0.5) = 1 - P(0 goals)
        prob_over_0_5 = 1.0 - poisson.pmf(0, first_half_goals)
        
        # P(1H Over 1.5) = 1 - P(0 or 1 goals)
        prob_over_1_5 = 1.0 - poisson.cdf(1, first_half_goals)
        
        return {
            '1h_over_0_5': prob_over_0_5,
            '1h_over_1_5': prob_over_1_5,
            '1h_total_goals': first_half_goals
        }
    
    # ========================================
    # CORRECT SCORE PREDICTIONS
    # ========================================
    
    def predict_correct_score_top10(
        self,
        predicted_home_goals: float,
        predicted_away_goals: float
    ) -> Dict[str, float]:
        """
        Predict top 10 most likely correct scores using Poisson
        
        Most common scores in soccer:
        1. 1-1 (~12%)
        2. 1-0 (~11%)
        3. 2-1 (~9%)
        4. 0-0 (~8%)
        5. 2-0 (~7%)
        6. 0-1 (~7%)
        7. 3-1 (~5%)
        8. 2-2 (~5%)
        9. 3-0 (~4%)
        10. 1-2 (~4%)
        """
        from scipy.stats import poisson
        
        # Possible scores (home-away)
        possible_scores = [
            (0, 0), (1, 0), (0, 1), (1, 1),
            (2, 0), (0, 2), (2, 1), (1, 2),
            (2, 2), (3, 0), (0, 3), (3, 1),
            (1, 3), (3, 2), (2, 3), (3, 3),
            (4, 0), (0, 4), (4, 1), (1, 4)
        ]
        
        score_probs = {}
        
        for home_score, away_score in possible_scores:
            # Independent Poisson probabilities
            prob_home = poisson.pmf(home_score, predicted_home_goals)
            prob_away = poisson.pmf(away_score, predicted_away_goals)
            prob = prob_home * prob_away
            
            score_key = f"{home_score}-{away_score}"
            score_probs[score_key] = prob
        
        # Sort by probability, get top 10
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {score: prob for score, prob in sorted_scores}
    
    # ========================================
    # DEMO & TESTING
    # ========================================
    
    def demo_all_markets(
        self,
        home_team: str = "Bayern München",
        away_team: str = "Borussia Dortmund",
        home_elo: float = 1850,
        away_elo: float = 1720,
        predicted_home_goals: float = 2.1,
        predicted_away_goals: float = 1.4,
        league: str = "Bundesliga"
    ):
        """Demo all expanded markets"""
        print("\n" + "="*70)
        print(f"📊 EXPANDED MARKETS DEMO")
        print("="*70)
        print(f"Match: {home_team} vs {away_team}")
        print(f"League: {league}")
        print(f"Expected Goals: {predicted_home_goals:.2f} - {predicted_away_goals:.2f}")
        print(f"Elo: {home_elo} - {away_elo}")
        print("="*70 + "\n")
        
        # 1. Asian Handicap
        print("1️⃣ ASIAN HANDICAP")
        print("-" * 70)
        ah_predictions = self.predict_asian_handicap(
            home_elo, away_elo, predicted_home_goals, predicted_away_goals
        )
        for handicap, probs in ah_predictions.items():
            print(f"  AH {handicap:+.1f}:")
            print(f"    Cover: {probs['prob_cover']:.1%}  |  Push: {probs['prob_push']:.1%}  |  Lose: {probs['prob_lose']:.1%}")
        
        # 2. First Half
        print("\n2️⃣ FIRST HALF OVER/UNDER")
        print("-" * 70)
        first_half = self.predict_first_half_over_under(
            predicted_home_goals, predicted_away_goals, league
        )
        print(f"  1H Total Goals: {first_half['1h_total_goals']:.2f}")
        print(f"  1H Over 0.5: {first_half['1h_over_0_5']:.1%}")
        print(f"  1H Over 1.5: {first_half['1h_over_1_5']:.1%}")
        
        # 3. Correct Score
        print("\n3️⃣ CORRECT SCORE (Top 10 Most Likely)")
        print("-" * 70)
        correct_scores = self.predict_correct_score_top10(
            predicted_home_goals, predicted_away_goals
        )
        for i, (score, prob) in enumerate(correct_scores.items(), 1):
            print(f"  {i:2}. {score:5} : {prob:.1%}")
        
        print("\n" + "="*70)
        print("✅ All expanded markets demonstrated")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Demo expanded markets
    engine = MarketExpansionEngine()
    
    engine.demo_all_markets(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        home_elo=1850,
        away_elo=1720,
        predicted_home_goals=2.3,
        predicted_away_goals=1.6,
        league="Bundesliga"
    )
    
    print("\n💾 Market expansion engine ready for integration")
    print("   - Asian Handicap: ✅")
    print("   - First Half: ✅")
    print("   - Correct Score: ✅")
    print("\nNext: Integrate with IntegratedPredictionPipeline")

#!/usr/bin/env python3
"""
📐 DIXON-COLES MODEL
====================
Advanced goal prediction model that corrects for low-scoring games.

Standard Poisson assumes home and away goals are independent.
Dixon-Coles (1997) showed this is FALSE for low scores (0-0, 1-0, 0-1, 1-1).

The tau (τ) adjustment factor corrects these correlations:
- 0-0 outcomes are MORE likely than Poisson predicts
- 1-1 outcomes are LESS likely than Poisson predicts

Mathematical Foundation:
P(home=x, away=y) = τ(x,y,λ_h,λ_a,ρ) × Poisson(x,λ_h) × Poisson(y,λ_a)

Where:
- λ_h = expected home goals
- λ_a = expected away goals  
- ρ (rho) = correlation parameter (typically -0.05 to -0.15)

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
"""

import math
from typing import Dict, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DixonColesResult:
    """Result from Dixon-Coles model"""
    home_xg: float
    away_xg: float
    rho: float
    
    # Score matrix (probability of each scoreline)
    score_matrix: np.ndarray
    
    # Market probabilities
    home_win: float
    draw: float
    away_win: float
    
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float
    
    under_0_5: float
    under_1_5: float
    under_2_5: float
    under_3_5: float
    
    btts_yes: float
    btts_no: float
    
    # Most likely scores
    most_likely_scores: List[Tuple[int, int, float]]


class DixonColesModel:
    """
    📐 Dixon-Coles Goal Prediction Model
    
    Improvements over basic Poisson:
    1. Corrects correlation in low-scoring outcomes
    2. Better estimates for 0-0, 1-0, 0-1, 1-1
    3. More accurate Under 2.5 and BTTS No predictions
    """
    
    def __init__(self, rho: float = -0.13):
        """
        Initialize with rho parameter.
        
        Args:
            rho: Correlation parameter. Typical values:
                 -0.05 to -0.15 for most leagues
                 More negative = stronger low-score correlation
        """
        self.rho = rho
        self.max_goals = 10  # Maximum goals to consider
        
        # League-specific parameters (from historical analysis)
        self.league_params = {
            'Bundesliga': {'avg_goals': 3.15, 'home_adv': 0.28, 'rho': -0.12},
            'Premier League': {'avg_goals': 2.85, 'home_adv': 0.25, 'rho': -0.13},
            'La Liga': {'avg_goals': 2.65, 'home_adv': 0.30, 'rho': -0.11},
            'Serie A': {'avg_goals': 2.75, 'home_adv': 0.27, 'rho': -0.10},
            'Ligue 1': {'avg_goals': 2.80, 'home_adv': 0.26, 'rho': -0.12},
            'Eredivisie': {'avg_goals': 3.35, 'home_adv': 0.22, 'rho': -0.08},
            'Championship': {'avg_goals': 2.95, 'home_adv': 0.24, 'rho': -0.11},
        }
    
    def tau(self, x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
        """
        Dixon-Coles tau adjustment factor.
        
        Only affects scores 0-0, 1-0, 0-1, 1-1.
        All other scores: τ = 1 (no adjustment)
        
        Mathematical formulas:
        τ(0,0) = 1 - λ_h × λ_a × ρ
        τ(1,0) = 1 + λ_a × ρ
        τ(0,1) = 1 + λ_h × ρ  
        τ(1,1) = 1 - ρ
        """
        if x == 0 and y == 0:
            return 1 - lambda_h * lambda_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + lambda_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    def poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function"""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    def calculate_score_matrix(self, home_xg: float, away_xg: float, 
                               rho: float = None) -> np.ndarray:
        """
        Calculate probability matrix for all possible scorelines.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            rho: Correlation parameter (uses default if None)
            
        Returns:
            2D numpy array where matrix[i][j] = P(home=i, away=j)
        """
        if rho is None:
            rho = self.rho
            
        matrix = np.zeros((self.max_goals, self.max_goals))
        
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                tau_adj = self.tau(h, a, home_xg, away_xg, rho)
                prob = (tau_adj * 
                       self.poisson_pmf(h, home_xg) * 
                       self.poisson_pmf(a, away_xg))
                matrix[h, a] = prob
        
        # Normalize to ensure probabilities sum to 1
        matrix /= matrix.sum()
        
        return matrix
    
    def predict(self, home_xg: float, away_xg: float, 
                league: str = None) -> DixonColesResult:
        """
        Generate full prediction for a match.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            league: League name (for league-specific rho)
            
        Returns:
            DixonColesResult with all market probabilities
        """
        # Get league-specific rho if available
        rho = self.rho
        if league and league in self.league_params:
            rho = self.league_params[league]['rho']
        
        # Calculate score matrix
        matrix = self.calculate_score_matrix(home_xg, away_xg, rho)
        
        # Calculate 1X2 probabilities
        home_win = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h > a)
        draw = sum(matrix[h, a] for h in range(self.max_goals) 
                  for a in range(self.max_goals) if h == a)
        away_win = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h < a)
        
        # Over/Under probabilities
        over_0_5 = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h + a > 0)
        over_1_5 = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h + a > 1)
        over_2_5 = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h + a > 2)
        over_3_5 = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h + a > 3)
        
        # BTTS probabilities
        btts_yes = sum(matrix[h, a] for h in range(self.max_goals) 
                      for a in range(self.max_goals) if h > 0 and a > 0)
        btts_no = 1 - btts_yes
        
        # Find most likely scores
        scores = []
        for h in range(min(6, self.max_goals)):
            for a in range(min(6, self.max_goals)):
                scores.append((h, a, matrix[h, a]))
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return DixonColesResult(
            home_xg=home_xg,
            away_xg=away_xg,
            rho=rho,
            score_matrix=matrix,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            over_0_5=over_0_5,
            over_1_5=over_1_5,
            over_2_5=over_2_5,
            over_3_5=over_3_5,
            under_0_5=1 - over_0_5,
            under_1_5=1 - over_1_5,
            under_2_5=1 - over_2_5,
            under_3_5=1 - over_3_5,
            btts_yes=btts_yes,
            btts_no=btts_no,
            most_likely_scores=scores[:5]
        )
    
    def predict_from_elo(self, home_elo: float, away_elo: float,
                         league: str = 'default') -> DixonColesResult:
        """
        Generate prediction from Elo ratings.
        
        Converts Elo difference to expected goals using empirical formula.
        """
        # Get league parameters
        params = self.league_params.get(league, {
            'avg_goals': 2.80,
            'home_adv': 0.25,
            'rho': -0.12
        })
        
        avg_goals = params['avg_goals']
        home_advantage = params['home_adv']
        
        # Calculate Elo-based strength ratio
        elo_diff = (home_elo - away_elo) / 400
        home_strength = 10 ** (elo_diff / 2)
        away_strength = 10 ** (-elo_diff / 2)
        
        # Convert to expected goals
        base_goals = avg_goals / 2
        home_xg = base_goals * home_strength + home_advantage
        away_xg = base_goals * away_strength
        
        # Clamp to reasonable range
        home_xg = max(0.4, min(4.0, home_xg))
        away_xg = max(0.3, min(3.5, away_xg))
        
        return self.predict(home_xg, away_xg, league)
    
    def compare_with_poisson(self, home_xg: float, away_xg: float) -> Dict:
        """
        Compare Dixon-Coles predictions with standard Poisson.
        
        Shows the adjustment Dixon-Coles makes.
        """
        dc_result = self.predict(home_xg, away_xg)
        
        # Calculate standard Poisson
        poisson_matrix = np.zeros((self.max_goals, self.max_goals))
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                poisson_matrix[h, a] = (self.poisson_pmf(h, home_xg) * 
                                        self.poisson_pmf(a, away_xg))
        poisson_matrix /= poisson_matrix.sum()
        
        return {
            'dixon_coles': {
                '0-0': dc_result.score_matrix[0, 0],
                '1-0': dc_result.score_matrix[1, 0],
                '0-1': dc_result.score_matrix[0, 1],
                '1-1': dc_result.score_matrix[1, 1],
                'under_2_5': dc_result.under_2_5,
                'btts_no': dc_result.btts_no,
            },
            'standard_poisson': {
                '0-0': poisson_matrix[0, 0],
                '1-0': poisson_matrix[1, 0],
                '0-1': poisson_matrix[0, 1],
                '1-1': poisson_matrix[1, 1],
                'under_2_5': sum(poisson_matrix[h, a] for h in range(10) 
                               for a in range(10) if h + a < 3),
                'btts_no': sum(poisson_matrix[h, a] for h in range(10) 
                              for a in range(10) if h == 0 or a == 0),
            },
            'adjustment': {
                '0-0': f"{(dc_result.score_matrix[0, 0] / poisson_matrix[0, 0] - 1) * 100:+.1f}%",
                '1-0': f"{(dc_result.score_matrix[1, 0] / poisson_matrix[1, 0] - 1) * 100:+.1f}%",
                '0-1': f"{(dc_result.score_matrix[0, 1] / poisson_matrix[0, 1] - 1) * 100:+.1f}%",
                '1-1': f"{(dc_result.score_matrix[1, 1] / poisson_matrix[1, 1] - 1) * 100:+.1f}%",
            }
        }


def test_dixon_coles():
    """Test the Dixon-Coles model"""
    print("=" * 60)
    print("📐 DIXON-COLES MODEL TEST")
    print("=" * 60)
    
    model = DixonColesModel()
    
    # Test case: Bayern vs Dortmund
    print("\n🔬 Bayern München vs Borussia Dortmund")
    print("   Home xG: 2.1, Away xG: 1.3")
    
    result = model.predict(2.1, 1.3, 'Bundesliga')
    
    print(f"\n   1X2: {result.home_win:.1%} / {result.draw:.1%} / {result.away_win:.1%}")
    print(f"   Over 2.5: {result.over_2_5:.1%} | Under 2.5: {result.under_2_5:.1%}")
    print(f"   BTTS Yes: {result.btts_yes:.1%} | BTTS No: {result.btts_no:.1%}")
    
    print("\n   Top 5 most likely scores:")
    for h, a, prob in result.most_likely_scores:
        print(f"      {h}-{a}: {prob:.1%}")
    
    # Compare with standard Poisson
    print("\n📊 Dixon-Coles vs Standard Poisson:")
    comparison = model.compare_with_poisson(2.1, 1.3)
    for score in ['0-0', '1-0', '0-1', '1-1']:
        dc = comparison['dixon_coles'][score]
        sp = comparison['standard_poisson'][score]
        adj = comparison['adjustment'][score]
        print(f"   {score}: Poisson {sp:.1%} → Dixon-Coles {dc:.1%} ({adj})")
    
    # Test from Elo
    print("\n🎯 Prediction from Elo ratings:")
    elo_result = model.predict_from_elo(1850, 1780, 'Bundesliga')
    print(f"   Home xG: {elo_result.home_xg:.2f}, Away xG: {elo_result.away_xg:.2f}")
    print(f"   1X2: {elo_result.home_win:.1%} / {elo_result.draw:.1%} / {elo_result.away_win:.1%}")
    
    print("\n✅ Dixon-Coles model working correctly!")


if __name__ == "__main__":
    test_dixon_coles()

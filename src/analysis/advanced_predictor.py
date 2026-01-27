#!/usr/bin/env python3
"""
🧠 ADVANCED SOCCER PREDICTOR
============================
Extreme Power Prediction System using ALL Knowledge Files

Features:
- ALL Betting Markets (Over/Under, BTTS, Double Chance, Asian Handicap, 1X2)
- Optimal Odds Filter: 1.30-1.70 (Golden Range)
- League-Specific Adjustments (Bundesliga favors Over, La Liga favors Under)
- Form & Momentum Analysis
- H2H Derby Detection
- Kelly Criterion Staking
- Value Bet Detection (Min 5% Edge)
- Poisson Goal Distribution

Based on Knowledge Files:
- BETTING_MATHEMATICS.md
- OPTIMAL_ODDS_RANGES.md
- ALL_BETTING_MARKETS.md
- LEAGUE_STATISTICS.md
- FORM_ANALYSIS.md
- HEAD_TO_HEAD.md
- BANKROLL_MANAGEMENT.md
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class Market(Enum):
    """All supported betting markets"""
    # Goal Markets
    OVER_0_5 = "over_0_5"
    OVER_1_5 = "over_1_5"
    OVER_2_5 = "over_2_5"
    OVER_3_5 = "over_3_5"
    UNDER_0_5 = "under_0_5"
    UNDER_1_5 = "under_1_5"
    UNDER_2_5 = "under_2_5"
    UNDER_3_5 = "under_3_5"
    
    # BTTS Markets
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"
    
    # Match Result (1X2)
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    
    # Double Chance
    HOME_OR_DRAW = "home_or_draw"
    AWAY_OR_DRAW = "away_or_draw"
    HOME_OR_AWAY = "home_or_away"
    
    # Draw No Bet (DNB)
    DNB_HOME = "dnb_home"
    DNB_AWAY = "dnb_away"
    
    # Asian Handicap (Most Common)
    AH_HOME_MINUS_0_5 = "ah_home_-0.5"
    AH_HOME_MINUS_1 = "ah_home_-1"
    AH_HOME_MINUS_1_5 = "ah_home_-1.5"
    AH_AWAY_PLUS_0_5 = "ah_away_+0.5"
    AH_AWAY_PLUS_1 = "ah_away_+1"
    AH_AWAY_PLUS_1_5 = "ah_away_+1.5"


# Optimal Odds Ranges (from OPTIMAL_ODDS_RANGES.md)
OPTIMAL_ODDS_RANGE = {
    'min': 1.30,
    'max': 1.70,
    'sweet_spot_min': 1.40,
    'sweet_spot_max': 1.60
}

# Market-Specific Optimal Ranges (from knowledge)
MARKET_ODDS_RANGES = {
    Market.OVER_1_5: {'min': 1.20, 'max': 1.50, 'tier': 1},
    Market.OVER_2_5: {'min': 1.70, 'max': 2.20, 'tier': 2},
    Market.BTTS_YES: {'min': 1.60, 'max': 2.00, 'tier': 2},
    Market.BTTS_NO: {'min': 1.65, 'max': 2.10, 'tier': 2},
    Market.HOME_OR_DRAW: {'min': 1.25, 'max': 1.55, 'tier': 1},
    Market.AWAY_OR_DRAW: {'min': 1.30, 'max': 1.70, 'tier': 2},
    Market.HOME_WIN: {'min': 1.35, 'max': 2.20, 'tier': 2},
    Market.AWAY_WIN: {'min': 1.80, 'max': 3.00, 'tier': 3},
    Market.DRAW: {'min': 3.00, 'max': 4.00, 'tier': 3},
    Market.DNB_HOME: {'min': 1.40, 'max': 1.80, 'tier': 2},
    Market.DNB_AWAY: {'min': 1.60, 'max': 2.20, 'tier': 3},
    Market.UNDER_2_5: {'min': 1.55, 'max': 2.10, 'tier': 2},
    Market.UNDER_3_5: {'min': 1.20, 'max': 1.45, 'tier': 1},
    Market.OVER_3_5: {'min': 2.20, 'max': 3.50, 'tier': 3},
}

# League Profiles (from LEAGUE_STATISTICS.md)
LEAGUE_PROFILES = {
    'bundesliga': {
        'avg_goals': 3.18,
        'over_2_5_rate': 0.552,
        'over_1_5_rate': 0.785,
        'btts_rate': 0.528,
        'home_win_rate': 0.453,
        'draw_rate': 0.221,
        'away_win_rate': 0.326,
        'best_markets': [Market.OVER_2_5, Market.BTTS_YES, Market.OVER_1_5],
        'avoid_markets': [Market.UNDER_2_5, Market.BTTS_NO],
        'confidence_boost': 1.05
    },
    'premier_league': {
        'avg_goals': 2.85,
        'over_2_5_rate': 0.501,
        'over_1_5_rate': 0.753,
        'btts_rate': 0.512,
        'home_win_rate': 0.428,
        'draw_rate': 0.245,
        'away_win_rate': 0.327,
        'best_markets': [Market.BTTS_YES, Market.OVER_2_5, Market.HOME_OR_DRAW],
        'avoid_markets': [],
        'confidence_boost': 1.00
    },
    'la_liga': {
        'avg_goals': 2.58,
        'over_2_5_rate': 0.458,
        'over_1_5_rate': 0.712,
        'btts_rate': 0.463,
        'home_win_rate': 0.472,
        'draw_rate': 0.248,
        'away_win_rate': 0.280,
        'best_markets': [Market.UNDER_2_5, Market.HOME_WIN, Market.BTTS_NO],
        'avoid_markets': [Market.OVER_2_5, Market.OVER_3_5],
        'confidence_boost': 0.95
    },
    'serie_a': {
        'avg_goals': 2.78,
        'over_2_5_rate': 0.482,
        'over_1_5_rate': 0.738,
        'btts_rate': 0.495,
        'home_win_rate': 0.445,
        'draw_rate': 0.268,
        'away_win_rate': 0.287,
        'best_markets': [Market.HOME_OR_DRAW, Market.UNDER_2_5],
        'avoid_markets': [],
        'confidence_boost': 1.00
    },
    'ligue_1': {
        'avg_goals': 2.72,
        'over_2_5_rate': 0.475,
        'over_1_5_rate': 0.728,
        'btts_rate': 0.478,
        'home_win_rate': 0.448,
        'draw_rate': 0.252,
        'away_win_rate': 0.300,
        'best_markets': [Market.OVER_1_5, Market.HOME_WIN],
        'avoid_markets': [Market.BTTS_YES],
        'confidence_boost': 1.00
    },
    'eredivisie': {
        'avg_goals': 3.25,
        'over_2_5_rate': 0.583,
        'over_1_5_rate': 0.812,
        'btts_rate': 0.558,
        'home_win_rate': 0.482,
        'draw_rate': 0.205,
        'away_win_rate': 0.313,
        'best_markets': [Market.OVER_2_5, Market.BTTS_YES, Market.OVER_3_5],
        'avoid_markets': [Market.UNDER_2_5, Market.UNDER_1_5],
        'confidence_boost': 1.08
    },
    'championship': {
        'avg_goals': 2.68,
        'over_2_5_rate': 0.465,
        'over_1_5_rate': 0.718,
        'btts_rate': 0.482,
        'home_win_rate': 0.435,
        'draw_rate': 0.278,
        'away_win_rate': 0.287,
        'best_markets': [Market.DRAW, Market.UNDER_2_5, Market.HOME_OR_DRAW],
        'avoid_markets': [Market.OVER_3_5],
        'confidence_boost': 0.95
    }
}

# Known Derbies (from HEAD_TO_HEAD.md)
KNOWN_DERBIES = {
    'bundesliga': [
        ('Bayern München', 'Borussia Dortmund', 'Der Klassiker'),
        ('Borussia Dortmund', 'Schalke 04', 'Revierderby'),
        ('Hamburger SV', 'Werder Bremen', 'Nordderby'),
        ('Eintracht Frankfurt', '1. FSV Mainz 05', 'Rhein-Main Derby'),
        ('1. FC Köln', 'Borussia Mönchengladbach', 'Rheinderby'),
        ('VfB Stuttgart', 'Karlsruher SC', 'Baden-Württemberg Derby'),
    ],
    'premier_league': [
        ('Liverpool', 'Manchester United', 'North-West Derby'),
        ('Manchester City', 'Manchester United', 'Manchester Derby'),
        ('Arsenal', 'Tottenham Hotspur', 'North London Derby'),
        ('Liverpool', 'Everton', 'Merseyside Derby'),
        ('Chelsea', 'Tottenham Hotspur', 'London Derby'),
        ('Aston Villa', 'Birmingham City', 'Second City Derby'),
    ],
    'la_liga': [
        ('Real Madrid', 'Barcelona', 'El Clásico'),
        ('Real Madrid', 'Atlético Madrid', 'Derby Madrileño'),
        ('Barcelona', 'Espanyol', 'Derbi Barceloní'),
        ('Athletic Bilbao', 'Real Sociedad', 'Derbi Vasco'),
        ('Sevilla', 'Real Betis', 'Derbi Sevillano'),
    ],
    'serie_a': [
        ('AC Milan', 'Inter Milan', 'Derby della Madonnina'),
        ('Roma', 'Lazio', 'Derby della Capitale'),
        ('Juventus', 'Torino', 'Derby della Mole'),
        ('Genoa', 'Sampdoria', 'Derby della Lanterna'),
    ]
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TeamStats:
    """Team statistics for predictions"""
    name: str
    elo: float = 1500.0
    form_points: int = 0  # Last 5 matches (max 15)
    home_form: int = 0
    away_form: int = 0
    goals_scored_last_5: float = 0.0
    goals_conceded_last_5: float = 0.0
    clean_sheet_rate: float = 0.0
    failed_to_score_rate: float = 0.0
    

@dataclass
class MatchContext:
    """Full context for a match"""
    home_team: TeamStats
    away_team: TeamStats
    league: str
    date: str
    is_derby: bool = False
    derby_name: str = ""
    h2h_data: Dict = field(default_factory=dict)


@dataclass
class Prediction:
    """Single market prediction"""
    market: Market
    probability: float
    odds: float
    implied_prob: float
    edge: float
    value: float
    kelly_stake: float
    confidence_tier: str  # HIGH, MEDIUM, LOW
    is_value_bet: bool
    is_in_optimal_range: bool
    recommendation: str  # BET, SKIP, AVOID


@dataclass
class MatchPredictions:
    """All predictions for a match"""
    match_context: MatchContext
    predictions: List[Prediction]
    best_bet: Optional[Prediction]
    accumulator_legs: List[Prediction]  # Good for accas
    analysis_notes: List[str]


# ============================================================================
# MATHEMATICAL ENGINE (from BETTING_MATHEMATICS.md)
# ============================================================================

class BettingMath:
    """All mathematical formulas from knowledge base"""
    
    @staticmethod
    def implied_probability(odds: float) -> float:
        """Calculate implied probability from decimal odds"""
        if odds <= 1:
            return 1.0
        return 1 / odds
    
    @staticmethod
    def fair_odds(probability: float) -> float:
        """Calculate fair odds from probability"""
        if probability <= 0:
            return float('inf')
        return 1 / probability
    
    @staticmethod
    def value(our_prob: float, odds: float) -> float:
        """Value = (Our Probability × Odds) - 1"""
        return (our_prob * odds) - 1
    
    @staticmethod
    def edge(our_prob: float, odds: float) -> float:
        """Edge = Our Probability - Implied Probability"""
        implied = BettingMath.implied_probability(odds)
        return our_prob - implied
    
    @staticmethod
    def kelly_criterion(probability: float, odds: float, fraction: float = 0.25) -> float:
        """
        Kelly Fraction = (p × b - q) / b
        
        p = Our probability
        q = 1 - p
        b = Odds - 1
        
        Returns fractional Kelly stake (25% by default for safety)
        """
        p = probability
        q = 1 - p
        b = odds - 1
        
        if b <= 0:
            return 0
        
        full_kelly = (p * b - q) / b
        fractional = full_kelly * fraction
        
        # Cap at 5% max stake
        return max(0, min(fractional, 0.05))
    
    @staticmethod
    def expected_value(probability: float, odds: float, stake: float = 1.0) -> float:
        """EV = Stake × [(Probability × Odds) - 1]"""
        return stake * ((probability * odds) - 1)
    
    @staticmethod
    def poisson_probability(k: int, lambda_: float) -> float:
        """
        Poisson Distribution for goal prediction
        P(X = k) = (λ^k × e^(-λ)) / k!
        """
        if lambda_ <= 0:
            return 0 if k > 0 else 1
        return (math.pow(lambda_, k) * math.exp(-lambda_)) / math.factorial(k)
    
    @staticmethod
    def over_under_probability(home_xg: float, away_xg: float, line: float) -> Tuple[float, float]:
        """
        Calculate Over/Under probabilities using Poisson distribution
        
        Returns: (over_prob, under_prob)
        """
        max_goals = 10
        under_prob = 0.0
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                total = home_goals + away_goals
                if total <= line:
                    home_prob = BettingMath.poisson_probability(home_goals, home_xg)
                    away_prob = BettingMath.poisson_probability(away_goals, away_xg)
                    under_prob += home_prob * away_prob
        
        over_prob = 1 - under_prob
        return (over_prob, under_prob)
    
    @staticmethod
    def btts_probability(home_xg: float, away_xg: float) -> Tuple[float, float]:
        """
        Calculate BTTS Yes/No probabilities
        
        BTTS Yes = P(Home Scores) × P(Away Scores)
        Returns: (btts_yes, btts_no)
        """
        # P(Team scores at least 1) = 1 - P(Team scores 0)
        home_scores = 1 - BettingMath.poisson_probability(0, home_xg)
        away_scores = 1 - BettingMath.poisson_probability(0, away_xg)
        
        btts_yes = home_scores * away_scores
        btts_no = 1 - btts_yes
        
        return (btts_yes, btts_no)
    
    @staticmethod
    def match_result_probabilities(home_xg: float, away_xg: float) -> Dict[str, float]:
        """
        Calculate 1X2 probabilities using Poisson
        
        Returns: {'home_win': p, 'draw': p, 'away_win': p}
        """
        max_goals = 8
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                home_prob = BettingMath.poisson_probability(home_goals, home_xg)
                away_prob = BettingMath.poisson_probability(away_goals, away_xg)
                score_prob = home_prob * away_prob
                
                if home_goals > away_goals:
                    home_win += score_prob
                elif home_goals == away_goals:
                    draw += score_prob
                else:
                    away_win += score_prob
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
    
    @staticmethod
    def is_value_bet(our_prob: float, odds: float, min_edge: float = 0.05) -> bool:
        """Check if bet has positive expected value"""
        edge = BettingMath.edge(our_prob, odds)
        return edge >= min_edge


# ============================================================================
# FORM ANALYZER (from FORM_ANALYSIS.md)
# ============================================================================

class FormAnalyzer:
    """Analyze team form and momentum"""
    
    @staticmethod
    def weighted_form(results: List[str]) -> float:
        """
        Weighted form - recent games matter more
        
        Weights: [0.35, 0.25, 0.20, 0.12, 0.08] (newest first)
        """
        weights = [0.35, 0.25, 0.20, 0.12, 0.08]
        points = {'W': 3, 'D': 1, 'L': 0}
        
        if not results:
            return 50.0
        
        weighted_sum = sum(
            weights[i] * points.get(results[i], 0)
            for i in range(min(5, len(results)))
        )
        
        max_weighted = sum(w * 3 for w in weights[:len(results)])
        if max_weighted == 0:
            return 50.0
        
        return (weighted_sum / max_weighted) * 100
    
    @staticmethod
    def momentum_score(results: List[str]) -> float:
        """
        Momentum Score: -100 (terrible) to +100 (excellent)
        """
        points = {'W': 10, 'D': 0, 'L': -10}
        weights = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        if not results:
            return 0.0
        
        score = sum(
            weights[i] * points.get(results[i], 0)
            for i in range(min(5, len(results)))
        )
        
        return score * 10  # Scale to -100 to +100
    
    @staticmethod
    def detect_trend(results: List[str]) -> Tuple[str, float]:
        """
        Detect form trend (rising, stable, falling)
        
        Returns: (trend, adjustment_factor)
        """
        points = {'W': 3, 'D': 1, 'L': 0}
        
        if len(results) < 3:
            return ('stable', 1.0)
        
        recent = sum(points.get(r, 0) for r in results[:3])
        older = sum(points.get(r, 0) for r in results[3:6]) if len(results) >= 6 else recent
        
        diff = recent - older
        
        if diff >= 4:
            return ('rising_strong', 1.10)
        elif diff >= 2:
            return ('rising', 1.05)
        elif diff <= -4:
            return ('falling_strong', 0.90)
        elif diff <= -2:
            return ('falling', 0.95)
        else:
            return ('stable', 1.00)
    
    @staticmethod
    def form_rating(form_points: int) -> str:
        """Convert form points to rating"""
        if form_points >= 13:
            return 'EXCELLENT'
        elif form_points >= 10:
            return 'GOOD'
        elif form_points >= 7:
            return 'AVERAGE'
        elif form_points >= 4:
            return 'POOR'
        else:
            return 'TERRIBLE'


# ============================================================================
# ADVANCED PREDICTOR ENGINE
# ============================================================================

class AdvancedPredictor:
    """
    🧠 Ultimate Prediction Engine
    
    Combines ALL knowledge from:
    - BETTING_MATHEMATICS.md
    - OPTIMAL_ODDS_RANGES.md
    - ALL_BETTING_MARKETS.md
    - LEAGUE_STATISTICS.md
    - FORM_ANALYSIS.md
    - HEAD_TO_HEAD.md
    - BANKROLL_MANAGEMENT.md
    """
    
    def __init__(self, bankroll: float = 1000.0):
        self.math = BettingMath()
        self.form_analyzer = FormAnalyzer()
        self.bankroll = bankroll
        
        # Settings
        self.min_edge = 0.05  # 5% minimum edge for value
        self.odds_filter_enabled = True
        self.optimal_odds_min = OPTIMAL_ODDS_RANGE['min']  # 1.30
        self.optimal_odds_max = OPTIMAL_ODDS_RANGE['max']  # 1.70
    
    def is_derby(self, home: str, away: str, league: str) -> Tuple[bool, str]:
        """Check if match is a derby"""
        league_key = league.lower().replace(' ', '_')
        derbies = KNOWN_DERBIES.get(league_key, [])
        
        for team_a, team_b, name in derbies:
            if (home in team_a or team_a in home) and (away in team_b or team_b in away):
                return (True, name)
            if (away in team_a or team_a in away) and (home in team_b or team_b in home):
                return (True, name)
        
        return (False, "")
    
    def get_league_profile(self, league: str) -> Dict:
        """Get league-specific statistics"""
        league_key = league.lower().replace(' ', '_')
        return LEAGUE_PROFILES.get(league_key, LEAGUE_PROFILES['premier_league'])
    
    def estimate_expected_goals(self, home: TeamStats, away: TeamStats, league_profile: Dict) -> Tuple[float, float]:
        """
        Estimate xG for each team based on:
        - Elo ratings
        - Form
        - League averages
        """
        league_avg = league_profile['avg_goals'] / 2  # Per team
        
        # Elo-based adjustment
        elo_diff = home.elo - away.elo
        home_elo_factor = 1 + (elo_diff / 800)  # Softer scaling
        away_elo_factor = 1 - (elo_diff / 800)
        
        # Form adjustment
        home_form_factor = (home.form_points / 15) * 0.3 + 0.85  # 0.85 to 1.15
        away_form_factor = (away.form_points / 15) * 0.3 + 0.85
        
        # Attack/Defense balance
        home_attack = home.goals_scored_last_5 / 5 if home.goals_scored_last_5 > 0 else league_avg
        away_attack = away.goals_scored_last_5 / 5 if away.goals_scored_last_5 > 0 else league_avg
        
        home_defense = away.goals_conceded_last_5 / 5 if away.goals_conceded_last_5 > 0 else league_avg
        away_defense = home.goals_conceded_last_5 / 5 if home.goals_conceded_last_5 > 0 else league_avg
        
        # Calculate xG
        home_xg = (league_avg * 0.3 + home_attack * 0.35 + home_defense * 0.35) * home_elo_factor * home_form_factor
        away_xg = (league_avg * 0.3 + away_attack * 0.35 + away_defense * 0.35) * away_elo_factor * away_form_factor
        
        # Home advantage
        home_xg *= 1.08
        away_xg *= 0.92
        
        # Cap reasonable range
        home_xg = max(0.5, min(3.5, home_xg))
        away_xg = max(0.3, min(3.0, away_xg))
        
        return (home_xg, away_xg)
    
    def calculate_all_probabilities(self, home_xg: float, away_xg: float) -> Dict[Market, float]:
        """Calculate probabilities for ALL markets using Poisson"""
        probs = {}
        
        # Over/Under Goals
        over_0_5, under_0_5 = self.math.over_under_probability(home_xg, away_xg, 0.5)
        over_1_5, under_1_5 = self.math.over_under_probability(home_xg, away_xg, 1.5)
        over_2_5, under_2_5 = self.math.over_under_probability(home_xg, away_xg, 2.5)
        over_3_5, under_3_5 = self.math.over_under_probability(home_xg, away_xg, 3.5)
        
        probs[Market.OVER_0_5] = over_0_5
        probs[Market.UNDER_0_5] = under_0_5
        probs[Market.OVER_1_5] = over_1_5
        probs[Market.UNDER_1_5] = under_1_5
        probs[Market.OVER_2_5] = over_2_5
        probs[Market.UNDER_2_5] = under_2_5
        probs[Market.OVER_3_5] = over_3_5
        probs[Market.UNDER_3_5] = under_3_5
        
        # BTTS
        btts_yes, btts_no = self.math.btts_probability(home_xg, away_xg)
        probs[Market.BTTS_YES] = btts_yes
        probs[Market.BTTS_NO] = btts_no
        
        # 1X2
        result_probs = self.math.match_result_probabilities(home_xg, away_xg)
        probs[Market.HOME_WIN] = result_probs['home_win']
        probs[Market.DRAW] = result_probs['draw']
        probs[Market.AWAY_WIN] = result_probs['away_win']
        
        # Double Chance
        probs[Market.HOME_OR_DRAW] = result_probs['home_win'] + result_probs['draw']
        probs[Market.AWAY_OR_DRAW] = result_probs['away_win'] + result_probs['draw']
        probs[Market.HOME_OR_AWAY] = result_probs['home_win'] + result_probs['away_win']
        
        # Draw No Bet (DNB) - Stake returned on draw
        non_draw = result_probs['home_win'] + result_probs['away_win']
        if non_draw > 0:
            probs[Market.DNB_HOME] = result_probs['home_win'] / non_draw
            probs[Market.DNB_AWAY] = result_probs['away_win'] / non_draw
        else:
            probs[Market.DNB_HOME] = 0.5
            probs[Market.DNB_AWAY] = 0.5
        
        # Asian Handicap (Simplified - full implementation would need more logic)
        # AH -0.5 = Win, AH +0.5 = Win or Draw
        probs[Market.AH_HOME_MINUS_0_5] = result_probs['home_win']
        probs[Market.AH_AWAY_PLUS_0_5] = result_probs['away_win'] + result_probs['draw']
        
        # AH -1 / +1 (Win by 2+ / Lose by <1)
        probs[Market.AH_HOME_MINUS_1] = self._asian_handicap_prob(home_xg, away_xg, -1.0)
        probs[Market.AH_AWAY_PLUS_1] = 1 - self._asian_handicap_prob(home_xg, away_xg, -1.0)
        
        # AH -1.5 / +1.5
        probs[Market.AH_HOME_MINUS_1_5] = self._asian_handicap_prob(home_xg, away_xg, -1.5)
        probs[Market.AH_AWAY_PLUS_1_5] = 1 - self._asian_handicap_prob(home_xg, away_xg, -1.5)
        
        return probs
    
    def _asian_handicap_prob(self, home_xg: float, away_xg: float, handicap: float) -> float:
        """Calculate Asian Handicap probability"""
        max_goals = 8
        prob = 0.0
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                adjusted_diff = home_goals - away_goals + handicap
                
                if adjusted_diff > 0:
                    home_prob = self.math.poisson_probability(home_goals, home_xg)
                    away_prob = self.math.poisson_probability(away_goals, away_xg)
                    prob += home_prob * away_prob
        
        return prob
    
    def apply_league_adjustment(self, probs: Dict[Market, float], league_profile: Dict) -> Dict[Market, float]:
        """Apply league-specific confidence adjustments"""
        adjusted = probs.copy()
        
        boost = league_profile.get('confidence_boost', 1.0)
        best_markets = league_profile.get('best_markets', [])
        avoid_markets = league_profile.get('avoid_markets', [])
        
        for market, prob in probs.items():
            if market in best_markets:
                adjusted[market] = min(0.95, prob * (boost * 1.05))
            elif market in avoid_markets:
                adjusted[market] = prob * 0.90
            else:
                adjusted[market] = prob * boost
        
        return adjusted
    
    def apply_derby_adjustment(self, probs: Dict[Market, float]) -> Dict[Market, float]:
        """Apply derby-specific adjustments (more unpredictable)"""
        adjusted = probs.copy()
        
        # Pull extreme probabilities towards center
        for market in [Market.HOME_WIN, Market.AWAY_WIN]:
            if market in adjusted:
                adjusted[market] = 0.33 + (adjusted[market] - 0.33) * 0.7
        
        # Increase draw probability
        if Market.DRAW in adjusted:
            adjusted[Market.DRAW] = min(0.40, adjusted[Market.DRAW] * 1.15)
        
        # Slightly reduce goal expectations
        for market in [Market.OVER_2_5, Market.OVER_3_5]:
            if market in adjusted:
                adjusted[market] *= 0.92
        
        # Recalculate double chances
        if Market.HOME_WIN in adjusted and Market.DRAW in adjusted:
            adjusted[Market.HOME_OR_DRAW] = adjusted[Market.HOME_WIN] + adjusted[Market.DRAW]
        if Market.AWAY_WIN in adjusted and Market.DRAW in adjusted:
            adjusted[Market.AWAY_OR_DRAW] = adjusted[Market.AWAY_WIN] + adjusted[Market.DRAW]
        
        return adjusted
    
    def simulate_market_odds(self, market: Market, probability: float, league_profile: Dict) -> float:
        """Generate realistic odds based on probability and market"""
        # Fair odds
        fair = self.math.fair_odds(probability) if probability > 0 else 10.0
        
        # Add margin (bookmaker profit)
        margin = np.random.uniform(0.03, 0.08)  # 3-8% margin
        odds_with_margin = fair / (1 + margin)
        
        # Market-specific adjustments
        market_range = MARKET_ODDS_RANGES.get(market, {'min': 1.20, 'max': 3.00})
        
        # Clip to realistic range
        odds = max(market_range['min'], min(market_range['max'], odds_with_margin))
        
        return round(odds, 2)
    
    def evaluate_prediction(self, market: Market, probability: float, odds: float) -> Prediction:
        """Create full prediction evaluation"""
        implied_prob = self.math.implied_probability(odds)
        edge = self.math.edge(probability, odds)
        value = self.math.value(probability, odds)
        kelly = self.math.kelly_criterion(probability, odds, fraction=0.25)
        is_value = edge >= self.min_edge
        
        # Check if in optimal range
        market_range = MARKET_ODDS_RANGES.get(market, {'min': 1.30, 'max': 1.70})
        in_optimal_range = self.optimal_odds_min <= odds <= self.optimal_odds_max
        in_market_range = market_range['min'] <= odds <= market_range['max']
        
        # Determine confidence tier
        if edge >= 0.10 and probability >= 0.70:
            confidence = 'HIGH'
        elif edge >= 0.05 and probability >= 0.60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Determine recommendation
        if is_value and in_optimal_range and probability >= 0.60:
            recommendation = 'BET'
        elif is_value and in_market_range and probability >= 0.55:
            recommendation = 'CONSIDER'
        elif edge < 0:
            recommendation = 'AVOID'
        else:
            recommendation = 'SKIP'
        
        return Prediction(
            market=market,
            probability=probability,
            odds=odds,
            implied_prob=implied_prob,
            edge=edge,
            value=value,
            kelly_stake=kelly,
            confidence_tier=confidence,
            is_value_bet=is_value,
            is_in_optimal_range=in_optimal_range,
            recommendation=recommendation
        )
    
    def predict_match(self, context: MatchContext) -> MatchPredictions:
        """
        Generate ALL market predictions for a match
        
        This is the main prediction function!
        """
        # Get league profile
        league_profile = self.get_league_profile(context.league)
        
        # Calculate expected goals
        home_xg, away_xg = self.estimate_expected_goals(
            context.home_team, 
            context.away_team, 
            league_profile
        )
        
        # Calculate base probabilities for all markets
        base_probs = self.calculate_all_probabilities(home_xg, away_xg)
        
        # Apply league adjustments
        adjusted_probs = self.apply_league_adjustment(base_probs, league_profile)
        
        # Apply derby adjustments if needed
        if context.is_derby:
            adjusted_probs = self.apply_derby_adjustment(adjusted_probs)
        
        # Generate predictions for all markets
        predictions = []
        notes = []
        
        notes.append(f"📊 Expected Goals: {context.home_team.name} {home_xg:.2f} - {away_xg:.2f} {context.away_team.name}")
        notes.append(f"🏆 League: {context.league} (Avg Goals: {league_profile['avg_goals']:.2f})")
        
        if context.is_derby:
            notes.append(f"⚔️ DERBY: {context.derby_name}")
        
        # Generate odds and evaluate each market
        for market, probability in adjusted_probs.items():
            odds = self.simulate_market_odds(market, probability, league_profile)
            prediction = self.evaluate_prediction(market, probability, odds)
            predictions.append(prediction)
        
        # Sort by edge (best value first)
        predictions.sort(key=lambda p: p.edge, reverse=True)
        
        # Find best bet
        value_bets = [p for p in predictions if p.recommendation == 'BET']
        best_bet = value_bets[0] if value_bets else None
        
        # Find good accumulator legs (high probability, decent odds)
        acca_legs = [
            p for p in predictions 
            if p.probability >= 0.70 
            and 1.20 <= p.odds <= 1.50 
            and p.edge >= 0.03
        ][:3]  # Max 3 legs per match for accas
        
        if best_bet:
            notes.append(f"🎯 Best Bet: {best_bet.market.value} @ {best_bet.odds} (Edge: {best_bet.edge:.1%})")
        
        return MatchPredictions(
            match_context=context,
            predictions=predictions,
            best_bet=best_bet,
            accumulator_legs=acca_legs,
            analysis_notes=notes
        )
    
    def filter_by_optimal_range(self, predictions: List[Prediction]) -> List[Prediction]:
        """Filter predictions to only include those in optimal odds range 1.30-1.70"""
        return [
            p for p in predictions 
            if self.optimal_odds_min <= p.odds <= self.optimal_odds_max
        ]
    
    def get_recommendations(self, match_predictions: MatchPredictions, 
                            max_picks: int = 3,
                            only_optimal_range: bool = True) -> List[Prediction]:
        """
        Get final betting recommendations
        
        Filters by:
        - Optimal odds range (1.30-1.70)
        - Positive edge (5%+ minimum)
        - High probability (60%+ minimum)
        """
        candidates = match_predictions.predictions
        
        if only_optimal_range:
            candidates = self.filter_by_optimal_range(candidates)
        
        # Filter value bets only
        value_bets = [
            p for p in candidates 
            if p.is_value_bet and p.probability >= 0.60
        ]
        
        # Sort by expected value
        value_bets.sort(key=lambda p: p.value, reverse=True)
        
        return value_bets[:max_picks]


# ============================================================================
# TELEGRAM FORMATTER
# ============================================================================

class TelegramFormatter:
    """Format predictions for Telegram"""
    
    @staticmethod
    def format_prediction(pred: Prediction, match_context: MatchContext) -> str:
        """Format single prediction for Telegram"""
        emoji_map = {
            'BET': '✅',
            'CONSIDER': '🤔',
            'SKIP': '⏭️',
            'AVOID': '❌'
        }
        
        market_names = {
            Market.OVER_1_5: 'Over 1.5 Goals',
            Market.OVER_2_5: 'Over 2.5 Goals',
            Market.BTTS_YES: 'Both Teams to Score',
            Market.BTTS_NO: 'No BTTS',
            Market.HOME_WIN: f'{match_context.home_team.name} Win',
            Market.AWAY_WIN: f'{match_context.away_team.name} Win',
            Market.DRAW: 'Draw',
            Market.HOME_OR_DRAW: f'{match_context.home_team.name} or Draw',
            Market.AWAY_OR_DRAW: f'{match_context.away_team.name} or Draw',
            Market.DNB_HOME: f'{match_context.home_team.name} DNB',
            Market.UNDER_2_5: 'Under 2.5 Goals',
        }
        
        market_name = market_names.get(pred.market, pred.market.value)
        rec_emoji = emoji_map.get(pred.recommendation, '❓')
        
        msg = f"""
{rec_emoji} **{market_name}**
⚽ {match_context.home_team.name} vs {match_context.away_team.name}
📈 Probability: {pred.probability:.1%}
💰 Odds: {pred.odds}
📊 Edge: {pred.edge:.1%}
🎯 Confidence: {pred.confidence_tier}
"""
        
        if pred.kelly_stake > 0:
            msg += f"💵 Kelly Stake: {pred.kelly_stake:.1%} of bankroll\n"
        
        return msg
    
    @staticmethod
    def format_match_summary(match_preds: MatchPredictions) -> str:
        """Format full match analysis for Telegram"""
        context = match_preds.match_context
        
        header = f"""
🏟️ **{context.home_team.name} vs {context.away_team.name}**
🏆 {context.league} | 📅 {context.date}
"""
        
        if context.is_derby:
            header += f"⚔️ *{context.derby_name}*\n"
        
        notes = "\n".join(match_preds.analysis_notes)
        
        # Best picks
        picks = ""
        if match_preds.best_bet:
            picks = "\n🎯 **BEST BET:**\n"
            picks += TelegramFormatter.format_prediction(match_preds.best_bet, context)
        
        return header + "\n" + notes + picks


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def demo_prediction():
    """Demo the advanced predictor"""
    
    print("="*60)
    print("🧠 ADVANCED SOCCER PREDICTOR - DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = AdvancedPredictor(bankroll=1000)
    
    # Create sample match context
    home_team = TeamStats(
        name="Bayern München",
        elo=1850,
        form_points=13,  # EXCELLENT
        home_form=14,
        goals_scored_last_5=12,
        goals_conceded_last_5=3,
        clean_sheet_rate=0.40
    )
    
    away_team = TeamStats(
        name="Borussia Dortmund",
        elo=1780,
        form_points=10,  # GOOD
        away_form=8,
        goals_scored_last_5=10,
        goals_conceded_last_5=6,
        clean_sheet_rate=0.25
    )
    
    is_derby, derby_name = predictor.is_derby(home_team.name, away_team.name, "Bundesliga")
    
    context = MatchContext(
        home_team=home_team,
        away_team=away_team,
        league="Bundesliga",
        date="2026-01-28",
        is_derby=is_derby,
        derby_name=derby_name
    )
    
    # Generate predictions
    match_predictions = predictor.predict_match(context)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"⚽ {context.home_team.name} vs {context.away_team.name}")
    print(f"🏆 {context.league}")
    if context.is_derby:
        print(f"⚔️ DERBY: {context.derby_name}")
    print(f"{'='*60}")
    
    print("\n📝 Analysis Notes:")
    for note in match_predictions.analysis_notes:
        print(f"   {note}")
    
    print(f"\n{'─'*60}")
    print("📊 ALL MARKET PREDICTIONS (sorted by Edge):")
    print(f"{'─'*60}")
    
    for pred in match_predictions.predictions[:15]:
        range_indicator = "🎯" if pred.is_in_optimal_range else "  "
        value_indicator = "✅" if pred.is_value_bet else "❌"
        print(f"{range_indicator} {pred.market.value:20} | Prob: {pred.probability:5.1%} | Odds: {pred.odds:5.2f} | Edge: {pred.edge:+6.1%} | {value_indicator} {pred.recommendation}")
    
    print(f"\n{'─'*60}")
    print("🎯 OPTIMAL RANGE PICKS (1.30-1.70 Odds):")
    print(f"{'─'*60}")
    
    recommendations = predictor.get_recommendations(match_predictions, max_picks=5)
    
    if recommendations:
        for pred in recommendations:
            kelly_stake = pred.kelly_stake * 1000  # For €1000 bankroll
            print(f"✅ {pred.market.value:20} @ {pred.odds:5.2f}")
            print(f"   Probability: {pred.probability:.1%} | Edge: {pred.edge:.1%} | Kelly: €{kelly_stake:.2f}")
    else:
        print("   No value bets in optimal range found")
    
    # Accumulator suggestions
    print(f"\n{'─'*60}")
    print("🔗 ACCUMULATOR LEGS (High Prob + Decent Odds):")
    print(f"{'─'*60}")
    
    for leg in match_predictions.accumulator_legs:
        print(f"   • {leg.market.value} @ {leg.odds} (Prob: {leg.probability:.1%})")
    
    if match_predictions.accumulator_legs:
        combined_odds = 1.0
        combined_prob = 1.0
        for leg in match_predictions.accumulator_legs:
            combined_odds *= leg.odds
            combined_prob *= leg.probability
        print(f"\n   Combined: {combined_odds:.2f} odds | {combined_prob:.1%} probability")


if __name__ == "__main__":
    demo_prediction()

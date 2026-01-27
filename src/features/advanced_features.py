"""
Advanced Feature Engineering für Professional Soccer Betting System
Implementiert: Elo Ratings, xG-basierte Features, Form Indices, Value Detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import math


@dataclass
class EloConfig:
    """Konfiguration für Elo Rating System"""
    initial_rating: float = 1500.0
    k_factor: float = 32.0  # Sensitivity to new results
    home_advantage: float = 100.0  # Points added for home team
    goal_diff_multiplier: float = 1.5  # Impact of goal difference


class EloRatingSystem:
    """
    Elo Rating System für Fußball Teams
    Basiert auf Chess Elo aber angepasst für Fußball (Goal Difference, Home Advantage)
    """
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict] = []
    
    def get_rating(self, team: str) -> float:
        """Hole aktuelles Elo Rating für Team"""
        return self.ratings.get(team, self.config.initial_rating)
    
    def expected_result(self, rating_a: float, rating_b: float, is_home: bool = False) -> float:
        """
        Berechne erwartetes Ergebnis (0-1) für Team A
        0 = Total loss, 0.5 = Draw, 1 = Total win
        """
        # Add home advantage
        if is_home:
            rating_a += self.config.home_advantage
        
        # Elo expected value formula
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        date: Optional[datetime] = None
    ) -> Tuple[float, float]:
        """
        Update Elo Ratings nach Match
        Returns: (new_home_rating, new_away_rating)
        """
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected results
        home_expected = self.expected_result(home_rating, away_rating, is_home=True)
        away_expected = 1 - home_expected
        
        # Actual results (normalized to 0-1)
        if home_goals > away_goals:
            home_actual = 1.0
            away_actual = 0.0
        elif away_goals > home_goals:
            home_actual = 0.0
            away_actual = 1.0
        else:
            home_actual = 0.5
            away_actual = 0.5
        
        # Goal difference multiplier (bigger wins = bigger rating change)
        goal_diff = abs(home_goals - away_goals)
        goal_multiplier = 1 + (goal_diff - 1) * self.config.goal_diff_multiplier if goal_diff > 0 else 1
        
        # Update ratings
        k_adjusted = self.config.k_factor * goal_multiplier
        home_change = k_adjusted * (home_actual - home_expected)
        away_change = k_adjusted * (away_actual - away_expected)
        
        new_home_rating = home_rating + home_change
        new_away_rating = away_rating + away_change
        
        # Store ratings
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        # Store history
        self.history.append({
            'date': date or datetime.now(),
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_rating_before': home_rating,
            'away_rating_before': away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_change': home_change,
            'away_change': away_change
        })
        
        return new_home_rating, new_away_rating
    
    def predict_match_outcome_probability(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Predict Match Outcome Probabilities basierend auf Elo
        Returns: {home_win_prob, draw_prob, away_win_prob, over_1_5_prob, over_2_5_prob}
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Home win probability
        home_win_prob = self.expected_result(home_rating, away_rating, is_home=True)
        
        # Draw probability (estimated as function of rating difference)
        rating_diff = abs(home_rating - away_rating)
        draw_prob = max(0.15, 0.30 - (rating_diff / 1000))  # Closer teams = more draws
        
        # Away win probability (remainder)
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Goal-based probabilities (derived from Elo difference)
        # Higher rated team = more goals expected
        avg_rating = (home_rating + away_rating) / 2
        rating_factor = avg_rating / self.config.initial_rating
        
        # Over 1.5 Goals probability (baseline + rating factor)
        over_1_5_prob = min(0.95, 0.65 + (rating_factor - 1) * 0.2 + (1 - draw_prob) * 0.1)
        
        # Over 2.5 Goals probability
        over_2_5_prob = min(0.90, 0.45 + (rating_factor - 1) * 0.15 + (1 - draw_prob) * 0.15)
        
        # BTTS probability (higher when both teams strong)
        btts_prob = min(0.90, 0.50 + min(home_rating, away_rating) / 3000)
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'over_1_5_prob': over_1_5_prob,
            'over_2_5_prob': over_2_5_prob,
            'btts_prob': btts_prob,
            'home_rating': home_rating,
            'away_rating': away_rating,
            'rating_difference': home_rating - away_rating
        }


class AdvancedFeatureEngineer:
    """
    Professional Feature Engineering für ML Models
    Generiert Features die von Top Betting Models genutzt werden
    """
    
    def __init__(self, elo_system: Optional[EloRatingSystem] = None):
        self.elo_system = elo_system or EloRatingSystem()
    
    def calculate_form_index(
        self,
        results: List[Dict],
        window: int = 5,
        decay_factor: float = 0.9
    ) -> float:
        """
        Berechne gewichteten Form Index
        Recent results = more weight (exponential decay)
        
        Args:
            results: List of match results [{result: 'W'/'D'/'L', goals_scored: int, goals_conceded: int}]
            window: Number of recent matches
            decay_factor: Weight decay for older matches
        
        Returns:
            Form index (0-100)
        """
        if not results:
            return 50.0  # Neutral
        
        recent_results = results[-window:]
        
        points = 0.0
        total_weight = 0.0
        
        for i, match in enumerate(reversed(recent_results)):
            # Weight: recent = 1.0, older = decay_factor^i
            weight = decay_factor ** i
            
            # Points: Win=3, Draw=1, Loss=0
            if match['result'] == 'W':
                match_points = 3.0
            elif match['result'] == 'D':
                match_points = 1.0
            else:
                match_points = 0.0
            
            # Goal difference bonus
            goal_diff = match['goals_scored'] - match['goals_conceded']
            goal_bonus = min(goal_diff * 0.5, 2.0) if goal_diff > 0 else 0
            
            points += (match_points + goal_bonus) * weight
            total_weight += weight
        
        # Normalize to 0-100 scale
        if total_weight > 0:
            form_index = (points / (total_weight * 5.0)) * 100  # Max 5 points per match
            return min(100.0, max(0.0, form_index))
        
        return 50.0
    
    def calculate_xg_features(
        self,
        xg_data: List[Dict],
        window: int = 10
    ) -> Dict[str, float]:
        """
        Berechne xG-basierte Features
        xG (Expected Goals) ist einer der besten Predictors für zukünftige Performance
        
        Args:
            xg_data: [{xg_for: float, xg_against: float, actual_goals_for: int, actual_goals_against: int}]
        
        Returns:
            Dict mit xG features
        """
        recent_data = xg_data[-window:]
        
        if not recent_data:
            return {
                'xg_for_avg': 1.5,
                'xg_against_avg': 1.5,
                'xg_diff_avg': 0.0,
                'xg_overperformance': 0.0,
                'xg_consistency': 0.5
            }
        
        xg_for_values = [d['xg_for'] for d in recent_data]
        xg_against_values = [d['xg_against'] for d in recent_data]
        actual_for_values = [d['actual_goals_for'] for d in recent_data]
        
        xg_for_avg = np.mean(xg_for_values)
        xg_against_avg = np.mean(xg_against_values)
        
        # xG difference (attack strength - defense weakness)
        xg_diff_avg = xg_for_avg - xg_against_avg
        
        # xG overperformance (actual goals vs expected)
        # Positive = lucky/clinical, Negative = unlucky/wasteful
        xg_overperformance = np.mean(actual_for_values) - xg_for_avg
        
        # xG consistency (lower std = more consistent)
        xg_consistency = 1 / (1 + np.std(xg_for_values))
        
        return {
            'xg_for_avg': xg_for_avg,
            'xg_against_avg': xg_against_avg,
            'xg_diff_avg': xg_diff_avg,
            'xg_overperformance': xg_overperformance,
            'xg_consistency': xg_consistency
        }
    
    def calculate_h2h_features(
        self,
        h2h_matches: List[Dict],
        max_history: int = 10
    ) -> Dict[str, float]:
        """
        Head-to-Head Historical Features
        Recent H2H results können sehr predictive sein (rivalries, tactical matchups)
        """
        recent_h2h = h2h_matches[-max_history:]
        
        if not recent_h2h:
            return {
                'h2h_win_rate': 0.33,
                'h2h_avg_goals': 2.5,
                'h2h_btts_rate': 0.50,
                'h2h_over_1_5_rate': 0.70,
                'h2h_over_2_5_rate': 0.45
            }
        
        wins = sum(1 for m in recent_h2h if m['result'] == 'W')
        total_goals = sum(m['total_goals'] for m in recent_h2h)
        btts_count = sum(1 for m in recent_h2h if m['both_scored'])
        over_1_5_count = sum(1 for m in recent_h2h if m['total_goals'] > 1.5)
        over_2_5_count = sum(1 for m in recent_h2h if m['total_goals'] > 2.5)
        
        return {
            'h2h_win_rate': wins / len(recent_h2h),
            'h2h_avg_goals': total_goals / len(recent_h2h),
            'h2h_btts_rate': btts_count / len(recent_h2h),
            'h2h_over_1_5_rate': over_1_5_count / len(recent_h2h),
            'h2h_over_2_5_rate': over_2_5_count / len(recent_h2h)
        }
    
    def calculate_contextual_features(
        self,
        match_data: Dict
    ) -> Dict[str, float]:
        """
        Contextual Features (environmental, psychological, tactical)
        Diese Factors wurden in der original instruction als critical identifiziert
        """
        features = {}
        
        # Days since last match (fatigue factor)
        days_rest_home = match_data.get('days_rest_home', 7)
        days_rest_away = match_data.get('days_rest_away', 7)
        features['rest_advantage'] = (days_rest_home - days_rest_away) / 7.0
        
        # Match importance (derby, title race, relegation)
        features['is_derby'] = 1.0 if match_data.get('is_derby', False) else 0.0
        features['importance_home'] = match_data.get('importance_home', 0.5)  # 0-1 scale
        features['importance_away'] = match_data.get('importance_away', 0.5)
        
        # Weather factor (impacts play style)
        # Temperature: ideal 15-20°C, extreme = more cautious
        temp = match_data.get('temperature', 15)
        features['weather_factor'] = 1.0 - min(abs(temp - 17.5) / 30, 1.0)
        
        # Rain factor (reduces goals slightly)
        features['is_raining'] = 1.0 if match_data.get('rain', False) else 0.0
        
        # Stadium capacity utilization (crowd pressure)
        features['attendance_factor'] = match_data.get('attendance_percentage', 0.75)
        
        # Managerial experience difference
        features['manager_exp_diff'] = match_data.get('manager_exp_home', 5) - match_data.get('manager_exp_away', 5)
        
        return features
    
    def build_complete_feature_vector(
        self,
        home_team: str,
        away_team: str,
        home_data: Dict,
        away_data: Dict,
        match_context: Dict
    ) -> Dict[str, float]:
        """
        Erstelle kompletten Feature Vector für ML Model
        Kombiniert alle Feature Engineering Methoden
        
        Returns:
            Dict mit 50+ features für XGBoost/CatBoost
        """
        features = {}
        
        # 1. Elo Ratings & Probabilities
        elo_pred = self.elo_system.predict_match_outcome_probability(home_team, away_team)
        for key, value in elo_pred.items():
            features[f'elo_{key}'] = value
        
        # 2. Form Indices
        features['form_home'] = self.calculate_form_index(home_data.get('recent_results', []))
        features['form_away'] = self.calculate_form_index(away_data.get('recent_results', []))
        features['form_difference'] = features['form_home'] - features['form_away']
        
        # 3. xG Features
        xg_home = self.calculate_xg_features(home_data.get('xg_data', []))
        xg_away = self.calculate_xg_features(away_data.get('xg_data', []))
        
        for key, value in xg_home.items():
            features[f'home_{key}'] = value
        for key, value in xg_away.items():
            features[f'away_{key}'] = value
        
        # Combined xG features
        features['xg_total_expected'] = xg_home['xg_for_avg'] + xg_away['xg_for_avg']
        features['xg_attack_vs_defense'] = xg_home['xg_for_avg'] - away_data.get('xg_against_avg', 1.5)
        
        # 4. Head-to-Head Features
        h2h_features = self.calculate_h2h_features(match_context.get('h2h_matches', []))
        for key, value in h2h_features.items():
            features[key] = value
        
        # 5. Contextual Features
        context_features = self.calculate_contextual_features(match_context)
        for key, value in context_features.items():
            features[key] = value
        
        # 6. Derived Statistical Features
        features['home_attack_strength'] = home_data.get('goals_scored_avg', 1.5) / home_data.get('league_avg_goals', 2.5)
        features['away_attack_strength'] = away_data.get('goals_scored_avg', 1.5) / away_data.get('league_avg_goals', 2.5)
        features['home_defense_weakness'] = home_data.get('goals_conceded_avg', 1.5) / home_data.get('league_avg_goals', 2.5)
        features['away_defense_weakness'] = away_data.get('goals_conceded_avg', 1.5) / away_data.get('league_avg_goals', 2.5)
        
        # 7. League Position Features
        features['home_league_position'] = home_data.get('position', 10)
        features['away_league_position'] = away_data.get('position', 10)
        features['position_difference'] = features['home_league_position'] - features['away_league_position']
        
        # 8. Recent Goal Trends
        features['home_goals_last_5'] = home_data.get('goals_last_5', 7.5)
        features['away_goals_last_5'] = away_data.get('goals_last_5', 7.5)
        features['home_conceded_last_5'] = home_data.get('conceded_last_5', 7.5)
        features['away_conceded_last_5'] = away_data.get('conceded_last_5', 7.5)
        
        return features


class ValueBettingCalculator:
    """
    Value Betting Calculator mit Kelly Criterion
    Core des professional betting systems
    """
    
    @staticmethod
    def calculate_fair_odds(probability: float) -> float:
        """Berechne faire Odds aus Probability"""
        if probability <= 0 or probability >= 1:
            return 1.0
        return 1.0 / probability
    
    @staticmethod
    def calculate_implied_probability(odds: float) -> float:
        """Berechne implizierte Probability aus Market Odds"""
        if odds <= 1.0:
            return 1.0
        return 1.0 / odds
    
    @staticmethod
    def calculate_expected_value(
        probability: float,
        odds: float,
        stake: float = 1.0
    ) -> float:
        """
        Berechne Expected Value (EV)
        EV = (Probability × Profit) - ((1 - Probability) × Stake)
        
        Positive EV = Value Bet
        """
        profit = stake * (odds - 1)
        loss = stake
        
        ev = (probability * profit) - ((1 - probability) * loss)
        return ev
    
    @staticmethod
    def has_value(
        model_probability: float,
        market_odds: float,
        min_edge: float = 0.05
    ) -> bool:
        """
        Check ob Bet Value hat
        Value exists wenn: model_probability > implied_probability + min_edge
        """
        implied_prob = ValueBettingCalculator.calculate_implied_probability(market_odds)
        edge = model_probability - implied_prob
        return edge >= min_edge
    
    @staticmethod
    def calculate_kelly_stake(
        probability: float,
        odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Kelly Criterion für optimal stake sizing
        Kelly = (bp - q) / b
        where b = odds - 1, p = probability, q = 1 - p
        
        Kelly Fraction (0.25 = 1/4 Kelly) für conservatism
        """
        b = odds - 1
        p = probability
        q = 1 - p
        
        if b <= 0 or p <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fraction and cap at 10% of bankroll
        kelly_adjusted = kelly * kelly_fraction
        stake = max(0.0, min(kelly_adjusted, 0.10)) * bankroll
        
        return stake
    
    @staticmethod
    def calculate_closing_line_value(
        opening_odds: float,
        closing_odds: float,
        bet_odds: float
    ) -> float:
        """
        Closing Line Value (CLV) - Ultimate metric of prediction skill
        CLV = (Your Odds - Opening Odds) / (Closing Odds - Opening Odds)
        
        Positive CLV = You beat the market
        """
        if closing_odds == opening_odds:
            return 0.0
        
        clv = (bet_odds - opening_odds) / (closing_odds - opening_odds)
        return clv

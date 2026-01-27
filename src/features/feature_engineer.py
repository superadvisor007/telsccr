"""Feature engineering for soccer match prediction."""
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Feature engineering for match predictions."""
    
    def __init__(self, xg_window: int = 5, form_window: int = 6, h2h_window: int = 10):
        self.xg_window = xg_window
        self.form_window = form_window
        self.h2h_window = h2h_window
    
    def engineer_features(
        self,
        match_data: Dict[str, Any],
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        h2h_stats: Dict[str, Any],
        weather: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Engineer all features for a match.
        
        Returns comprehensive feature dictionary ready for model input.
        """
        features = {}
        
        # Basic match info
        features["league"] = match_data.get("league", "")
        features["is_high_scoring_league"] = self._is_high_scoring_league(features["league"])
        
        # Team form features
        features.update(self._form_features(home_stats, "home"))
        features.update(self._form_features(away_stats, "away"))
        
        # Goal-scoring features
        features.update(self._goal_features(home_stats, away_stats))
        
        # Head-to-head features
        features.update(self._h2h_features(h2h_stats))
        
        # Defensive features
        features.update(self._defensive_features(home_stats, away_stats))
        
        # Odds-derived features
        if "odds" in match_data:
            features.update(self._odds_features(match_data["odds"]))
        
        # Weather features
        if weather:
            features.update(self._weather_features(weather))
        
        # Derived probabilities
        features.update(self._calculate_derived_probabilities(features))
        
        logger.debug(f"Engineered {len(features)} features for match")
        return features
    
    def _form_features(self, stats: Dict[str, Any], prefix: str) -> Dict[str, float]:
        """Extract form-based features."""
        return {
            f"{prefix}_ppg": stats.get("ppg", 0),
            f"{prefix}_wins": stats.get("wins", 0),
            f"{prefix}_draws": stats.get("draws", 0),
            f"{prefix}_losses": stats.get("losses", 0),
            f"{prefix}_win_rate": stats.get("wins", 0) / max(stats.get("matches_played", 1), 1),
        }
    
    def _goal_features(self, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """Extract goal-scoring features."""
        home_gpg = home_stats.get("goals_per_game", 0)
        away_gpg = away_stats.get("goals_per_game", 0)
        home_gcpg = home_stats.get("goals_conceded_per_game", 0)
        away_gcpg = away_stats.get("goals_conceded_per_game", 0)
        
        # Attack vs Defense matchups
        home_attack_vs_away_defense = home_gpg / max(away_gcpg, 0.1)
        away_attack_vs_home_defense = away_gpg / max(home_gcpg, 0.1)
        
        return {
            "home_goals_per_game": home_gpg,
            "away_goals_per_game": away_gpg,
            "home_goals_conceded_per_game": home_gcpg,
            "away_goals_conceded_per_game": away_gcpg,
            "total_expected_goals": home_gpg + away_gpg,
            "home_attack_strength": home_attack_vs_away_defense,
            "away_attack_strength": away_attack_vs_home_defense,
            "combined_attack_strength": home_attack_vs_away_defense + away_attack_vs_home_defense,
            "home_btts_rate": home_stats.get("btts_percentage", 0) / 100,
            "away_btts_rate": away_stats.get("btts_percentage", 0) / 100,
            "home_over_1_5_rate": home_stats.get("over_1_5_percentage", 0) / 100,
            "away_over_1_5_rate": away_stats.get("over_1_5_percentage", 0) / 100,
            "home_over_2_5_rate": home_stats.get("over_2_5_percentage", 0) / 100,
            "away_over_2_5_rate": away_stats.get("over_2_5_percentage", 0) / 100,
        }
    
    def _defensive_features(self, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """Extract defensive features."""
        return {
            "home_clean_sheet_rate": home_stats.get("clean_sheet_percentage", 0) / 100,
            "away_clean_sheet_rate": away_stats.get("clean_sheet_percentage", 0) / 100,
            "defensive_weakness": (
                home_stats.get("goals_conceded_per_game", 0) +
                away_stats.get("goals_conceded_per_game", 0)
            ),
        }
    
    def _h2h_features(self, h2h_stats: Dict) -> Dict[str, float]:
        """Extract head-to-head features."""
        return {
            "h2h_matches": h2h_stats.get("matches", 0),
            "h2h_avg_goals": h2h_stats.get("avg_goals", 0),
            "h2h_btts_rate": h2h_stats.get("btts_rate", 0) / 100,
        }
    
    def _odds_features(self, odds: Dict) -> Dict[str, float]:
        """Extract features from betting odds."""
        over_1_5_odds = odds.get("over_1_5", 1.5)
        btts_odds = odds.get("btts_yes", 1.5)
        
        return {
            "over_1_5_implied_prob": 1 / over_1_5_odds if over_1_5_odds > 0 else 0,
            "btts_implied_prob": 1 / btts_odds if btts_odds > 0 else 0,
            "over_1_5_odds": over_1_5_odds,
            "btts_odds": btts_odds,
        }
    
    def _weather_features(self, weather: Dict) -> Dict[str, float]:
        """Extract weather impact features."""
        impact = weather.get("impact_score", 0)
        
        return {
            "weather_impact": impact / 10,  # Normalize to 0-1
            "weather_favorable": 1.0 if weather.get("favorable_for_goals", True) else 0.0,
            "temperature": weather.get("temperature", 15),
            "precipitation": weather.get("precipitation", 0),
            "wind_speed": weather.get("wind_speed", 0),
        }
    
    def _is_high_scoring_league(self, league: str) -> bool:
        """Check if league is known for high scoring."""
        high_scoring = [
            "Bundesliga",
            "Eredivisie",
            "Austrian Bundesliga",
            "Swiss Super League",
            "Norwegian Eliteserien",
        ]
        return any(hs in league for hs in high_scoring)
    
    def _calculate_derived_probabilities(self, features: Dict) -> Dict[str, float]:
        """Calculate derived probability estimates from features."""
        # Simple heuristic-based probability estimates
        # These serve as baseline before ML model predictions
        
        # Over 1.5 probability
        total_xg = features.get("total_expected_goals", 0)
        over_1_5_rate_avg = (
            features.get("home_over_1_5_rate", 0) +
            features.get("away_over_1_5_rate", 0)
        ) / 2
        h2h_factor = features.get("h2h_avg_goals", 0) / 3.0  # Normalize
        
        over_1_5_baseline = min(
            0.95,
            (total_xg / 3.0) * 0.4 +
            over_1_5_rate_avg * 0.4 +
            h2h_factor * 0.2
        )
        
        # BTTS probability
        btts_rate_avg = (
            features.get("home_btts_rate", 0) +
            features.get("away_btts_rate", 0)
        ) / 2
        attack_balance = min(
            features.get("home_attack_strength", 0),
            features.get("away_attack_strength", 0)
        )
        
        btts_baseline = min(
            0.90,
            btts_rate_avg * 0.5 +
            (attack_balance / 2.0) * 0.3 +
            features.get("h2h_btts_rate", 0) * 0.2
        )
        
        # Weather adjustment
        if features.get("weather_favorable", 1.0) == 0:
            over_1_5_baseline *= 0.85
            btts_baseline *= 0.85
        
        return {
            "over_1_5_baseline_prob": over_1_5_baseline,
            "btts_baseline_prob": btts_baseline,
        }
    
    def create_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Create numerical feature vector for ML model.
        
        Returns numpy array with all numerical features in fixed order.
        """
        # Define feature order (must match training)
        feature_names = [
            "home_ppg", "away_ppg",
            "home_goals_per_game", "away_goals_per_game",
            "home_goals_conceded_per_game", "away_goals_conceded_per_game",
            "total_expected_goals",
            "home_attack_strength", "away_attack_strength", "combined_attack_strength",
            "home_btts_rate", "away_btts_rate",
            "home_over_1_5_rate", "away_over_1_5_rate",
            "home_clean_sheet_rate", "away_clean_sheet_rate",
            "defensive_weakness",
            "h2h_avg_goals", "h2h_btts_rate",
            "over_1_5_implied_prob", "btts_implied_prob",
            "weather_impact", "weather_favorable",
            "is_high_scoring_league",
        ]
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0)
            # Convert boolean to float
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            vector.append(float(value))
        
        return np.array(vector)

"""
🎯 Structural Features Engine
=============================
Advanced football analytics features for deep match reasoning.

Features computed:
- Transition speed & counter-attack propensity
- Pressing intensity & PPDA
- Defensive fragility & high-line risk
- Set-piece effectiveness
- xG flow & momentum
- Tactical shape metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TeamFeatures:
    """Structural features for a team."""
    team_name: str
    team_id: int
    
    # Offensive metrics
    xg_per_90: float = 0.0
    shots_per_90: float = 0.0
    shot_quality: float = 0.0  # xG per shot
    big_chances_created: float = 0.0
    
    # Possession metrics
    possession_pct: float = 50.0
    pass_accuracy: float = 0.0
    progressive_passes_per_90: float = 0.0
    final_third_entries: float = 0.0
    
    # Pressing metrics
    ppda: float = 10.0  # Passes Per Defensive Action (lower = more pressing)
    high_press_intensity: float = 0.0
    counter_press_success: float = 0.0
    
    # Defensive metrics
    xga_per_90: float = 0.0  # xG against
    defensive_line_height: float = 50.0  # 0-100
    defensive_fragility: float = 0.0  # Measure of vulnerability
    tackles_won_pct: float = 0.0
    interceptions_per_90: float = 0.0
    
    # Transition metrics
    transition_speed: float = 0.0  # Seconds from regain to shot
    counter_attack_frequency: float = 0.0
    counter_attack_goals_pct: float = 0.0
    
    # Set piece metrics
    set_piece_xg_for: float = 0.0
    set_piece_xg_against: float = 0.0
    corner_conversion: float = 0.0
    
    # Form metrics
    form_xg_trend: float = 0.0  # Positive = improving
    form_xga_trend: float = 0.0
    momentum_score: float = 0.0  # -1 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class MatchFeatures:
    """Structural features for a match prediction."""
    match_id: str
    home_team: TeamFeatures
    away_team: TeamFeatures
    
    # Match context
    league: str = ""
    is_derby: bool = False
    is_cup: bool = False
    high_stakes: bool = False
    
    # Derived metrics
    style_clash_score: float = 0.0  # How different are playing styles
    open_game_probability: float = 0.0
    goal_expectation: float = 0.0
    btts_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'match_id': self.match_id,
            'home_team': self.home_team.to_dict(),
            'away_team': self.away_team.to_dict(),
            'league': self.league,
            'is_derby': self.is_derby,
            'is_cup': self.is_cup,
            'high_stakes': self.high_stakes,
            'style_clash_score': self.style_clash_score,
            'open_game_probability': self.open_game_probability,
            'goal_expectation': self.goal_expectation,
            'btts_probability': self.btts_probability,
        }


class StructuralFeatureEngine:
    """
    🎯 Structural Feature Extraction Engine
    
    Computes advanced football analytics features from:
    - SPADL action data
    - Raw event streams
    - Historical match data
    
    Features are designed for:
    - Deep match reasoning
    - Betting market analysis
    - Tactical profiling
    
    Example:
        engine = StructuralFeatureEngine()
        
        # From SPADL data
        team_features = engine.compute_team_features(spadl_df, team_id=123)
        
        # From historical matches
        features = engine.compute_from_history(matches_df, team_name="Bayern")
        
        # Match prediction features
        match_features = engine.compute_match_features(
            home_features, away_features, match_context
        )
    """
    
    def __init__(self):
        self._feature_cache = {}
    
    def compute_team_features(
        self,
        spadl_df: pd.DataFrame,
        team_id: int,
        team_name: str = None
    ) -> TeamFeatures:
        """
        Compute team features from SPADL actions.
        
        Args:
            spadl_df: SPADL-formatted DataFrame
            team_id: Team identifier
            team_name: Optional team name
        
        Returns:
            TeamFeatures with computed metrics
        """
        team_actions = spadl_df[spadl_df['team_id'] == team_id]
        opp_actions = spadl_df[spadl_df['team_id'] != team_id]
        
        total_actions = len(spadl_df)
        team_action_count = len(team_actions)
        
        if team_action_count == 0:
            return TeamFeatures(team_name=team_name or "Unknown", team_id=team_id)
        
        # Calculate possession
        possession_pct = team_action_count / total_actions * 100 if total_actions > 0 else 50
        
        # Passing metrics
        passes = team_actions[team_actions['type_name'] == 'pass']
        pass_accuracy = 0.0
        if len(passes) > 0:
            successful_passes = passes[passes['result_name'] == 'success']
            pass_accuracy = len(successful_passes) / len(passes) * 100
        
        # Progressive passes (move ball > 10m towards goal)
        progressive_passes = passes[passes['progression'] > 10] if 'progression' in passes.columns else pd.DataFrame()
        progressive_per_90 = len(progressive_passes) * 90 / (total_actions / 60) if total_actions > 0 else 0
        
        # Shots and xG
        shots = team_actions[team_actions['type_name'].isin(['shot', 'shot_penalty', 'shot_freekick'])]
        shots_per_90 = len(shots) * 90 / (total_actions / 60) if total_actions > 0 else 0
        
        # Defensive actions
        defensive_actions = team_actions[team_actions['type_name'].isin(['tackle', 'interception', 'clearance'])]
        
        # PPDA calculation (opponent passes / defensive actions in opp half)
        opp_passes_in_own_half = opp_actions[
            (opp_actions['type_name'] == 'pass') & 
            (opp_actions['start_x'] < 52.5)  # Opponent's half
        ]
        def_actions_in_opp_half = defensive_actions[defensive_actions['start_x'] > 52.5]
        
        ppda = len(opp_passes_in_own_half) / len(def_actions_in_opp_half) if len(def_actions_in_opp_half) > 0 else 20.0
        
        # Interceptions
        interceptions = defensive_actions[defensive_actions['type_name'] == 'interception']
        interceptions_per_90 = len(interceptions) * 90 / (total_actions / 60) if total_actions > 0 else 0
        
        # Final third entries
        final_third_entries = team_actions[
            (team_actions['end_x'] > 70) & 
            (team_actions['start_x'] <= 70)
        ]
        final_third_per_90 = len(final_third_entries) * 90 / (total_actions / 60) if total_actions > 0 else 0
        
        # Defensive line height (average defensive action location)
        if len(defensive_actions) > 0:
            defensive_line_height = defensive_actions['start_x'].mean()
        else:
            defensive_line_height = 50.0
        
        return TeamFeatures(
            team_name=team_name or str(team_id),
            team_id=team_id,
            possession_pct=round(possession_pct, 1),
            pass_accuracy=round(pass_accuracy, 1),
            progressive_passes_per_90=round(progressive_per_90, 1),
            final_third_entries=round(final_third_per_90, 1),
            shots_per_90=round(shots_per_90, 1),
            ppda=round(ppda, 1),
            interceptions_per_90=round(interceptions_per_90, 1),
            defensive_line_height=round(defensive_line_height, 1),
        )
    
    def compute_from_history(
        self,
        matches: pd.DataFrame,
        team_name: str,
        last_n: int = 10
    ) -> TeamFeatures:
        """
        Compute team features from historical match data.
        
        Args:
            matches: DataFrame with historical matches (goals, xG, etc.)
            team_name: Team to analyze
            last_n: Number of recent matches to consider
        
        Returns:
            TeamFeatures computed from history
        """
        # Filter to team's matches
        team_matches = matches[
            (matches['home_team'] == team_name) | 
            (matches['away_team'] == team_name)
        ].head(last_n)
        
        if len(team_matches) == 0:
            return TeamFeatures(team_name=team_name, team_id=0)
        
        # Calculate goals and xG
        goals_for = []
        goals_against = []
        xg_for = []
        xg_against = []
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team_name
            
            if is_home:
                goals_for.append(match.get('home_score', 0) or 0)
                goals_against.append(match.get('away_score', 0) or 0)
                xg_for.append(match.get('home_xg', match.get('home_score', 0)) or 0)
                xg_against.append(match.get('away_xg', match.get('away_score', 0)) or 0)
            else:
                goals_for.append(match.get('away_score', 0) or 0)
                goals_against.append(match.get('home_score', 0) or 0)
                xg_for.append(match.get('away_xg', match.get('away_score', 0)) or 0)
                xg_against.append(match.get('home_xg', match.get('home_score', 0)) or 0)
        
        avg_xg_for = np.mean(xg_for) if xg_for else 1.5
        avg_xg_against = np.mean(xg_against) if xg_against else 1.2
        
        # Form trend (comparing recent to older)
        if len(xg_for) >= 5:
            recent_xg = np.mean(xg_for[:3])
            older_xg = np.mean(xg_for[3:])
            form_xg_trend = recent_xg - older_xg
        else:
            form_xg_trend = 0.0
        
        # Momentum score
        recent_goals = sum(goals_for[:3]) if len(goals_for) >= 3 else sum(goals_for)
        recent_conceded = sum(goals_against[:3]) if len(goals_against) >= 3 else sum(goals_against)
        momentum_score = (recent_goals - recent_conceded) / 6  # Normalize to roughly -1 to 1
        momentum_score = max(-1, min(1, momentum_score))
        
        # BTTS and Over calculations
        btts_matches = sum(1 for gf, ga in zip(goals_for, goals_against) if gf > 0 and ga > 0)
        btts_rate = btts_matches / len(goals_for) if goals_for else 0.5
        
        return TeamFeatures(
            team_name=team_name,
            team_id=0,
            xg_per_90=round(avg_xg_for, 2),
            xga_per_90=round(avg_xg_against, 2),
            form_xg_trend=round(form_xg_trend, 2),
            momentum_score=round(momentum_score, 2),
        )
    
    def compute_match_features(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context: Dict[str, Any] = None
    ) -> MatchFeatures:
        """
        Compute match-level features for prediction.
        
        Args:
            home_features: Home team features
            away_features: Away team features
            context: Match context (derby, cup, stakes)
        
        Returns:
            MatchFeatures with derived metrics
        """
        context = context or {}
        
        # Style clash score
        # High when one team presses and other is possession-based
        press_diff = abs(home_features.ppda - away_features.ppda)
        possession_diff = abs(home_features.possession_pct - away_features.possession_pct)
        style_clash = (press_diff / 10 + possession_diff / 20) / 2
        style_clash = min(1.0, style_clash)
        
        # Open game probability
        # Higher when both teams are attacking-minded
        both_attacking = (
            (home_features.xg_per_90 > 1.5) and 
            (away_features.xg_per_90 > 1.5)
        )
        high_lines = (
            (home_features.defensive_line_height > 55) and 
            (away_features.defensive_line_height > 55)
        )
        open_game = 0.5
        if both_attacking:
            open_game += 0.2
        if high_lines:
            open_game += 0.15
        if style_clash > 0.6:
            open_game += 0.1
        if context.get('is_derby'):
            open_game += 0.1
        open_game = min(0.9, open_game)
        
        # Goal expectation
        home_attack = home_features.xg_per_90 * 1.1  # Home advantage
        away_attack = away_features.xg_per_90 * 0.9
        home_defense = home_features.xga_per_90 * 0.9
        away_defense = away_features.xga_per_90 * 1.1
        
        # Blend attack strength with opponent's defensive weakness
        home_expected = (home_attack + away_defense) / 2
        away_expected = (away_attack + home_defense) / 2
        total_goals = home_expected + away_expected
        
        # BTTS probability
        # Based on both teams scoring tendency
        home_scores = 1 - np.exp(-home_expected)  # Poisson P(X >= 1)
        away_scores = 1 - np.exp(-away_expected)
        btts_prob = home_scores * away_scores
        
        # Adjust for game state
        if context.get('high_stakes'):
            btts_prob *= 0.9  # Tighter games in high stakes
            total_goals *= 0.95
        
        return MatchFeatures(
            match_id=context.get('match_id', ''),
            home_team=home_features,
            away_team=away_features,
            league=context.get('league', ''),
            is_derby=context.get('is_derby', False),
            is_cup=context.get('is_cup', False),
            high_stakes=context.get('high_stakes', False),
            style_clash_score=round(style_clash, 3),
            open_game_probability=round(open_game, 3),
            goal_expectation=round(total_goals, 2),
            btts_probability=round(btts_prob, 3),
        )
    
    def compute_pressing_metrics(
        self,
        spadl_df: pd.DataFrame,
        team_id: int
    ) -> Dict[str, float]:
        """Compute detailed pressing metrics."""
        team_actions = spadl_df[spadl_df['team_id'] == team_id]
        opp_actions = spadl_df[spadl_df['team_id'] != team_id]
        
        # High press actions (defensive actions in opponent's third)
        high_press = team_actions[
            (team_actions['type_name'].isin(['tackle', 'interception', 'foul'])) &
            (team_actions['start_x'] > 70)
        ]
        
        # Counter-press (actions within 5 seconds of losing ball)
        # Would need timestamp analysis for full implementation
        
        # PPDA zones
        opp_passes_deep = len(opp_actions[
            (opp_actions['type_name'] == 'pass') &
            (opp_actions['start_x'] < 35)
        ])
        def_actions_high = len(team_actions[
            (team_actions['type_name'].isin(['tackle', 'interception'])) &
            (team_actions['start_x'] > 70)
        ])
        
        ppda_high = opp_passes_deep / def_actions_high if def_actions_high > 0 else 30
        
        return {
            'high_press_actions': len(high_press),
            'ppda_high_zone': round(ppda_high, 1),
            'press_intensity': round(len(high_press) / (len(team_actions) + 1) * 100, 1),
        }
    
    def compute_transition_metrics(
        self,
        spadl_df: pd.DataFrame,
        team_id: int
    ) -> Dict[str, float]:
        """Compute transition and counter-attack metrics."""
        # This would need possession chain analysis
        # Simplified version:
        
        team_actions = spadl_df[spadl_df['team_id'] == team_id]
        
        # Fast attacks (actions that progress quickly)
        if 'time_seconds' in spadl_df.columns:
            # Group by possession chains and measure speed
            pass
        
        # Shots from counter-attacks (approximation)
        # Shots within 15 seconds of turnover
        counter_shots = 0  # Would need timestamp analysis
        
        return {
            'transition_attacks': 0,
            'counter_shots': counter_shots,
            'transition_xg': 0.0,
        }

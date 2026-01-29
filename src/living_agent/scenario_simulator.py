"""
🎯 SCENARIO SIMULATOR - Forward-Thinking Match Analysis
=======================================================
Simulates multiple match scenarios to understand dynamic outcomes.
Goes beyond stats - models actual match flow and chaos moments.

This is what makes the agent "alive" - it doesn't just crunch numbers,
it imagines how matches might unfold.

Features:
- Multi-scenario simulation (attack, defense, chaos)
- Structural reasoning (wing exposure, transitions)
- Chaos moment detection (set pieces, red cards, etc.)
- Dynamic probability adjustment based on scenarios
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import random
import math


@dataclass
class TeamProfile:
    """Tactical profile for scenario simulation."""
    name: str
    attack_strength: float  # 0-1, how dangerous in attack
    defense_strength: float  # 0-1, how solid defensively
    transition_speed: float  # 0-1, quick counter ability
    wing_exposure: float  # 0-1, vulnerability on wings
    set_piece_threat: float  # 0-1, danger from set pieces
    pressing_intensity: float  # 0-1, high press tendency
    defensive_line_height: float  # 0-1, high line = risky
    chaos_factor: float  # 0-1, unpredictability
    form_momentum: float  # -1 to 1, recent form
    mental_state: float  # -1 to 1, confidence/pressure


@dataclass  
class MatchScenario:
    """A single simulated match scenario."""
    scenario_type: str  # 'base', 'high_scoring', 'defensive', 'chaos'
    description: str
    home_goals: Tuple[float, float]  # (expected, variance)
    away_goals: Tuple[float, float]
    btts_probability: float
    over_1_5_probability: float
    over_2_5_probability: float
    key_factors: List[str]
    confidence: float
    chaos_events: List[str]


@dataclass
class SimulationResult:
    """Complete simulation result across all scenarios."""
    match_id: str
    home_team: str
    away_team: str
    scenarios: List[MatchScenario]
    weighted_predictions: Dict[str, float]
    confidence_scores: Dict[str, float]
    key_insights: List[str]
    simulation_time: str
    

class ScenarioSimulator:
    """
    🎯 Forward-Thinking Match Simulator
    
    Simulates multiple ways a match could unfold:
    1. Base scenario (most likely outcome)
    2. High-scoring scenario (attack dominance)
    3. Defensive scenario (tight match)
    4. Chaos scenario (unpredictable events)
    
    Each scenario contributes to final predictions with weights.
    """
    
    def __init__(self):
        self.scenario_weights = {
            'base': 0.50,      # 50% weight on most likely
            'high_scoring': 0.20,
            'defensive': 0.15,
            'chaos': 0.15,
        }
        
        # Chaos events that can flip predictions
        self.chaos_events = [
            'early_goal', 'red_card', 'penalty_miss', 'goalkeeper_error',
            'injury_to_key_player', 'weather_change', 'var_drama',
            'tactical_switch', 'momentum_shift', 'late_drama'
        ]
        
        # Tactical patterns that influence scenarios
        self.tactical_patterns = {
            'high_press_vs_slow_buildup': {'goals': 1.2, 'btts': 1.1},
            'counter_vs_possession': {'goals': 0.9, 'btts': 1.0},
            'open_game': {'goals': 1.3, 'btts': 1.2},
            'cagey_tactical': {'goals': 0.7, 'btts': 0.8},
            'one_sided_dominance': {'goals': 1.0, 'btts': 0.7},
        }
    
    def simulate_match(
        self,
        home_profile: TeamProfile,
        away_profile: TeamProfile,
        league_priors: Dict[str, float],
        context: Dict[str, Any] = None
    ) -> SimulationResult:
        """
        Run complete match simulation across multiple scenarios.
        
        Args:
            home_profile: Tactical profile of home team
            away_profile: Tactical profile of away team
            league_priors: League-level tendencies (btts_rate, etc.)
            context: Additional context (derby, cup, stakes)
        
        Returns:
            SimulationResult with weighted predictions
        """
        context = context or {}
        match_id = f"{home_profile.name}_vs_{away_profile.name}_{datetime.now().strftime('%Y%m%d')}"
        
        scenarios = []
        
        # 1. Base Scenario (most likely)
        base = self._simulate_base_scenario(home_profile, away_profile, league_priors)
        scenarios.append(base)
        
        # 2. High-Scoring Scenario
        high_scoring = self._simulate_high_scoring(home_profile, away_profile, league_priors)
        scenarios.append(high_scoring)
        
        # 3. Defensive Scenario
        defensive = self._simulate_defensive(home_profile, away_profile, league_priors)
        scenarios.append(defensive)
        
        # 4. Chaos Scenario (unpredictable)
        chaos = self._simulate_chaos(home_profile, away_profile, league_priors, context)
        scenarios.append(chaos)
        
        # Calculate weighted predictions
        weighted = self._calculate_weighted_predictions(scenarios)
        
        # Calculate confidence scores
        confidence = self._calculate_confidence_scores(scenarios, home_profile, away_profile)
        
        # Generate key insights
        insights = self._generate_insights(scenarios, home_profile, away_profile, context)
        
        return SimulationResult(
            match_id=match_id,
            home_team=home_profile.name,
            away_team=away_profile.name,
            scenarios=scenarios,
            weighted_predictions=weighted,
            confidence_scores=confidence,
            key_insights=insights,
            simulation_time=datetime.now().isoformat()
        )
    
    def _simulate_base_scenario(
        self,
        home: TeamProfile,
        away: TeamProfile,
        priors: Dict[str, float]
    ) -> MatchScenario:
        """Simulate the most likely match outcome."""
        
        # Expected goals based on attack/defense balance
        home_xg = 1.3 * home.attack_strength * (1 - away.defense_strength * 0.7)
        away_xg = 1.0 * away.attack_strength * (1 - home.defense_strength * 0.7)
        
        # Adjust for form
        home_xg *= (1 + home.form_momentum * 0.2)
        away_xg *= (1 + away.form_momentum * 0.2)
        
        # Home advantage
        home_xg *= 1.15
        
        # Calculate market probabilities
        total_xg = home_xg + away_xg
        
        # Over 1.5: P(goals >= 2)
        over_1_5 = 1 - self._poisson_prob(0, total_xg) - self._poisson_prob(1, total_xg)
        
        # Over 2.5: P(goals >= 3)
        over_2_5 = 1 - sum(self._poisson_prob(g, total_xg) for g in range(3))
        
        # BTTS: P(both score) = P(home >= 1) * P(away >= 1)
        home_scores = 1 - self._poisson_prob(0, home_xg)
        away_scores = 1 - self._poisson_prob(0, away_xg)
        btts = home_scores * away_scores
        
        # Blend with league priors
        btts = 0.6 * btts + 0.4 * priors.get('btts_rate', 0.50)
        over_1_5 = 0.6 * over_1_5 + 0.4 * priors.get('over_1_5_rate', 0.75)
        over_2_5 = 0.6 * over_2_5 + 0.4 * priors.get('over_2_5_rate', 0.50)
        
        return MatchScenario(
            scenario_type='base',
            description='Most likely outcome based on team strengths and form',
            home_goals=(home_xg, 0.8),
            away_goals=(away_xg, 0.7),
            btts_probability=min(0.95, max(0.10, btts)),
            over_1_5_probability=min(0.98, max(0.30, over_1_5)),
            over_2_5_probability=min(0.90, max(0.15, over_2_5)),
            key_factors=[
                f"Home attack: {home.attack_strength:.0%}",
                f"Away defense: {away.defense_strength:.0%}",
                f"Expected total goals: {total_xg:.1f}"
            ],
            confidence=0.75,
            chaos_events=[]
        )
    
    def _simulate_high_scoring(
        self,
        home: TeamProfile,
        away: TeamProfile,
        priors: Dict[str, float]
    ) -> MatchScenario:
        """Simulate a high-scoring scenario (attacking dominance)."""
        
        # Amplify attacking factors
        attack_factor = (home.attack_strength + away.attack_strength) / 2
        defense_weakness = 2 - (home.defense_strength + away.defense_strength)
        
        # High line + fast transitions = more goals
        line_risk = (home.defensive_line_height + away.defensive_line_height) / 2
        transition_danger = (home.transition_speed + away.transition_speed) / 2
        
        # Calculate boosted xG
        home_xg = 1.5 * (1 + attack_factor * 0.5) * (1 + defense_weakness * 0.3)
        away_xg = 1.3 * (1 + attack_factor * 0.5) * (1 + defense_weakness * 0.3)
        
        if line_risk > 0.6:
            home_xg *= 1.2
            away_xg *= 1.2
        
        total_xg = home_xg + away_xg
        
        over_1_5 = 1 - self._poisson_prob(0, total_xg) - self._poisson_prob(1, total_xg)
        over_2_5 = 1 - sum(self._poisson_prob(g, total_xg) for g in range(3))
        
        home_scores = 1 - self._poisson_prob(0, home_xg)
        away_scores = 1 - self._poisson_prob(0, away_xg)
        btts = home_scores * away_scores
        
        key_factors = [
            f"Combined attack power: {attack_factor:.0%}",
            f"Defense vulnerability: {defense_weakness/2:.0%}",
        ]
        
        if line_risk > 0.6:
            key_factors.append(f"⚠️ High defensive lines - counter-attack opportunities")
        if transition_danger > 0.6:
            key_factors.append(f"⚡ Fast transitions on both sides")
        
        return MatchScenario(
            scenario_type='high_scoring',
            description='Open, attacking game with defensive vulnerabilities',
            home_goals=(home_xg, 1.0),
            away_goals=(away_xg, 0.9),
            btts_probability=min(0.95, btts * 1.1),
            over_1_5_probability=min(0.98, over_1_5),
            over_2_5_probability=min(0.90, over_2_5),
            key_factors=key_factors,
            confidence=0.60,
            chaos_events=['early_goal', 'momentum_shift']
        )
    
    def _simulate_defensive(
        self,
        home: TeamProfile,
        away: TeamProfile,
        priors: Dict[str, float]
    ) -> MatchScenario:
        """Simulate a tight, defensive game."""
        
        # Amplify defensive factors
        combined_defense = (home.defense_strength + away.defense_strength) / 2
        pressing = (home.pressing_intensity + away.pressing_intensity) / 2
        
        # Tight games have fewer goals
        home_xg = 0.9 * (1 - combined_defense * 0.4)
        away_xg = 0.7 * (1 - combined_defense * 0.4)
        
        # High pressing reduces space
        if pressing > 0.6:
            home_xg *= 0.85
            away_xg *= 0.85
        
        total_xg = home_xg + away_xg
        
        over_1_5 = 1 - self._poisson_prob(0, total_xg) - self._poisson_prob(1, total_xg)
        over_2_5 = 1 - sum(self._poisson_prob(g, total_xg) for g in range(3))
        
        home_scores = 1 - self._poisson_prob(0, home_xg)
        away_scores = 1 - self._poisson_prob(0, away_xg)
        btts = home_scores * away_scores
        
        key_factors = [
            f"Combined defense: {combined_defense:.0%}",
            f"Expected goals suppressed: {total_xg:.1f}",
        ]
        
        if pressing > 0.6:
            key_factors.append("⚠️ High pressing - reduced space for attacks")
        
        return MatchScenario(
            scenario_type='defensive',
            description='Tight, tactical game with few scoring chances',
            home_goals=(home_xg, 0.5),
            away_goals=(away_xg, 0.5),
            btts_probability=max(0.15, btts * 0.8),
            over_1_5_probability=max(0.35, over_1_5),
            over_2_5_probability=max(0.15, over_2_5),
            key_factors=key_factors,
            confidence=0.55,
            chaos_events=[]
        )
    
    def _simulate_chaos(
        self,
        home: TeamProfile,
        away: TeamProfile,
        priors: Dict[str, float],
        context: Dict[str, Any]
    ) -> MatchScenario:
        """Simulate unpredictable chaos scenario."""
        
        # Base chaos from team profiles
        chaos_level = (home.chaos_factor + away.chaos_factor) / 2
        
        # Context-based chaos amplifiers
        if context.get('is_derby'):
            chaos_level += 0.2
        if context.get('high_stakes'):
            chaos_level += 0.15
        if context.get('bad_weather'):
            chaos_level += 0.1
        
        chaos_level = min(0.9, chaos_level)
        
        # Chaos means higher variance
        variance_multiplier = 1 + chaos_level * 0.5
        
        # Random walk around base expectations
        base_home_xg = 1.3 * home.attack_strength
        base_away_xg = 1.0 * away.attack_strength
        
        # Chaos can swing either way
        home_xg = base_home_xg * (0.7 + chaos_level * 0.8)
        away_xg = base_away_xg * (0.8 + chaos_level * 0.6)
        
        total_xg = home_xg + away_xg
        
        # In chaos, BTTS is more likely (unpredictable events)
        over_1_5 = 1 - self._poisson_prob(0, total_xg) - self._poisson_prob(1, total_xg)
        over_2_5 = 1 - sum(self._poisson_prob(g, total_xg) for g in range(3))
        btts = 0.5 + chaos_level * 0.3
        
        # Select random chaos events
        num_events = int(2 + chaos_level * 3)
        events = random.sample(self.chaos_events, min(num_events, len(self.chaos_events)))
        
        key_factors = [
            f"Chaos factor: {chaos_level:.0%}",
            f"⚡ Unpredictable match conditions",
        ]
        
        if context.get('is_derby'):
            key_factors.append("🔥 Derby intensity - emotions high")
        if context.get('high_stakes'):
            key_factors.append("🏆 High stakes - pressure situation")
        
        return MatchScenario(
            scenario_type='chaos',
            description='Unpredictable game with potential for drama',
            home_goals=(home_xg, 1.5 * variance_multiplier),
            away_goals=(away_xg, 1.3 * variance_multiplier),
            btts_probability=min(0.85, btts),
            over_1_5_probability=min(0.90, over_1_5 + chaos_level * 0.1),
            over_2_5_probability=min(0.80, over_2_5 + chaos_level * 0.1),
            key_factors=key_factors,
            confidence=0.40,  # Low confidence in chaos
            chaos_events=events
        )
    
    def _calculate_weighted_predictions(
        self,
        scenarios: List[MatchScenario]
    ) -> Dict[str, float]:
        """Calculate weighted predictions across all scenarios."""
        
        predictions = {
            'btts': 0,
            'over_1_5': 0,
            'over_2_5': 0,
            'expected_home_goals': 0,
            'expected_away_goals': 0,
        }
        
        for scenario in scenarios:
            weight = self.scenario_weights.get(scenario.scenario_type, 0.25)
            
            predictions['btts'] += scenario.btts_probability * weight
            predictions['over_1_5'] += scenario.over_1_5_probability * weight
            predictions['over_2_5'] += scenario.over_2_5_probability * weight
            predictions['expected_home_goals'] += scenario.home_goals[0] * weight
            predictions['expected_away_goals'] += scenario.away_goals[0] * weight
        
        return predictions
    
    def _calculate_confidence_scores(
        self,
        scenarios: List[MatchScenario],
        home: TeamProfile,
        away: TeamProfile
    ) -> Dict[str, float]:
        """Calculate confidence in each prediction."""
        
        # Variance across scenarios indicates uncertainty
        btts_values = [s.btts_probability for s in scenarios]
        over_1_5_values = [s.over_1_5_probability for s in scenarios]
        over_2_5_values = [s.over_2_5_probability for s in scenarios]
        
        # Lower variance = higher confidence
        btts_variance = self._variance(btts_values)
        over_1_5_variance = self._variance(over_1_5_values)
        over_2_5_variance = self._variance(over_2_5_values)
        
        # Adjust for team predictability
        predictability = 1 - (home.chaos_factor + away.chaos_factor) / 2
        
        return {
            'btts': min(0.90, (1 - btts_variance * 2) * predictability),
            'over_1_5': min(0.95, (1 - over_1_5_variance * 2) * predictability),
            'over_2_5': min(0.85, (1 - over_2_5_variance * 2) * predictability),
        }
    
    def _generate_insights(
        self,
        scenarios: List[MatchScenario],
        home: TeamProfile,
        away: TeamProfile,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from simulation."""
        
        insights = []
        
        # High-scoring potential
        base = next(s for s in scenarios if s.scenario_type == 'base')
        total_xg = base.home_goals[0] + base.away_goals[0]
        
        if total_xg > 3.0:
            insights.append(f"🔥 High-scoring potential: {total_xg:.1f} expected goals")
        elif total_xg < 2.0:
            insights.append(f"🛡️ Tight game expected: only {total_xg:.1f} expected goals")
        
        # BTTS analysis
        if base.btts_probability > 0.65:
            insights.append(f"⚽ Strong BTTS signal: {base.btts_probability:.0%} probability")
        elif base.btts_probability < 0.35:
            insights.append(f"🧱 Likely clean sheet scenario: only {base.btts_probability:.0%} BTTS")
        
        # Defensive analysis
        if home.defense_strength > 0.7 or away.defense_strength > 0.7:
            insights.append("🛡️ Strong defensive presence - goals may be limited")
        
        # Chaos warning
        chaos = next(s for s in scenarios if s.scenario_type == 'chaos')
        if len(chaos.chaos_events) > 3:
            insights.append(f"⚠️ High unpredictability - {len(chaos.chaos_events)} chaos factors")
        
        # Context-specific
        if context.get('is_derby'):
            insights.append("🔥 Derby match - expect intensity and unpredictability")
        if context.get('high_stakes'):
            insights.append("🏆 High stakes - teams may play differently than usual")
        
        # Scenario divergence warning
        btts_range = max(s.btts_probability for s in scenarios) - min(s.btts_probability for s in scenarios)
        if btts_range > 0.3:
            insights.append("⚡ High scenario divergence - consider lower stake")
        
        return insights[:5]  # Max 5 insights
    
    def _poisson_prob(self, k: int, lambda_: float) -> float:
        """Calculate Poisson probability P(X = k)."""
        import math
        if lambda_ <= 0:
            return 1.0 if k == 0 else 0.0
        return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)
    
    def _variance(self, values: List[float]) -> float:
        """Calculate variance of a list."""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


# ==================== HELPER FUNCTIONS ====================

def create_team_profile_from_stats(
    team_name: str,
    goals_scored: float,
    goals_conceded: float,
    form_points: float,
    league_avg_goals: float = 2.8
) -> TeamProfile:
    """Create a TeamProfile from basic stats."""
    
    # Normalize attack strength (goals vs league average)
    attack = min(1.0, goals_scored / (league_avg_goals * 0.6))
    
    # Defense is inverse of goals conceded
    defense = max(0.0, 1 - (goals_conceded / (league_avg_goals * 0.6)))
    
    # Form from recent points (out of 15 for last 5 games)
    form = (form_points / 15 - 0.5) * 2  # Normalize to -1 to 1
    
    return TeamProfile(
        name=team_name,
        attack_strength=attack,
        defense_strength=defense,
        transition_speed=0.5,  # Default
        wing_exposure=1 - defense * 0.5,
        set_piece_threat=attack * 0.6,
        pressing_intensity=0.5,
        defensive_line_height=0.5,
        chaos_factor=0.3,  # Default
        form_momentum=form,
        mental_state=form * 0.5
    )

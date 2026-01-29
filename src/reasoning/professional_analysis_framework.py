"""
PROFESSIONAL MATCH ANALYSIS FRAMEWORK
=====================================

8-Ebenen-Methodik für Top 1% Fußball-Wetten-Analyse

Basiert auf Profi-Analysemethodik:
1. Fundament: Klare Zielsetzung (BTTS, Over/Under, AH, etc.)
2. Strukturelle Ebene: Spielstil-Kompatibilität, Transition Speed
3. Psychologische Ebene: Tabellenkontext, Momentum, Spielphase
4. Mikro-Ebene: Zonen-Analyse, wiederkehrende Muster
5. Statistiken: xG nach Spielstand, PPDA, Field Tilt
6. Szenario-Modellierung: Multi-Outcome Probability Distribution
7. Evaluierung: Erwartungswert-basiert, nicht Ergebnis-basiert
8. Langzeit-Tracking: Systematische Fehleranalyse über 100+ Spiele

Diese Methodik transformiert ML-Vorhersagen von reinen Stats zu
strukturierten Spielverlauf-Prognosen mit taktischem, psychologischem
und kontextuellem Verständnis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from pathlib import Path


class PlayingStyle(Enum):
    """Spielstil-Kategorien für Kompatibilitätsanalyse"""
    HIGH_PRESSING = "high_pressing"           # Gegenpressing, hohes Anlaufen
    POSSESSION_BASED = "possession_based"     # Ballbesitz-dominiert, geduldig
    COUNTER_ATTACKING = "counter_attacking"   # Tiefstehen + Konter
    TRANSITION_FAST = "transition_fast"       # Schnelle Umschaltmomente
    BALANCED = "balanced"                     # Ausgewogen
    DEFENSIVE_COMPACT = "defensive_compact"   # Tief + kompakt


class MatchContext(Enum):
    """Psychologischer und taktischer Kontext"""
    MUST_WIN = "must_win"                     # Sieg zwingend (Abstiegskampf, Qualifikation)
    POINT_SUFFICIENT = "point_sufficient"     # Punkt reicht (Tabellenführer)
    ALREADY_QUALIFIED = "already_qualified"   # Bereits qualifiziert (Rotation wahrscheinlich)
    GOAL_DIFFERENCE_CRITICAL = "gd_critical"  # Torverhältnis entscheidend
    DERBY = "derby"                           # Derby (emotionale Intensität)
    NEUTRAL = "neutral"                       # Normale Spielsituation


@dataclass
class TransitionMetrics:
    """Übergangsphasen-Kennzahlen (kritischer als pure Stats)"""
    time_to_shot_after_turnover: float = 0.0    # Sekunden von Ballgewinn → Abschluss
    vertical_passes_per_transition: float = 0.0  # Anzahl vertikaler Pässe
    defensive_transition_speed: float = 0.0      # Geschwindigkeit Rückwärtsbewegung
    counter_attack_frequency: float = 0.0        # Konter pro Spiel
    transition_xg: float = 0.0                   # xG aus Umschaltmomenten
    
    def effectiveness_score(self) -> float:
        """Composite Score für Transition-Effektivität"""
        # Schnellere Zeit = besser (inverser Faktor)
        speed_factor = 10.0 / (self.time_to_shot_after_turnover + 1.0)
        return (
            speed_factor * 0.3 +
            self.vertical_passes_per_transition * 0.2 +
            self.counter_attack_frequency * 0.2 +
            self.transition_xg * 0.3
        )


@dataclass
class ZoneAnalysis:
    """Mikro-Ebene: Wo entstehen Chancen, wo sind Schwächen?"""
    left_wing_weakness: float = 0.0      # 0-1: Schwäche linker Flügel
    right_wing_weakness: float = 0.0     # 0-1: Schwäche rechter Flügel
    central_penetration: float = 0.0     # 0-1: Zentrale Durchschlagskraft
    set_piece_strength: float = 0.0      # 0-1: Standardsituationen
    high_line_vulnerability: float = 0.0 # 0-1: Anfälligkeit bei hoher Linie
    
    def exploit_potential(self, opponent_zone: 'ZoneAnalysis') -> float:
        """Wie gut kann Team die Schwächen des Gegners ausnutzen?"""
        return (
            self.right_wing_weakness * opponent_zone.left_wing_weakness +
            self.left_wing_weakness * opponent_zone.right_wing_weakness +
            self.central_penetration * (1.0 - opponent_zone.central_penetration) +
            self.high_line_vulnerability * opponent_zone.counter_attack_frequency
        ) / 4.0


@dataclass
class GoalPattern:
    """Wiederkehrende Muster: WIE entstehen Tore?"""
    from_set_pieces: float = 0.0        # % Tore aus Standards
    from_cutbacks: float = 0.0          # % Tore aus Rückpässen
    from_long_shots: float = 0.0        # % Tore aus Distanz
    from_counters: float = 0.0          # % Tore aus Kontern
    from_individual: float = 0.0        # % Tore durch Einzelaktionen
    
    def pattern_stability(self) -> float:
        """Wie stabil sind die Muster? (höhere Varianz = instabiler)"""
        values = [self.from_set_pieces, self.from_cutbacks, self.from_long_shots,
                  self.from_counters, self.from_individual]
        return 1.0 - np.std(values)  # Niedrige Standardabweichung = stabil


@dataclass
class MatchScenario:
    """Szenario-Modellierung: Multiple Outcomes mit Wahrscheinlichkeiten"""
    name: str
    probability: float
    expected_home_goals: float
    expected_away_goals: float
    triggers: List[str] = field(default_factory=list)
    
    def supports_bet(self, bet_type: str, threshold: float = None) -> bool:
        """Prüft, ob Szenario einen bestimmten Bet-Typ unterstützt"""
        total_goals = self.expected_home_goals + self.expected_away_goals
        
        if bet_type == "over_1_5":
            return total_goals > 1.5
        elif bet_type == "over_2_5":
            return total_goals > 2.5
        elif bet_type == "btts":
            return self.expected_home_goals >= 1.0 and self.expected_away_goals >= 1.0
        elif bet_type == "home_win":
            return self.expected_home_goals > self.expected_away_goals
        else:
            return False


class ProfessionalAnalysisEngine:
    """
    Kern-Engine für professionelle Match-Analyse
    
    Transformiert ML-Statistiken in strukturierte Spielverlauf-Prognosen
    """
    
    def __init__(self):
        self.style_compatibility_matrix = self._build_style_matrix()
        self.context_modifiers = self._build_context_modifiers()
        
    def _build_style_matrix(self) -> Dict[Tuple[PlayingStyle, PlayingStyle], Dict]:
        """
        Spielstil-Kompatibilitätsmatrix
        
        Definiert, was passiert, wenn bestimmte Stile aufeinandertreffen
        """
        return {
            # HIGH_PRESSING vs andere
            (PlayingStyle.HIGH_PRESSING, PlayingStyle.POSSESSION_BASED): {
                "chaos_factor": 0.8,
                "goals_expectation": 1.2,  # Multiplikator
                "btts_boost": 0.15,
                "variance": 0.7,
                "description": "Pressing vs. Aufbau → Chaos, Fehler, Tore"
            },
            (PlayingStyle.HIGH_PRESSING, PlayingStyle.COUNTER_ATTACKING): {
                "chaos_factor": 0.6,
                "goals_expectation": 1.1,
                "btts_boost": 0.20,
                "variance": 0.8,
                "description": "Beide Teams risikoreich → hohe Varianz, BTTS-freundlich"
            },
            (PlayingStyle.HIGH_PRESSING, PlayingStyle.DEFENSIVE_COMPACT): {
                "chaos_factor": 0.3,
                "goals_expectation": 0.85,
                "btts_boost": -0.10,
                "variance": 0.4,
                "description": "Pressing vs. Mauer → wenige Chancen, niedriger Score"
            },
            
            # POSSESSION_BASED vs andere
            (PlayingStyle.POSSESSION_BASED, PlayingStyle.DEFENSIVE_COMPACT): {
                "chaos_factor": 0.2,
                "goals_expectation": 0.75,
                "btts_boost": -0.15,
                "variance": 0.3,
                "description": "Ballbesitz ohne Tempo vs. Mauer → sehr wenig Tore"
            },
            (PlayingStyle.POSSESSION_BASED, PlayingStyle.COUNTER_ATTACKING): {
                "chaos_factor": 0.5,
                "goals_expectation": 1.05,
                "btts_boost": 0.10,
                "variance": 0.6,
                "description": "Ballbesitz öffnet Räume für Konter → ausgeglichen"
            },
            
            # TRANSITION_FAST vs andere (Umschaltspiel)
            (PlayingStyle.TRANSITION_FAST, PlayingStyle.TRANSITION_FAST): {
                "chaos_factor": 0.9,
                "goals_expectation": 1.3,
                "btts_boost": 0.25,
                "variance": 0.9,
                "description": "Beide schnell umschaltend → extrem hohe Varianz, BTTS sehr wahrscheinlich"
            },
            
            # Balanced (Fallback)
            (PlayingStyle.BALANCED, PlayingStyle.BALANCED): {
                "chaos_factor": 0.5,
                "goals_expectation": 1.0,
                "btts_boost": 0.0,
                "variance": 0.5,
                "description": "Ausgeglichenes Spiel → Standard-Erwartungen"
            },
        }
    
    def _build_context_modifiers(self) -> Dict[MatchContext, Dict]:
        """
        Psychologische Kontext-Modifikatoren
        
        Wie verändert die Situation das Spielverhalten?
        """
        return {
            MatchContext.MUST_WIN: {
                "risk_increase": 0.3,
                "goals_expectation": 1.15,
                "btts_boost": 0.10,
                "description": "Team muss gewinnen → höheres Risiko, mehr Tore"
            },
            MatchContext.POINT_SUFFICIENT: {
                "risk_increase": -0.2,
                "goals_expectation": 0.85,
                "btts_boost": -0.10,
                "description": "Punkt reicht → defensiver, weniger Risiko"
            },
            MatchContext.ALREADY_QUALIFIED: {
                "risk_increase": -0.3,
                "goals_expectation": 0.80,
                "btts_boost": -0.05,
                "description": "Bereits qualifiziert → Rotation, niedrigere Intensität"
            },
            MatchContext.GOAL_DIFFERENCE_CRITICAL: {
                "risk_increase": 0.4,
                "goals_expectation": 1.25,
                "btts_boost": 0.15,
                "description": "Torverhältnis zählt → sehr offensiv, viele Tore"
            },
            MatchContext.DERBY: {
                "risk_increase": 0.2,
                "goals_expectation": 1.1,
                "btts_boost": 0.05,
                "variance_increase": 0.3,
                "description": "Derby → emotional, unvorhersehbar"
            },
            MatchContext.NEUTRAL: {
                "risk_increase": 0.0,
                "goals_expectation": 1.0,
                "btts_boost": 0.0,
                "description": "Normale Spielsituation"
            },
        }
    
    def analyze_style_compatibility(
        self,
        home_style: PlayingStyle,
        away_style: PlayingStyle
    ) -> Dict:
        """
        Analysiere Spielstil-Kompatibilität
        
        Returns:
            Dict mit chaos_factor, goals_expectation, btts_boost, variance
        """
        # Direkte Paarung
        if (home_style, away_style) in self.style_compatibility_matrix:
            return self.style_compatibility_matrix[(home_style, away_style)]
        
        # Umgekehrte Paarung
        if (away_style, home_style) in self.style_compatibility_matrix:
            result = self.style_compatibility_matrix[(away_style, home_style)].copy()
            result["description"] = f"Inverse: {result['description']}"
            return result
        
        # Fallback: Balanced vs Balanced
        return self.style_compatibility_matrix[(PlayingStyle.BALANCED, PlayingStyle.BALANCED)]
    
    def generate_match_scenarios(
        self,
        base_home_xg: float,
        base_away_xg: float,
        home_style: PlayingStyle,
        away_style: PlayingStyle,
        home_context: MatchContext,
        away_context: MatchContext,
        home_transition: TransitionMetrics,
        away_transition: TransitionMetrics
    ) -> List[MatchScenario]:
        """
        Szenario-Modellierung: Generiere Multiple Outcomes
        
        Nicht ein einziges Ergebnis, sondern Verteilung von Szenarien
        mit Wahrscheinlichkeiten
        """
        scenarios = []
        
        # 1. Style Compatibility Analysis
        style_compat = self.analyze_style_compatibility(home_style, away_style)
        
        # 2. Context Modifiers
        home_mod = self.context_modifiers[home_context]
        away_mod = self.context_modifiers[away_context]
        
        # 3. Transition Effectiveness
        home_transition_score = home_transition.effectiveness_score()
        away_transition_score = away_transition.effectiveness_score()
        
        # Base Scenario (Expected Value)
        adjusted_home_xg = base_home_xg * style_compat["goals_expectation"] * home_mod["goals_expectation"]
        adjusted_away_xg = base_away_xg * style_compat["goals_expectation"] * away_mod["goals_expectation"]
        
        # Transition Boost
        adjusted_home_xg += home_transition_score * 0.15
        adjusted_away_xg += away_transition_score * 0.15
        
        # === SZENARIO A: Spiel öffnet sich (Higher Variance) ===
        if style_compat["chaos_factor"] > 0.6 or style_compat["variance"] > 0.7:
            scenarios.append(MatchScenario(
                name="Spiel öffnet sich",
                probability=0.40,
                expected_home_goals=adjusted_home_xg * 1.2,
                expected_away_goals=adjusted_away_xg * 1.2,
                triggers=[
                    "Hohe Chaos-Factor",
                    style_compat["description"],
                    "Schnelle Übergangsphasen"
                ]
            ))
        
        # === SZENARIO B: Favorit kontrolliert ===
        if abs(adjusted_home_xg - adjusted_away_xg) > 0.5:
            favorite_goals = max(adjusted_home_xg, adjusted_away_xg) * 1.1
            underdog_goals = min(adjusted_home_xg, adjusted_away_xg) * 0.7
            
            scenarios.append(MatchScenario(
                name="Favorit kontrolliert",
                probability=0.35,
                expected_home_goals=favorite_goals if adjusted_home_xg > adjusted_away_xg else underdog_goals,
                expected_away_goals=underdog_goals if adjusted_home_xg > adjusted_away_xg else favorite_goals,
                triggers=[
                    "Klarer Qualitätsunterschied",
                    "Ballbesitzdominanz",
                    "Territoriale Kontrolle"
                ]
            ))
        
        # === SZENARIO C: Stalemate / Niedriger Score ===
        if style_compat["chaos_factor"] < 0.4:
            scenarios.append(MatchScenario(
                name="Stalemate / Wenig Tore",
                probability=0.25,
                expected_home_goals=adjusted_home_xg * 0.6,
                expected_away_goals=adjusted_away_xg * 0.6,
                triggers=[
                    "Niedriger Chaos-Factor",
                    style_compat["description"],
                    "Beide Teams defensiv stabil"
                ]
            ))
        
        # === SZENARIO D: Early Goal Changes Game ===
        # Frühes Tor verändert Risikoprofil komplett
        scenarios.append(MatchScenario(
            name="Frühes Tor verändert Spiel",
            probability=0.20,
            expected_home_goals=adjusted_home_xg * 0.8,
            expected_away_goals=adjusted_away_xg * 1.3,  # Trailing team erhöht Risiko
            triggers=[
                "Tor in ersten 15 Minuten",
                "Trailing team erhöht Pressing",
                "Räume für Konter"
            ]
        ))
        
        # Normalize probabilities to 1.0
        total_prob = sum(s.probability for s in scenarios)
        for scenario in scenarios:
            scenario.probability /= total_prob
        
        return scenarios
    
    def evaluate_bet_value(
        self,
        scenarios: List[MatchScenario],
        bet_type: str,
        offered_odds: float
    ) -> Dict:
        """
        Evaluiere Bet Value über alle Szenarien
        
        Profis denken:
        "Welche Wett-Option profitiert von MEHREREN Szenarien gleichzeitig?"
        
        Returns:
            {
                "expected_probability": float,
                "implied_probability": float,
                "edge": float,
                "ev_percentage": float,
                "supporting_scenarios": List[str],
                "confidence": str
            }
        """
        # Berechne erwartete Wahrscheinlichkeit über alle Szenarien
        expected_prob = sum(
            scenario.probability
            for scenario in scenarios
            if scenario.supports_bet(bet_type)
        )
        
        # Implied Probability aus Odds
        implied_prob = 1.0 / offered_odds
        
        # Edge = Expected - Implied
        edge = expected_prob - implied_prob
        
        # Expected Value (EV) in Prozent
        ev_percentage = (expected_prob * offered_odds - 1.0) * 100
        
        # Supporting Scenarios
        supporting = [
            f"{s.name} ({s.probability:.1%})"
            for s in scenarios
            if s.supports_bet(bet_type)
        ]
        
        # Confidence Level
        if expected_prob > 0.70 and edge > 0.10:
            confidence = "HIGH"
        elif expected_prob > 0.60 and edge > 0.05:
            confidence = "MEDIUM"
        elif expected_prob > 0.50 and edge > 0.00:
            confidence = "LOW"
        else:
            confidence = "NO_VALUE"
        
        return {
            "expected_probability": expected_prob,
            "implied_probability": implied_prob,
            "edge": edge,
            "ev_percentage": ev_percentage,
            "supporting_scenarios": supporting,
            "confidence": confidence,
            "recommendation": "BET" if edge > 0.05 else "SKIP"
        }
    
    def generate_analysis_report(
        self,
        home_team: str,
        away_team: str,
        scenarios: List[MatchScenario],
        bet_evaluations: Dict[str, Dict]
    ) -> str:
        """
        Generiere strukturierten Analyse-Report
        
        Format: Wie Profis analysieren (nicht nur Tipp, sondern Begründung)
        """
        report = f"""
{'='*70}
PROFESSIONAL MATCH ANALYSIS
{'='*70}

Match: {home_team} vs {away_team}

SZENARIO-MODELLIERUNG
----------------------
"""
        
        for i, scenario in enumerate(scenarios, 1):
            report += f"""
Szenario {i}: {scenario.name} ({scenario.probability:.1%})
  Expected Goals: {scenario.expected_home_goals:.1f} - {scenario.expected_away_goals:.1f}
  Triggers: {', '.join(scenario.triggers)}
"""
        
        report += f"""
BET VALUE ANALYSIS
------------------
"""
        
        for bet_type, evaluation in bet_evaluations.items():
            report += f"""
{bet_type.upper()}:
  Expected Probability: {evaluation['expected_probability']:.1%}
  Implied Probability: {evaluation['implied_probability']:.1%}
  Edge: {evaluation['edge']:+.1%}
  EV: {evaluation['ev_percentage']:+.1f}%
  Confidence: {evaluation['confidence']}
  Recommendation: {evaluation['recommendation']}
  Supporting Scenarios: {', '.join(evaluation['supporting_scenarios'])}
"""
        
        report += f"""
{'='*70}
"""
        
        return report


# Long-term Evaluation System
class LongTermEvaluator:
    """
    Langzeit-Evaluierung über 100+ Spiele
    
    Tracking von:
    - Erwarteter vs. tatsächlicher Trefferquote
    - Systematischen Fehlern
    - ROI über Zeit
    """
    
    def __init__(self, db_path: str = "data/tracking/professional_analysis.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(
        self,
        match_id: str,
        bet_type: str,
        expected_prob: float,
        offered_odds: float,
        scenarios: List[MatchScenario],
        timestamp: datetime = None
    ):
        """Log eine Vorhersage für spätere Evaluierung"""
        if timestamp is None:
            timestamp = datetime.now()
        
        prediction = {
            "match_id": match_id,
            "bet_type": bet_type,
            "expected_probability": expected_prob,
            "offered_odds": offered_odds,
            "implied_probability": 1.0 / offered_odds,
            "edge": expected_prob - (1.0 / offered_odds),
            "scenarios": [
                {
                    "name": s.name,
                    "probability": s.probability,
                    "expected_home_goals": s.expected_home_goals,
                    "expected_away_goals": s.expected_away_goals
                }
                for s in scenarios
            ],
            "timestamp": timestamp.isoformat()
        }
        
        # Append to JSON Lines file
        predictions_file = self.db_path.parent / "predictions.jsonl"
        with open(predictions_file, 'a') as f:
            f.write(json.dumps(prediction) + '\n')
    
    def evaluate_performance(self, min_bets: int = 100) -> Dict:
        """
        Evaluiere Performance über mindestens N Bets
        
        Returns:
            {
                "total_predictions": int,
                "actual_win_rate": float,
                "expected_win_rate": float,
                "calibration_error": float,
                "roi": float,
                "systematic_errors": List[str]
            }
        """
        predictions_file = self.db_path.parent / "predictions.jsonl"
        
        if not predictions_file.exists():
            return {"error": "No predictions logged yet"}
        
        # Load all predictions
        predictions = []
        with open(predictions_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if len(predictions) < min_bets:
            return {
                "error": f"Insufficient data ({len(predictions)} predictions, need {min_bets})"
            }
        
        # TODO: Implement actual result matching + ROI calculation
        # This requires integration with live bet tracker
        
        return {
            "total_predictions": len(predictions),
            "status": "Awaiting result integration"
        }


if __name__ == "__main__":
    # Demo: Professional Analysis in Action
    engine = ProfessionalAnalysisEngine()
    
    # Example Match: Bayern München vs Borussia Dortmund
    scenarios = engine.generate_match_scenarios(
        base_home_xg=2.1,
        base_away_xg=1.8,
        home_style=PlayingStyle.HIGH_PRESSING,
        away_style=PlayingStyle.TRANSITION_FAST,
        home_context=MatchContext.MUST_WIN,
        away_context=MatchContext.NEUTRAL,
        home_transition=TransitionMetrics(
            time_to_shot_after_turnover=8.5,
            vertical_passes_per_transition=3.2,
            counter_attack_frequency=4.5,
            transition_xg=0.6
        ),
        away_transition=TransitionMetrics(
            time_to_shot_after_turnover=7.2,
            vertical_passes_per_transition=3.8,
            counter_attack_frequency=5.1,
            transition_xg=0.8
        )
    )
    
    # Evaluate Bets
    bet_evaluations = {
        "over_2_5": engine.evaluate_bet_value(scenarios, "over_2_5", offered_odds=1.75),
        "btts": engine.evaluate_bet_value(scenarios, "btts", offered_odds=1.55),
    }
    
    # Generate Report
    report = engine.generate_analysis_report(
        "Bayern München",
        "Borussia Dortmund",
        scenarios,
        bet_evaluations
    )
    
    print(report)

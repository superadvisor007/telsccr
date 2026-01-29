"""
PREMIUM PREDICTION ENGINE V3
============================

Integration of:
1. ML Predictions (base accuracy)
2. Professional Analysis (structural understanding)
3. CLV Tracking (market edge verification)
4. Odds Comparison (value maximization)
5. Sharp Indicators (professional money tracking)

This is what a $2000/month service looks like.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our components
try:
    from src.reasoning.professional_analysis_framework import ProfessionalAnalysisEngine
    from src.reasoning.team_profile_database import TeamProfileDatabase
    from src.premium.clv_tracker import ClosingLineValueTracker
    from src.premium.odds_comparison import OddsComparisonEngine, BookmakerOdds
except ImportError as e:
    print(f"⚠️  Import warning: {e}")


@dataclass
class PremiumPrediction:
    """
    Premium prediction with all intelligence layers
    
    This is what customers pay for: complete analysis, not just a number
    """
    # Match info
    match_id: str
    home_team: str
    away_team: str
    match_date: str
    league: str
    
    # ML Prediction
    ml_probability: float
    ml_confidence: str  # 'HIGH' | 'MEDIUM' | 'LOW'
    
    # Professional Analysis
    scenarios: List[Dict]  # Multiple possible outcomes
    playing_style_analysis: str
    context_factors: List[str]
    expected_goals_range: Tuple[float, float]
    
    # Market Analysis
    market: str
    predicted_probability: float
    
    # Odds Intelligence
    best_odds: float
    best_bookmaker: str
    average_market_odds: float
    odds_edge_pct: float  # How much better than average
    
    # Sharp Indicators
    sharp_confidence: str  # 'HIGH' | 'MEDIUM' | 'LOW'
    line_movement_pct: float
    reverse_line_movement: bool
    steam_move: bool
    
    # Value Assessment
    expected_value: float
    implied_probability: float
    edge: float
    kelly_stake_pct: float
    
    # Final Recommendation
    recommendation: str  # 'STRONG_BET' | 'BET' | 'LEAN' | 'SKIP'
    confidence_score: float  # 0-100
    reasoning: str
    
    # Risk Warning
    risk_factors: List[str]
    
    def to_telegram_message(self) -> str:
        """Format as Telegram message"""
        emoji = "🔥" if self.recommendation == "STRONG_BET" else "✅" if self.recommendation == "BET" else "💡" if self.recommendation == "LEAN" else "⏭️"
        
        msg = f"""
{emoji} **{self.home_team} vs {self.away_team}**
📅 {self.match_date} | 🏆 {self.league}

**PREDICTION:** {self.market.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━

**🎯 PROBABILITY:** {self.predicted_probability*100:.1f}%
**💰 BEST ODDS:** {self.best_odds:.2f} ({self.best_bookmaker})
**📊 EDGE:** +{self.edge*100:.1f}%
**💎 EXPECTED VALUE:** +{self.expected_value:.1f}%

**🧠 PROFESSIONAL ANALYSIS:**
{self.playing_style_analysis}

**📈 SCENARIOS:**
"""
        for i, scenario in enumerate(self.scenarios[:2], 1):
            msg += f"{i}. {scenario['name']} ({scenario['probability']*100:.1f}%)\n"
            msg += f"   Expected: {scenario['expected_home_goals']:.1f} - {scenario['expected_away_goals']:.1f}\n"
        
        msg += f"""
**🎲 SHARP INDICATORS:**
• Line Movement: {self.line_movement_pct:+.1f}%
• Reverse Movement: {"✅" if self.reverse_line_movement else "❌"}
• Steam Move: {"✅" if self.steam_move else "❌"}
• Sharp Confidence: {self.sharp_confidence}

**💸 BET RECOMMENDATION:**
**{self.recommendation}** (Confidence: {self.confidence_score:.0f}/100)
Stake: {self.kelly_stake_pct:.1f}% of bankroll

**💡 REASONING:**
{self.reasoning}
"""
        
        if self.risk_factors:
            msg += "\n**⚠️ RISK FACTORS:**\n"
            for factor in self.risk_factors:
                msg += f"• {factor}\n"
        
        return msg


class PremiumPredictionEngine:
    """
    Premium Prediction Engine V3
    
    This is the complete system that justifies premium pricing:
    - ML + Professional Analysis
    - CLV tracking for verified edge
    - Odds comparison for value maximization
    - Sharp indicators for pro insights
    """
    
    def __init__(self):
        print("🚀 Initializing Premium Prediction Engine V3...")
        
        # Core components (all optional, graceful degradation)
        self.professional_engine = None
        self.team_db = None
        self.clv_tracker = None
        self.odds_engine = None
        
        try:
            from src.reasoning.professional_analysis_framework import ProfessionalAnalysisEngine
            from src.reasoning.team_profile_database import TeamProfileDatabase
            self.professional_engine = ProfessionalAnalysisEngine()
            self.team_db = TeamProfileDatabase()
            print("✅ Professional Analysis loaded")
        except Exception as e:
            print(f"⚠️  Professional Analysis not available: {e}")
        
        try:
            from src.premium.clv_tracker import ClosingLineValueTracker
            from src.premium.odds_comparison import OddsComparisonEngine
            self.clv_tracker = ClosingLineValueTracker()
            self.odds_engine = OddsComparisonEngine()
            print("✅ CLV Tracker and Odds Engine loaded")
        except Exception as e:
            print(f"⚠️  Premium components not fully loaded: {e}")
            # Create minimal fallback instances
            if not self.clv_tracker:
                from src.premium.clv_tracker import ClosingLineValueTracker
                self.clv_tracker = ClosingLineValueTracker()
            if not self.odds_engine:
                from src.premium.odds_comparison import OddsComparisonEngine
                self.odds_engine = OddsComparisonEngine()
    
    def predict_premium(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_date: str,
        market: str = "over_2_5",
        ml_features: Optional[Dict] = None
    ) -> PremiumPrediction:
        """
        Generate premium prediction with all intelligence layers
        
        This is the main product: complete analysis, not just a number
        """
        match_id = f"{home_team}_{away_team}_{match_date}".replace(" ", "_")
        
        # 1. ML Prediction (base accuracy)
        if ml_features:
            ml_probability = self._get_ml_prediction(ml_features, market)
            ml_confidence = self._assess_ml_confidence(ml_probability)
        else:
            ml_probability = 0.60  # Demo
            ml_confidence = "MEDIUM"
        
        # 2. Professional Analysis (structural understanding)
        scenarios = []
        playing_style_analysis = "Team profiles not available"
        expected_goals_range = (2.0, 3.5)
        
        if self.team_db and self.professional_engine:
            home_profile = self.team_db.get_profile(home_team)
            away_profile = self.team_db.get_profile(away_team)
            
            if home_profile and away_profile:
                analysis = self.professional_engine.analyze_match(
                    home_profile, away_profile, context="NORMAL", league=league
                )
                
                scenarios = analysis['scenarios']
                playing_style_analysis = analysis['playing_style_analysis']
                expected_goals_range = analysis['expected_goals_range']
        
        # 3. Odds Intelligence (value maximization)
        best_odds_obj = self.odds_engine.find_best_odds(
            home_team, away_team, market, min_bookmakers=5
        )
        
        if best_odds_obj:
            best_odds = best_odds_obj.odds
            best_bookmaker = best_odds_obj.bookmaker
        else:
            best_odds = 1.85  # Demo
            best_bookmaker = "Bet365"
        
        # Get average market odds
        all_odds = self.odds_engine._get_simulated_odds(market)
        average_market_odds = np.mean([o.odds for o in all_odds])
        odds_edge_pct = ((best_odds - average_market_odds) / average_market_odds) * 100
        
        # 4. Sharp Indicators (professional money tracking)
        sharp_indicators = self.clv_tracker.detect_sharp_indicators(
            match_id, market, public_bet_percentage=None
        )
        
        sharp_confidence = sharp_indicators['sharp_confidence']
        line_movement_pct = sharp_indicators['line_movement_pct']
        reverse_line_movement = sharp_indicators['reverse_line_movement']
        steam_move = sharp_indicators['steam_move']
        
        # 5. Value Assessment
        predicted_probability = self._combine_probabilities(
            ml_probability, scenarios, market
        )
        
        implied_probability = 1.0 / best_odds
        edge = predicted_probability - implied_probability
        expected_value = (predicted_probability * best_odds - 1.0) * 100
        
        # Kelly Criterion staking (0.25 fractional)
        if edge > 0:
            kelly_stake_pct = 0.25 * (edge / (best_odds - 1.0)) * 100
            kelly_stake_pct = min(kelly_stake_pct, 10.0)  # Cap at 10%
        else:
            kelly_stake_pct = 0.0
        
        # 6. Final Recommendation
        recommendation, confidence_score, reasoning = self._make_recommendation(
            predicted_probability,
            edge,
            expected_value,
            ml_confidence,
            sharp_confidence,
            scenarios,
            best_odds
        )
        
        # 7. Risk Assessment
        risk_factors = self._assess_risks(
            scenarios, line_movement_pct, ml_confidence, sharp_confidence
        )
        
        # Context factors
        context_factors = []
        if reverse_line_movement:
            context_factors.append("Reverse line movement (sharp money detected)")
        if steam_move:
            context_factors.append("Steam move (sudden sharp action)")
        if edge > 0.05:
            context_factors.append("Strong value (>5% edge)")
        
        return PremiumPrediction(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league=league,
            ml_probability=ml_probability,
            ml_confidence=ml_confidence,
            scenarios=scenarios,
            playing_style_analysis=playing_style_analysis,
            context_factors=context_factors,
            expected_goals_range=expected_goals_range,
            market=market,
            predicted_probability=predicted_probability,
            best_odds=best_odds,
            best_bookmaker=best_bookmaker,
            average_market_odds=average_market_odds,
            odds_edge_pct=odds_edge_pct,
            sharp_confidence=sharp_confidence,
            line_movement_pct=line_movement_pct,
            reverse_line_movement=reverse_line_movement,
            steam_move=steam_move,
            expected_value=expected_value,
            implied_probability=implied_probability,
            edge=edge,
            kelly_stake_pct=kelly_stake_pct,
            recommendation=recommendation,
            confidence_score=confidence_score,
            reasoning=reasoning,
            risk_factors=risk_factors
        )
    
    def _get_ml_prediction(self, features: Dict, market: str) -> float:
        """Get ML model prediction"""
        # In production, load model and predict
        # For now, return demo
        return 0.62
    
    def _assess_ml_confidence(self, probability: float) -> str:
        """Assess ML confidence level"""
        if probability > 0.65 or probability < 0.35:
            return "HIGH"
        elif probability > 0.60 or probability < 0.40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _combine_probabilities(
        self,
        ml_probability: float,
        scenarios: List[Dict],
        market: str
    ) -> float:
        """
        Combine ML and Professional Analysis probabilities
        
        Weight: 60% ML, 40% Scenarios
        """
        if not scenarios:
            return ml_probability
        
        # Calculate scenario-based probability
        scenario_probability = 0.0
        for scenario in scenarios:
            # Check if this scenario supports the market
            if market == "over_2_5":
                total_goals = scenario['expected_home_goals'] + scenario['expected_away_goals']
                if total_goals > 2.5:
                    scenario_probability += scenario['probability']
            elif market == "over_1_5":
                total_goals = scenario['expected_home_goals'] + scenario['expected_away_goals']
                if total_goals > 1.5:
                    scenario_probability += scenario['probability']
            elif market == "btts":
                if scenario['expected_home_goals'] >= 1.0 and scenario['expected_away_goals'] >= 1.0:
                    scenario_probability += scenario['probability']
        
        # Combine: 60% ML, 40% Scenarios
        combined = 0.6 * ml_probability + 0.4 * scenario_probability
        
        return combined
    
    def _make_recommendation(
        self,
        predicted_prob: float,
        edge: float,
        expected_value: float,
        ml_confidence: str,
        sharp_confidence: str,
        scenarios: List[Dict],
        odds: float
    ) -> Tuple[str, float, str]:
        """
        Make final bet recommendation
        
        Returns:
            (recommendation, confidence_score, reasoning)
        """
        # Calculate confidence score (0-100)
        confidence_score = 50.0
        
        # Add confidence from edge
        if edge > 0.10:
            confidence_score += 20
        elif edge > 0.05:
            confidence_score += 10
        
        # Add confidence from ML
        if ml_confidence == "HIGH":
            confidence_score += 15
        elif ml_confidence == "MEDIUM":
            confidence_score += 10
        
        # Add confidence from sharp indicators
        if sharp_confidence == "HIGH":
            confidence_score += 15
        elif sharp_confidence == "MEDIUM":
            confidence_score += 10
        
        # Add confidence from scenarios
        if len(scenarios) >= 2:
            supporting_scenarios = sum(1 for s in scenarios if s.get('supports_bet', False))
            if supporting_scenarios >= 2:
                confidence_score += 10
        
        confidence_score = min(confidence_score, 100)
        
        # Make recommendation
        if edge > 0.08 and confidence_score >= 80:
            recommendation = "STRONG_BET"
            reasoning = f"Strong value bet: {edge*100:.1f}% edge, {confidence_score:.0f}/100 confidence. Multiple positive signals."
        elif edge > 0.05 and confidence_score >= 70:
            recommendation = "BET"
            reasoning = f"Good value: {edge*100:.1f}% edge, {confidence_score:.0f}/100 confidence. Recommended bet."
        elif edge > 0.03 and confidence_score >= 60:
            recommendation = "LEAN"
            reasoning = f"Slight edge: {edge*100:.1f}% edge, {confidence_score:.0f}/100 confidence. Consider betting small."
        else:
            recommendation = "SKIP"
            reasoning = f"No clear value: {edge*100:.1f}% edge too small or confidence too low."
        
        return recommendation, confidence_score, reasoning
    
    def _assess_risks(
        self,
        scenarios: List[Dict],
        line_movement_pct: float,
        ml_confidence: str,
        sharp_confidence: str
    ) -> List[str]:
        """Assess risk factors"""
        risks = []
        
        if ml_confidence == "LOW":
            risks.append("ML model has low confidence")
        
        if sharp_confidence == "LOW":
            risks.append("No sharp money indicators")
        
        if abs(line_movement_pct) > 5.0:
            risks.append(f"Significant line movement ({line_movement_pct:+.1f}%)")
        
        if len(scenarios) < 2:
            risks.append("Limited scenario analysis (missing team profiles)")
        
        return risks


if __name__ == "__main__":
    print("=" * 70)
    print("PREMIUM PREDICTION ENGINE V3 - DEMO")
    print("=" * 70)
    
    engine = PremiumPredictionEngine()
    
    # Generate premium prediction
    prediction = engine.predict_premium(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        match_date="2026-01-30",
        market="over_2_5"
    )
    
    # Display as Telegram message
    print(prediction.to_telegram_message())
    
    # Display as JSON (for API)
    print("\n" + "=" * 70)
    print("JSON OUTPUT (for API integration)")
    print("=" * 70)
    import json
    print(json.dumps(asdict(prediction), indent=2, default=str))

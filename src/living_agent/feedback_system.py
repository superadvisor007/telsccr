"""
🔄 FEEDBACK SYSTEM - Self-Improving Learning Loop
================================================
Tracks predictions, collects results, and improves over time.

This is what makes the agent "alive" - it learns from its mistakes.

Features:
- Result collection and verification
- Performance tracking by market/league
- Error analysis and pattern detection
- Confidence calibration adjustment
- Weekly performance summaries
- Reasoning pattern improvement
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from living_agent.knowledge_cache import KnowledgeCache


@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    market: str
    predicted_prob: float
    predicted_confidence: float
    odds_at_prediction: float
    reasoning_summary: str
    ticket_id: Optional[str] = None


@dataclass
class ResultRecord:
    """Result of a prediction after match completion."""
    match_id: str
    market: str
    actual_outcome: bool  # True if prediction was correct
    actual_score: str  # e.g., "2-1"
    home_goals: int
    away_goals: int
    profit_loss: float
    verified_at: str


@dataclass
class PerformanceReport:
    """Performance report for a time period."""
    period_start: str
    period_end: str
    total_predictions: int
    correct_predictions: int
    win_rate: float
    total_stake: float
    total_profit: float
    roi: float
    best_market: str
    worst_market: str
    market_breakdown: Dict[str, Dict[str, float]]
    confidence_calibration: Dict[str, float]
    improvement_suggestions: List[str]


class FeedbackSystem:
    """
    🔄 Self-Improving Feedback Loop
    
    Tracks every prediction, collects results, and learns:
    1. Records all predictions with reasoning
    2. Verifies results after matches
    3. Analyzes performance by market/league
    4. Detects patterns in errors
    5. Adjusts confidence calibration
    6. Generates improvement suggestions
    
    "The system gets smarter over time."
    """
    
    def __init__(self, cache: KnowledgeCache = None):
        self.cache = cache or KnowledgeCache()
        self.predictions_file = "data/predictions/pending.json"
        self.results_file = "data/predictions/results.json"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.predictions_file), exist_ok=True)
        
        # Load pending predictions
        self.pending_predictions = self._load_pending()
        
        # Calibration adjustments (learned over time)
        self.calibration_adjustments = {
            'btts': 1.0,
            'over_1_5': 1.0,
            'over_2_5': 1.0,
        }
        
    def _load_pending(self) -> List[PredictionRecord]:
        """Load pending predictions from file."""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                return [PredictionRecord(**p) for p in data]
        except Exception as e:
            print(f"Error loading pending: {e}")
        return []
    
    def _save_pending(self):
        """Save pending predictions to file."""
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump([asdict(p) for p in self.pending_predictions], f, indent=2)
        except Exception as e:
            print(f"Error saving pending: {e}")
    
    # ==================== PREDICTION RECORDING ====================
    
    def record_prediction(self, prediction: PredictionRecord) -> bool:
        """Record a new prediction for future verification."""
        self.pending_predictions.append(prediction)
        self._save_pending()
        
        print(f"📝 Recorded prediction: {prediction.home_team} vs {prediction.away_team} - {prediction.market}")
        return True
    
    def record_ticket_predictions(self, ticket: Any) -> bool:
        """Record all predictions from a multi-bet ticket."""
        for leg in ticket.legs:
            prediction = PredictionRecord(
                match_id=leg.match_id,
                home_team=leg.home_team,
                away_team=leg.away_team,
                league=leg.league,
                match_date=datetime.now().strftime('%Y-%m-%d'),  # Would get from leg
                market=leg.market,
                predicted_prob=leg.probability,
                predicted_confidence=leg.confidence,
                odds_at_prediction=leg.odds,
                reasoning_summary=leg.reasoning,
                ticket_id=ticket.ticket_id
            )
            self.record_prediction(prediction)
        
        return True
    
    # ==================== RESULT COLLECTION ====================
    
    def verify_result(
        self,
        match_id: str,
        home_goals: int,
        away_goals: int
    ) -> List[ResultRecord]:
        """Verify predictions for a completed match."""
        
        results = []
        actual_score = f"{home_goals}-{away_goals}"
        
        # Find all predictions for this match
        matched_predictions = [p for p in self.pending_predictions if p.match_id == match_id]
        
        for prediction in matched_predictions:
            # Determine outcome based on market
            actual_outcome = self._evaluate_market_outcome(
                prediction.market, home_goals, away_goals
            )
            
            # Calculate profit/loss
            if actual_outcome:
                profit_loss = prediction.odds_at_prediction - 1  # Net profit per unit
            else:
                profit_loss = -1  # Lost stake
            
            # Create result record
            result = ResultRecord(
                match_id=match_id,
                market=prediction.market,
                actual_outcome=actual_outcome,
                actual_score=actual_score,
                home_goals=home_goals,
                away_goals=away_goals,
                profit_loss=profit_loss,
                verified_at=datetime.now().isoformat()
            )
            results.append(result)
            
            # Record in cache for learning
            self.cache.record_prediction_result(
                match_id=match_id,
                market=prediction.market,
                predicted_prob=prediction.predicted_prob,
                predicted_confidence=prediction.predicted_confidence,
                actual_outcome=1 if actual_outcome else 0,
                odds=prediction.odds_at_prediction,
                reasoning_summary=prediction.reasoning_summary
            )
            
            # Remove from pending
            self.pending_predictions.remove(prediction)
            
            # Print result
            emoji = "✅" if actual_outcome else "❌"
            print(f"{emoji} {prediction.home_team} vs {prediction.away_team} ({actual_score}): {prediction.market} - {'WON' if actual_outcome else 'LOST'}")
        
        self._save_pending()
        return results
    
    def _evaluate_market_outcome(
        self,
        market: str,
        home_goals: int,
        away_goals: int
    ) -> bool:
        """Evaluate if a market prediction was correct."""
        
        total_goals = home_goals + away_goals
        btts = home_goals > 0 and away_goals > 0
        
        outcomes = {
            'btts': btts,
            'btts_yes': btts,
            'btts_no': not btts,
            'over_1_5': total_goals > 1.5,
            'over_2_5': total_goals > 2.5,
            'over_3_5': total_goals > 3.5,
            'under_1_5': total_goals < 1.5,
            'under_2_5': total_goals < 2.5,
            'under_3_5': total_goals < 3.5,
            'home_win': home_goals > away_goals,
            'away_win': away_goals > home_goals,
            'draw': home_goals == away_goals,
        }
        
        return outcomes.get(market, False)
    
    # ==================== PERFORMANCE ANALYSIS ====================
    
    def generate_performance_report(self, days: int = 30) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        # Get stats from cache
        overall_stats = self.cache.get_performance_stats(days=days)
        
        # Get per-market stats
        market_breakdown = {}
        for market in ['btts', 'over_1_5', 'over_2_5']:
            market_stats = self.cache.get_performance_stats(market=market, days=days)
            if market_stats['total_bets'] > 0:
                market_breakdown[market] = {
                    'bets': market_stats['total_bets'],
                    'wins': market_stats['wins'],
                    'win_rate': market_stats['win_rate'],
                    'profit': market_stats['total_profit'],
                    'roi': market_stats['roi']
                }
        
        # Find best and worst markets
        if market_breakdown:
            best_market = max(market_breakdown, key=lambda m: market_breakdown[m]['roi'])
            worst_market = min(market_breakdown, key=lambda m: market_breakdown[m]['roi'])
        else:
            best_market = worst_market = 'N/A'
        
        # Confidence calibration analysis
        calibration = self._analyze_calibration(days)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            overall_stats, market_breakdown, calibration
        )
        
        # Calculate ROI
        total_bets = overall_stats.get('total_bets', 0)
        total_profit = overall_stats.get('total_profit', 0)
        roi = (total_profit / total_bets) if total_bets > 0 else 0
        
        return PerformanceReport(
            period_start=(datetime.now() - timedelta(days=days)).isoformat(),
            period_end=datetime.now().isoformat(),
            total_predictions=total_bets,
            correct_predictions=overall_stats.get('wins', 0),
            win_rate=overall_stats.get('win_rate', 0),
            total_stake=float(total_bets),  # Assuming unit stakes
            total_profit=total_profit,
            roi=roi,
            best_market=best_market,
            worst_market=worst_market,
            market_breakdown=market_breakdown,
            confidence_calibration=calibration,
            improvement_suggestions=suggestions
        )
    
    def _analyze_calibration(self, days: int = 30) -> Dict[str, float]:
        """Analyze how well confidence scores match actual outcomes."""
        
        # This would query the database for predictions grouped by confidence level
        # and compare predicted probabilities to actual hit rates
        
        # Simplified version:
        calibration = {}
        
        for market in ['btts', 'over_1_5', 'over_2_5']:
            stats = self.cache.get_performance_stats(market=market, days=days)
            
            if stats['total_bets'] > 10:
                # Compare average predicted probability to actual win rate
                predicted = stats.get('avg_predicted_prob', 0.5)
                actual = stats.get('win_rate', 0.5)
                
                # Calibration ratio: >1 means overconfident, <1 means underconfident
                calibration[market] = predicted / actual if actual > 0 else 1.0
            else:
                calibration[market] = 1.0  # Insufficient data
        
        return calibration
    
    def _generate_improvement_suggestions(
        self,
        overall_stats: Dict[str, Any],
        market_breakdown: Dict[str, Dict[str, float]],
        calibration: Dict[str, float]
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        
        suggestions = []
        
        # Win rate suggestions
        win_rate = overall_stats.get('win_rate', 0)
        if win_rate < 0.50:
            suggestions.append("⚠️ Win rate below 50% - review selection criteria")
        elif win_rate > 0.65:
            suggestions.append("✅ Strong win rate - consider increasing stakes slightly")
        
        # Market-specific suggestions
        for market, stats in market_breakdown.items():
            if stats['win_rate'] < 0.45:
                suggestions.append(f"❌ {market.upper()}: Poor performance - consider avoiding")
            elif stats['win_rate'] > 0.60:
                suggestions.append(f"✅ {market.upper()}: Strong market - prioritize selections")
        
        # Calibration suggestions
        for market, cal_ratio in calibration.items():
            if cal_ratio > 1.15:
                suggestions.append(f"📊 {market.upper()}: Overconfident - reduce probability estimates by {(cal_ratio-1)*100:.0f}%")
            elif cal_ratio < 0.85:
                suggestions.append(f"📊 {market.upper()}: Underconfident - increase probability estimates by {(1-cal_ratio)*100:.0f}%")
        
        # ROI suggestions
        roi = overall_stats.get('roi', 0)
        if roi < -0.10:
            suggestions.append("🔴 Negative ROI - review value detection threshold")
        elif roi > 0.10:
            suggestions.append("🟢 Positive ROI - strategy is working")
        
        # Sample size suggestion
        total_bets = overall_stats.get('total_bets', 0)
        if total_bets < 50:
            suggestions.append("📈 Insufficient data - need more predictions for reliable analysis")
        
        return suggestions[:5]  # Max 5 suggestions
    
    # ==================== CONFIDENCE ADJUSTMENT ====================
    
    def get_calibrated_probability(
        self,
        market: str,
        raw_probability: float
    ) -> float:
        """Apply calibration adjustment to raw probability."""
        
        adjustment = self.calibration_adjustments.get(market, 1.0)
        
        # Apply adjustment
        calibrated = raw_probability / adjustment
        
        # Clamp to valid range
        return max(0.01, min(0.99, calibrated))
    
    def update_calibration_from_results(self, min_samples: int = 30):
        """Update calibration adjustments based on recent results."""
        
        for market in ['btts', 'over_1_5', 'over_2_5']:
            stats = self.cache.get_performance_stats(market=market, days=60)
            
            if stats['total_bets'] >= min_samples:
                predicted = stats.get('avg_predicted_prob', 0.5)
                actual = stats.get('win_rate', 0.5)
                
                if actual > 0:
                    new_adjustment = predicted / actual
                    
                    # Smooth update (don't change too fast)
                    old_adjustment = self.calibration_adjustments[market]
                    self.calibration_adjustments[market] = 0.7 * old_adjustment + 0.3 * new_adjustment
                    
                    print(f"📊 Updated {market} calibration: {old_adjustment:.2f} → {self.calibration_adjustments[market]:.2f}")
    
    # ==================== WEEKLY SUMMARY ====================
    
    def generate_weekly_summary(self) -> str:
        """Generate weekly performance summary for Telegram."""
        
        report = self.generate_performance_report(days=7)
        
        lines = [
            "═══════════════════════════════",
            "   📊 WEEKLY PERFORMANCE 📊",
            "═══════════════════════════════",
            "",
            f"📅 {report.period_start[:10]} → {report.period_end[:10]}",
            "",
            "─────────────────────────────────",
            "",
            f"📈 Total Predictions: {report.total_predictions}",
            f"✅ Correct: {report.correct_predictions}",
            f"🎯 Win Rate: {report.win_rate:.1%}",
            "",
            f"💰 Total Profit: €{report.total_profit:.2f}",
            f"📊 ROI: {report.roi:.1%}",
            "",
            "─────────────────────────────────",
            "",
            "📊 MARKET BREAKDOWN",
            "",
        ]
        
        for market, stats in report.market_breakdown.items():
            emoji = "✅" if stats['roi'] > 0 else "❌"
            lines.append(f"  {emoji} {market.upper():12} | {stats['win_rate']:.0%} | {stats['roi']:+.0%} ROI")
        
        lines.extend([
            "",
            "─────────────────────────────────",
            "",
            "💡 INSIGHTS",
            "",
        ])
        
        for suggestion in report.improvement_suggestions[:3]:
            lines.append(f"  {suggestion}")
        
        lines.extend([
            "",
            "═══════════════════════════════",
            "  🤖 DeepSeek 7B Analysis",
            "═══════════════════════════════",
        ])
        
        return "\n".join(lines)
    
    # ==================== CURIOSITY LEARNING ====================
    
    def learn_from_curiosity_finding(
        self,
        finding_type: str,
        description: str,
        was_successful: bool
    ):
        """Update curiosity findings based on outcomes."""
        
        # This would update the curiosity_findings table
        # Increasing confidence for successful findings
        # Decreasing for unsuccessful ones
        
        pass  # Implementation would query and update database

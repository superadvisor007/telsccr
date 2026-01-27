#!/usr/bin/env python3
"""
🔄 SELF-CORRECTION ENGINE - Learning from Mistakes
===================================================

Analyzes WHY predictions failed and automatically adjusts:
1. Overconfidence Detection
2. Market-specific Bias
3. League-specific Issues
4. Team Blind Spots
5. Time-based Patterns

Based on:
- Academic: Merkle & Steyvers calibration studies
- Practical: Professional bettor error logs
- Research: datarootsio/your-best-bet feedback loops

"A system that doesn't learn from failure will repeat failure."
"""

import json
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ErrorPattern:
    """A detected error pattern"""
    pattern_type: str  # overconfidence, market_bias, league_issue, etc.
    description: str
    affected_bets: int
    total_loss: float
    severity: str  # HIGH, MEDIUM, LOW
    correction: str  # How to fix
    confidence: float  # How sure we are this is a real pattern


@dataclass
class CalibrationReport:
    """Probability calibration analysis"""
    bucket: str  # e.g., "70-80%"
    predicted_prob: float
    actual_win_rate: float
    sample_size: int
    calibration_error: float  # predicted - actual
    bias_direction: str  # "overconfident" or "underconfident"


class SelfCorrectionEngine:
    """
    🔄 Self-Correction Engine
    
    Continuously analyzes prediction errors to:
    1. Detect systematic biases
    2. Adjust probability estimates
    3. Improve market selection
    4. Fix blind spots
    """
    
    def __init__(self):
        self.data_dir = Path("data/self_correction")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.errors_file = self.data_dir / "error_analysis.json"
        self.adjustments_file = self.data_dir / "probability_adjustments.json"
        self.corrections_log = self.data_dir / "corrections_log.json"
        
        self.error_patterns = []
        self.probability_adjustments = self._load_adjustments()
        
        # Thresholds
        self.overconfidence_threshold = 0.10  # 10% gap between predicted and actual
        self.min_sample_for_pattern = 20
        self.significant_bias_threshold = 0.05
    
    def analyze_errors(self, predictions: List[Dict]) -> Dict:
        """
        Comprehensive error analysis.
        
        Input: List of predictions with outcomes
            {
                'prediction_id': str,
                'market': str,
                'league': str,
                'predicted_probability': float,
                'odds': float,
                'was_correct': bool,
                'home_team': str,
                'away_team': str,
                'timestamp': str,
            }
        """
        
        if len(predictions) < self.min_sample_for_pattern:
            return {"status": "insufficient_data", "count": len(predictions)}
        
        verified = [p for p in predictions if p.get('was_correct') is not None]
        
        analysis = {
            "total_analyzed": len(verified),
            "overall_win_rate": self._calculate_win_rate(verified),
            "calibration": self._analyze_calibration(verified),
            "market_analysis": self._analyze_by_market(verified),
            "league_analysis": self._analyze_by_league(verified),
            "team_analysis": self._analyze_by_team(verified),
            "time_analysis": self._analyze_by_time(verified),
            "error_patterns": self._detect_error_patterns(verified),
            "recommendations": [],
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Update probability adjustments
        self._update_adjustments(analysis)
        
        return analysis
    
    def _analyze_calibration(self, predictions: List[Dict]) -> List[CalibrationReport]:
        """
        Analyze probability calibration.
        
        A well-calibrated predictor should have:
        - 70% predicted probability → 70% actual win rate
        """
        
        buckets = [
            (0.40, 0.50, "40-50%"),
            (0.50, 0.60, "50-60%"),
            (0.60, 0.70, "60-70%"),
            (0.70, 0.80, "70-80%"),
            (0.80, 0.90, "80-90%"),
        ]
        
        reports = []
        
        for low, high, label in buckets:
            bucket_preds = [
                p for p in predictions 
                if low <= p.get('predicted_probability', 0) < high
            ]
            
            if len(bucket_preds) < 10:
                continue
            
            wins = sum(1 for p in bucket_preds if p.get('was_correct'))
            actual_rate = wins / len(bucket_preds)
            predicted_avg = np.mean([p.get('predicted_probability', 0) for p in bucket_preds])
            
            cal_error = predicted_avg - actual_rate
            
            bias = "overconfident" if cal_error > 0.02 else "underconfident" if cal_error < -0.02 else "calibrated"
            
            reports.append(CalibrationReport(
                bucket=label,
                predicted_prob=predicted_avg,
                actual_win_rate=actual_rate,
                sample_size=len(bucket_preds),
                calibration_error=cal_error,
                bias_direction=bias
            ))
        
        return reports
    
    def _analyze_by_market(self, predictions: List[Dict]) -> Dict:
        """Analyze performance by market type"""
        
        markets = defaultdict(lambda: {'wins': 0, 'total': 0, 'profit': 0})
        
        for p in predictions:
            market = p.get('market', 'unknown')
            markets[market]['total'] += 1
            
            if p.get('was_correct'):
                markets[market]['wins'] += 1
                markets[market]['profit'] += (p.get('odds', 2) - 1)
            else:
                markets[market]['profit'] -= 1
        
        analysis = {}
        for market, data in markets.items():
            if data['total'] >= 10:
                win_rate = data['wins'] / data['total']
                roi = data['profit'] / data['total']
                
                analysis[market] = {
                    'win_rate': round(win_rate, 3),
                    'roi': round(roi, 3),
                    'sample': data['total'],
                    'status': 'PROFITABLE' if roi > 0 else 'LOSING',
                    'recommendation': self._market_recommendation(market, win_rate, roi)
                }
        
        return analysis
    
    def _analyze_by_league(self, predictions: List[Dict]) -> Dict:
        """Analyze performance by league"""
        
        leagues = defaultdict(lambda: {'wins': 0, 'total': 0, 'profit': 0})
        
        for p in predictions:
            league = p.get('league', 'unknown')
            leagues[league]['total'] += 1
            
            if p.get('was_correct'):
                leagues[league]['wins'] += 1
                leagues[league]['profit'] += (p.get('odds', 2) - 1)
            else:
                leagues[league]['profit'] -= 1
        
        analysis = {}
        for league, data in leagues.items():
            if data['total'] >= 10:
                win_rate = data['wins'] / data['total']
                roi = data['profit'] / data['total']
                
                analysis[league] = {
                    'win_rate': round(win_rate, 3),
                    'roi': round(roi, 3),
                    'sample': data['total'],
                    'status': 'PROFITABLE' if roi > 0 else 'LOSING',
                }
        
        return analysis
    
    def _analyze_by_team(self, predictions: List[Dict]) -> Dict:
        """Find team blind spots"""
        
        teams = defaultdict(lambda: {'wins': 0, 'total': 0, 'as_home': 0, 'as_away': 0})
        
        for p in predictions:
            home = p.get('home_team', '')
            away = p.get('away_team', '')
            
            for team, role in [(home, 'home'), (away, 'away')]:
                if team:
                    teams[team]['total'] += 1
                    teams[team][f'as_{role}'] += 1
                    if p.get('was_correct'):
                        teams[team]['wins'] += 1
        
        # Find problematic teams
        blind_spots = {}
        for team, data in teams.items():
            if data['total'] >= 5:
                win_rate = data['wins'] / data['total']
                
                if win_rate < 0.35:  # Very poor performance
                    blind_spots[team] = {
                        'win_rate': round(win_rate, 3),
                        'sample': data['total'],
                        'issue': 'CONSISTENT_MISS',
                        'recommendation': f'Review analysis approach for {team}'
                    }
                elif win_rate > 0.75:  # Very good - might be overfit
                    blind_spots[team] = {
                        'win_rate': round(win_rate, 3),
                        'sample': data['total'],
                        'issue': 'POSSIBLE_OVERFIT',
                        'recommendation': f'Monitor {team} for regression'
                    }
        
        return blind_spots
    
    def _analyze_by_time(self, predictions: List[Dict]) -> Dict:
        """Analyze performance by time patterns"""
        
        # Day of week analysis
        days = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for p in predictions:
            timestamp = p.get('timestamp', p.get('match_date', ''))
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    day = dt.strftime('%A')
                    days[day]['total'] += 1
                    if p.get('was_correct'):
                        days[day]['wins'] += 1
                except:
                    pass
        
        day_analysis = {}
        for day, data in days.items():
            if data['total'] >= 10:
                win_rate = data['wins'] / data['total']
                day_analysis[day] = {
                    'win_rate': round(win_rate, 3),
                    'sample': data['total'],
                    'recommendation': 'REDUCE_BETS' if win_rate < 0.45 else 'NORMAL'
                }
        
        return {'by_day': day_analysis}
    
    def _detect_error_patterns(self, predictions: List[Dict]) -> List[ErrorPattern]:
        """Detect systematic error patterns"""
        
        patterns = []
        errors = [p for p in predictions if not p.get('was_correct')]
        
        if len(errors) < 10:
            return patterns
        
        # Pattern 1: Overconfidence in high probability bets
        high_conf_errors = [e for e in errors if e.get('predicted_probability', 0) > 0.75]
        if len(high_conf_errors) > len(errors) * 0.3:
            patterns.append(ErrorPattern(
                pattern_type="overconfidence",
                description="High confidence predictions (>75%) failing at elevated rate",
                affected_bets=len(high_conf_errors),
                total_loss=sum(1 for _ in high_conf_errors),  # Simplified
                severity="HIGH",
                correction="Reduce probability estimates by 5-10% for high confidence bets",
                confidence=0.8
            ))
        
        # Pattern 2: Specific market underperformance
        market_errors = defaultdict(int)
        market_totals = defaultdict(int)
        for p in predictions:
            market = p.get('market', 'unknown')
            market_totals[market] += 1
            if not p.get('was_correct'):
                market_errors[market] += 1
        
        for market, error_count in market_errors.items():
            total = market_totals[market]
            if total >= 15:
                error_rate = error_count / total
                if error_rate > 0.55:  # >55% error rate
                    patterns.append(ErrorPattern(
                        pattern_type="market_bias",
                        description=f"Market '{market}' has {error_rate:.1%} error rate",
                        affected_bets=error_count,
                        total_loss=error_count,
                        severity="HIGH" if error_rate > 0.6 else "MEDIUM",
                        correction=f"Increase probability threshold for {market} by 5%",
                        confidence=0.7
                    ))
        
        # Pattern 3: Favorite-Longshot Bias
        longshot_errors = [e for e in errors if e.get('odds', 0) > 3.0]
        longshot_total = [p for p in predictions if p.get('odds', 0) > 3.0]
        
        if len(longshot_total) >= 10:
            longshot_error_rate = len(longshot_errors) / len(longshot_total)
            if longshot_error_rate > 0.7:
                patterns.append(ErrorPattern(
                    pattern_type="longshot_bias",
                    description=f"Longshots (odds >3.0) losing at {longshot_error_rate:.1%} rate",
                    affected_bets=len(longshot_errors),
                    total_loss=len(longshot_errors),
                    severity="MEDIUM",
                    correction="Avoid bets with odds > 3.0 or reduce stake significantly",
                    confidence=0.75
                ))
        
        # Pattern 4: Time-based issues (e.g., poor Monday performance)
        # Add more patterns as data accumulates
        
        self.error_patterns = patterns
        return patterns
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # From calibration
        calibration = analysis.get('calibration', [])
        for cal in calibration:
            if cal.calibration_error > 0.10:
                recommendations.append(
                    f"🔴 OVERCONFIDENT in {cal.bucket} range: "
                    f"Predicted {cal.predicted_prob:.1%}, actual {cal.actual_win_rate:.1%}. "
                    f"REDUCE predictions by {cal.calibration_error:.1%}"
                )
            elif cal.calibration_error < -0.10:
                recommendations.append(
                    f"🟢 UNDERCONFIDENT in {cal.bucket} range: "
                    f"Predicted {cal.predicted_prob:.1%}, actual {cal.actual_win_rate:.1%}. "
                    f"You can be more confident here."
                )
        
        # From market analysis
        markets = analysis.get('market_analysis', {})
        for market, data in markets.items():
            if data.get('roi', 0) < -0.10:
                recommendations.append(
                    f"⚠️ LOSING MARKET: {market} has {data['roi']:.1%} ROI. "
                    f"Consider removing from strategy or adjusting thresholds."
                )
            elif data.get('roi', 0) > 0.10:
                recommendations.append(
                    f"✅ WINNING MARKET: {market} has {data['roi']:.1%} ROI. "
                    f"Consider increasing allocation."
                )
        
        # From error patterns
        for pattern in analysis.get('error_patterns', []):
            if pattern.severity == "HIGH":
                recommendations.append(
                    f"🚨 {pattern.pattern_type.upper()}: {pattern.correction}"
                )
        
        # From league analysis
        leagues = analysis.get('league_analysis', {})
        for league, data in leagues.items():
            if data.get('roi', 0) < -0.15:
                recommendations.append(
                    f"📉 LEAGUE ISSUE: {league} losing {-data['roi']:.1%}. "
                    f"Review data quality or consider excluding."
                )
        
        return recommendations
    
    def _update_adjustments(self, analysis: Dict):
        """Update probability adjustments based on analysis"""
        
        adjustments = self.probability_adjustments
        
        # Calibration-based adjustments
        for cal in analysis.get('calibration', []):
            bucket_key = cal.bucket.replace('%', '').replace('-', '_')
            
            if abs(cal.calibration_error) > 0.05:
                adjustments[f'calibration_{bucket_key}'] = -cal.calibration_error
        
        # Market-based adjustments
        for market, data in analysis.get('market_analysis', {}).items():
            if data.get('sample', 0) >= 20:
                if data.get('roi', 0) < -0.10:
                    adjustments[f'market_{market}'] = 0.05  # Require 5% higher prob
                elif data.get('roi', 0) > 0.10:
                    adjustments[f'market_{market}'] = -0.03  # Can accept 3% lower
        
        # Save adjustments
        self._save_adjustments()
    
    def get_adjusted_probability(self, raw_probability: float, market: str, 
                                  probability_bucket: str = None) -> float:
        """
        Apply learned adjustments to a probability estimate.
        
        This is the KEY OUTPUT - use this to correct predictions!
        """
        
        adjusted = raw_probability
        
        # Apply calibration adjustment
        if probability_bucket:
            bucket_key = probability_bucket.replace('%', '').replace('-', '_')
            cal_adj = self.probability_adjustments.get(f'calibration_{bucket_key}', 0)
            adjusted += cal_adj
        
        # Apply market adjustment
        market_adj = self.probability_adjustments.get(f'market_{market}', 0)
        adjusted += market_adj
        
        # Clamp to valid range
        adjusted = max(0.30, min(0.95, adjusted))
        
        return adjusted
    
    def _market_recommendation(self, market: str, win_rate: float, roi: float) -> str:
        """Generate market-specific recommendation"""
        
        if roi > 0.10:
            return "INCREASE allocation - consistent profits"
        elif roi < -0.15:
            return "REDUCE or EXCLUDE - consistent losses"
        elif win_rate > 0.55:
            return "MAINTAIN - good win rate, optimize stake sizing"
        else:
            return "REVIEW - marginal performance"
    
    def _calculate_win_rate(self, predictions: List[Dict]) -> float:
        """Calculate overall win rate"""
        wins = sum(1 for p in predictions if p.get('was_correct'))
        return wins / len(predictions) if predictions else 0
    
    def _load_adjustments(self) -> Dict:
        """Load probability adjustments"""
        if self.adjustments_file.exists():
            try:
                with open(self.adjustments_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_adjustments(self):
        """Save probability adjustments"""
        with open(self.adjustments_file, 'w') as f:
            json.dump(self.probability_adjustments, f, indent=2)
    
    def generate_correction_report(self, predictions: List[Dict]) -> str:
        """Generate human-readable correction report"""
        
        analysis = self.analyze_errors(predictions)
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("🔄 SELF-CORRECTION REPORT")
        report.append("=" * 70)
        report.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"   Predictions Analyzed: {analysis.get('total_analyzed', 0)}")
        report.append(f"   Overall Win Rate: {analysis.get('overall_win_rate', 0):.1%}")
        report.append("")
        
        # Calibration
        report.append("📊 CALIBRATION ANALYSIS:")
        report.append("-" * 50)
        for cal in analysis.get('calibration', []):
            emoji = "🔴" if cal.bias_direction == "overconfident" else "🟢" if cal.bias_direction == "underconfident" else "⚪"
            report.append(f"   {emoji} {cal.bucket}: Predicted {cal.predicted_prob:.1%} → Actual {cal.actual_win_rate:.1%}")
            report.append(f"      Error: {cal.calibration_error:+.1%} ({cal.bias_direction}) [n={cal.sample_size}]")
        report.append("")
        
        # Error Patterns
        if analysis.get('error_patterns'):
            report.append("🚨 ERROR PATTERNS DETECTED:")
            report.append("-" * 50)
            for pattern in analysis['error_patterns']:
                severity_emoji = "🔴" if pattern.severity == "HIGH" else "🟡" if pattern.severity == "MEDIUM" else "🟢"
                report.append(f"   {severity_emoji} {pattern.pattern_type.upper()}")
                report.append(f"      {pattern.description}")
                report.append(f"      Fix: {pattern.correction}")
            report.append("")
        
        # Market Performance
        report.append("📈 MARKET PERFORMANCE:")
        report.append("-" * 50)
        for market, data in sorted(analysis.get('market_analysis', {}).items()):
            status = "✅" if data.get('status') == 'PROFITABLE' else "❌"
            report.append(f"   {status} {market:15} WR: {data['win_rate']:.1%}  ROI: {data['roi']:+.1%}  [n={data['sample']}]")
        report.append("")
        
        # Recommendations
        if analysis.get('recommendations'):
            report.append("💡 RECOMMENDATIONS:")
            report.append("-" * 50)
            for rec in analysis['recommendations'][:10]:
                report.append(f"   • {rec}")
            report.append("")
        
        # Active Adjustments
        if self.probability_adjustments:
            report.append("🎯 ACTIVE PROBABILITY ADJUSTMENTS:")
            report.append("-" * 50)
            for key, adj in self.probability_adjustments.items():
                direction = "↑" if adj > 0 else "↓"
                report.append(f"   {key}: {direction} {abs(adj):.1%}")
        
        return "\n".join(report)


# Test
if __name__ == "__main__":
    engine = SelfCorrectionEngine()
    
    # Simulate some predictions
    test_predictions = [
        {"market": "over_2_5", "league": "Bundesliga", "predicted_probability": 0.72, "odds": 1.80, "was_correct": True, "home_team": "Bayern", "away_team": "Dortmund"},
        {"market": "over_2_5", "league": "Bundesliga", "predicted_probability": 0.75, "odds": 1.75, "was_correct": False, "home_team": "Leipzig", "away_team": "Leverkusen"},
        {"market": "btts_yes", "league": "Premier League", "predicted_probability": 0.68, "odds": 1.85, "was_correct": True, "home_team": "Liverpool", "away_team": "Chelsea"},
        {"market": "over_2_5", "league": "Bundesliga", "predicted_probability": 0.78, "odds": 1.65, "was_correct": False, "home_team": "Stuttgart", "away_team": "Frankfurt"},
        {"market": "btts_yes", "league": "La Liga", "predicted_probability": 0.65, "odds": 1.90, "was_correct": True, "home_team": "Barcelona", "away_team": "Madrid"},
        # Add more for realistic testing...
    ] * 5  # Multiply for sample size
    
    print(engine.generate_correction_report(test_predictions))

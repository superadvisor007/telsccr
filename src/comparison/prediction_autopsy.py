"""
🔍 Prediction Autopsy - Deep Error Analysis
============================================

Post-mortem analysis of prediction failures:
- Error categorization (overconfidence, defensive surprises, etc.)
- Pattern detection in failures
- Feature importance for failed predictions
- Confidence vs accuracy correlation
- Actionable insights generation

Pattern from FiveThirtyEight error analysis methodology.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of prediction errors"""
    OVERCONFIDENT_FAIL = "High confidence prediction failed"
    DEFENSIVE_SURPRISE = "Expected goals but game was defensive"
    GOAL_FEST_SURPRISE = "Expected defensive but game was high-scoring"
    BLOWOUT_MATCH = "One team dominated unexpectedly"
    BTTS_SHUTOUT = "Expected both teams to score but one was shut out"
    UNDERDOG_WIN = "Underdog won against expectations"
    DRAW_SURPRISE = "Unexpected draw"
    LATE_DRAMA = "Result changed in final minutes"
    MARGINAL_MISS = "Prediction just missed the threshold"
    GENERAL_ERROR = "No specific pattern identified"


@dataclass
class FailureAnalysis:
    """Analysis of a single failed prediction"""
    prediction_id: str
    match_date: str
    home_team: str
    away_team: str
    market: str
    predicted_probability: float
    actual_score: str
    error_category: ErrorCategory
    confidence_bucket: str
    odds_bucket: str
    
    # Context
    league: str
    was_high_confidence: bool
    was_low_odds: bool
    goal_difference: int
    total_goals: int
    
    # Insights
    insights: List[str] = field(default_factory=list)


@dataclass
class AutopsyReport:
    """Complete autopsy report for a set of predictions"""
    timestamp: str
    total_predictions: int
    total_failures: int
    failure_rate: float
    
    # Error breakdown
    error_categories: Dict[str, int]
    error_percentages: Dict[str, float]
    
    # Pattern analysis
    worst_confidence_bucket: str
    worst_market: str
    worst_league: str
    
    # Overconfidence analysis
    overconfidence_analysis: Dict
    
    # Specific failure patterns
    failure_patterns: Dict
    
    # Top failures (worst predictions)
    top_failures: List[FailureAnalysis]
    
    # Recommendations
    recommendations: List[str]
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        report = f"""
# 🔍 Prediction Autopsy Report

**Generated:** {self.timestamp}
**Total Predictions:** {self.total_predictions}
**Failures:** {self.total_failures} ({self.failure_rate:.1%})

## 📊 Error Category Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
"""
        for cat, count in sorted(self.error_categories.items(), key=lambda x: -x[1]):
            pct = self.error_percentages.get(cat, 0)
            report += f"| {cat} | {count} | {pct:.1%} |\n"
        
        report += f"""
## 🎯 Overconfidence Analysis

- **High-confidence predictions:** {self.overconfidence_analysis.get('high_conf_count', 0)}
- **High-confidence failures:** {self.overconfidence_analysis.get('high_conf_failures', 0)}
- **High-confidence accuracy:** {self.overconfidence_analysis.get('high_conf_accuracy', 0):.1%}
- **Overconfidence gap:** {self.overconfidence_analysis.get('overconfidence_gap', 0):.1%}
- **Assessment:** {self.overconfidence_analysis.get('assessment', 'N/A')}

## ⚠️ Worst Performing Areas

- **Worst Confidence Bucket:** {self.worst_confidence_bucket}
- **Worst Market:** {self.worst_market}
- **Worst League:** {self.worst_league}

## 🔬 Failure Patterns

| Pattern | Occurrences |
|---------|-------------|
"""
        for pattern, count in self.failure_patterns.items():
            report += f"| {pattern} | {count} |\n"
        
        report += "\n## 💀 Top Failures (Worst Predictions)\n\n"
        
        for i, f in enumerate(self.top_failures[:5], 1):
            report += f"""
### {i}. {f.home_team} vs {f.away_team}
- **Date:** {f.match_date}
- **Market:** {f.market}
- **Predicted:** {f.predicted_probability:.1%}
- **Actual:** {f.actual_score}
- **Error:** {f.error_category.value}
- **Insights:** {', '.join(f.insights) if f.insights else 'None'}
"""
        
        report += "\n## 💡 Recommendations\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report


class PredictionAutopsy:
    """
    Deep analysis of prediction failures to understand why the model was wrong.
    
    Inspired by FiveThirtyEight's forecast methodology and professional
    betting operations that conduct post-mortem analysis.
    """
    
    def __init__(self):
        self.failures: List[FailureAnalysis] = []
        self.successes: List[Dict] = []
    
    def analyze(
        self,
        predictions: List[Dict],
        results: List[Dict]
    ) -> AutopsyReport:
        """
        Conduct full autopsy analysis.
        
        Args:
            predictions: List of prediction dicts with keys:
                - prediction_id, match_date, home_team, away_team
                - market, predicted_probability, odds
            results: List of result dicts with keys:
                - home_team, away_team, date
                - home_goals, away_goals
        
        Returns:
            AutopsyReport with detailed analysis
        """
        logger.info(f"Conducting autopsy on {len(predictions)} predictions...")
        
        self.failures = []
        self.successes = []
        
        # Match predictions to results
        for pred in predictions:
            result = self._find_matching_result(pred, results)
            if result is None:
                continue
            
            was_correct = self._evaluate_prediction(pred, result)
            
            if was_correct:
                self.successes.append({'prediction': pred, 'result': result})
            else:
                failure = self._analyze_failure(pred, result)
                self.failures.append(failure)
        
        return self._generate_report()
    
    def _find_matching_result(self, prediction: Dict, results: List[Dict]) -> Optional[Dict]:
        """Find matching result for a prediction"""
        pred_date = prediction.get('match_date', prediction.get('date', ''))
        pred_home = prediction.get('home_team', '').lower()
        pred_away = prediction.get('away_team', '').lower()
        
        for result in results:
            result_date = result.get('date', result.get('match_date', ''))
            result_home = result.get('home_team', '').lower()
            result_away = result.get('away_team', '').lower()
            
            # Fuzzy match
            if (pred_date == result_date and
                (pred_home in result_home or result_home in pred_home) and
                (pred_away in result_away or result_away in pred_away)):
                return result
        
        return None
    
    def _evaluate_prediction(self, prediction: Dict, result: Dict) -> bool:
        """Check if prediction was correct"""
        market = prediction.get('market', '').lower()
        prob = prediction.get('predicted_probability', 0.5)
        
        home_goals = result.get('home_goals', 0)
        away_goals = result.get('away_goals', 0)
        total_goals = home_goals + away_goals
        
        # Predicted YES if probability > 0.5
        predicted_yes = prob > 0.5
        
        # Determine actual outcome
        actual_yes = False
        if 'over_1_5' in market or 'over_15' in market:
            actual_yes = total_goals > 1.5
        elif 'over_2_5' in market or 'over_25' in market:
            actual_yes = total_goals > 2.5
        elif 'over_3_5' in market or 'over_35' in market:
            actual_yes = total_goals > 3.5
        elif 'btts' in market or 'both_teams' in market:
            actual_yes = home_goals > 0 and away_goals > 0
        elif 'home_win' in market:
            actual_yes = home_goals > away_goals
        elif 'away_win' in market:
            actual_yes = away_goals > home_goals
        elif 'draw' in market:
            actual_yes = home_goals == away_goals
        
        return predicted_yes == actual_yes
    
    def _analyze_failure(self, prediction: Dict, result: Dict) -> FailureAnalysis:
        """Analyze a single failure in detail"""
        home_goals = result.get('home_goals', 0)
        away_goals = result.get('away_goals', 0)
        total_goals = home_goals + away_goals
        goal_diff = abs(home_goals - away_goals)
        
        market = prediction.get('market', '')
        prob = prediction.get('predicted_probability', 0.5)
        odds = prediction.get('odds', 1.5)
        
        # Categorize error
        error_category = self._categorize_error(prediction, result)
        
        # Buckets
        confidence_bucket = self._get_confidence_bucket(prob)
        odds_bucket = self._get_odds_bucket(odds)
        
        # Flags
        was_high_confidence = prob >= 0.70
        was_low_odds = odds <= 1.40
        
        # Generate insights
        insights = self._generate_insights(prediction, result, error_category)
        
        return FailureAnalysis(
            prediction_id=prediction.get('prediction_id', 'unknown'),
            match_date=prediction.get('match_date', ''),
            home_team=prediction.get('home_team', ''),
            away_team=prediction.get('away_team', ''),
            market=market,
            predicted_probability=prob,
            actual_score=f"{home_goals}-{away_goals}",
            error_category=error_category,
            confidence_bucket=confidence_bucket,
            odds_bucket=odds_bucket,
            league=prediction.get('league', ''),
            was_high_confidence=was_high_confidence,
            was_low_odds=was_low_odds,
            goal_difference=goal_diff,
            total_goals=total_goals,
            insights=insights
        )
    
    def _categorize_error(self, prediction: Dict, result: Dict) -> ErrorCategory:
        """Categorize the type of error"""
        market = prediction.get('market', '').lower()
        prob = prediction.get('predicted_probability', 0.5)
        
        home_goals = result.get('home_goals', 0)
        away_goals = result.get('away_goals', 0)
        total_goals = home_goals + away_goals
        goal_diff = abs(home_goals - away_goals)
        
        # Overconfidence
        if prob >= 0.75:
            return ErrorCategory.OVERCONFIDENT_FAIL
        
        # Market-specific errors
        if 'over' in market:
            if total_goals <= 1:
                return ErrorCategory.DEFENSIVE_SURPRISE
            elif prob > 0.5 and total_goals == 2:
                return ErrorCategory.MARGINAL_MISS
        
        if 'under' in market and total_goals >= 4:
            return ErrorCategory.GOAL_FEST_SURPRISE
        
        if 'btts' in market:
            if home_goals == 0 or away_goals == 0:
                return ErrorCategory.BTTS_SHUTOUT
        
        # Blowout
        if goal_diff >= 3:
            return ErrorCategory.BLOWOUT_MATCH
        
        # Draw surprise
        if home_goals == away_goals and 'draw' not in market:
            return ErrorCategory.DRAW_SURPRISE
        
        return ErrorCategory.GENERAL_ERROR
    
    def _generate_insights(
        self,
        prediction: Dict,
        result: Dict,
        error_category: ErrorCategory
    ) -> List[str]:
        """Generate specific insights for this failure"""
        insights = []
        
        home_goals = result.get('home_goals', 0)
        away_goals = result.get('away_goals', 0)
        total_goals = home_goals + away_goals
        
        prob = prediction.get('predicted_probability', 0.5)
        market = prediction.get('market', '')
        
        if error_category == ErrorCategory.OVERCONFIDENT_FAIL:
            insights.append(f"Model was {prob:.0%} confident but wrong")
            insights.append("Consider lowering confidence in similar situations")
        
        if error_category == ErrorCategory.DEFENSIVE_SURPRISE:
            insights.append(f"Only {total_goals} goals in match")
            insights.append("Check pre-match defensive form indicators")
        
        if error_category == ErrorCategory.GOAL_FEST_SURPRISE:
            insights.append(f"Match had {total_goals} goals - very high")
            insights.append("May have underestimated attacking potential")
        
        if error_category == ErrorCategory.BTTS_SHUTOUT:
            shutout_team = prediction.get('away_team') if home_goals == 0 else prediction.get('home_team')
            insights.append(f"{shutout_team} failed to score")
            insights.append("Review team's scoring form before similar bets")
        
        if error_category == ErrorCategory.BLOWOUT_MATCH:
            winner = prediction.get('home_team') if home_goals > away_goals else prediction.get('away_team')
            insights.append(f"{winner} dominated with {max(home_goals, away_goals)} goals")
        
        return insights
    
    @staticmethod
    def _get_confidence_bucket(prob: float) -> str:
        if prob >= 0.80: return '80-100%'
        elif prob >= 0.70: return '70-80%'
        elif prob >= 0.60: return '60-70%'
        elif prob >= 0.50: return '50-60%'
        else: return '<50%'
    
    @staticmethod
    def _get_odds_bucket(odds: float) -> str:
        if odds <= 1.30: return '1.00-1.30'
        elif odds <= 1.50: return '1.30-1.50'
        elif odds <= 1.80: return '1.50-1.80'
        elif odds <= 2.20: return '1.80-2.20'
        else: return '>2.20'
    
    def _generate_report(self) -> AutopsyReport:
        """Generate complete autopsy report"""
        total = len(self.failures) + len(self.successes)
        failure_rate = len(self.failures) / total if total > 0 else 0
        
        # Error category breakdown
        error_counts = Counter(f.error_category.name for f in self.failures)
        error_percentages = {k: v / len(self.failures) for k, v in error_counts.items()} if self.failures else {}
        
        # Find worst areas
        worst_conf_bucket = self._find_worst_bucket([f.confidence_bucket for f in self.failures])
        worst_market = self._find_worst_bucket([f.market for f in self.failures])
        worst_league = self._find_worst_bucket([f.league for f in self.failures if f.league])
        
        # Overconfidence analysis
        overconf = self._analyze_overconfidence()
        
        # Failure patterns
        patterns = self._detect_patterns()
        
        # Sort failures by confidence (worst first)
        top_failures = sorted(self.failures, key=lambda f: -f.predicted_probability)[:10]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            error_counts, overconf, worst_market, worst_league
        )
        
        return AutopsyReport(
            timestamp=datetime.now().isoformat(),
            total_predictions=total,
            total_failures=len(self.failures),
            failure_rate=failure_rate,
            error_categories=dict(error_counts),
            error_percentages=error_percentages,
            worst_confidence_bucket=worst_conf_bucket,
            worst_market=worst_market,
            worst_league=worst_league,
            overconfidence_analysis=overconf,
            failure_patterns=patterns,
            top_failures=top_failures,
            recommendations=recommendations
        )
    
    def _find_worst_bucket(self, values: List[str]) -> str:
        """Find the most common value (worst performer)"""
        if not values:
            return "N/A"
        counter = Counter(values)
        return counter.most_common(1)[0][0]
    
    def _analyze_overconfidence(self) -> Dict:
        """Analyze overconfidence in predictions"""
        all_preds = [(True, f.predicted_probability) for f in self.failures]
        all_preds += [(False, s['prediction'].get('predicted_probability', 0.5)) for s in self.successes]
        
        # High confidence (>70%)
        high_conf = [(failed, prob) for failed, prob in all_preds if prob >= 0.70]
        
        if not high_conf:
            return {'has_data': False}
        
        high_conf_count = len(high_conf)
        high_conf_failures = sum(1 for failed, _ in high_conf if failed)
        high_conf_accuracy = 1 - (high_conf_failures / high_conf_count)
        
        avg_high_conf = np.mean([prob for _, prob in high_conf])
        overconfidence_gap = avg_high_conf - high_conf_accuracy
        
        return {
            'has_data': True,
            'high_conf_count': high_conf_count,
            'high_conf_failures': high_conf_failures,
            'high_conf_accuracy': high_conf_accuracy,
            'avg_confidence': avg_high_conf,
            'overconfidence_gap': overconfidence_gap,
            'is_overconfident': overconfidence_gap > 0.05,
            'assessment': 'SEVERE' if overconfidence_gap > 0.15 else ('MODERATE' if overconfidence_gap > 0.05 else 'OK')
        }
    
    def _detect_patterns(self) -> Dict:
        """Detect common patterns in failures"""
        patterns = {
            'high_confidence_failures': sum(1 for f in self.failures if f.was_high_confidence),
            'low_odds_failures': sum(1 for f in self.failures if f.was_low_odds),
            'zero_zero_draws': sum(1 for f in self.failures if f.actual_score == '0-0'),
            'high_scoring_surprises': sum(1 for f in self.failures if f.total_goals >= 5),
            'blowouts': sum(1 for f in self.failures if f.goal_difference >= 3),
        }
        return patterns
    
    def _generate_recommendations(
        self,
        error_counts: Dict,
        overconf: Dict,
        worst_market: str,
        worst_league: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overconfidence
        if overconf.get('is_overconfident'):
            gap = overconf.get('overconfidence_gap', 0)
            recommendations.append(f"🔻 Reduce high-confidence predictions by ~{gap:.0%}")
        
        # Error type specific
        if error_counts.get('DEFENSIVE_SURPRISE', 0) > 3:
            recommendations.append("🛡️ Improve defensive form analysis for Over market")
        
        if error_counts.get('BTTS_SHUTOUT', 0) > 3:
            recommendations.append("⚽ Check team scoring consistency before BTTS bets")
        
        if error_counts.get('OVERCONFIDENT_FAIL', 0) > 5:
            recommendations.append("📉 Apply probability calibration to reduce overconfidence")
        
        # Market/League specific
        if worst_market and worst_market != "N/A":
            recommendations.append(f"⚠️ Review strategy for {worst_market} market")
        
        if worst_league and worst_league != "N/A":
            recommendations.append(f"🌍 Consider reducing exposure to {worst_league}")
        
        if not recommendations:
            recommendations.append("✅ No critical patterns detected - continue monitoring")
        
        return recommendations


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test autopsy with sample data"""
    # Sample data
    predictions = [
        {
            'prediction_id': '1',
            'match_date': '2026-01-25',
            'home_team': 'Bayern Munich',
            'away_team': 'Wolfsburg',
            'market': 'over_2_5',
            'predicted_probability': 0.78,
            'odds': 1.50,
            'league': 'Bundesliga'
        },
        {
            'prediction_id': '2',
            'match_date': '2026-01-25',
            'home_team': 'Dortmund',
            'away_team': 'Frankfurt',
            'market': 'btts',
            'predicted_probability': 0.72,
            'odds': 1.65,
            'league': 'Bundesliga'
        }
    ]
    
    results = [
        {
            'date': '2026-01-25',
            'home_team': 'Bayern Munich',
            'away_team': 'Wolfsburg',
            'home_goals': 1,
            'away_goals': 0
        },
        {
            'date': '2026-01-25',
            'home_team': 'Dortmund',
            'away_team': 'Frankfurt',
            'home_goals': 2,
            'away_goals': 0
        }
    ]
    
    autopsy = PredictionAutopsy()
    report = autopsy.analyze(predictions, results)
    
    print(report.to_markdown())


if __name__ == '__main__':
    main()

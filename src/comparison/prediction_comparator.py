"""
🎯 Prediction Comparator - Core Engine
=======================================

The heart of the self-improvement loop. Compares predictions against
actual results with professional-grade metrics:

- Brier Score decomposition (reliability, resolution, uncertainty)
- Expected Calibration Error (ECE)
- ROI / Profit-Loss tracking
- Market-specific performance
- League-specific performance
- Confidence-binned accuracy

Pattern sources: FiveThirtyEight, datarootsio/your-best-bet
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path

from .result_fetcher import MultiSourceResultFetcher, MatchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction to be verified"""
    prediction_id: str
    match_date: str
    home_team: str
    away_team: str
    league: str
    market: str  # 'over_2_5', 'btts', etc.
    predicted_probability: float
    odds: float
    stake: float = 10.0
    model_version: str = "v1"
    
    # Optional: for CLV tracking
    opening_odds: Optional[float] = None
    closing_odds: Optional[float] = None
    
    # Computed
    implied_probability: float = field(init=False)
    expected_value: float = field(init=False)
    
    def __post_init__(self):
        self.implied_probability = 1 / self.odds if self.odds > 0 else 0
        self.expected_value = (self.predicted_probability * (self.odds - 1) - 
                               (1 - self.predicted_probability))


@dataclass
class ComparisonResult:
    """Result of comparing a prediction to actual outcome"""
    prediction: Prediction
    actual_result: Optional[MatchResult]
    
    # Core outcomes
    prediction_correct: bool
    actual_outcome: bool  # Did the predicted event happen?
    
    # Financial
    profit_loss: float
    
    # Metrics
    brier_contribution: float  # (predicted_prob - outcome)²
    log_loss_contribution: float
    
    # CLV (if available)
    clv: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction.prediction_id,
            'match_date': self.prediction.match_date,
            'home_team': self.prediction.home_team,
            'away_team': self.prediction.away_team,
            'league': self.prediction.league,
            'market': self.prediction.market,
            'predicted_probability': self.prediction.predicted_probability,
            'odds': self.prediction.odds,
            'stake': self.prediction.stake,
            'actual_outcome': self.actual_outcome,
            'prediction_correct': self.prediction_correct,
            'profit_loss': self.profit_loss,
            'brier_contribution': self.brier_contribution,
            'clv': self.clv,
            'actual_score': f"{self.actual_result.home_goals}-{self.actual_result.away_goals}" if self.actual_result else None
        }


@dataclass
class BrierDecomposition:
    """Brier Score broken into components"""
    brier_score: float
    reliability: float  # Lower = better calibrated
    resolution: float   # Higher = better discrimination
    uncertainty: float  # Fixed for dataset
    skill_score: float  # 1 - BS/uncertainty (>0 = better than climatology)


@dataclass
class ComparisonReport:
    """Complete comparison report"""
    timestamp: str
    period_start: str
    period_end: str
    total_predictions: int
    verified_predictions: int
    unverified_predictions: int
    
    # Core metrics
    accuracy: float
    brier_score: float
    brier_decomposition: BrierDecomposition
    log_loss: float
    expected_calibration_error: float
    
    # Financial
    total_staked: float
    total_profit: float
    roi: float
    win_rate: float
    
    # Detailed breakdowns
    market_performance: Dict[str, Dict]
    league_performance: Dict[str, Dict]
    confidence_calibration: Dict[str, Dict]
    
    # Individual results
    results: List[ComparisonResult]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'period': f"{self.period_start} to {self.period_end}",
            'total_predictions': self.total_predictions,
            'verified': self.verified_predictions,
            'metrics': {
                'accuracy': self.accuracy,
                'brier_score': self.brier_score,
                'log_loss': self.log_loss,
                'ece': self.expected_calibration_error,
                'roi': self.roi,
                'win_rate': self.win_rate
            },
            'brier_decomposition': asdict(self.brier_decomposition),
            'market_performance': self.market_performance,
            'league_performance': self.league_performance,
            'confidence_calibration': self.confidence_calibration,
            'recommendations': self.recommendations
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        grade = self._get_grade()
        
        report = f"""
# 📊 Prediction Comparison Report

**Generated:** {self.timestamp}
**Period:** {self.period_start} to {self.period_end}
**Grade:** {grade['emoji']} {grade['grade']}

## 📈 Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Predictions | {self.total_predictions} | - |
| Verified | {self.verified_predictions} ({self.verified_predictions/self.total_predictions*100:.0f}%) | - |
| Accuracy | {self.accuracy:.1%} | {'✅' if self.accuracy > 0.55 else '⚠️'} |
| Win Rate | {self.win_rate:.1%} | {'✅' if self.win_rate > 0.50 else '⚠️'} |
| ROI | {self.roi:+.2%} | {'✅' if self.roi > 0 else '❌'} |
| Total Profit | €{self.total_profit:+.2f} | {'✅' if self.total_profit > 0 else '❌'} |

## 🎯 Calibration Metrics

| Metric | Value | Quality |
|--------|-------|---------|
| Brier Score | {self.brier_score:.4f} | {self._brier_quality()} |
| Log Loss | {self.log_loss:.4f} | - |
| ECE | {self.expected_calibration_error:.4f} | {self._ece_quality()} |

### Brier Decomposition
- **Reliability:** {self.brier_decomposition.reliability:.4f} (lower = better)
- **Resolution:** {self.brier_decomposition.resolution:.4f} (higher = better)
- **Skill Score:** {self.brier_decomposition.skill_score:.4f} (>0 = beats random)

## 📊 Market Performance

| Market | Predictions | Accuracy | ROI |
|--------|-------------|----------|-----|
"""
        for market, stats in self.market_performance.items():
            report += f"| {market} | {stats['count']} | {stats['accuracy']:.1%} | {stats['roi']:+.2%} |\n"
        
        report += """
## 🌍 League Performance

| League | Predictions | Accuracy | ROI |
|--------|-------------|----------|-----|
"""
        for league, stats in self.league_performance.items():
            report += f"| {league} | {stats['count']} | {stats['accuracy']:.1%} | {stats['roi']:+.2%} |\n"
        
        report += """
## 🎚️ Confidence Calibration

| Confidence | Predictions | Predicted | Actual | Gap |
|------------|-------------|-----------|--------|-----|
"""
        for bucket, stats in self.confidence_calibration.items():
            gap = stats['avg_predicted'] - stats['actual_rate']
            gap_emoji = '✅' if abs(gap) < 0.05 else ('⚠️' if abs(gap) < 0.10 else '❌')
            report += f"| {bucket} | {stats['count']} | {stats['avg_predicted']:.1%} | {stats['actual_rate']:.1%} | {gap:+.1%} {gap_emoji} |\n"
        
        report += "\n## 💡 Recommendations\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def _get_grade(self) -> Dict:
        """Calculate overall grade"""
        score = 0
        if self.accuracy > 0.55: score += 2
        elif self.accuracy > 0.50: score += 1
        
        if self.roi > 0.05: score += 2
        elif self.roi > 0: score += 1
        
        if self.brier_score < 0.20: score += 1
        if self.expected_calibration_error < 0.10: score += 1
        
        grades = {
            6: {'grade': 'A+', 'emoji': '🌟'},
            5: {'grade': 'A', 'emoji': '⭐'},
            4: {'grade': 'B', 'emoji': '✅'},
            3: {'grade': 'C', 'emoji': '⚠️'},
            2: {'grade': 'D', 'emoji': '❌'},
        }
        return grades.get(score, {'grade': 'F', 'emoji': '💀'})
    
    def _brier_quality(self) -> str:
        if self.brier_score < 0.15: return '🌟 Excellent'
        elif self.brier_score < 0.20: return '✅ Good'
        elif self.brier_score < 0.25: return '⚠️ Fair'
        else: return '❌ Poor'
    
    def _ece_quality(self) -> str:
        if self.expected_calibration_error < 0.05: return '🌟 Excellent'
        elif self.expected_calibration_error < 0.10: return '✅ Good'
        elif self.expected_calibration_error < 0.15: return '⚠️ Fair'
        else: return '❌ Poor'


class PredictionComparator:
    """
    Core engine for comparing predictions against actual results.
    
    Implements professional-grade evaluation metrics used by
    FiveThirtyEight and professional betting operations.
    """
    
    def __init__(self, result_fetcher: Optional[MultiSourceResultFetcher] = None):
        self.result_fetcher = result_fetcher or MultiSourceResultFetcher()
        self.results_cache: Dict[str, List[MatchResult]] = {}
    
    def compare_predictions(
        self,
        predictions: List[Prediction],
        fetch_results: bool = True
    ) -> ComparisonReport:
        """
        Compare a list of predictions against actual results.
        
        Args:
            predictions: List of Prediction objects
            fetch_results: Whether to fetch results from APIs
        
        Returns:
            ComparisonReport with detailed analysis
        """
        if not predictions:
            raise ValueError("No predictions to compare")
        
        logger.info(f"Comparing {len(predictions)} predictions...")
        
        # Group predictions by date
        dates = set(p.match_date for p in predictions)
        
        # Fetch results for all dates
        if fetch_results:
            for date in dates:
                if date not in self.results_cache:
                    logger.info(f"Fetching results for {date}...")
                    self.results_cache[date] = self.result_fetcher.fetch_results(date)
        
        # Compare each prediction
        comparison_results = []
        verified_count = 0
        
        for pred in predictions:
            result = self._compare_single(pred)
            comparison_results.append(result)
            if result.actual_result is not None:
                verified_count += 1
        
        # Calculate metrics
        report = self._generate_report(predictions, comparison_results, verified_count)
        
        return report
    
    def _compare_single(self, prediction: Prediction) -> ComparisonResult:
        """Compare a single prediction to actual result"""
        # Find matching result
        results = self.results_cache.get(prediction.match_date, [])
        actual_result = self.result_fetcher.find_result(
            prediction.home_team,
            prediction.away_team,
            prediction.match_date,
            results
        )
        
        if actual_result is None:
            # Could not verify
            return ComparisonResult(
                prediction=prediction,
                actual_result=None,
                prediction_correct=False,
                actual_outcome=False,
                profit_loss=0,
                brier_contribution=0,
                log_loss_contribution=0,
                clv=None
            )
        
        # Get actual outcome for the market
        actual_outcome = actual_result.get_outcome(prediction.market)
        
        # Was prediction correct?
        # If predicted probability > 0.5, we predicted YES
        predicted_yes = prediction.predicted_probability > 0.5
        prediction_correct = predicted_yes == actual_outcome
        
        # Calculate profit/loss
        if prediction_correct:
            profit_loss = prediction.stake * (prediction.odds - 1)
        else:
            profit_loss = -prediction.stake
        
        # Brier score contribution
        outcome_numeric = 1.0 if actual_outcome else 0.0
        brier_contribution = (prediction.predicted_probability - outcome_numeric) ** 2
        
        # Log loss contribution
        eps = 1e-15
        prob_clipped = np.clip(prediction.predicted_probability, eps, 1 - eps)
        if actual_outcome:
            log_loss_contribution = -np.log(prob_clipped)
        else:
            log_loss_contribution = -np.log(1 - prob_clipped)
        
        # CLV (if closing odds available)
        clv = None
        if prediction.closing_odds and prediction.closing_odds > 0:
            clv = (prediction.odds / prediction.closing_odds) - 1
        
        return ComparisonResult(
            prediction=prediction,
            actual_result=actual_result,
            prediction_correct=prediction_correct,
            actual_outcome=actual_outcome,
            profit_loss=profit_loss,
            brier_contribution=brier_contribution,
            log_loss_contribution=log_loss_contribution,
            clv=clv
        )
    
    def _generate_report(
        self,
        predictions: List[Prediction],
        results: List[ComparisonResult],
        verified_count: int
    ) -> ComparisonReport:
        """Generate comprehensive comparison report"""
        
        # Filter to verified only
        verified = [r for r in results if r.actual_result is not None]
        
        if not verified:
            # Return empty report
            return self._empty_report(predictions)
        
        # Core metrics
        correct_count = sum(1 for r in verified if r.prediction_correct)
        accuracy = correct_count / len(verified)
        
        # Brier score
        brier_score = np.mean([r.brier_contribution for r in verified])
        brier_decomp = self._brier_decomposition(verified)
        
        # Log loss
        log_loss = np.mean([r.log_loss_contribution for r in verified])
        
        # ECE
        ece = self._calculate_ece(verified)
        
        # Financial
        total_staked = sum(r.prediction.stake for r in verified)
        total_profit = sum(r.profit_loss for r in verified)
        roi = total_profit / total_staked if total_staked > 0 else 0
        win_count = sum(1 for r in verified if r.profit_loss > 0)
        win_rate = win_count / len(verified)
        
        # Market breakdown
        market_performance = self._breakdown_by_market(verified)
        
        # League breakdown
        league_performance = self._breakdown_by_league(verified)
        
        # Confidence calibration
        confidence_calibration = self._calibration_by_confidence(verified)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy, roi, brier_decomp, ece, 
            market_performance, league_performance, confidence_calibration
        )
        
        # Get date range
        dates = [r.prediction.match_date for r in verified]
        period_start = min(dates)
        period_end = max(dates)
        
        return ComparisonReport(
            timestamp=datetime.now().isoformat(),
            period_start=period_start,
            period_end=period_end,
            total_predictions=len(predictions),
            verified_predictions=verified_count,
            unverified_predictions=len(predictions) - verified_count,
            accuracy=accuracy,
            brier_score=brier_score,
            brier_decomposition=brier_decomp,
            log_loss=log_loss,
            expected_calibration_error=ece,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            market_performance=market_performance,
            league_performance=league_performance,
            confidence_calibration=confidence_calibration,
            results=results,
            recommendations=recommendations
        )
    
    def _brier_decomposition(self, results: List[ComparisonResult]) -> BrierDecomposition:
        """
        Murphy decomposition of Brier Score.
        
        BS = Reliability - Resolution + Uncertainty
        """
        n = len(results)
        if n == 0:
            return BrierDecomposition(0, 0, 0, 0, 0)
        
        # Base rate (climatology)
        outcomes = [1.0 if r.actual_outcome else 0.0 for r in results]
        base_rate = np.mean(outcomes)
        uncertainty = base_rate * (1 - base_rate)
        
        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        reliability = 0.0
        resolution = 0.0
        
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            
            bin_results = [r for r in results 
                          if low <= r.prediction.predicted_probability < high]
            
            if not bin_results:
                continue
            
            n_k = len(bin_results)
            f_k = np.mean([r.prediction.predicted_probability for r in bin_results])
            o_k = np.mean([1.0 if r.actual_outcome else 0.0 for r in bin_results])
            
            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2
        
        reliability /= n
        resolution /= n
        
        brier_score = reliability - resolution + uncertainty
        skill_score = 1 - brier_score / uncertainty if uncertainty > 0 else 0
        
        return BrierDecomposition(
            brier_score=brier_score,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            skill_score=skill_score
        )
    
    def _calculate_ece(self, results: List[ComparisonResult], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        n = len(results)
        if n == 0:
            return 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            
            bin_results = [r for r in results 
                          if low <= r.prediction.predicted_probability < high]
            
            if not bin_results:
                continue
            
            bin_conf = np.mean([r.prediction.predicted_probability for r in bin_results])
            bin_acc = np.mean([1.0 if r.actual_outcome else 0.0 for r in bin_results])
            bin_size = len(bin_results) / n
            
            ece += bin_size * abs(bin_conf - bin_acc)
        
        return ece
    
    def _breakdown_by_market(self, results: List[ComparisonResult]) -> Dict[str, Dict]:
        """Performance breakdown by market type"""
        breakdown = {}
        
        markets = set(r.prediction.market for r in results)
        
        for market in markets:
            market_results = [r for r in results if r.prediction.market == market]
            
            correct = sum(1 for r in market_results if r.prediction_correct)
            total_staked = sum(r.prediction.stake for r in market_results)
            total_profit = sum(r.profit_loss for r in market_results)
            
            breakdown[market] = {
                'count': len(market_results),
                'accuracy': correct / len(market_results),
                'roi': total_profit / total_staked if total_staked > 0 else 0,
                'total_profit': total_profit,
                'avg_odds': np.mean([r.prediction.odds for r in market_results])
            }
        
        return breakdown
    
    def _breakdown_by_league(self, results: List[ComparisonResult]) -> Dict[str, Dict]:
        """Performance breakdown by league"""
        breakdown = {}
        
        leagues = set(r.prediction.league for r in results if r.prediction.league)
        
        for league in leagues:
            league_results = [r for r in results if r.prediction.league == league]
            
            correct = sum(1 for r in league_results if r.prediction_correct)
            total_staked = sum(r.prediction.stake for r in league_results)
            total_profit = sum(r.profit_loss for r in league_results)
            
            breakdown[league] = {
                'count': len(league_results),
                'accuracy': correct / len(league_results),
                'roi': total_profit / total_staked if total_staked > 0 else 0,
                'total_profit': total_profit
            }
        
        return breakdown
    
    def _calibration_by_confidence(self, results: List[ComparisonResult]) -> Dict[str, Dict]:
        """Calibration analysis by confidence bucket"""
        buckets = {
            '50-60%': (0.50, 0.60),
            '60-70%': (0.60, 0.70),
            '70-80%': (0.70, 0.80),
            '80-90%': (0.80, 0.90),
            '90-100%': (0.90, 1.01)
        }
        
        calibration = {}
        
        for bucket_name, (low, high) in buckets.items():
            bucket_results = [r for r in results 
                            if low <= r.prediction.predicted_probability < high]
            
            if not bucket_results:
                continue
            
            avg_predicted = np.mean([r.prediction.predicted_probability for r in bucket_results])
            actual_rate = np.mean([1.0 if r.actual_outcome else 0.0 for r in bucket_results])
            
            calibration[bucket_name] = {
                'count': len(bucket_results),
                'avg_predicted': avg_predicted,
                'actual_rate': actual_rate,
                'calibration_error': avg_predicted - actual_rate,
                'is_overconfident': avg_predicted > actual_rate
            }
        
        return calibration
    
    def _generate_recommendations(
        self,
        accuracy: float,
        roi: float,
        brier: BrierDecomposition,
        ece: float,
        market_perf: Dict,
        league_perf: Dict,
        calibration: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall performance
        if accuracy < 0.50:
            recommendations.append("⚠️ Accuracy below 50% - model may be worse than random")
        
        if roi < 0:
            recommendations.append("❌ Negative ROI - review stake sizing and odds selection")
        elif roi < 0.05:
            recommendations.append("⚠️ Marginal ROI - consider higher edge threshold")
        
        # Calibration
        if brier.reliability > 0.02:
            recommendations.append("🎯 High reliability error - consider probability calibration (Platt/Isotonic)")
        
        if brier.resolution < 0.05:
            recommendations.append("📊 Low resolution - model not distinguishing outcomes well")
        
        if ece > 0.10:
            recommendations.append("⚠️ Poor calibration (ECE > 10%) - predictions not matching outcomes")
        
        # Overconfidence check
        overconf_buckets = [b for b, s in calibration.items() if s.get('is_overconfident') and s['count'] > 5]
        if overconf_buckets:
            recommendations.append(f"🔻 Overconfident in: {', '.join(overconf_buckets)}")
        
        # Market-specific
        worst_market = min(market_perf.items(), key=lambda x: x[1]['roi'], default=(None, {}))
        if worst_market[0] and worst_market[1].get('roi', 0) < -0.10:
            recommendations.append(f"❌ Poor performance in {worst_market[0]} market (ROI: {worst_market[1]['roi']:.1%})")
        
        best_market = max(market_perf.items(), key=lambda x: x[1]['roi'], default=(None, {}))
        if best_market[0] and best_market[1].get('roi', 0) > 0.10:
            recommendations.append(f"✅ Strong performance in {best_market[0]} market - consider increasing exposure")
        
        # League-specific
        worst_league = min(league_perf.items(), key=lambda x: x[1]['roi'], default=(None, {}))
        if worst_league[0] and worst_league[1].get('roi', 0) < -0.15:
            recommendations.append(f"❌ Avoid {worst_league[0]} - consistent losses")
        
        if not recommendations:
            recommendations.append("✅ System performing well - maintain current strategy")
        
        return recommendations
    
    def _empty_report(self, predictions: List[Prediction]) -> ComparisonReport:
        """Generate empty report when no results verified"""
        return ComparisonReport(
            timestamp=datetime.now().isoformat(),
            period_start=min(p.match_date for p in predictions),
            period_end=max(p.match_date for p in predictions),
            total_predictions=len(predictions),
            verified_predictions=0,
            unverified_predictions=len(predictions),
            accuracy=0,
            brier_score=0,
            brier_decomposition=BrierDecomposition(0, 0, 0, 0, 0),
            log_loss=0,
            expected_calibration_error=0,
            total_staked=0,
            total_profit=0,
            roi=0,
            win_rate=0,
            market_performance={},
            league_performance={},
            confidence_calibration={},
            results=[],
            recommendations=["⚠️ No predictions could be verified - check team name matching"]
        )
    
    def save_report(self, report: ComparisonReport, output_dir: str = "reports"):
        """Save report to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_path = output_path / f"comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Save Markdown
        md_path = output_path / f"comparison_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(report.to_markdown())
        
        logger.info(f"Reports saved to {output_path}")
        return json_path, md_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test the comparator with sample data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare predictions to results')
    parser.add_argument('--sample', action='store_true', help='Run with sample data')
    args = parser.parse_args()
    
    if args.sample:
        # Sample predictions for testing
        predictions = [
            Prediction(
                prediction_id="test_1",
                match_date="2026-01-26",
                home_team="Bayern Munich",
                away_team="RB Leipzig",
                league="Bundesliga",
                market="over_2_5",
                predicted_probability=0.72,
                odds=1.65,
                stake=10
            ),
            Prediction(
                prediction_id="test_2",
                match_date="2026-01-26",
                home_team="Borussia Dortmund",
                away_team="Eintracht Frankfurt",
                league="Bundesliga",
                market="btts",
                predicted_probability=0.68,
                odds=1.75,
                stake=10
            ),
        ]
        
        comparator = PredictionComparator()
        report = comparator.compare_predictions(predictions)
        
        print(report.to_markdown())


if __name__ == '__main__':
    main()

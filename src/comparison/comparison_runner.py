"""
🎯 Comparison Runner - CLI for Daily Prediction Verification
=============================================================

Automated runner for:
- Fetching yesterday's results
- Comparing against predictions
- Generating autopsy reports
- Updating CLV tracking
- Creating visualizations
- Sending Telegram summary

Usage:
    python -m src.comparison.comparison_runner --date 2026-01-25
    python -m src.comparison.comparison_runner --yesterday
    python -m src.comparison.comparison_runner --batch --days 7
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.comparison.result_fetcher import MultiSourceResultFetcher, MatchResult
from src.comparison.prediction_comparator import PredictionComparator, Prediction, ComparisonReport
from src.comparison.prediction_autopsy import PredictionAutopsy, AutopsyReport
from src.comparison.clv_tracker import CLVTracker, CLVAnalysis
from src.comparison.visualizer import ComparisonVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComparisonRunner:
    """
    Orchestrates the full comparison pipeline.
    
    Flow:
    1. Load predictions (from JSON/DB)
    2. Fetch results (from multiple APIs)
    3. Match predictions to results
    4. Generate comparison report
    5. Run autopsy on failures
    6. Update CLV tracker
    7. Generate visualizations
    8. Save reports
    9. Send Telegram summary
    """
    
    def __init__(
        self,
        predictions_dir: str = "data/predictions",
        results_dir: str = "data/results",
        reports_dir: str = "reports",
        api_key: Optional[str] = None
    ):
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create directories
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.result_fetcher = MultiSourceResultFetcher(api_key=api_key)
        self.comparator = PredictionComparator(result_fetcher=self.result_fetcher)
        self.autopsy = PredictionAutopsy()
        self.clv_tracker = CLVTracker()
        self.visualizer = ComparisonVisualizer()
    
    def run(
        self,
        date: str,
        leagues: Optional[List[str]] = None,
        send_telegram: bool = False
    ) -> Dict[str, Any]:
        """
        Run full comparison pipeline for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            leagues: Optional list of leagues to filter
            send_telegram: Whether to send Telegram summary
        
        Returns:
            Dict with all reports
        """
        logger.info(f"🎯 Running comparison pipeline for {date}")
        
        results = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        }
        
        try:
            # 1. Load predictions
            predictions = self._load_predictions(date)
            logger.info(f"Loaded {len(predictions)} predictions for {date}")
            
            if not predictions:
                logger.warning(f"No predictions found for {date}")
                results['status'] = 'no_predictions'
                return results
            
            # 2. Run comparison
            comparison_report = self.comparator.compare_predictions(predictions)
            results['comparison'] = self._serialize_report(comparison_report)
            logger.info(f"Comparison complete: {comparison_report.accuracy:.1%} accuracy")
            
            # 3. Run autopsy on failures
            match_results = self._fetch_results_for_autopsy(date, leagues)
            autopsy_report = self.autopsy.analyze(
                [self._prediction_to_dict(p) for p in predictions],
                match_results
            )
            results['autopsy'] = self._serialize_autopsy(autopsy_report)
            logger.info(f"Autopsy complete: {autopsy_report.total_failures} failures analyzed")
            
            # 4. Update CLV tracker
            self._update_clv_tracker(predictions, comparison_report)
            clv_analysis = self.clv_tracker.analyze()
            results['clv'] = self._serialize_clv(clv_analysis)
            logger.info(f"CLV analysis: {clv_analysis.avg_clv:+.2%} average")
            
            # 5. Generate visualizations
            viz_paths = self._generate_visualizations(date, comparison_report, clv_analysis)
            results['visualizations'] = viz_paths
            
            # 6. Save reports
            self._save_reports(date, results)
            
            # 7. Send Telegram summary
            if send_telegram:
                self._send_telegram_summary(date, comparison_report, autopsy_report, clv_analysis)
            
            results['status'] = 'completed'
            logger.info(f"✅ Pipeline complete for {date}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def run_batch(
        self,
        start_date: str,
        end_date: str,
        leagues: Optional[List[str]] = None
    ) -> List[Dict]:
        """Run comparison for multiple dates"""
        results = []
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            result = self.run(date_str, leagues)
            results.append(result)
            current += timedelta(days=1)
        
        return results
    
    def _load_predictions(self, date: str) -> List[Prediction]:
        """Load predictions for a date from JSON file"""
        pred_file = self.predictions_dir / f"predictions_{date}.json"
        
        if not pred_file.exists():
            # Try alternate naming
            alt_file = self.predictions_dir / f"{date}_predictions.json"
            if alt_file.exists():
                pred_file = alt_file
            else:
                logger.warning(f"No prediction file found: {pred_file}")
                return []
        
        with open(pred_file) as f:
            data = json.load(f)
        
        predictions = []
        for item in data.get('predictions', data if isinstance(data, list) else []):
            try:
                pred = Prediction(
                    prediction_id=item.get('prediction_id', item.get('id', '')),
                    match_date=item.get('match_date', item.get('date', date)),
                    home_team=item.get('home_team', ''),
                    away_team=item.get('away_team', ''),
                    market=item.get('market', ''),
                    predicted_probability=float(item.get('predicted_probability', item.get('probability', 0.5))),
                    odds=float(item.get('odds', 1.5)),
                    league=item.get('league', ''),
                    closing_odds=item.get('closing_odds')
                )
                predictions.append(pred)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid prediction: {e}")
        
        return predictions
    
    def _fetch_results_for_autopsy(
        self,
        date: str,
        leagues: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetch results and convert to dict format for autopsy"""
        results = self.result_fetcher.fetch_results(date, leagues)
        
        return [
            {
                'date': r.match_date,
                'home_team': r.home_team,
                'away_team': r.away_team,
                'home_goals': r.home_goals,
                'away_goals': r.away_goals
            }
            for r in results
        ]
    
    def _update_clv_tracker(
        self,
        predictions: List[Prediction],
        report: ComparisonReport
    ):
        """Update CLV tracker with comparison results"""
        for result in report.results:
            pred = next((p for p in predictions if p.prediction_id == result.prediction_id), None)
            if pred:
                self.clv_tracker.add_bet(
                    prediction_id=result.prediction_id,
                    match_date=result.match_date,
                    home_team=result.home_team,
                    away_team=result.away_team,
                    market=result.market,
                    predicted_probability=result.predicted_probability,
                    odds_at_bet=result.odds,
                    closing_odds=pred.closing_odds,
                    won=result.correct,
                    stake=1.0
                )
    
    def _generate_visualizations(
        self,
        date: str,
        report: ComparisonReport,
        clv: CLVAnalysis
    ) -> Dict[str, str]:
        """Generate and save visualizations"""
        viz_dir = self.reports_dir / "visualizations" / date
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Extract data for visualizations
        predictions = [r.predicted_probability for r in report.results]
        outcomes = [r.correct for r in report.results]
        pnls = [r.pnl for r in report.results]
        
        # Reliability diagram
        if predictions and outcomes:
            path = self.visualizer.reliability_diagram(
                predictions, outcomes,
                title=f"Reliability Diagram - {date}",
                save_path=str(viz_dir / "reliability.png")
            )
            if path:
                paths['reliability'] = path
        
        # Equity curve
        if pnls:
            path = self.visualizer.equity_curve(
                pnls,
                title=f"Equity Curve - {date}",
                save_path=str(viz_dir / "equity.png")
            )
            if path:
                paths['equity'] = path
        
        # CLV distribution
        clvs = [e.clv_vs_close for e in self.clv_tracker.entries if e.clv_vs_close is not None]
        if clvs:
            path = self.visualizer.clv_distribution(
                clvs,
                title=f"CLV Distribution",
                save_path=str(viz_dir / "clv_dist.png")
            )
            if path:
                paths['clv_distribution'] = path
        
        return paths
    
    def _save_reports(self, date: str, results: Dict):
        """Save all reports to files"""
        report_dir = self.reports_dir / date
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = report_dir / "comparison_report.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save markdown reports
        if 'comparison' in results:
            md_file = report_dir / "comparison.md"
            # Generate markdown from comparison data
            with open(md_file, 'w') as f:
                f.write(f"# Comparison Report - {date}\n\n")
                f.write(json.dumps(results['comparison'], indent=2, default=str))
        
        if 'autopsy' in results:
            md_file = report_dir / "autopsy.md"
            with open(md_file, 'w') as f:
                f.write(f"# Autopsy Report - {date}\n\n")
                f.write(json.dumps(results['autopsy'], indent=2, default=str))
        
        logger.info(f"Reports saved to {report_dir}")
    
    def _send_telegram_summary(
        self,
        date: str,
        comparison: ComparisonReport,
        autopsy: AutopsyReport,
        clv: CLVAnalysis
    ):
        """Send summary to Telegram"""
        try:
            # Import telegram bot
            from src.bot.telegram_sender import send_message
            
            summary = f"""
📊 **Comparison Report - {date}**

**Performance:**
- Predictions: {comparison.total_predictions}
- Accuracy: {comparison.accuracy:.1%}
- ROI: {comparison.roi:+.2%}
- Brier Score: {comparison.brier_score:.4f}

**Failures ({autopsy.total_failures}):**
- Top error: {list(autopsy.error_categories.keys())[0] if autopsy.error_categories else 'N/A'}

**CLV:**
- Average: {clv.avg_clv:+.2%}
- Quality: {clv.clv_quality.value}
"""
            send_message(summary)
            logger.info("Telegram summary sent")
            
        except Exception as e:
            logger.warning(f"Failed to send Telegram: {e}")
    
    @staticmethod
    def _prediction_to_dict(pred: Prediction) -> Dict:
        """Convert Prediction to dict for autopsy"""
        return {
            'prediction_id': pred.prediction_id,
            'match_date': pred.match_date,
            'home_team': pred.home_team,
            'away_team': pred.away_team,
            'market': pred.market,
            'predicted_probability': pred.predicted_probability,
            'odds': pred.odds,
            'league': pred.league
        }
    
    @staticmethod
    def _serialize_report(report: ComparisonReport) -> Dict:
        """Serialize ComparisonReport for JSON"""
        return {
            'total_predictions': report.total_predictions,
            'matched_predictions': report.matched_predictions,
            'accuracy': report.accuracy,
            'roi': report.roi,
            'brier_score': report.brier_score,
            'log_loss': report.log_loss,
            'ece': report.ece,
            'total_pnl': report.total_pnl
        }
    
    @staticmethod
    def _serialize_autopsy(report: AutopsyReport) -> Dict:
        """Serialize AutopsyReport for JSON"""
        return {
            'total_predictions': report.total_predictions,
            'total_failures': report.total_failures,
            'failure_rate': report.failure_rate,
            'error_categories': report.error_categories,
            'recommendations': report.recommendations
        }
    
    @staticmethod
    def _serialize_clv(analysis: CLVAnalysis) -> Dict:
        """Serialize CLVAnalysis for JSON"""
        return {
            'avg_clv': analysis.avg_clv,
            'median_clv': analysis.median_clv,
            'clv_positive_rate': analysis.clv_positive_rate,
            'quality': analysis.clv_quality.value,
            'quality_score': analysis.quality_score,
            'theoretical_roi': analysis.theoretical_roi
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run prediction comparison pipeline"
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date to compare (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--yesterday',
        action='store_true',
        help='Compare yesterday\'s predictions'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch comparison'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days for batch (default: 7)'
    )
    parser.add_argument(
        '--leagues',
        nargs='+',
        help='Filter by leagues'
    )
    parser.add_argument(
        '--telegram',
        action='store_true',
        help='Send Telegram summary'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Football-Data.org API key'
    )
    
    args = parser.parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get('FOOTBALL_DATA_API_KEY')
    
    runner = ComparisonRunner(api_key=api_key)
    
    if args.yesterday:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = runner.run(date, args.leagues, args.telegram)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.batch:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days - 1)
        results = runner.run_batch(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            args.leagues
        )
        print(json.dumps(results, indent=2, default=str))
    
    elif args.date:
        result = runner.run(args.date, args.leagues, args.telegram)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

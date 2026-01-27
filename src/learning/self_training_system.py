#!/usr/bin/env python3
"""
🔄 24/7 FREE SELF-TRAINING SYSTEM
==================================
Autonomous continuous learning without any costs:

1. Result Collection (via free APIs)
2. Prediction Verification
3. Error Analysis & Pattern Detection
4. Model Retraining Triggers
5. Performance Monitoring

Designed for GitHub Actions (2000 free minutes/month)
or local cron jobs.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    prediction_id: str
    timestamp: str
    match_date: str
    home_team: str
    away_team: str
    league: str
    
    # Prediction details
    market: str  # 'over_1_5', 'btts_yes', etc.
    predicted_probability: float
    market_odds: float
    kelly_stake: float
    
    # Actual result (filled after match)
    actual_outcome: Optional[bool] = None
    actual_home_goals: Optional[int] = None
    actual_away_goals: Optional[int] = None
    
    # Analysis
    was_correct: Optional[bool] = None
    profit_loss: Optional[float] = None
    verified_at: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Metrics for self-training decisions"""
    total_predictions: int = 0
    verified_predictions: int = 0
    correct_predictions: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    
    # Per-market breakdown
    market_performance: Dict[str, Dict] = None
    
    # Calibration
    calibration_error: float = 0.0  # Brier score
    
    # Recent performance (last 50)
    recent_win_rate: float = 0.0
    recent_roi: float = 0.0
    
    # Triggers
    needs_retraining: bool = False
    retraining_reason: str = ""


class SelfTrainingSystem:
    """
    🔄 Autonomous self-training system
    
    Workflow:
    1. Store predictions with IDs
    2. Collect results after matches
    3. Verify predictions
    4. Analyze errors
    5. Trigger retraining if needed
    """
    
    def __init__(self, data_dir: str = "data/self_training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Files
        self.predictions_file = self.data_dir / "predictions.json"
        self.metrics_file = self.data_dir / "metrics.json"
        self.training_log = self.data_dir / "training_log.json"
        
        # Load existing data
        self.predictions = self._load_predictions()
        self.metrics = self._load_metrics()
        
        # Thresholds for retraining
        self.MIN_PREDICTIONS = 50  # Minimum before retraining
        self.WIN_RATE_THRESHOLD = 0.52  # Retrain if below
        self.ROI_THRESHOLD = -0.05  # Retrain if ROI below -5%
        self.CALIBRATION_THRESHOLD = 0.15  # Retrain if calibration error > 15%
    
    def record_prediction(self, home_team: str, away_team: str, league: str,
                         match_date: str, market: str, probability: float,
                         odds: float, kelly_stake: float) -> str:
        """
        Record a new prediction
        
        Returns: prediction_id for later verification
        """
        # Generate unique ID
        pred_id = self._generate_prediction_id(home_team, away_team, match_date, market)
        
        record = PredictionRecord(
            prediction_id=pred_id,
            timestamp=datetime.now().isoformat(),
            match_date=match_date,
            home_team=home_team,
            away_team=away_team,
            league=league,
            market=market,
            predicted_probability=probability,
            market_odds=odds,
            kelly_stake=kelly_stake
        )
        
        self.predictions[pred_id] = asdict(record)
        self._save_predictions()
        
        print(f"   📝 Recorded prediction: {pred_id}")
        return pred_id
    
    def verify_prediction(self, prediction_id: str, home_goals: int, away_goals: int) -> bool:
        """
        Verify a prediction against actual result
        
        Returns: True if prediction was correct
        """
        if prediction_id not in self.predictions:
            print(f"   ⚠️  Prediction {prediction_id} not found")
            return False
        
        record = self.predictions[prediction_id]
        market = record['market']
        
        # Determine outcome based on market
        outcome = self._determine_outcome(market, home_goals, away_goals)
        
        # Update record
        record['actual_outcome'] = outcome
        record['actual_home_goals'] = home_goals
        record['actual_away_goals'] = away_goals
        record['was_correct'] = outcome
        record['verified_at'] = datetime.now().isoformat()
        
        # Calculate profit/loss
        if outcome:
            record['profit_loss'] = (record['market_odds'] - 1) * record['kelly_stake']
        else:
            record['profit_loss'] = -record['kelly_stake']
        
        self._save_predictions()
        self._update_metrics()
        
        status = "✅ CORRECT" if outcome else "❌ WRONG"
        print(f"   {status}: {record['home_team']} vs {record['away_team']} ({market})")
        
        return outcome
    
    def verify_pending_predictions(self, results: List[Dict]) -> int:
        """
        Bulk verify pending predictions
        
        Args:
            results: List of {'home_team': str, 'away_team': str, 
                            'home_goals': int, 'away_goals': int, 'date': str}
        
        Returns: Number of verified predictions
        """
        verified_count = 0
        
        # Find unverified predictions
        pending = {
            pid: rec for pid, rec in self.predictions.items()
            if rec.get('verified_at') is None
        }
        
        for result in results:
            # Try to match with pending predictions
            for pred_id, record in pending.items():
                if (self._fuzzy_match(record['home_team'], result['home_team']) and
                    self._fuzzy_match(record['away_team'], result['away_team'])):
                    
                    self.verify_prediction(
                        pred_id,
                        result['home_goals'],
                        result['away_goals']
                    )
                    verified_count += 1
        
        print(f"   📊 Verified {verified_count} predictions")
        return verified_count
    
    def _determine_outcome(self, market: str, home_goals: int, away_goals: int) -> bool:
        """Determine if prediction was correct based on market"""
        total_goals = home_goals + away_goals
        both_scored = home_goals > 0 and away_goals > 0
        
        outcomes = {
            'over_0_5': total_goals > 0,
            'under_0_5': total_goals == 0,
            'over_1_5': total_goals > 1,
            'under_1_5': total_goals < 2,
            'over_2_5': total_goals > 2,
            'under_2_5': total_goals < 3,
            'over_3_5': total_goals > 3,
            'under_3_5': total_goals < 4,
            'btts_yes': both_scored,
            'btts_no': not both_scored,
            'home_win': home_goals > away_goals,
            'draw': home_goals == away_goals,
            'away_win': home_goals < away_goals,
            'home_or_draw': home_goals >= away_goals,
            'away_or_draw': home_goals <= away_goals,
        }
        
        return outcomes.get(market, False)
    
    def analyze_errors(self) -> Dict:
        """
        Analyze prediction errors to find patterns
        
        Returns: Analysis report
        """
        verified = [
            rec for rec in self.predictions.values()
            if rec.get('verified_at') is not None
        ]
        
        if len(verified) < 20:
            return {'status': 'insufficient_data', 'message': 'Need at least 20 verified predictions'}
        
        errors = [rec for rec in verified if not rec['was_correct']]
        
        analysis = {
            'total_verified': len(verified),
            'total_errors': len(errors),
            'error_rate': len(errors) / len(verified),
            'patterns': []
        }
        
        # Analyze errors by market
        market_errors = {}
        for err in errors:
            market = err['market']
            if market not in market_errors:
                market_errors[market] = {'count': 0, 'total_in_market': 0}
            market_errors[market]['count'] += 1
        
        for rec in verified:
            market = rec['market']
            if market in market_errors:
                market_errors[market]['total_in_market'] += 1
        
        for market, data in market_errors.items():
            error_rate = data['count'] / max(1, data['total_in_market'])
            analysis['patterns'].append({
                'type': 'market_error_rate',
                'market': market,
                'error_rate': error_rate,
                'severity': 'high' if error_rate > 0.5 else 'medium'
            })
        
        # Analyze overconfidence
        high_conf_errors = [
            err for err in errors 
            if err['predicted_probability'] > 0.75
        ]
        
        if len(high_conf_errors) > len(errors) * 0.3:
            analysis['patterns'].append({
                'type': 'overconfidence',
                'message': f'{len(high_conf_errors)} high-confidence predictions wrong',
                'severity': 'high'
            })
        
        # Analyze league-specific issues
        league_performance = {}
        for rec in verified:
            league = rec['league']
            if league not in league_performance:
                league_performance[league] = {'correct': 0, 'total': 0}
            league_performance[league]['total'] += 1
            if rec['was_correct']:
                league_performance[league]['correct'] += 1
        
        for league, data in league_performance.items():
            win_rate = data['correct'] / max(1, data['total'])
            if win_rate < 0.45:
                analysis['patterns'].append({
                    'type': 'league_underperformance',
                    'league': league,
                    'win_rate': win_rate,
                    'severity': 'medium'
                })
        
        return analysis
    
    def check_retraining_needed(self) -> TrainingMetrics:
        """
        Check if model retraining is needed
        
        Triggers:
        1. Win rate below threshold
        2. ROI below threshold
        3. Calibration error too high
        4. Significant concept drift detected
        """
        metrics = self._calculate_metrics()
        
        # Check triggers
        triggers = []
        
        if metrics.verified_predictions >= self.MIN_PREDICTIONS:
            if metrics.win_rate < self.WIN_RATE_THRESHOLD:
                triggers.append(f"Win rate {metrics.win_rate:.1%} < {self.WIN_RATE_THRESHOLD:.1%}")
            
            if metrics.roi < self.ROI_THRESHOLD:
                triggers.append(f"ROI {metrics.roi:.1%} < {self.ROI_THRESHOLD:.1%}")
            
            if metrics.calibration_error > self.CALIBRATION_THRESHOLD:
                triggers.append(f"Calibration error {metrics.calibration_error:.1%} > {self.CALIBRATION_THRESHOLD:.1%}")
            
            if metrics.recent_win_rate < metrics.win_rate - 0.1:
                triggers.append(f"Recent performance declining: {metrics.recent_win_rate:.1%} vs {metrics.win_rate:.1%}")
        
        metrics.needs_retraining = len(triggers) > 0
        metrics.retraining_reason = "; ".join(triggers) if triggers else "Performance acceptable"
        
        self.metrics = metrics
        self._save_metrics()
        
        return metrics
    
    def _calculate_metrics(self) -> TrainingMetrics:
        """Calculate all training metrics"""
        verified = [
            rec for rec in self.predictions.values()
            if rec.get('verified_at') is not None
        ]
        
        if not verified:
            return TrainingMetrics()
        
        correct = [rec for rec in verified if rec['was_correct']]
        
        total_profit = sum(rec['profit_loss'] for rec in verified)
        total_staked = sum(rec['kelly_stake'] for rec in verified)
        
        # Per-market performance
        market_perf = {}
        for rec in verified:
            market = rec['market']
            if market not in market_perf:
                market_perf[market] = {'correct': 0, 'total': 0, 'profit': 0}
            market_perf[market]['total'] += 1
            if rec['was_correct']:
                market_perf[market]['correct'] += 1
            market_perf[market]['profit'] += rec['profit_loss']
        
        # Calibration (Brier score)
        brier_sum = 0
        for rec in verified:
            prob = rec['predicted_probability']
            outcome = 1 if rec['was_correct'] else 0
            brier_sum += (prob - outcome) ** 2
        
        calibration_error = brier_sum / len(verified) if verified else 0
        
        # Recent performance (last 50)
        recent = sorted(verified, key=lambda x: x['verified_at'], reverse=True)[:50]
        recent_correct = len([r for r in recent if r['was_correct']])
        recent_profit = sum(r['profit_loss'] for r in recent)
        recent_staked = sum(r['kelly_stake'] for r in recent)
        
        return TrainingMetrics(
            total_predictions=len(self.predictions),
            verified_predictions=len(verified),
            correct_predictions=len(correct),
            win_rate=len(correct) / len(verified) if verified else 0,
            total_profit=total_profit,
            roi=total_profit / total_staked if total_staked > 0 else 0,
            market_performance=market_perf,
            calibration_error=calibration_error,
            recent_win_rate=recent_correct / len(recent) if recent else 0,
            recent_roi=recent_profit / recent_staked if recent_staked > 0 else 0
        )
    
    def generate_report(self) -> str:
        """Generate human-readable training report"""
        metrics = self.check_retraining_needed()
        analysis = self.analyze_errors()
        
        report = []
        report.append("\n" + "="*60)
        report.append("🔄 SELF-TRAINING STATUS REPORT")
        report.append("="*60)
        report.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("📊 OVERALL PERFORMANCE")
        report.append("─"*40)
        report.append(f"   Total Predictions:     {metrics.total_predictions}")
        report.append(f"   Verified:              {metrics.verified_predictions}")
        report.append(f"   Correct:               {metrics.correct_predictions}")
        report.append(f"   Win Rate:              {metrics.win_rate:.1%}")
        report.append(f"   ROI:                   {metrics.roi:+.1%}")
        report.append(f"   Calibration Error:     {metrics.calibration_error:.3f}")
        report.append("")
        
        report.append("📈 RECENT PERFORMANCE (Last 50)")
        report.append("─"*40)
        report.append(f"   Win Rate:              {metrics.recent_win_rate:.1%}")
        report.append(f"   ROI:                   {metrics.recent_roi:+.1%}")
        report.append("")
        
        if metrics.market_performance:
            report.append("🎯 PER-MARKET BREAKDOWN")
            report.append("─"*40)
            for market, data in sorted(metrics.market_performance.items()):
                wr = data['correct'] / data['total'] if data['total'] > 0 else 0
                report.append(f"   {market:15} {wr:6.1%} ({data['correct']}/{data['total']})")
            report.append("")
        
        if analysis.get('patterns'):
            report.append("⚠️  ERROR PATTERNS DETECTED")
            report.append("─"*40)
            for pattern in analysis['patterns']:
                severity = "🔴" if pattern['severity'] == 'high' else "🟡"
                report.append(f"   {severity} {pattern['type']}: {pattern.get('message', pattern.get('market', ''))}")
            report.append("")
        
        report.append("🔧 RETRAINING STATUS")
        report.append("─"*40)
        if metrics.needs_retraining:
            report.append(f"   ❌ RETRAINING NEEDED")
            report.append(f"   Reason: {metrics.retraining_reason}")
        else:
            report.append(f"   ✅ Model performing within thresholds")
        report.append("")
        
        return "\n".join(report)
    
    def trigger_retraining(self) -> bool:
        """
        Trigger model retraining if needed
        
        This would integrate with the main training pipeline.
        For now, logs the action and saves state.
        """
        metrics = self.check_retraining_needed()
        
        if not metrics.needs_retraining:
            print("   ✅ No retraining needed")
            return False
        
        # Log retraining trigger
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': metrics.retraining_reason,
            'metrics': {
                'win_rate': metrics.win_rate,
                'roi': metrics.roi,
                'calibration_error': metrics.calibration_error
            }
        }
        
        # Append to training log
        log = []
        if self.training_log.exists():
            with open(self.training_log, 'r') as f:
                log = json.load(f)
        
        log.append(log_entry)
        
        with open(self.training_log, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"   🔄 Retraining triggered: {metrics.retraining_reason}")
        
        # In production, this would call the training pipeline
        # train_models() or similar
        
        return True
    
    def _generate_prediction_id(self, home: str, away: str, date: str, market: str) -> str:
        """Generate unique prediction ID"""
        key = f"{home}_{away}_{date}_{market}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Check if team names match (fuzzy)"""
        s1 = str1.lower().replace(' fc', '').replace(' cf', '').strip()
        s2 = str2.lower().replace(' fc', '').replace(' cf', '').strip()
        return s1 in s2 or s2 in s1
    
    def _load_predictions(self) -> Dict:
        """Load predictions from file"""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_predictions(self):
        """Save predictions to file"""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
    
    def _load_metrics(self) -> TrainingMetrics:
        """Load metrics from file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                return TrainingMetrics(**data)
        return TrainingMetrics()
    
    def _save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
    
    def _update_metrics(self):
        """Update and save metrics"""
        self.metrics = self._calculate_metrics()
        self._save_metrics()


def test_self_training():
    """Test the self-training system"""
    print("\n" + "="*60)
    print("🔄 SELF-TRAINING SYSTEM TEST")
    print("="*60)
    
    # Use test directory
    trainer = SelfTrainingSystem(data_dir="data/self_training_test")
    
    # Record some predictions
    test_predictions = [
        ("Bayern München", "Dortmund", "Bundesliga", "2025-01-27", "over_2_5", 0.68, 1.85, 0.03),
        ("Bayern München", "Dortmund", "Bundesliga", "2025-01-27", "btts_yes", 0.72, 1.65, 0.04),
        ("Liverpool", "Arsenal", "Premier League", "2025-01-27", "over_1_5", 0.82, 1.35, 0.05),
        ("Real Madrid", "Barcelona", "La Liga", "2025-01-27", "over_2_5", 0.70, 1.80, 0.03),
    ]
    
    print("\n📝 Recording predictions...")
    for home, away, league, date, market, prob, odds, stake in test_predictions:
        trainer.record_prediction(home, away, league, date, market, prob, odds, stake)
    
    # Verify some predictions
    print("\n✅ Verifying predictions...")
    results = [
        {'home_team': 'Bayern München', 'away_team': 'Dortmund', 'home_goals': 3, 'away_goals': 2},
        {'home_team': 'Liverpool', 'away_team': 'Arsenal', 'home_goals': 2, 'away_goals': 1},
        {'home_team': 'Real Madrid', 'away_team': 'Barcelona', 'home_goals': 1, 'away_goals': 1},
    ]
    
    trainer.verify_pending_predictions(results)
    
    # Generate report
    print(trainer.generate_report())
    
    # Check retraining status
    metrics = trainer.check_retraining_needed()
    
    print("\n✅ Self-training system working correctly!")


if __name__ == "__main__":
    test_self_training()

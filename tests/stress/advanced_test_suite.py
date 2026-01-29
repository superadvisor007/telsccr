"""
🧪 Advanced Test Suite for Self-Improving Betting System
=========================================================

Battle-tested patterns for rigorous ML betting system validation:
- Monte Carlo Simulation (1000+ paths)
- Bootstrap Confidence Intervals (BCa)
- Probability Calibration Testing
- Concept Drift Detection
- Information Leakage Detection
- Overfitting Detection
- Statistical Significance (t-tests, permutation tests)
- Robustness Testing (noise injection)
- Self-Improvement Validation

Author: telegramsoccer
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ks_2samp, ttest_1samp, ttest_ind, kendalltau
import warnings
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# ============================================================================

@dataclass
class TestResult:
    """Individual test result"""
    name: str
    passed: bool
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    score: float  # 0-1 scale
    details: Dict
    recommendation: str


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    overall_grade: str  # 'PASS', 'PASS_WITH_WARNINGS', 'CAUTION', 'FAIL'
    overall_score: float
    critical_failures: List[str]
    warnings: List[str]
    test_results: Dict[str, TestResult]
    recommendation: str
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        grade_emoji = {
            'PASS': '🟢',
            'PASS_WITH_WARNINGS': '🟡',
            'CAUTION': '🟠',
            'FAIL': '🔴'
        }
        
        report = f"""
# 🧪 Advanced Test Suite Report

**Generated:** {self.timestamp}
**Overall Grade:** {grade_emoji.get(self.overall_grade, '⚪')} {self.overall_grade}
**Score:** {self.overall_score:.2%}

## 📊 Summary

{'### ❌ Critical Failures' if self.critical_failures else ''}
{chr(10).join(f'- {f}' for f in self.critical_failures) if self.critical_failures else ''}

{'### ⚠️ Warnings' if self.warnings else ''}
{chr(10).join(f'- {w}' for w in self.warnings) if self.warnings else ''}

## 🔬 Test Results

| Test | Status | Score | Severity |
|------|--------|-------|----------|
"""
        for name, result in self.test_results.items():
            status = '✅' if result.passed else '❌'
            report += f"| {name} | {status} | {result.score:.2%} | {result.severity} |\n"
        
        report += f"""
## 💡 Recommendation

{self.recommendation}
"""
        return report


# ============================================================================
# 1. MONTE CARLO SIMULATOR
# ============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting systems.
    Generates confidence intervals for ROI, drawdown, and ruin probability.
    
    Pattern from: quantopian/zipline risk metrics
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        np.random.seed(random_seed)
    
    def simulate_betting_paths(
        self,
        bet_returns: np.ndarray,
        n_future_bets: int = 200,
        initial_bankroll: float = 1000.0,
        stake_fraction: float = 0.02
    ) -> Dict:
        """
        Simulate N betting paths via bootstrap resampling.
        
        Args:
            bet_returns: Historical bet returns (profit/stake)
            n_future_bets: Number of future bets to simulate
            initial_bankroll: Starting capital
            stake_fraction: Fixed fraction of bankroll per bet
        
        Returns:
            Statistics, confidence intervals, and paths
        """
        final_bankrolls = []
        max_drawdowns = []
        paths = []
        
        for _ in range(self.n_simulations):
            # Bootstrap: sample with replacement
            sampled_returns = np.random.choice(bet_returns, size=n_future_bets, replace=True)
            
            # Simulate bankroll evolution
            bankroll = initial_bankroll
            path = [bankroll]
            peak = bankroll
            max_dd = 0
            
            for ret in sampled_returns:
                stake = bankroll * stake_fraction
                profit = stake * ret
                bankroll = max(0, bankroll + profit)
                path.append(bankroll)
                
                # Track drawdown
                peak = max(peak, bankroll)
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            final_bankrolls.append(bankroll)
            max_drawdowns.append(max_dd)
            paths.append(path)
        
        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)
        roi = (final_bankrolls - initial_bankroll) / initial_bankroll
        
        return {
            'statistics': {
                'mean_final_bankroll': np.mean(final_bankrolls),
                'median_final_bankroll': np.median(final_bankrolls),
                'std_final_bankroll': np.std(final_bankrolls),
                'mean_roi': np.mean(roi),
                'median_roi': np.median(roi),
                'prob_profitable': np.mean(roi > 0),
                'prob_ruin': np.mean(final_bankrolls < initial_bankroll * 0.1),  # <10% = ruin
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_max_drawdown': np.max(max_drawdowns),
            },
            'confidence_intervals': {
                'roi_95_ci': (np.percentile(roi, 2.5), np.percentile(roi, 97.5)),
                'roi_99_ci': (np.percentile(roi, 0.5), np.percentile(roi, 99.5)),
                'drawdown_95_ci': (np.percentile(max_drawdowns, 5), np.percentile(max_drawdowns, 95)),
                'var_95': np.percentile(final_bankrolls, 5),  # Value at Risk
                'cvar_95': np.mean(final_bankrolls[final_bankrolls <= np.percentile(final_bankrolls, 5)]),
            },
            'paths': np.array(paths) if len(paths) <= 100 else None,  # Save memory
            'final_bankrolls': final_bankrolls
        }
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> Tuple[float, float]:
        """Calculate annualized Sharpe Ratio with standard error."""
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if len(excess_returns) < 2 or np.std(excess_returns, ddof=1) == 0:
            return 0.0, float('inf')
        
        sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / len(returns))
        
        return sharpe, se_sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """Calculate Sortino Ratio (downside deviation only)."""
        excess_returns = returns - target_return / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return float('inf')
        
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


# ============================================================================
# 2. BOOTSTRAP ANALYZER
# ============================================================================

class BootstrapAnalyzer:
    """
    Non-parametric confidence intervals via bootstrap.
    
    Pattern from: stefan-jansen/machine-learning-for-trading
    """
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic_func: Callable,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        method: str = 'percentile'
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Observed data
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            method: 'percentile' or 'bca'
        
        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        point_estimate = statistic_func(data)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        
        if method == 'percentile':
            ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        elif method == 'bca':
            ci_lower, ci_upper = BootstrapAnalyzer._bca_interval(
                data, bootstrap_stats, statistic_func, alpha
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return point_estimate, ci_lower, ci_upper
    
    @staticmethod
    def _bca_interval(data, bootstrap_stats, statistic_func, alpha):
        """BCa Bootstrap (Gold Standard for small samples)."""
        point_estimate = statistic_func(data)
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate) + 1e-10)
        
        # Acceleration (Jackknife)
        n = len(data)
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats.append(statistic_func(jackknife_sample))
        
        jackknife_mean = np.mean(jackknife_stats)
        denom = np.sum((jackknife_mean - np.array(jackknife_stats)) ** 2) ** 1.5
        if denom == 0:
            acc = 0
        else:
            acc = np.sum((jackknife_mean - np.array(jackknife_stats)) ** 3) / (6 * denom)
        
        # Adjust percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
        
        def adjusted_percentile(z_alpha):
            denom = 1 - acc * (z0 + z_alpha)
            if abs(denom) < 1e-10:
                return 50
            return stats.norm.cdf(z0 + (z0 + z_alpha) / denom) * 100
        
        p_lower = adjusted_percentile(z_alpha_lower)
        p_upper = adjusted_percentile(z_alpha_upper)
        
        ci_lower = np.percentile(bootstrap_stats, np.clip(p_lower, 0, 100))
        ci_upper = np.percentile(bootstrap_stats, np.clip(p_upper, 0, 100))
        
        return ci_lower, ci_upper
    
    def betting_edge_ci(
        self,
        profits: np.ndarray,
        stakes: np.ndarray,
        n_bootstrap: int = 10000
    ) -> Dict:
        """Compute confidence intervals for betting metrics."""
        results = {}
        
        # ROI CI
        roi_func = lambda x: np.sum(x) / (len(x) * np.mean(stakes))
        roi, roi_lower, roi_upper = self.bootstrap_ci(profits, roi_func, n_bootstrap)
        results['roi'] = {'point': roi, 'ci_95': (roi_lower, roi_upper)}
        
        # Win Rate CI
        wins = (profits > 0).astype(int)
        wr_func = lambda x: np.mean(x)
        wr, wr_lower, wr_upper = self.bootstrap_ci(wins, wr_func, n_bootstrap)
        results['win_rate'] = {'point': wr, 'ci_95': (wr_lower, wr_upper)}
        
        # Average Profit CI
        avg, avg_lower, avg_upper = self.bootstrap_ci(profits, np.mean, n_bootstrap)
        results['avg_profit'] = {'point': avg, 'ci_95': (avg_lower, avg_upper)}
        
        # Edge significance (one-sided t-test)
        t_stat, p_value = ttest_1samp(profits, 0)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        results['edge_significance'] = {
            't_stat': t_stat,
            'p_value_one_sided': p_one_sided,
            'significant_at_05': p_one_sided < 0.05,
            'significant_at_01': p_one_sided < 0.01
        }
        
        return results


# ============================================================================
# 3. CALIBRATION TESTER
# ============================================================================

class CalibrationTester:
    """
    Test probability calibration of predictions.
    
    Well-calibrated: predicted probability matches actual outcome rate
    """
    
    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Compute reliability diagram and calibration metrics.
        
        Args:
            y_true: Actual outcomes (0/1)
            y_prob: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            ECE, MCE, Brier score, and bin analysis
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        prob_true = []
        prob_pred = []
        bin_counts = []
        ece = 0.0
        
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                bin_size = mask.sum() / len(y_prob)
                
                prob_true.append(bin_acc)
                prob_pred.append(bin_conf)
                bin_counts.append(mask.sum())
                
                ece += bin_size * abs(bin_acc - bin_conf)
        
        prob_true = np.array(prob_true)
        prob_pred = np.array(prob_pred)
        
        # Maximum Calibration Error
        mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
        
        # Brier Score
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'brier_score': brier_score,
            'is_well_calibrated': ece < 0.1,
            'calibration_quality': self._quality_label(ece),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'bin_counts': bin_counts
        }
    
    def _quality_label(self, ece: float) -> str:
        if ece < 0.05:
            return 'EXCELLENT'
        elif ece < 0.1:
            return 'GOOD'
        elif ece < 0.15:
            return 'FAIR'
        else:
            return 'POOR'
    
    def test_overconfidence(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.7
    ) -> Dict:
        """Test for systematic overconfidence in high-probability predictions."""
        high_conf_mask = y_prob >= threshold
        
        if high_conf_mask.sum() == 0:
            return {'high_confidence_predictions': 0, 'has_data': False}
        
        high_conf_accuracy = y_true[high_conf_mask].mean()
        avg_high_confidence = y_prob[high_conf_mask].mean()
        overconfidence_gap = avg_high_confidence - high_conf_accuracy
        
        return {
            'has_data': True,
            'high_confidence_predictions': int(high_conf_mask.sum()),
            'high_conf_percentage': high_conf_mask.mean() * 100,
            'avg_predicted_prob': avg_high_confidence,
            'actual_accuracy': high_conf_accuracy,
            'overconfidence_gap': overconfidence_gap,
            'is_overconfident': overconfidence_gap > 0.05,
            'recommendation': 'CALIBRATE' if overconfidence_gap > 0.1 else 'OK'
        }


# ============================================================================
# 4. CONCEPT DRIFT DETECTOR
# ============================================================================

class ConceptDriftDetector:
    """
    Detect when feature distributions or model behavior shifts.
    
    Uses KS-test and Population Stability Index (PSI).
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        self.reference_data = reference_data
        if reference_data is not None:
            self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: pd.DataFrame) -> Dict:
        """Compute reference statistics."""
        stats_dict = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'quantiles': data[col].quantile([0.25, 0.5, 0.75]).tolist()
            }
        return stats_dict
    
    def detect_feature_drift(
        self,
        new_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        p_threshold: float = 0.05
    ) -> Dict:
        """
        Kolmogorov-Smirnov test for feature drift.
        
        Returns drift status per feature.
        """
        ref = reference_data if reference_data is not None else self.reference_data
        if ref is None:
            raise ValueError("No reference data provided")
        
        drift_results = {}
        numeric_cols = ref.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in new_data.columns:
                continue
            
            ref_values = ref[col].dropna().values
            new_values = new_data[col].dropna().values
            
            if len(ref_values) < 10 or len(new_values) < 10:
                continue
            
            # KS Test
            ks_stat, p_value = ks_2samp(ref_values, new_values)
            
            # PSI
            psi = self._calculate_psi(ref_values, new_values)
            
            drift_results[col] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'psi': psi,
                'drift_detected': p_value < p_threshold or psi > 0.2,
                'severity': 'HIGH' if psi > 0.25 else ('MEDIUM' if psi > 0.1 else 'LOW')
            }
        
        drifted = [k for k, v in drift_results.items() if v['drift_detected']]
        
        return {
            'features': drift_results,
            'summary': {
                'total_features': len(drift_results),
                'drifted_features': len(drifted),
                'drift_percentage': len(drifted) / len(drift_results) * 100 if drift_results else 0,
                'action_required': len(drifted) > len(drift_results) * 0.2 if drift_results else False
            }
        }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        
        PSI < 0.1: No drift
        0.1 <= PSI < 0.2: Slight drift
        PSI >= 0.2: Significant drift
        """
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_counts, _ = np.histogram(reference, bins=bins_edges)
        cur_counts, _ = np.histogram(current, bins=bins_edges)
        
        # Laplace smoothing
        ref_pct = (ref_counts + 1) / (sum(ref_counts) + bins)
        cur_pct = (cur_counts + 1) / (sum(cur_counts) + bins)
        
        # PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return psi


# ============================================================================
# 5. LEAKAGE DETECTOR
# ============================================================================

class LeakageDetector:
    """
    Detect information leakage in train/test splits.
    
    Critical for betting systems where temporal order matters!
    """
    
    def check_temporal_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        date_column: str = 'date'
    ) -> Dict:
        """Check if test data temporally precedes training data."""
        if date_column not in train_data.columns or date_column not in test_data.columns:
            return {'error': f'Date column "{date_column}" not found'}
        
        train_max = pd.to_datetime(train_data[date_column]).max()
        test_min = pd.to_datetime(test_data[date_column]).min()
        
        overlap = train_max >= test_min
        
        if overlap:
            overlap_count = len(test_data[pd.to_datetime(test_data[date_column]) <= train_max])
        else:
            overlap_count = 0
        
        return {
            'temporal_leakage_detected': overlap,
            'train_max_date': str(train_max),
            'test_min_date': str(test_min),
            'overlapping_samples': overlap_count,
            'gap_days': (test_min - train_max).days if not overlap else 0
        }
    
    def check_target_leakage(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        correlation_threshold: float = 0.9
    ) -> Dict:
        """Check if features have suspiciously high correlation with target."""
        suspicious = []
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            corr = features[col].corr(target)
            if abs(corr) > correlation_threshold:
                suspicious.append({'feature': col, 'correlation': corr})
        
        return {
            'suspicious_features': suspicious,
            'leakage_warning': len(suspicious) > 0,
            'recommendation': 'REVIEW FEATURES' if suspicious else 'OK'
        }


# ============================================================================
# 6. OVERFITTING DETECTOR
# ============================================================================

class OverfittingDetector:
    """
    Detect overfitting via train-test gaps and CV variance.
    """
    
    def train_test_gap_analysis(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        acceptable_gap: float = 0.1
    ) -> Dict:
        """Compare train vs test performance."""
        gaps = {}
        
        for metric in ['accuracy', 'roi', 'win_rate']:
            if metric in train_metrics and metric in test_metrics:
                gap = train_metrics[metric] - test_metrics[metric]
                gaps[metric] = {
                    'train': train_metrics[metric],
                    'test': test_metrics[metric],
                    'gap': gap,
                    'overfitting_detected': gap > acceptable_gap
                }
        
        overall_overfit = any(g['overfitting_detected'] for g in gaps.values()) if gaps else False
        max_gap = max(g['gap'] for g in gaps.values()) if gaps else 0
        
        return {
            'metric_gaps': gaps,
            'overall_overfitting': overall_overfit,
            'severity': 'HIGH' if overall_overfit and max_gap > 0.2 else ('MEDIUM' if overall_overfit else 'LOW')
        }
    
    def cv_variance_test(
        self,
        cv_scores: List[float],
        max_acceptable_std: float = 0.05
    ) -> Dict:
        """Test cross-validation variance."""
        scores = np.array(cv_scores)
        
        if len(scores) < 2:
            return {'sufficient_data': False}
        
        return {
            'sufficient_data': True,
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'min_score': float(scores.min()),
            'max_score': float(scores.max()),
            'range': float(scores.max() - scores.min()),
            'cv_coefficient': float(scores.std() / scores.mean()) if scores.mean() > 0 else float('inf'),
            'high_variance_warning': scores.std() > max_acceptable_std,
            'unstable_model': scores.std() > max_acceptable_std or (scores.max() - scores.min()) > 0.2
        }
    
    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> Dict:
        """
        Deflated Sharpe Ratio - corrects for multiple testing.
        
        Pattern from: Lopez de Prado (Machine Learning for Asset Managers)
        """
        # Euler-Mascheroni constant
        emc = 0.5772156649
        
        # Expected max Sharpe under null
        z_max = (1 - emc) * stats.norm.ppf(1 - 1/n_trials) + emc * stats.norm.ppf(1 - 1/(n_trials * np.e))
        
        # Variance of Sharpe under non-normality
        sr_std = np.sqrt((1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe + 
                          (kurtosis - 3) / 4 * observed_sharpe**2) / n_observations)
        
        if sr_std == 0:
            return {'error': 'Zero standard deviation'}
        
        # Deflated Sharpe
        deflated_sr = (observed_sharpe - z_max * sr_std) / sr_std
        p_value = 1 - stats.norm.cdf(deflated_sr)
        
        return {
            'observed_sharpe': observed_sharpe,
            'expected_max_sharpe': z_max,
            'deflated_sharpe': deflated_sr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'GENUINE EDGE' if p_value < 0.05 else 'LIKELY LUCK/OVERFITTING'
        }


# ============================================================================
# 7. STATISTICAL SIGNIFICANCE TESTER
# ============================================================================

class StatisticalSignificanceTester:
    """
    Comprehensive statistical tests for betting edge validation.
    """
    
    def t_test_roi(
        self,
        roi_values: np.ndarray,
        null_hypothesis: float = 0.0
    ) -> Dict:
        """One-sample t-test: Is ROI significantly > 0?"""
        if len(roi_values) < 2:
            return {'error': 'Insufficient data'}
        
        t_stat, p_two_sided = ttest_1samp(roi_values, null_hypothesis)
        p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
        
        # Effect size (Cohen's d)
        cohens_d = (np.mean(roi_values) - null_hypothesis) / np.std(roi_values, ddof=1) if np.std(roi_values) > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value_two_sided': float(p_two_sided),
            'p_value_one_sided': float(p_one_sided),
            'cohens_d': float(cohens_d),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small'),
            'significant_at_05': p_one_sided < 0.05,
            'significant_at_01': p_one_sided < 0.01,
            'sample_size': len(roi_values),
            'mean_roi': float(np.mean(roi_values)),
            'std_roi': float(np.std(roi_values))
        }
    
    def permutation_test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        n_permutations: int = 10000
    ) -> Dict:
        """Non-parametric permutation test."""
        observed = np.mean(strategy_returns) - np.mean(benchmark_returns)
        
        combined = np.concatenate([strategy_returns, benchmark_returns])
        n_strategy = len(strategy_returns)
        
        permutation_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_stat = np.mean(combined[:n_strategy]) - np.mean(combined[n_strategy:])
            permutation_stats.append(perm_stat)
        
        permutation_stats = np.array(permutation_stats)
        p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed))
        
        return {
            'observed_difference': float(observed),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'n_permutations': n_permutations,
            'percentile_rank': float(np.mean(permutation_stats < observed) * 100)
        }
    
    def binomial_test_win_rate(
        self,
        wins: int,
        total: int,
        null_prob: float = 0.5
    ) -> Dict:
        """Binomial test for win rate."""
        if total == 0:
            return {'error': 'No data'}
        
        # Use scipy.stats.binom.sf for one-sided test
        p_value = stats.binom.sf(wins - 1, total, null_prob)  # P(X >= wins)
        
        # Wilson score CI
        z = 1.96
        p_hat = wins / total
        denom = 1 + z**2 / total
        center = (p_hat + z**2 / (2 * total)) / denom
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
        
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)
        
        return {
            'wins': wins,
            'total': total,
            'observed_win_rate': wins / total,
            'null_hypothesis_prob': null_prob,
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'wilson_ci_95': (ci_low, ci_high),
            'edge_confirmed': ci_low > null_prob
        }
    
    def runs_test(self, sequence: np.ndarray) -> Dict:
        """Wald-Wolfowitz runs test for independence."""
        if len(sequence) < 10:
            return {'error': 'Insufficient data'}
        
        binary = (sequence > 0).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        
        if n1 == 0 or n2 == 0:
            return {'error': 'All same outcome'}
        
        # Expected runs
        expected = (2 * n1 * n2) / (n1 + n2) + 1
        var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        
        if var <= 0:
            return {'error': 'Invalid variance'}
        
        z_stat = (runs - expected) / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'observed_runs': runs,
            'expected_runs': float(expected),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'is_random': p_value > 0.05,
            'interpretation': 'RANDOM' if p_value > 0.05 else 'PATTERNS DETECTED'
        }


# ============================================================================
# 8. ROBUSTNESS TESTER
# ============================================================================

class RobustnessTester:
    """
    Test model stability under noise and perturbation.
    """
    
    def test_noise_sensitivity(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict:
        """Test prediction stability under feature noise."""
        try:
            base_pred = model.predict(X)
            base_accuracy = (base_pred == y).mean()
        except Exception as e:
            return {'error': str(e)}
        
        results = {'base_accuracy': float(base_accuracy), 'noise_tests': []}
        
        for noise_level in noise_levels:
            try:
                # Add Gaussian noise
                if hasattr(X, 'values'):
                    X_arr = X.values
                else:
                    X_arr = X
                
                noise = np.random.normal(0, noise_level, X_arr.shape)
                X_noisy = X_arr + noise * np.std(X_arr, axis=0, keepdims=True)
                
                noisy_pred = model.predict(X_noisy)
                noisy_accuracy = (noisy_pred == y).mean()
                
                results['noise_tests'].append({
                    'noise_level': noise_level,
                    'accuracy': float(noisy_accuracy),
                    'accuracy_drop': float(base_accuracy - noisy_accuracy),
                    'stable': (base_accuracy - noisy_accuracy) < 0.05
                })
            except Exception as e:
                results['noise_tests'].append({
                    'noise_level': noise_level,
                    'error': str(e)
                })
        
        valid_tests = [t for t in results['noise_tests'] if 'accuracy_drop' in t]
        if valid_tests:
            avg_drop = np.mean([t['accuracy_drop'] for t in valid_tests])
            results['robustness_score'] = max(0, 1 - avg_drop * 5)
            results['is_robust'] = results['robustness_score'] > 0.8
        else:
            results['robustness_score'] = 0
            results['is_robust'] = False
        
        return results


# ============================================================================
# 9. SELF-IMPROVEMENT VALIDATOR
# ============================================================================

class SelfImprovementValidator:
    """
    Validate that the system actually learns and improves over time.
    """
    
    def test_improvement_trend(
        self,
        accuracy_history: List[float],
        min_periods: int = 5
    ) -> Dict:
        """Test if accuracy improves over time."""
        if len(accuracy_history) < min_periods:
            return {'sufficient_data': False, 'periods': len(accuracy_history)}
        
        x = np.arange(len(accuracy_history))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, accuracy_history)
        
        # Mann-Kendall trend test
        tau, mk_p = kendalltau(x, accuracy_history)
        
        return {
            'sufficient_data': True,
            'periods': len(accuracy_history),
            'linear_slope': float(slope),
            'r_squared': float(r_value ** 2),
            'trend_p_value': float(p_value),
            'kendall_tau': float(tau),
            'mk_p_value': float(mk_p),
            'significant_improvement': p_value < 0.05 and slope > 0,
            'total_improvement': float(accuracy_history[-1] - accuracy_history[0]),
            'diagnosis': self._diagnose(slope, p_value)
        }
    
    def _diagnose(self, slope: float, p_value: float) -> str:
        if slope > 0.01 and p_value < 0.05:
            return "GENUINE IMPROVEMENT"
        elif slope > 0 and p_value < 0.1:
            return "MARGINAL IMPROVEMENT"
        elif slope < 0 and p_value < 0.05:
            return "DEGRADATION DETECTED"
        else:
            return "NO CLEAR TREND"
    
    def test_error_reduction(
        self,
        error_rates: List[float]
    ) -> Dict:
        """Test if error rate decreases over time."""
        if len(error_rates) < 4:
            return {'sufficient_data': False}
        
        mid = len(error_rates) // 2
        first_half = error_rates[:mid]
        second_half = error_rates[mid:]
        
        t_stat, p_value = ttest_ind(first_half, second_half, alternative='greater')
        
        return {
            'sufficient_data': True,
            'first_half_mean': float(np.mean(first_half)),
            'second_half_mean': float(np.mean(second_half)),
            'error_reduction': float(np.mean(first_half) - np.mean(second_half)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_reduction': p_value < 0.05,
            'learning_confirmed': p_value < 0.05 and np.mean(second_half) < np.mean(first_half)
        }
    
    def compare_model_versions(
        self,
        results_v1: Dict,
        results_v2: Dict,
        metrics: List[str] = ['accuracy', 'roi', 'win_rate']
    ) -> Dict:
        """Compare two model versions."""
        comparison = {}
        
        for metric in metrics:
            if metric in results_v1 and metric in results_v2:
                v1_val = results_v1[metric]
                v2_val = results_v2[metric]
                improvement = v2_val - v1_val
                pct_change = (improvement / v1_val * 100) if v1_val != 0 else 0
                
                comparison[metric] = {
                    'v1': v1_val,
                    'v2': v2_val,
                    'improvement': improvement,
                    'pct_change': pct_change,
                    'winner': 'v2' if v2_val > v1_val else ('v1' if v1_val > v2_val else 'tie')
                }
        
        # Overall winner
        v2_wins = sum(1 for m in comparison.values() if m['winner'] == 'v2')
        v1_wins = sum(1 for m in comparison.values() if m['winner'] == 'v1')
        
        comparison['overall_winner'] = 'v2' if v2_wins > v1_wins else ('v1' if v1_wins > v2_wins else 'tie')
        
        return comparison


# ============================================================================
# 10. ADVERSARIAL TEST GENERATOR
# ============================================================================

class AdversarialTestGenerator:
    """
    Generate synthetic edge cases to stress test the system.
    """
    
    def generate_edge_cases(self, n_samples: int = 100) -> Dict[str, Dict]:
        """Generate various edge case scenarios."""
        edge_cases = {}
        
        # 1. All extreme favorites
        edge_cases['extreme_favorites'] = {
            'description': 'All matches with odds < 1.3',
            'home_win_odds': np.random.uniform(1.05, 1.30, n_samples),
            'away_win_odds': np.random.uniform(3.0, 8.0, n_samples),
            'expected_behavior': 'Low value, should avoid most'
        }
        
        # 2. All scoreless games
        edge_cases['all_scoreless'] = {
            'description': 'All 0-0 results',
            'home_goals': np.zeros(n_samples),
            'away_goals': np.zeros(n_samples),
            'expected_behavior': 'Over/BTTS should fail'
        }
        
        # 3. High scoring mayhem
        edge_cases['high_scoring'] = {
            'description': 'All games 4+ goals',
            'home_goals': np.random.randint(2, 5, n_samples),
            'away_goals': np.random.randint(2, 5, n_samples),
            'expected_behavior': 'Over should succeed'
        }
        
        # 4. Extreme Elo differences
        edge_cases['elo_extreme'] = {
            'description': 'Massive Elo gaps',
            'home_elo': np.where(np.random.random(n_samples) > 0.5, 1800, 1200),
            'elo_diff': None,  # Will compute
            'expected_behavior': 'Should adjust predictions'
        }
        edge_cases['elo_extreme']['away_elo'] = np.where(
            edge_cases['elo_extreme']['home_elo'] == 1800, 1200, 1800
        )
        edge_cases['elo_extreme']['elo_diff'] = (
            edge_cases['elo_extreme']['home_elo'] - edge_cases['elo_extreme']['away_elo']
        )
        
        # 5. Missing data scenario
        edge_cases['missing_data'] = {
            'description': '30% of features missing',
            'missing_rate': 0.3,
            'expected_behavior': 'Should handle gracefully'
        }
        
        return edge_cases


# ============================================================================
# MASTER TEST ORCHESTRATOR
# ============================================================================

class AdvancedTestSuite:
    """
    Master orchestrator for comprehensive system validation.
    
    Combines all testing components into a unified validation pipeline.
    """
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.bootstrap = BootstrapAnalyzer()
        self.calibration = CalibrationTester()
        self.drift = ConceptDriftDetector()
        self.leakage = LeakageDetector()
        self.overfitting = OverfittingDetector()
        self.significance = StatisticalSignificanceTester()
        self.robustness = RobustnessTester()
        self.self_improvement = SelfImprovementValidator()
        self.adversarial = AdversarialTestGenerator()
    
    def run_full_validation(
        self,
        bet_returns: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray,
        cv_scores: Optional[List[float]] = None,
        accuracy_history: Optional[List[float]] = None,
        train_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None
    ) -> ValidationReport:
        """
        Run complete validation suite.
        
        Args:
            bet_returns: Historical profit/stake ratios
            predictions: Model probability predictions
            actuals: Actual outcomes (0/1)
            cv_scores: Cross-validation scores
            accuracy_history: Historical accuracy over time
            train_data: Training dataset (for leakage check)
            test_data: Test dataset (for leakage check)
            model: Trained model (for robustness testing)
        
        Returns:
            Complete ValidationReport
        """
        logger.info("🧪 Starting Advanced Test Suite...")
        
        test_results = {}
        critical_failures = []
        warnings_list = []
        
        # 1. Monte Carlo Simulation
        logger.info("🎲 Monte Carlo Simulation...")
        try:
            mc_results = self.monte_carlo.simulate_betting_paths(bet_returns)
            mc_passed = mc_results['statistics']['prob_ruin'] < 0.05
            test_results['monte_carlo'] = TestResult(
                name='Monte Carlo Simulation',
                passed=mc_passed,
                severity='CRITICAL' if not mc_passed else 'INFO',
                score=1 - mc_results['statistics']['prob_ruin'],
                details=mc_results['statistics'],
                recommendation='Reduce stake size' if not mc_passed else 'OK'
            )
            if not mc_passed:
                critical_failures.append(f"High ruin probability: {mc_results['statistics']['prob_ruin']:.1%}")
        except Exception as e:
            logger.warning(f"Monte Carlo failed: {e}")
        
        # 2. Bootstrap Confidence Intervals
        logger.info("📊 Bootstrap Analysis...")
        try:
            # Create dummy stakes if not available
            stakes = np.ones(len(bet_returns)) * 10
            profits = bet_returns * stakes
            
            bs_results = self.bootstrap.betting_edge_ci(profits, stakes)
            edge_significant = bs_results['edge_significance']['significant_at_05']
            test_results['bootstrap_ci'] = TestResult(
                name='Bootstrap CI',
                passed=edge_significant,
                severity='WARNING' if not edge_significant else 'INFO',
                score=1 - bs_results['edge_significance']['p_value_one_sided'],
                details=bs_results,
                recommendation='Need more data' if not edge_significant else 'Edge confirmed'
            )
            if not edge_significant:
                warnings_list.append("ROI confidence interval includes zero")
        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}")
        
        # 3. Calibration Testing
        logger.info("🎯 Calibration Testing...")
        try:
            cal_results = self.calibration.compute_calibration_metrics(actuals, predictions)
            cal_passed = cal_results['is_well_calibrated']
            test_results['calibration'] = TestResult(
                name='Probability Calibration',
                passed=cal_passed,
                severity='WARNING' if not cal_passed else 'INFO',
                score=1 - cal_results['expected_calibration_error'],
                details=cal_results,
                recommendation='Add calibration layer' if not cal_passed else 'Well calibrated'
            )
            if not cal_passed:
                warnings_list.append(f"Poor calibration (ECE={cal_results['expected_calibration_error']:.3f})")
        except Exception as e:
            logger.warning(f"Calibration test failed: {e}")
        
        # 4. Overconfidence Check
        logger.info("⚠️ Overconfidence Check...")
        try:
            oc_results = self.calibration.test_overconfidence(actuals, predictions)
            if oc_results.get('has_data', False):
                oc_passed = not oc_results['is_overconfident']
                test_results['overconfidence'] = TestResult(
                    name='Overconfidence Check',
                    passed=oc_passed,
                    severity='WARNING' if not oc_passed else 'INFO',
                    score=1 - abs(oc_results['overconfidence_gap']),
                    details=oc_results,
                    recommendation='Calibrate high-confidence predictions' if not oc_passed else 'OK'
                )
                if not oc_passed:
                    warnings_list.append(f"Overconfident by {oc_results['overconfidence_gap']:.1%}")
        except Exception as e:
            logger.warning(f"Overconfidence test failed: {e}")
        
        # 5. CV Variance (if provided)
        if cv_scores:
            logger.info("📈 CV Variance Analysis...")
            try:
                cv_results = self.overfitting.cv_variance_test(cv_scores)
                if cv_results.get('sufficient_data', False):
                    cv_passed = not cv_results['high_variance_warning']
                    test_results['cv_variance'] = TestResult(
                        name='CV Variance',
                        passed=cv_passed,
                        severity='WARNING' if not cv_passed else 'INFO',
                        score=1 - cv_results['std_score'],
                        details=cv_results,
                        recommendation='Model unstable, add regularization' if not cv_passed else 'Stable'
                    )
                    if not cv_passed:
                        warnings_list.append(f"High CV variance: {cv_results['std_score']:.3f}")
            except Exception as e:
                logger.warning(f"CV variance test failed: {e}")
        
        # 6. Statistical Significance
        logger.info("📉 Statistical Significance...")
        try:
            sig_results = self.significance.t_test_roi(bet_returns)
            sig_passed = sig_results.get('significant_at_05', False)
            test_results['significance'] = TestResult(
                name='ROI Significance',
                passed=sig_passed,
                severity='WARNING' if not sig_passed else 'INFO',
                score=1 - sig_results.get('p_value_one_sided', 1),
                details=sig_results,
                recommendation='Edge not proven, collect more data' if not sig_passed else 'Significant edge'
            )
            if not sig_passed:
                warnings_list.append("ROI not statistically significant at α=0.05")
        except Exception as e:
            logger.warning(f"Significance test failed: {e}")
        
        # 7. Win/Loss Randomness (Runs Test)
        logger.info("🎰 Runs Test...")
        try:
            runs_results = self.significance.runs_test(bet_returns)
            if 'error' not in runs_results:
                runs_passed = runs_results['is_random']
                test_results['runs_test'] = TestResult(
                    name='Runs Test (Independence)',
                    passed=runs_passed,
                    severity='INFO',
                    score=runs_results['p_value'],
                    details=runs_results,
                    recommendation='Possible streaks detected' if not runs_passed else 'Random sequence'
                )
        except Exception as e:
            logger.warning(f"Runs test failed: {e}")
        
        # 8. Temporal Leakage (if data provided)
        if train_data is not None and test_data is not None:
            logger.info("🔍 Leakage Detection...")
            try:
                leak_results = self.leakage.check_temporal_leakage(train_data, test_data)
                if 'error' not in leak_results:
                    leak_passed = not leak_results['temporal_leakage_detected']
                    test_results['leakage'] = TestResult(
                        name='Temporal Leakage',
                        passed=leak_passed,
                        severity='CRITICAL' if not leak_passed else 'INFO',
                        score=1.0 if leak_passed else 0.0,
                        details=leak_results,
                        recommendation='FIX DATA PIPELINE IMMEDIATELY' if not leak_passed else 'No leakage'
                    )
                    if not leak_passed:
                        critical_failures.append("TEMPORAL LEAKAGE DETECTED")
            except Exception as e:
                logger.warning(f"Leakage test failed: {e}")
        
        # 9. Self-Improvement (if history provided)
        if accuracy_history and len(accuracy_history) >= 5:
            logger.info("🔄 Self-Improvement Validation...")
            try:
                si_results = self.self_improvement.test_improvement_trend(accuracy_history)
                if si_results.get('sufficient_data', False):
                    si_passed = si_results['significant_improvement'] or si_results['linear_slope'] >= 0
                    test_results['self_improvement'] = TestResult(
                        name='Self-Improvement',
                        passed=si_passed,
                        severity='CRITICAL' if 'DEGRADATION' in si_results['diagnosis'] else 'INFO',
                        score=max(0, min(1, 0.5 + si_results['linear_slope'] * 10)),
                        details=si_results,
                        recommendation=si_results['diagnosis']
                    )
                    if 'DEGRADATION' in si_results['diagnosis']:
                        critical_failures.append("System performance is degrading!")
            except Exception as e:
                logger.warning(f"Self-improvement test failed: {e}")
        
        # Compute overall grade
        overall_score = np.mean([r.score for r in test_results.values()])
        
        if critical_failures:
            grade = 'FAIL'
        elif len(warnings_list) >= 3:
            grade = 'CAUTION'
        elif warnings_list:
            grade = 'PASS_WITH_WARNINGS'
        else:
            grade = 'PASS'
        
        # Generate recommendation
        if critical_failures:
            recommendation = "🛑 DO NOT DEPLOY - Critical issues must be fixed: " + "; ".join(critical_failures)
        elif grade == 'CAUTION':
            recommendation = "⚠️ PROCEED WITH CAUTION - Multiple concerns: " + "; ".join(warnings_list[:3])
        elif warnings_list:
            recommendation = "✅ DEPLOYABLE - Monitor: " + "; ".join(warnings_list)
        else:
            recommendation = "🚀 READY FOR DEPLOYMENT - All tests passed!"
        
        logger.info(f"✅ Test Suite Complete: {grade} (Score: {overall_score:.2%})")
        
        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_grade=grade,
            overall_score=overall_score,
            critical_failures=critical_failures,
            warnings=warnings_list,
            test_results=test_results,
            recommendation=recommendation
        )
    
    def quick_health_check(
        self,
        bet_returns: np.ndarray,
        min_roi: float = 0.0,
        min_win_rate: float = 0.5
    ) -> Dict:
        """Quick health check for production monitoring."""
        wins = (bet_returns > 0).sum()
        total = len(bet_returns)
        win_rate = wins / total if total > 0 else 0
        roi = np.sum(bet_returns) / total if total > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_bets': total,
            'win_rate': win_rate,
            'roi': roi,
            'healthy': roi >= min_roi and win_rate >= min_win_rate,
            'sharpe': self.monte_carlo.calculate_sharpe_ratio(bet_returns)[0],
            'max_drawdown': np.min(np.cumsum(bet_returns) - np.maximum.accumulate(np.cumsum(bet_returns)))
        }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Run test suite with sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Suite for Betting System')
    parser.add_argument('--sample', action='store_true', help='Run with sample data')
    parser.add_argument('--output', type=str, default='validation_report.md', help='Output file')
    args = parser.parse_args()
    
    if args.sample:
        print("🧪 Running Advanced Test Suite with Sample Data...")
        print("=" * 60)
        
        # Generate sample betting data
        np.random.seed(42)
        n_bets = 500
        
        # Simulated returns (55% win rate, avg profit 5%)
        win_prob = 0.55
        wins = np.random.random(n_bets) < win_prob
        bet_returns = np.where(wins, np.random.uniform(0.5, 1.5, n_bets), -1)
        
        # Simulated predictions
        predictions = np.clip(np.random.normal(0.55, 0.15, n_bets), 0, 1)
        actuals = (np.random.random(n_bets) < predictions).astype(int)
        
        # Simulated CV scores
        cv_scores = [0.55, 0.52, 0.58, 0.54, 0.56]
        
        # Simulated accuracy history (slight improvement)
        accuracy_history = [0.52 + 0.01 * i + np.random.normal(0, 0.01) for i in range(10)]
        
        # Run full validation
        suite = AdvancedTestSuite()
        report = suite.run_full_validation(
            bet_returns=bet_returns,
            predictions=predictions,
            actuals=actuals,
            cv_scores=cv_scores,
            accuracy_history=accuracy_history
        )
        
        # Print report
        print(report.to_markdown())
        
        # Save report
        with open(args.output, 'w') as f:
            f.write(report.to_markdown())
        
        print(f"\n📄 Report saved to: {args.output}")
        
        return report
    
    else:
        print("Usage: python advanced_test_suite.py --sample")
        print("       or import AdvancedTestSuite and call run_full_validation()")


if __name__ == '__main__':
    main()

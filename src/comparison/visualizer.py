"""
📊 Comparison Visualizer - Professional Betting Analytics Charts
================================================================

Publication-quality visualizations for prediction analysis:
- Reliability Diagrams (calibration curves)
- Equity Curves with drawdown
- CLV Distribution histograms
- Error Heatmaps by market/league
- ROI over time with confidence bands

Based on FiveThirtyEight and professional quant fund visualizations.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - visualizations will be text-only")


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8-whitegrid'
    color_primary: str = '#1f77b4'
    color_secondary: str = '#ff7f0e'
    color_positive: str = '#2ca02c'
    color_negative: str = '#d62728'
    font_size: int = 12
    save_path: str = 'reports/visualizations'


class ComparisonVisualizer:
    """
    Generate professional visualizations for betting analysis.
    
    Features:
    - Reliability diagrams (calibration curves)
    - Equity curves with drawdown overlay
    - CLV distribution histograms
    - Market/League performance heatmaps
    - Rolling ROI with confidence bands
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(self.config.style)
            except OSError:
                plt.style.use('seaborn-v0_8')
    
    def reliability_diagram(
        self,
        predictions: List[float],
        outcomes: List[bool],
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create calibration curve (reliability diagram).
        
        Shows predicted probability vs actual frequency.
        Perfect calibration = diagonal line.
        """
        if not HAS_MATPLOTLIB:
            return self._text_reliability(predictions, outcomes, n_bins)
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (np.array(predictions) >= bin_edges[i]) & (np.array(predictions) < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(np.array(outcomes)[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Plot actual calibration
        valid_mask = ~np.isnan(bin_accuracies)
        ax.plot(
            np.array(bin_centers)[valid_mask],
            np.array(bin_accuracies)[valid_mask],
            'o-',
            color=self.config.color_primary,
            markersize=8,
            linewidth=2,
            label='Model Calibration'
        )
        
        # Add histogram at bottom
        ax2 = ax.twinx()
        ax2.bar(
            bin_centers,
            bin_counts,
            width=0.08,
            alpha=0.3,
            color=self.config.color_secondary,
            label='Sample Count'
        )
        ax2.set_ylabel('Sample Count', fontsize=self.config.font_size)
        
        # Labels
        ax.set_xlabel('Predicted Probability', fontsize=self.config.font_size)
        ax.set_ylabel('Actual Frequency', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        
        # Add ECE annotation
        ece = self._calculate_ece(predictions, outcomes, n_bins)
        ax.annotate(
            f'ECE: {ece:.3f}',
            xy=(0.05, 0.95),
            fontsize=self.config.font_size,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved reliability diagram to {save_path}")
        
        plt.close()
        return save_path
    
    def equity_curve(
        self,
        pnls: List[float],
        dates: Optional[List[str]] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create equity curve with drawdown overlay.
        
        Shows cumulative PnL over time with underwater equity.
        """
        if not HAS_MATPLOTLIB:
            return self._text_equity_curve(pnls)
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(self.config.figsize[0], self.config.figsize[1]),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
        
        # Cumulative PnL
        cumulative = np.cumsum(pnls)
        x = range(len(cumulative))
        
        # Color based on final result
        color = self.config.color_positive if cumulative[-1] >= 0 else self.config.color_negative
        
        ax1.fill_between(x, 0, cumulative, alpha=0.3, color=color)
        ax1.plot(x, cumulative, color=color, linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Cumulative PnL (units)', fontsize=self.config.font_size)
        ax1.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        
        # Add stats annotation
        stats_text = f"Total: {cumulative[-1]:+.2f}\nMax: {max(cumulative):+.2f}\nMin: {min(cumulative):+.2f}"
        ax1.annotate(
            stats_text,
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            verticalalignment='top',
            fontsize=self.config.font_size - 1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        ax2.fill_between(x, drawdown, 0, color=self.config.color_negative, alpha=0.5)
        ax2.set_ylabel('Drawdown', fontsize=self.config.font_size)
        ax2.set_xlabel('Bet Number', fontsize=self.config.font_size)
        
        # Max drawdown annotation
        max_dd = min(drawdown)
        ax2.annotate(
            f'Max DD: {max_dd:.2f}',
            xy=(np.argmin(drawdown), max_dd),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=self.config.font_size - 1,
            arrowprops=dict(arrowstyle='->', color='red')
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")
        
        plt.close()
        return save_path
    
    def clv_distribution(
        self,
        clvs: List[float],
        title: str = "CLV Distribution",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create CLV distribution histogram.
        
        Shows distribution of Closing Line Value with benchmark lines.
        """
        if not HAS_MATPLOTLIB:
            return self._text_clv_distribution(clvs)
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Histogram
        n, bins, patches = ax.hist(
            clvs,
            bins=30,
            edgecolor='white',
            alpha=0.7
        )
        
        # Color bars by positive/negative
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor(self.config.color_positive)
            else:
                patch.set_facecolor(self.config.color_negative)
        
        # Add benchmark lines
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Break-even')
        ax.axvline(x=0.03, color='gold', linestyle='--', linewidth=1.5, label='Elite (+3%)')
        ax.axvline(x=0.015, color='silver', linestyle='--', linewidth=1.5, label='Professional (+1.5%)')
        
        # Average line
        avg_clv = np.mean(clvs)
        ax.axvline(x=avg_clv, color='blue', linestyle=':', linewidth=2, label=f'Your Avg ({avg_clv:+.1%})')
        
        ax.set_xlabel('CLV (Closing Line Value)', fontsize=self.config.font_size)
        ax.set_ylabel('Frequency', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Stats annotation
        positive_rate = sum(1 for c in clvs if c > 0) / len(clvs)
        stats_text = f"N: {len(clvs)}\nAvg: {avg_clv:+.2%}\nCLV+ Rate: {positive_rate:.0%}"
        ax.annotate(
            stats_text,
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            verticalalignment='top',
            fontsize=self.config.font_size - 1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved CLV distribution to {save_path}")
        
        plt.close()
        return save_path
    
    def market_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        metric_name: str = "ROI",
        title: str = "Performance Heatmap",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create heatmap of performance by market and league.
        
        Args:
            data: Dict[market][league] = value
        """
        if not HAS_MATPLOTLIB:
            return self._text_heatmap(data, metric_name)
        
        # Convert to matrix
        markets = sorted(data.keys())
        leagues = sorted(set(l for m in data.values() for l in m.keys()))
        
        matrix = np.zeros((len(markets), len(leagues)))
        for i, market in enumerate(markets):
            for j, league in enumerate(leagues):
                matrix[i, j] = data.get(market, {}).get(league, np.nan)
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(leagues)))
        ax.set_yticks(np.arange(len(markets)))
        ax.set_xticklabels(leagues, rotation=45, ha='right')
        ax.set_yticklabels(markets)
        
        # Add text annotations
        for i in range(len(markets)):
            for j in range(len(leagues)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(
                        j, i, f'{matrix[i, j]:.1%}',
                        ha='center', va='center',
                        color='white' if abs(matrix[i, j]) > 0.1 else 'black'
                    )
        
        ax.set_title(f'{title} ({metric_name})', fontsize=self.config.font_size + 2, fontweight='bold')
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric_name, rotation=-90, va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        plt.close()
        return save_path
    
    def rolling_roi(
        self,
        pnls: List[float],
        window: int = 50,
        title: str = "Rolling ROI",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create rolling ROI chart with confidence bands.
        """
        if not HAS_MATPLOTLIB:
            return self._text_rolling_roi(pnls, window)
        
        if len(pnls) < window:
            logger.warning(f"Not enough data for rolling ROI (need {window}, have {len(pnls)})")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Calculate rolling ROI
        rolling_roi = []
        rolling_std = []
        
        for i in range(window, len(pnls) + 1):
            window_pnls = pnls[i-window:i]
            rolling_roi.append(np.mean(window_pnls))
            rolling_std.append(np.std(window_pnls))
        
        x = range(window, len(pnls) + 1)
        rolling_roi = np.array(rolling_roi)
        rolling_std = np.array(rolling_std)
        
        # Plot with confidence band
        ax.fill_between(
            x,
            rolling_roi - rolling_std,
            rolling_roi + rolling_std,
            alpha=0.3,
            color=self.config.color_primary,
            label='±1 Std Dev'
        )
        ax.plot(x, rolling_roi, color=self.config.color_primary, linewidth=2, label=f'Rolling {window}-bet ROI')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Break-even')
        
        # Color fill above/below zero
        ax.fill_between(x, rolling_roi, 0, where=rolling_roi > 0, alpha=0.1, color='green')
        ax.fill_between(x, rolling_roi, 0, where=rolling_roi < 0, alpha=0.1, color='red')
        
        ax.set_xlabel('Bet Number', fontsize=self.config.font_size)
        ax.set_ylabel('ROI', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.legend(loc='upper right')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved rolling ROI to {save_path}")
        
        plt.close()
        return save_path
    
    def win_rate_by_odds(
        self,
        odds: List[float],
        outcomes: List[bool],
        n_bins: int = 8,
        title: str = "Win Rate vs Implied Probability",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Show win rate compared to implied probability from odds.
        """
        if not HAS_MATPLOTLIB:
            return self._text_win_rate_by_odds(odds, outcomes, n_bins)
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Convert odds to implied probability
        implied_probs = [1/o for o in odds]
        
        # Bin by implied probability
        bin_edges = np.linspace(min(implied_probs), max(implied_probs), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        actual_rates = []
        implied_rates = []
        counts = []
        
        for i in range(n_bins):
            mask = (np.array(implied_probs) >= bin_edges[i]) & (np.array(implied_probs) < bin_edges[i+1])
            if np.sum(mask) > 0:
                actual_rates.append(np.mean(np.array(outcomes)[mask]))
                implied_rates.append(bin_centers[i])
                counts.append(np.sum(mask))
            else:
                actual_rates.append(np.nan)
                implied_rates.append(bin_centers[i])
                counts.append(0)
        
        # Plot
        width = (bin_edges[1] - bin_edges[0]) * 0.4
        
        ax.bar(
            bin_centers - width/2,
            implied_rates,
            width=width,
            label='Implied (Odds)',
            alpha=0.7,
            color=self.config.color_secondary
        )
        ax.bar(
            bin_centers + width/2,
            actual_rates,
            width=width,
            label='Actual Win Rate',
            alpha=0.7,
            color=self.config.color_primary
        )
        
        ax.set_xlabel('Implied Probability Bucket', fontsize=self.config.font_size)
        ax.set_ylabel('Rate', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.legend()
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved win rate chart to {save_path}")
        
        plt.close()
        return save_path
    
    # =========================================================================
    # Text-only fallbacks
    # =========================================================================
    
    def _text_reliability(self, predictions: List[float], outcomes: List[bool], n_bins: int) -> str:
        """Text-based reliability diagram"""
        ece = self._calculate_ece(predictions, outcomes, n_bins)
        
        output = "\n📊 RELIABILITY DIAGRAM (Text)\n" + "="*40 + "\n"
        output += f"ECE: {ece:.4f}\n\n"
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            mask = (np.array(predictions) >= bin_edges[i]) & (np.array(predictions) < bin_edges[i+1])
            count = np.sum(mask)
            if count > 0:
                actual = np.mean(np.array(outcomes)[mask])
                expected = (bin_edges[i] + bin_edges[i+1]) / 2
                gap = actual - expected
                bar = '█' * int(actual * 20)
                output += f"{bin_edges[i]:.0%}-{bin_edges[i+1]:.0%}: {bar} {actual:.0%} (n={count}, gap={gap:+.0%})\n"
        
        return output
    
    def _text_equity_curve(self, pnls: List[float]) -> str:
        """Text-based equity curve"""
        cumulative = np.cumsum(pnls)
        
        output = "\n📈 EQUITY CURVE (Text)\n" + "="*40 + "\n"
        output += f"Total PnL: {cumulative[-1]:+.2f}\n"
        output += f"Max: {max(cumulative):+.2f}\n"
        output += f"Min: {min(cumulative):+.2f}\n"
        output += f"Max Drawdown: {min(cumulative - np.maximum.accumulate(cumulative)):.2f}\n"
        
        return output
    
    def _text_clv_distribution(self, clvs: List[float]) -> str:
        """Text-based CLV distribution"""
        output = "\n📊 CLV DISTRIBUTION (Text)\n" + "="*40 + "\n"
        output += f"N: {len(clvs)}\n"
        output += f"Mean: {np.mean(clvs):+.2%}\n"
        output += f"Median: {np.median(clvs):+.2%}\n"
        output += f"Std: {np.std(clvs):.2%}\n"
        output += f"CLV+ Rate: {sum(1 for c in clvs if c > 0) / len(clvs):.0%}\n"
        
        return output
    
    def _text_heatmap(self, data: Dict, metric_name: str) -> str:
        """Text-based heatmap"""
        output = f"\n📊 {metric_name} HEATMAP (Text)\n" + "="*40 + "\n"
        
        for market, leagues in data.items():
            output += f"\n{market}:\n"
            for league, value in sorted(leagues.items(), key=lambda x: -x[1]):
                bar = '█' * int(abs(value) * 100)
                sign = '+' if value >= 0 else '-'
                output += f"  {league}: {sign}{abs(value):.1%} {bar}\n"
        
        return output
    
    def _text_rolling_roi(self, pnls: List[float], window: int) -> str:
        """Text-based rolling ROI"""
        output = f"\n📈 ROLLING {window}-BET ROI (Text)\n" + "="*40 + "\n"
        
        if len(pnls) < window:
            return output + "Insufficient data\n"
        
        rois = [np.mean(pnls[i-window:i]) for i in range(window, len(pnls) + 1)]
        
        output += f"Current: {rois[-1]:+.2%}\n"
        output += f"Max: {max(rois):+.2%}\n"
        output += f"Min: {min(rois):+.2%}\n"
        output += f"Avg: {np.mean(rois):+.2%}\n"
        
        return output
    
    def _text_win_rate_by_odds(self, odds: List[float], outcomes: List[bool], n_bins: int) -> str:
        """Text-based win rate by odds"""
        output = "\n📊 WIN RATE VS ODDS (Text)\n" + "="*40 + "\n"
        
        implied = [1/o for o in odds]
        bin_edges = np.linspace(min(implied), max(implied), n_bins + 1)
        
        for i in range(n_bins):
            mask = (np.array(implied) >= bin_edges[i]) & (np.array(implied) < bin_edges[i+1])
            if np.sum(mask) > 0:
                actual = np.mean(np.array(outcomes)[mask])
                expected = (bin_edges[i] + bin_edges[i+1]) / 2
                edge = actual - expected
                output += f"{bin_edges[i]:.0%}-{bin_edges[i+1]:.0%}: Actual={actual:.0%} vs Implied={expected:.0%} (edge={edge:+.0%})\n"
        
        return output
    
    @staticmethod
    def _calculate_ece(predictions: List[float], outcomes: List[bool], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        total = len(predictions)
        
        for i in range(n_bins):
            mask = (np.array(predictions) >= bin_edges[i]) & (np.array(predictions) < bin_edges[i+1])
            count = np.sum(mask)
            if count > 0:
                actual = np.mean(np.array(outcomes)[mask])
                predicted = np.mean(np.array(predictions)[mask])
                ece += (count / total) * abs(actual - predicted)
        
        return ece


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test visualizer with sample data"""
    viz = ComparisonVisualizer()
    
    # Sample data
    predictions = np.random.uniform(0.5, 0.9, 100).tolist()
    outcomes = [np.random.random() < p for p in predictions]
    pnls = np.random.normal(0.02, 0.5, 100).tolist()
    clvs = np.random.normal(0.01, 0.03, 100).tolist()
    
    # Generate visualizations
    print(viz.reliability_diagram(predictions, outcomes))
    print(viz.equity_curve(pnls))
    print(viz.clv_distribution(clvs))
    
    # Heatmap data
    heatmap_data = {
        'over_1_5': {'Premier League': 0.05, 'Bundesliga': 0.08, 'La Liga': -0.02},
        'over_2_5': {'Premier League': -0.03, 'Bundesliga': 0.04, 'La Liga': 0.01},
        'btts': {'Premier League': 0.02, 'Bundesliga': 0.06, 'La Liga': -0.01}
    }
    print(viz.market_heatmap(heatmap_data))


if __name__ == '__main__':
    main()

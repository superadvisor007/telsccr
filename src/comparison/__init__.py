"""
🎯 Prediction Comparison Module
===============================

Intelligent comparison of predictions vs actual results with:
- Brier Score decomposition (reliability, resolution, uncertainty)
- Closing Line Value (CLV) tracking
- Prediction Autopsy (error categorization)
- Multi-source result verification
- Professional visualizations

This module is the "feedback loop" for the self-improving betting system.
"""

from .prediction_comparator import PredictionComparator, ComparisonResult
from .result_fetcher import MultiSourceResultFetcher, MatchResult
from .prediction_autopsy import PredictionAutopsy, ErrorCategory
from .clv_tracker import CLVTracker, CLVAnalysis
from .visualizer import ComparisonVisualizer

__all__ = [
    'PredictionComparator',
    'ComparisonResult',
    'MultiSourceResultFetcher',
    'MatchResult',
    'PredictionAutopsy',
    'ErrorCategory',
    'CLVTracker',
    'CLVAnalysis',
    'ComparisonVisualizer'
]

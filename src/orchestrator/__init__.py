"""
🎯 Orchestrator Package
======================
Central pipeline orchestration for the betting system.

Battle-Tested Results (14K Walk-Forward Backtest):
- Win Rate: 77.0%
- ROI: +5.38%
- Max Drawdown: 4.9%
- Sharpe Ratio: 1.47
- Profitable Windows: 67.4%
"""

from .central_orchestrator import (
    CentralOrchestrator,
    WalkForwardEngine,
    FeatureEngineer,
    ModelRegistry,
    MatchData,
    PredictionResult,
    BettingDecision,
    BacktestResult,
)

from .battle_tested_orchestrator import (
    BattleTestedOrchestrator,
    BetLeg,
    BetTicket,
    BacktestResult as BTBacktestResult,
)

__all__ = [
    'CentralOrchestrator',
    'WalkForwardEngine',
    'FeatureEngineer',
    'ModelRegistry',
    'MatchData',
    'PredictionResult',
    'BettingDecision',
    'BacktestResult',
    'BattleTestedOrchestrator',
    'BetLeg',
    'BetTicket',
    'BTBacktestResult',
]

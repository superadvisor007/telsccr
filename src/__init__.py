"""
🎯 TELEGRAMSOCCER - AI Soccer Betting Assistant
==============================================
Goal-directed reasoning + Walk-forward validated ML models.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Ingestion    → Raw matches, odds, stats                 │
│ 2. Feature Engine    → SPADL, tactical metrics                  │
│ 3. Knowledge Base    → Team identities, priors                  │
│ 4. Goal Reasoning    → Multi-step LLM analysis (Chain-of-Thought)│
│ 5. ML Models         → Walk-forward GradientBoost               │
│ 6. Market Scoring    → Confidence + edge calculation            │
│ 7. Multi-Bet Builder → Optimal leg selection                    │
│ 8. Telegram Delivery → Bot + feedback loop                      │
└─────────────────────────────────────────────────────────────────┘

Battle-Tested Results (14K matches, 276 windows):
- 77% Win Rate
- +5.38% ROI  
- 4.9% Max Drawdown
- 1.47 Sharpe Ratio

Key Modules:
- unified_pipeline: Complete integration entry point
- reasoning: Goal-directed LLM reasoning
- orchestrator: Battle-tested walk-forward validation
- betting: Multi-bet ticket builder
- bot: Telegram delivery
"""

__version__ = '2.0.0'
__author__ = 'TelegramSoccer'

# Main entry point
from .unified_pipeline import UnifiedPipeline, UnifiedConfig

# Core components
from .reasoning import (
    GoalDirectedReasoningEngine,
    GoalDirectedAnalysis,
    SYSTEM_GOAL,
    DOMAIN_REFERENCES
)

from .orchestrator import (
    BattleTestedOrchestrator,
    BacktestResult
)

from .betting import (
    MultiBetTicketBuilder,
    EnhancedTicket,
    EnhancedBetLeg,
    TicketConfig
)

__all__ = [
    # Version
    '__version__',
    
    # Main pipeline
    'UnifiedPipeline',
    'UnifiedConfig',
    
    # Reasoning
    'GoalDirectedReasoningEngine',
    'GoalDirectedAnalysis',
    'SYSTEM_GOAL',
    'DOMAIN_REFERENCES',
    
    # Orchestrator
    'BattleTestedOrchestrator',
    'BacktestResult',
    
    # Betting
    'MultiBetTicketBuilder',
    'EnhancedTicket',
    'EnhancedBetLeg',
    'TicketConfig'
]

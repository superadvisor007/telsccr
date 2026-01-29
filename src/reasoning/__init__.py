"""
🧠 REASONING MODULE
==================
Goal-directed reasoning for betting decisions.

Components:
- goal_directed_reasoning: Hardcoded goals + multi-step LLM analysis
- match_context_collector: Comprehensive data gathering
- reasoning_agent: ReAct pattern reasoning
- schemas: Pydantic models for structured output

Battle-Tested Results:
- 77% Win Rate
- +5.38% ROI
- 1.47 Sharpe Ratio
"""

# Goal-directed reasoning (new)
from .goal_directed_reasoning import (
    SYSTEM_GOAL,
    DOMAIN_REFERENCES,
    TeamTacticalAnalysis,
    MatchScenario,
    MarketRecommendation,
    GoalDirectedAnalysis,
    GoalDirectedReasoningEngine,
    GoalDirectedOrchestrator,
    PromptTemplates,
    BetLeg as ReasoningBetLeg,
    MultiBeTicket
)

# Legacy components
from .match_context_collector import MatchContextCollector, MatchContext
from .reasoning_agent import ReasoningAgent, BettingDecision
from .schemas import ReasoningStep, ChainOfThought

__all__ = [
    # Goal-directed (new)
    'SYSTEM_GOAL',
    'DOMAIN_REFERENCES',
    'TeamTacticalAnalysis',
    'MatchScenario',
    'MarketRecommendation',
    'GoalDirectedAnalysis',
    'GoalDirectedReasoningEngine',
    'GoalDirectedOrchestrator',
    'PromptTemplates',
    'ReasoningBetLeg',
    'MultiBeTicket',
    
    # Legacy
    'MatchContextCollector',
    'MatchContext', 
    'ReasoningAgent',
    'BettingDecision',
    'ReasoningStep',
    'ChainOfThought',
]

"""
🤖 LIVING BETTING AGENT
======================
A proactive, self-improving AI betting system that thinks, learns, and adapts.

Architecture:
1. Data Collection → Scrapers, APIs, cached knowledge
2. Knowledge Base → League tendencies, market patterns, historical insights
3. Structural Reasoning → DeepSeek 7B multi-step chain-of-thought
4. Market Scoring → Probability & confidence assignment
5. Multi-Bet Builder → Optimal leg selection (1.4-1.7 odds range)
6. Delivery & Feedback → Telegram tickets + results tracking

Key Principles:
- Forward-thinking simulation (not just stat crunching)
- Curiosity prompts for hidden edges
- Persistent memory across sessions
- Multi-step reasoning chains
- Self-improving feedback loop

100% FREE: DeepSeek 7B via Ollama, no API costs.
"""

# Import components (lazy to avoid circular imports)
def get_living_agent():
    from living_agent.living_agent import LivingBettingAgent
    return LivingBettingAgent

def get_knowledge_cache():
    from living_agent.knowledge_cache import KnowledgeCache
    return KnowledgeCache

def get_scenario_simulator():
    from living_agent.scenario_simulator import ScenarioSimulator
    return ScenarioSimulator

def get_multi_bet_builder():
    from living_agent.multi_bet_builder import MultiBetBuilder
    return MultiBetBuilder

def get_feedback_system():
    from living_agent.feedback_system import FeedbackSystem
    return FeedbackSystem

__all__ = [
    'get_living_agent',
    'get_knowledge_cache',
    'get_scenario_simulator',
    'get_multi_bet_builder',
    'get_feedback_system',
]

#!/usr/bin/env python3
"""
📋 PYDANTIC SCHEMAS FOR REASONING
=================================
Structured output models for betting decisions.

Based on research patterns from:
- LangChain ReAct agents
- Microsoft AutoGen MagenticOne
- Professional betting systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class BettingMarket(Enum):
    """Supported betting markets"""
    OVER_0_5 = "over_0_5"
    OVER_1_5 = "over_1_5"
    OVER_2_5 = "over_2_5"
    OVER_3_5 = "over_3_5"
    UNDER_1_5 = "under_1_5"
    UNDER_2_5 = "under_2_5"
    UNDER_3_5 = "under_3_5"
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    HOME_OR_DRAW = "home_or_draw"
    AWAY_OR_DRAW = "away_or_draw"


class ActionType(Enum):
    """ReAct action types"""
    SEARCH_STATS = "search_team_stats"
    CHECK_FORM = "check_recent_form"
    GET_H2H = "get_head_to_head"
    GET_PSYCHOLOGY = "analyze_psychological_factors"
    CALCULATE_VALUE = "calculate_value_probability"
    CHECK_WEATHER = "check_weather_conditions"
    GET_NEWS = "get_team_news"
    FINAL_DECISION = "make_final_decision"


class Recommendation(Enum):
    """Final recommendation"""
    STRONG_BET = "STRONG_BET"
    BET = "BET"
    MONITOR = "MONITOR"
    AVOID = "AVOID"
    STRONG_AVOID = "STRONG_AVOID"


@dataclass
class ReasoningStep:
    """Single step in the Chain-of-Thought"""
    step_number: int
    thought: str
    action: ActionType
    action_input: Dict[str, Any]
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_string(self) -> str:
        """Format for display"""
        return f"""
**Step {self.step_number}**
💭 Thought: {self.thought}
🔧 Action: {self.action.value}
📥 Input: {self.action_input}
👁️ Observation: {self.observation}
"""


@dataclass
class ChainOfThought:
    """Complete Chain-of-Thought for a match"""
    match_id: str
    home_team: str
    away_team: str
    
    # The reasoning chain
    steps: List[ReasoningStep] = field(default_factory=list)
    
    # Summary
    total_steps: int = 0
    reasoning_time_ms: int = 0
    
    def add_step(self, thought: str, action: ActionType, 
                 action_input: Dict, observation: str) -> ReasoningStep:
        """Add a step to the chain"""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        self.steps.append(step)
        self.total_steps = len(self.steps)
        return step
    
    def to_string(self) -> str:
        """Format entire chain"""
        lines = [f"# 🧠 Chain-of-Thought: {self.home_team} vs {self.away_team}\n"]
        for step in self.steps:
            lines.append(step.to_string())
        lines.append(f"\n**Total Steps**: {self.total_steps} | **Time**: {self.reasoning_time_ms}ms")
        return "\n".join(lines)


@dataclass
class ValueAnalysis:
    """Value bet analysis"""
    market: BettingMarket
    our_probability: float
    market_odds: float
    implied_probability: float
    expected_value: float
    edge_percentage: float
    kelly_stake: float
    is_value: bool
    
    def to_string(self) -> str:
        return f"""
📊 **{self.market.value.upper()}**
- Our Probability: {self.our_probability:.1%}
- Market Odds: {self.market_odds:.2f} (Implied: {self.implied_probability:.1%})
- Expected Value: {self.expected_value:+.2%}
- Edge: {self.edge_percentage:+.1%}
- Kelly Stake: {self.kelly_stake:.1%}
- Value Bet: {'✅ YES' if self.is_value else '❌ NO'}
"""


@dataclass
class ContrarianCheck:
    """Devil's advocate analysis"""
    concerns: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    potential_blindspots: List[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    should_proceed: bool = True
    reasoning: str = ""
    
    def to_string(self) -> str:
        lines = ["## 😈 Contrarian Analysis (Devil's Advocate)\n"]
        
        if self.concerns:
            lines.append("### ⚠️ Concerns:")
            for c in self.concerns:
                lines.append(f"- {c}")
        
        if self.risk_factors:
            lines.append("\n### 🚨 Risk Factors:")
            for r in self.risk_factors:
                lines.append(f"- {r}")
        
        if self.potential_blindspots:
            lines.append("\n### 🔍 Potential Blindspots:")
            for b in self.potential_blindspots:
                lines.append(f"- {b}")
        
        lines.append(f"\n**Confidence Adjustment**: {self.confidence_adjustment:+.1%}")
        lines.append(f"**Proceed with Bet**: {'✅ Yes' if self.should_proceed else '❌ No'}")
        lines.append(f"\n**Reasoning**: {self.reasoning}")
        
        return "\n".join(lines)


@dataclass
class BettingDecision:
    """Complete betting decision with full reasoning"""
    # Match info
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    
    # Decision
    market: BettingMarket
    recommendation: Recommendation
    confidence_score: float  # 0-100
    confidence_stars: int  # 1-5
    
    # Probabilities
    our_probability: float
    market_odds: float
    implied_probability: float
    expected_value: float
    edge_percentage: float
    kelly_stake: float
    
    # Reasoning
    chain_of_thought: ChainOfThought = None
    value_analysis: ValueAnalysis = None
    contrarian_check: ContrarianCheck = None
    
    # Key factors
    why_this_bet: List[str] = field(default_factory=list)
    why_might_fail: List[str] = field(default_factory=list)
    key_factors: List[str] = field(default_factory=list)
    
    # Meta
    model_version: str = "v2.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_telegram_message(self) -> str:
        """Format for Telegram"""
        stars = "⭐" * self.confidence_stars
        
        market_names = {
            BettingMarket.OVER_1_5: "Over 1.5 Goals",
            BettingMarket.OVER_2_5: "Over 2.5 Goals",
            BettingMarket.BTTS_YES: "Both Teams to Score",
            BettingMarket.BTTS_NO: "Clean Sheet",
            BettingMarket.HOME_WIN: "Home Win",
            BettingMarket.AWAY_WIN: "Away Win",
        }
        
        rec_emoji = {
            Recommendation.STRONG_BET: "🔥🔥",
            Recommendation.BET: "✅",
            Recommendation.MONITOR: "👀",
            Recommendation.AVOID: "⚠️",
            Recommendation.STRONG_AVOID: "🚫",
        }
        
        msg = f"""
{rec_emoji.get(self.recommendation, '📊')} **{self.home_team} vs {self.away_team}**
📍 {self.league} | 📅 {self.match_date}

🎯 **Market**: {market_names.get(self.market, self.market.value)}
💰 **Odds**: {self.market_odds:.2f}
📊 **Our Probability**: {self.our_probability:.1%}
📈 **Edge**: {self.edge_percentage:+.1%}
💎 **EV**: {self.expected_value:+.1%}

{stars} **Confidence**: {self.confidence_score:.0f}/100

✅ **Why This Bet**:
"""
        for reason in self.why_this_bet[:3]:
            msg += f"• {reason}\n"
        
        msg += "\n⚠️ **Risks**:\n"
        for risk in self.why_might_fail[:2]:
            msg += f"• {risk}\n"
        
        msg += f"\n📝 **Recommendation**: {self.recommendation.value}"
        
        return msg
    
    def to_full_report(self) -> str:
        """Full report with reasoning chain"""
        report = f"""
# 🧠 BETTING DECISION REPORT
## {self.home_team} vs {self.away_team}
**League**: {self.league} | **Date**: {self.match_date}

---

## 📊 DECISION SUMMARY

| Metric | Value |
|--------|-------|
| Market | {self.market.value} |
| Recommendation | {self.recommendation.value} |
| Our Probability | {self.our_probability:.1%} |
| Market Odds | {self.market_odds:.2f} |
| Implied Probability | {self.implied_probability:.1%} |
| Edge | {self.edge_percentage:+.1%} |
| Expected Value | {self.expected_value:+.1%} |
| Kelly Stake | {self.kelly_stake:.1%} |
| Confidence | {self.confidence_score:.0f}/100 ({'⭐' * self.confidence_stars}) |

---

## ✅ WHY THIS BET
"""
        for reason in self.why_this_bet:
            report += f"1. {reason}\n"
        
        report += "\n## ⚠️ WHY IT MIGHT FAIL\n"
        for risk in self.why_might_fail:
            report += f"1. {risk}\n"
        
        if self.chain_of_thought:
            report += f"\n---\n\n{self.chain_of_thought.to_string()}"
        
        if self.contrarian_check:
            report += f"\n---\n\n{self.contrarian_check.to_string()}"
        
        report += f"\n\n---\n*Generated: {self.created_at} | Model: {self.model_version}*"
        
        return report


@dataclass
class AccumulatorBet:
    """Accumulator (parlay) bet"""
    selections: List[BettingDecision]
    combined_odds: float
    combined_probability: float
    expected_value: float
    stake_recommendation: float
    
    def to_telegram_message(self) -> str:
        """Format accumulator for Telegram"""
        msg = f"""
🎰 **ACCUMULATOR BET**
💰 Combined Odds: {self.combined_odds:.2f}
📊 Combined Probability: {self.combined_probability:.1%}
📈 Expected Value: {self.expected_value:+.1%}
💵 Suggested Stake: {self.stake_recommendation:.1%} of bankroll

**Selections**:
"""
        for i, sel in enumerate(self.selections, 1):
            msg += f"{i}. {sel.home_team} vs {sel.away_team}: {sel.market.value} @ {sel.market_odds:.2f}\n"
        
        return msg


# Test
if __name__ == "__main__":
    # Test decision creation
    decision = BettingDecision(
        match_id="test_001",
        home_team="Bayern Munich",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        match_date="2025-01-28",
        market=BettingMarket.OVER_2_5,
        recommendation=Recommendation.BET,
        confidence_score=78,
        confidence_stars=4,
        our_probability=0.62,
        market_odds=1.75,
        implied_probability=0.57,
        expected_value=0.085,
        edge_percentage=5.0,
        kelly_stake=0.02,
        why_this_bet=[
            "Both teams average 3.2 goals combined in last 5 meetings",
            "Bundesliga has 55.2% Over 2.5 rate",
            "Both teams in title race - high-stakes, attacking football"
        ],
        why_might_fail=[
            "Derby matches can be tighter than usual",
            "Weather could impact play style"
        ]
    )
    
    print(decision.to_telegram_message())
    print("\n" + "="*50 + "\n")
    print(decision.to_full_report())

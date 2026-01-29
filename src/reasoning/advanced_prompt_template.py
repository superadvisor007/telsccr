#!/usr/bin/env python3
"""
🎯 ADVANCED PROMPT TEMPLATE FOR DEEPSEEK-7B
===========================================
Hardcoded goal + reference layer for tactical football reasoning.

This module provides structured, battle-tested prompts that:
1. Embed system objectives and constraints
2. Reference historical/tactical/structural data
3. Guide LLM through multi-step reasoning (CoT)
4. Enforce JSON schema outputs
5. Integrate with multi-bet ticket builder

Architecture:
    Step 0: System Goal + References (Hardcoded)
    Step 1: Team Tactical Analysis
    Step 2: Scenario Simulation (Tree-of-Thought)
    Step 3: Market Evaluation with Confidence
    Step 4: Multi-Bet Integration

Author: telegramsoccer team
Date: 2026-01-28
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 0: SYSTEM GOAL & REFERENCES (HARDCODED)
# =============================================================================

SYSTEM_OBJECTIVE = """
You are a tactical football reasoning agent. Your task is to analyze each match 
and produce actionable market recommendations (BTTS, Over 1.5, Over 2.5, etc.) 
for multi-bet tickets.

CRITICAL CONSTRAINTS (MUST FOLLOW):
1. Single-leg odds must be between 1.40 and 1.70
2. Total multi-bet odds target ≈ 10x
3. Markets should be structurally justified (no guesses)
4. Provide reasoning based on tactical, statistical, and historical patterns
5. Include scenario simulation with probabilities and confidence scores
6. Output must always be valid JSON (schema below)
7. Include brief explanations for each recommendation

REFERENCES (HARDCODED):
- Historical team performance, league trends, and previous encounters
- Tactical features: formation, pressing intensity, transition speed, wing usage
- Structural priors: league BTTS rates, over 1.5 hit rates, home advantage factors
- Market logic rules: prioritize BTTS / Over 1.5 if structure justifies; cap single-leg odds at 1.7
- Match context: injuries, recent form, weather conditions

OPTIMIZATION OBJECTIVE:
- Maximize expected value while maintaining bankroll safety
- Prioritize high-confidence, structurally-supported bets
- Avoid markets without clear statistical edge
- Target markets with positive edge > 5% and confidence > 65%
"""


# =============================================================================
# STEP 1: TEAM TACTICAL ANALYSIS TEMPLATE
# =============================================================================

STEP_1_TEMPLATE = """
STEP 1: TEAM TACTICAL ANALYSIS
================================

Analyze the tactical strengths and weaknesses of each team.

Instructions:
- Evaluate attacking strength, defensive risk, transition danger.
- Consider formation, wings, pressing, open systems, and chaos in transitions.
- Use historical priors where relevant.
- Output must be valid JSON following the schema below.

Input Data:
{input_data}

Historical Context:
{historical_context}

League Priors:
{league_priors}

Output JSON Schema:
{{
  "home_team_analysis": {{
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "wing_usage": 0.0-1.0,
    "notes": "Brief tactical justification (2-3 sentences)"
  }},
  "away_team_analysis": {{
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "wing_usage": 0.0-1.0,
    "notes": "Brief tactical justification (2-3 sentences)"
  }}
}}

Respond ONLY with valid JSON. No other text.
"""


# =============================================================================
# STEP 2: SCENARIO SIMULATION TEMPLATE
# =============================================================================

STEP_2_TEMPLATE = """
STEP 2: SCENARIO SIMULATION (TREE-OF-THOUGHT)
=============================================

Simulate 3 plausible match outcomes using the tactical analysis above.

Instructions:
- Include scoring sequences, open system vulnerabilities, counterattacks.
- Assign probability (0.0-1.0) to each scenario.
- Probabilities must sum to approximately 1.0.
- Consider tactical matchups, recent form, and historical patterns.

Team Analysis:
{team_analysis}

Historical Head-to-Head:
{h2h_context}

Recent Form:
{form_context}

Output JSON Schema:
{{
  "scenarios": [
    {{
      "description": "Home scores early, away counters late",
      "home_goals": 2,
      "away_goals": 1,
      "probability": 0.35,
      "key_events": ["Early home goal from set piece", "Away counter-attack goal", "Home seals win"]
    }},
    {{
      "description": "Away dominates possession, defensive struggle",
      "home_goals": 1,
      "away_goals": 2,
      "probability": 0.40,
      "key_events": ["Away early goal", "Home equalizes", "Away late winner"]
    }},
    {{
      "description": "Balanced game, late drama",
      "home_goals": 1,
      "away_goals": 1,
      "probability": 0.25,
      "key_events": ["Cagey first half", "Home goal", "Away equalizer"]
    }}
  ]
}}

Respond ONLY with valid JSON. No other text.
"""


# =============================================================================
# STEP 3: MARKET EVALUATION TEMPLATE
# =============================================================================

STEP_3_TEMPLATE = """
STEP 3: MARKET EVALUATION WITH CONFIDENCE SCORING
=================================================

Evaluate requested markets using scenario probabilities and hardcoded rules.

Instructions:
- Markets to evaluate: BTTS, Over 1.5, Over 2.5
- Assign probability (0.0-1.0) and confidence (0.0-1.0) for each
- Justify recommendations using tactical, structural, and historical references
- Calculate edge: (your_probability - implied_probability) / implied_probability
- Only recommend markets with edge > 5% and confidence > 65%

Scenarios:
{scenarios}

Market Odds:
{market_odds}

League Structural Priors:
{structural_priors}

Tactical Rules:
{tactical_rules}

Output JSON Schema:
{{
  "market_recommendations": {{
    "BTTS": {{
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": 0.0-1.0,
      "reasoning": "Explanation using structural, tactical, and historical context (2-3 sentences)",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Reference to league priors, tactical features (1-2 sentences)"
    }},
    "Over_1_5": {{
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": 0.0-1.0,
      "reasoning": "Explanation using scenario simulation and priors (2-3 sentences)",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Reference to league trends, scoring patterns (1-2 sentences)"
    }},
    "Over_2_5": {{
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": 0.0-1.0,
      "reasoning": "Explanation using low/high scenario probability and structural factors (2-3 sentences)",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Reference to defensive strength, league averages (1-2 sentences)"
    }}
  }}
}}

Calculation Examples:
- BTTS probability = sum(scenario.probability where scenario.btts == true)
- Over 1.5 probability = sum(scenario.probability where scenario.total_goals > 1.5)
- Edge = (your_probability - implied_probability) / implied_probability
  Example: If odds = 1.50, implied_prob = 1/1.50 = 0.667
           If your_prob = 0.75, edge = (0.75 - 0.667) / 0.667 = 0.124 = 12.4%

Respond ONLY with valid JSON. No other text.
"""


# =============================================================================
# STEP 4: MULTI-BET INTEGRATION TEMPLATE
# =============================================================================

STEP_4_TEMPLATE = """
STEP 4: MULTI-BET TICKET INTEGRATION
====================================

Format JSON output for multi-bet selection.

Instructions:
- Filter markets where confidence >= 0.65 and edge >= 0.05
- Ensure single-leg odds between 1.4–1.7
- Combine legs to target total multi-bet odds ≈ 10x
- Include match details, market, tip, odds, confidence

Market Recommendations:
{market_recommendations}

Available Odds:
{available_odds}

Output JSON Schema:
{{
  "ticket_legs": [
    {{
      "match": "Home Team vs Away Team",
      "match_id": "unique_id",
      "league": "Premier League",
      "market": "BTTS",
      "tip": "Yes",
      "odds": 1.55,
      "probability": 0.72,
      "confidence": 0.82,
      "edge": 0.124,
      "reasoning": "Brief explanation (1 sentence)"
    }}
  ],
  "total_odds": 10.05,
  "recommended_stake": 50,
  "potential_win": 502.5,
  "expected_value": 0.089,
  "footer": "Play responsibly. Past performance does not guarantee future results."
}}

Respond ONLY with valid JSON. No other text.
"""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

@dataclass
class AdvancedPromptBuilder:
    """Builds structured prompts for DeepSeek-7B with embedded goals and references."""
    
    def __init__(self):
        self.system_objective = SYSTEM_OBJECTIVE
        
    def build_step_1_prompt(
        self,
        match_data: Dict[str, Any],
        historical_context: Dict[str, Any],
        league_priors: Dict[str, Any]
    ) -> str:
        """Build Step 1: Team Tactical Analysis prompt."""
        
        input_data_str = json.dumps(match_data, indent=2)
        historical_str = json.dumps(historical_context, indent=2)
        league_priors_str = json.dumps(league_priors, indent=2)
        
        prompt = f"{self.system_objective}\n\n"
        prompt += STEP_1_TEMPLATE.format(
            input_data=input_data_str,
            historical_context=historical_str,
            league_priors=league_priors_str
        )
        
        return prompt
    
    def build_step_2_prompt(
        self,
        team_analysis: Dict[str, Any],
        h2h_context: Dict[str, Any],
        form_context: Dict[str, Any]
    ) -> str:
        """Build Step 2: Scenario Simulation prompt."""
        
        team_analysis_str = json.dumps(team_analysis, indent=2)
        h2h_str = json.dumps(h2h_context, indent=2)
        form_str = json.dumps(form_context, indent=2)
        
        prompt = f"{self.system_objective}\n\n"
        prompt += STEP_2_TEMPLATE.format(
            team_analysis=team_analysis_str,
            h2h_context=h2h_str,
            form_context=form_str
        )
        
        return prompt
    
    def build_step_3_prompt(
        self,
        scenarios: List[Dict[str, Any]],
        market_odds: Dict[str, float],
        structural_priors: Dict[str, Any],
        tactical_rules: Dict[str, Any]
    ) -> str:
        """Build Step 3: Market Evaluation prompt."""
        
        scenarios_str = json.dumps(scenarios, indent=2)
        odds_str = json.dumps(market_odds, indent=2)
        priors_str = json.dumps(structural_priors, indent=2)
        rules_str = json.dumps(tactical_rules, indent=2)
        
        prompt = f"{self.system_objective}\n\n"
        prompt += STEP_3_TEMPLATE.format(
            scenarios=scenarios_str,
            market_odds=odds_str,
            structural_priors=priors_str,
            tactical_rules=rules_str
        )
        
        return prompt
    
    def build_step_4_prompt(
        self,
        market_recommendations: Dict[str, Any],
        available_odds: Dict[str, float]
    ) -> str:
        """Build Step 4: Multi-Bet Integration prompt."""
        
        recs_str = json.dumps(market_recommendations, indent=2)
        odds_str = json.dumps(available_odds, indent=2)
        
        prompt = f"{self.system_objective}\n\n"
        prompt += STEP_4_TEMPLATE.format(
            market_recommendations=recs_str,
            available_odds=odds_str
        )
        
        return prompt
    
    def build_full_pipeline_prompt(
        self,
        match_data: Dict[str, Any],
        historical_context: Dict[str, Any],
        league_priors: Dict[str, Any],
        market_odds: Dict[str, float],
        structural_priors: Dict[str, Any],
        tactical_rules: Dict[str, Any]
    ) -> str:
        """Build complete pipeline prompt (all steps combined)."""
        
        prompt = f"{self.system_objective}\n\n"
        prompt += "=" * 70 + "\n"
        prompt += "COMPLETE MATCH ANALYSIS PIPELINE\n"
        prompt += "=" * 70 + "\n\n"
        
        prompt += "Follow these steps in order:\n\n"
        
        # Step 1
        prompt += STEP_1_TEMPLATE.format(
            input_data=json.dumps(match_data, indent=2),
            historical_context=json.dumps(historical_context, indent=2),
            league_priors=json.dumps(league_priors, indent=2)
        )
        prompt += "\n\n" + "=" * 70 + "\n\n"
        
        # Step 2
        prompt += STEP_2_TEMPLATE.format(
            team_analysis="<Use output from Step 1>",
            h2h_context=json.dumps(historical_context.get('h2h', {}), indent=2),
            form_context=json.dumps(historical_context.get('form', {}), indent=2)
        )
        prompt += "\n\n" + "=" * 70 + "\n\n"
        
        # Step 3
        prompt += STEP_3_TEMPLATE.format(
            scenarios="<Use output from Step 2>",
            market_odds=json.dumps(market_odds, indent=2),
            structural_priors=json.dumps(structural_priors, indent=2),
            tactical_rules=json.dumps(tactical_rules, indent=2)
        )
        prompt += "\n\n" + "=" * 70 + "\n\n"
        
        # Step 4
        prompt += STEP_4_TEMPLATE.format(
            market_recommendations="<Use output from Step 3>",
            available_odds=json.dumps(market_odds, indent=2)
        )
        
        prompt += "\n\n" + "=" * 70 + "\n"
        prompt += "OUTPUT FORMAT: Provide complete JSON with all steps completed.\n"
        prompt += "=" * 70 + "\n"
        
        return prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response."""
    try:
        # Try direct parse first
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find first { to last }
        start = response.find('{')
        end = response.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to extract JSON from response: {response[:200]}...")
        return None


def validate_step_1_output(data: Dict[str, Any]) -> bool:
    """Validate Step 1 JSON output."""
    required_keys = ['home_team_analysis', 'away_team_analysis']
    team_keys = ['attacking_strength', 'defensive_risk', 'transition_danger', 'notes']
    
    if not all(key in data for key in required_keys):
        return False
    
    for team_key in required_keys:
        if not all(key in data[team_key] for key in team_keys):
            return False
    
    return True


def validate_step_2_output(data: Dict[str, Any]) -> bool:
    """Validate Step 2 JSON output."""
    if 'scenarios' not in data:
        return False
    
    if not isinstance(data['scenarios'], list) or len(data['scenarios']) < 3:
        return False
    
    scenario_keys = ['description', 'home_goals', 'away_goals', 'probability']
    
    for scenario in data['scenarios']:
        if not all(key in scenario for key in scenario_keys):
            return False
    
    # Check probabilities sum to approximately 1.0
    total_prob = sum(s['probability'] for s in data['scenarios'])
    if not (0.95 <= total_prob <= 1.05):
        logger.warning(f"Scenario probabilities sum to {total_prob}, not ≈1.0")
    
    return True


def validate_step_3_output(data: Dict[str, Any]) -> bool:
    """Validate Step 3 JSON output."""
    if 'market_recommendations' not in data:
        return False
    
    markets = data['market_recommendations']
    required_markets = ['BTTS', 'Over_1_5', 'Over_2_5']
    market_keys = ['tip', 'probability', 'confidence', 'edge', 'reasoning']
    
    if not all(market in markets for market in required_markets):
        return False
    
    for market in required_markets:
        if not all(key in markets[market] for key in market_keys):
            return False
    
    return True


def validate_step_4_output(data: Dict[str, Any]) -> bool:
    """Validate Step 4 JSON output."""
    if 'ticket_legs' not in data:
        return False
    
    if not isinstance(data['ticket_legs'], list):
        return False
    
    leg_keys = ['match', 'market', 'tip', 'odds', 'confidence']
    
    for leg in data['ticket_legs']:
        if not all(key in leg for key in leg_keys):
            return False
    
    return True


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'SYSTEM_OBJECTIVE',
    'AdvancedPromptBuilder',
    'extract_json_from_response',
    'validate_step_1_output',
    'validate_step_2_output',
    'validate_step_3_output',
    'validate_step_4_output',
]


if __name__ == "__main__":
    # Demo usage
    builder = AdvancedPromptBuilder()
    
    match_data = {
        "home_team": "Arsenal",
        "away_team": "Juventus",
        "league": "Champions League",
        "date": "2026-01-30"
    }
    
    historical_context = {
        "h2h": {"matches": 3, "home_wins": 1, "draws": 1, "away_wins": 1},
        "form": {"home_last_5": "WWDWL", "away_last_5": "WLWDD"}
    }
    
    league_priors = {
        "btts_rate": 0.52,
        "over_1_5_rate": 0.75,
        "avg_goals": 2.85
    }
    
    prompt = builder.build_step_1_prompt(match_data, historical_context, league_priors)
    print("=" * 70)
    print("STEP 1 PROMPT EXAMPLE")
    print("=" * 70)
    print(prompt[:500] + "...")

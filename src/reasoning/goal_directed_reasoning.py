"""
🎯 Goal-Directed LLM Reasoning System
======================================
Hardcoded objectives, structured prompts, and battle-tested patterns.

This module implements:
- Step 0: System Goal + References (Hardcoded)
- Step 1: Team Tactical Analysis
- Step 2: Scenario Simulation (Tree-of-Thought)
- Step 3: Market Evaluation with Confidence Scoring
- Step 4: Multi-Bet Builder Integration

The system ensures LLM never drifts from the optimization objective.
"""

import json
import hashlib
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging
import re

# Import advanced prompt templates
from .advanced_prompt_template import (
    AdvancedPromptBuilder,
    extract_json_from_response,
    validate_step_1_output,
    validate_step_2_output,
    validate_step_3_output,
    validate_step_4_output
)

logger = logging.getLogger(__name__)


# =============================================================================
# HARDCODED SYSTEM GOAL & REFERENCES
# =============================================================================

SYSTEM_GOAL = """
System Objective:
You are a tactical football reasoning agent. Your task is to analyze each match 
and produce actionable market recommendations (BTTS, Over 1.5, Over 2.5) for 
multi-bet tickets.

CRITICAL CONSTRAINTS (MUST FOLLOW):
1. Single-leg odds MUST be between 1.40 and 1.70
2. Total multi-bet odds target ≈ 10x
3. Markets MUST be structurally justified (no guesses)
4. Provide reasoning based on tactical, statistical, and historical patterns
5. Include scenario simulation with probabilities and confidence scores
6. Output MUST always be valid JSON (schema provided)
7. Include brief explanations for each recommendation

OPTIMIZATION OBJECTIVE:
- Maximize expected value while maintaining bankroll safety
- Prioritize high-confidence, structurally-supported bets
- Avoid markets without clear statistical edge
"""

DOMAIN_REFERENCES = {
    "league_priors": {
        "bundesliga": {
            "btts_rate": 0.53,
            "over_1_5_rate": 0.76,
            "over_2_5_rate": 0.52,
            "avg_goals": 2.95,
            "home_win_rate": 0.45,
            "pressing_league": True,
            "open_play_tendency": "high"
        },
        "premier_league": {
            "btts_rate": 0.52,
            "over_1_5_rate": 0.75,
            "over_2_5_rate": 0.51,
            "avg_goals": 2.85,
            "home_win_rate": 0.44,
            "pressing_league": True,
            "open_play_tendency": "medium-high"
        },
        "la_liga": {
            "btts_rate": 0.48,
            "over_1_5_rate": 0.72,
            "over_2_5_rate": 0.47,
            "avg_goals": 2.65,
            "home_win_rate": 0.47,
            "pressing_league": False,
            "open_play_tendency": "medium"
        },
        "serie_a": {
            "btts_rate": 0.49,
            "over_1_5_rate": 0.73,
            "over_2_5_rate": 0.49,
            "avg_goals": 2.75,
            "home_win_rate": 0.46,
            "pressing_league": False,
            "open_play_tendency": "medium"
        },
        "ligue_1": {
            "btts_rate": 0.47,
            "over_1_5_rate": 0.71,
            "over_2_5_rate": 0.46,
            "avg_goals": 2.60,
            "home_win_rate": 0.46,
            "pressing_league": False,
            "open_play_tendency": "low-medium"
        },
        "eredivisie": {
            "btts_rate": 0.58,
            "over_1_5_rate": 0.82,
            "over_2_5_rate": 0.60,
            "avg_goals": 3.20,
            "home_win_rate": 0.48,
            "pressing_league": True,
            "open_play_tendency": "very_high"
        },
        "championship": {
            "btts_rate": 0.51,
            "over_1_5_rate": 0.74,
            "over_2_5_rate": 0.50,
            "avg_goals": 2.80,
            "home_win_rate": 0.44,
            "pressing_league": True,
            "open_play_tendency": "high"
        }
    },
    "market_rules": {
        "min_single_odds": 1.40,
        "max_single_odds": 1.70,
        "target_total_odds": 10.0,
        "min_confidence": 0.65,
        "min_edge": 0.05,
        "max_legs": 6,
        "min_legs": 3
    },
    "tactical_rules": {
        "high_pressing_btts_boost": 0.08,
        "high_transition_over_boost": 0.06,
        "defensive_fragility_threshold": 0.65,
        "open_system_btts_threshold": 0.60,
        "counter_attack_goal_threshold": 0.55
    },
    "structural_priors": {
        "if_both_teams_transition_danger_gt_0.6": "BTTS_YES_boost_0.1",
        "if_pressing_intensity_gt_0.65": "OVER_1.5_boost_0.08",
        "if_defensive_risk_gt_0.7_both": "BTTS_YES_boost_0.12",
        "if_league_open_play_high": "OVER_2.5_boost_0.05",
        "if_home_attacking_gt_0.7_away_defensive_gt_0.6": "HOME_GOALS_boost"
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TeamTacticalAnalysis:
    """Step 1 output: Team tactical profile."""
    team_name: str
    attacking_strength: float  # 0-1
    defensive_risk: float      # 0-1
    transition_danger: float   # 0-1
    pressing_intensity: float  # 0-1
    set_piece_threat: float    # 0-1
    wing_usage: float          # 0-1
    tactical_notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TeamTacticalAnalysis':
        return cls(**data)


@dataclass
class MatchScenario:
    """Step 2 output: Simulated match scenario."""
    description: str
    home_goals: int
    away_goals: int
    probability: float
    key_events: List[str] = field(default_factory=list)
    
    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals
    
    @property
    def btts(self) -> bool:
        return self.home_goals > 0 and self.away_goals > 0
    
    @property
    def over_1_5(self) -> bool:
        return self.total_goals > 1.5
    
    @property
    def over_2_5(self) -> bool:
        return self.total_goals > 2.5
    
    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'home_goals': self.home_goals,
            'away_goals': self.away_goals,
            'total_goals': self.total_goals,
            'probability': self.probability,
            'btts': self.btts,
            'over_1_5': self.over_1_5,
            'over_2_5': self.over_2_5,
            'key_events': self.key_events
        }


@dataclass
class MarketRecommendation:
    """Step 3 output: Market recommendation with confidence."""
    market: str
    tip: str
    probability: float
    confidence: float
    edge: float
    reasoning: str
    scenario_support: float
    structural_justification: str
    
    @property
    def is_actionable(self) -> bool:
        """Check if this market meets actionable criteria."""
        rules = DOMAIN_REFERENCES['market_rules']
        return (
            self.confidence >= rules['min_confidence'] and
            self.edge >= rules['min_edge']
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BetLeg:
    """Single leg for multi-bet ticket."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    market: str
    tip: str
    odds: float
    probability: float
    confidence: float
    edge: float
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultiBeTicket:
    """Step 4 output: Complete multi-bet ticket."""
    ticket_id: str
    legs: List[BetLeg]
    total_odds: float
    stake: float
    potential_win: float
    expected_value: float
    overall_confidence: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @classmethod
    def from_legs(cls, legs: List[BetLeg], stake: float = 50.0) -> 'MultiBeTicket':
        total_odds = 1.0
        combined_prob = 1.0
        confidence_sum = 0.0
        
        for leg in legs:
            total_odds *= leg.odds
            combined_prob *= leg.probability
            confidence_sum += leg.confidence
        
        potential_win = stake * total_odds
        ev = combined_prob * potential_win - stake
        avg_confidence = confidence_sum / len(legs) if legs else 0
        
        ticket_id = f"TS-{datetime.now().strftime('%Y%m%d%H%M')}-{len(legs)}L"
        
        return cls(
            ticket_id=ticket_id,
            legs=legs,
            total_odds=total_odds,
            stake=stake,
            potential_win=potential_win,
            expected_value=ev,
            overall_confidence=avg_confidence
        )
    
    def to_dict(self) -> Dict:
        return {
            'ticket_id': self.ticket_id,
            'legs': [l.to_dict() for l in self.legs],
            'total_odds': round(self.total_odds, 2),
            'stake': self.stake,
            'potential_win': round(self.potential_win, 2),
            'expected_value': round(self.expected_value, 2),
            'overall_confidence': round(self.overall_confidence, 2),
            'created_at': self.created_at,
            'footer': 'Play responsibly - Gamble aware'
        }


@dataclass
class GoalDirectedAnalysis:
    """Complete goal-directed analysis result."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    
    # Analysis outputs
    home_analysis: TeamTacticalAnalysis
    away_analysis: TeamTacticalAnalysis
    scenarios: List[MatchScenario]
    market_recommendations: Dict[str, MarketRecommendation]
    
    # Meta
    system_goal: str
    references_used: List[str]
    reasoning_chain: List[str]
    processing_time_ms: int
    model_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_actionable_markets(self) -> List[MarketRecommendation]:
        """Get only actionable market recommendations."""
        return [m for m in self.market_recommendations.values() if m.is_actionable]
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'league': self.league,
            'date': self.date,
            'home_analysis': self.home_analysis.to_dict(),
            'away_analysis': self.away_analysis.to_dict(),
            'scenarios': [s.to_dict() for s in self.scenarios],
            'market_recommendations': {k: v.to_dict() for k, v in self.market_recommendations.items()},
            'system_goal': self.system_goal[:100] + '...',
            'references_used': self.references_used,
            'reasoning_chain': self.reasoning_chain,
            'processing_time_ms': self.processing_time_ms,
            'model_used': self.model_used,
            'timestamp': self.timestamp
        }


# =============================================================================
# PROMPT TEMPLATES (HARDCODED GOAL + REFERENCES)
# =============================================================================

class PromptTemplates:
    """Hardcoded prompt templates with goal and references."""
    
    @staticmethod
    def step0_system_instruction(league: str) -> str:
        """Step 0: System goal and references."""
        priors = DOMAIN_REFERENCES['league_priors'].get(
            league.lower().replace(' ', '_').replace('-', '_'),
            DOMAIN_REFERENCES['league_priors']['premier_league']
        )
        rules = DOMAIN_REFERENCES['market_rules']
        tactical = DOMAIN_REFERENCES['tactical_rules']
        
        return f"""{SYSTEM_GOAL}

DOMAIN REFERENCES FOR THIS ANALYSIS:

League Historical Priors ({league}):
- BTTS Rate: {priors['btts_rate']:.0%}
- Over 1.5 Rate: {priors['over_1_5_rate']:.0%}
- Over 2.5 Rate: {priors['over_2_5_rate']:.0%}
- Average Goals: {priors['avg_goals']:.2f}
- Home Win Rate: {priors['home_win_rate']:.0%}
- Pressing League: {priors['pressing_league']}
- Open Play Tendency: {priors['open_play_tendency']}

Market Rules:
- Single Leg Odds: {rules['min_single_odds']} - {rules['max_single_odds']}
- Target Total Odds: ~{rules['target_total_odds']}x
- Min Confidence: {rules['min_confidence']:.0%}
- Min Edge: {rules['min_edge']:.0%}

Tactical Rules:
- High Pressing BTTS Boost: +{tactical['high_pressing_btts_boost']:.0%}
- High Transition Over Boost: +{tactical['high_transition_over_boost']:.0%}
- Defensive Fragility Threshold: {tactical['defensive_fragility_threshold']}
- Open System BTTS Threshold: {tactical['open_system_btts_threshold']}

Structural Priors:
{json.dumps(DOMAIN_REFERENCES['structural_priors'], indent=2)}
"""

    @staticmethod
    def step1_team_analysis(
        home_team: str,
        away_team: str,
        home_stats: Dict,
        away_stats: Dict
    ) -> str:
        """Step 1: Team tactical analysis prompt."""
        return f"""
STEP 1: TEAM TACTICAL ANALYSIS

Analyze the tactical profile of each team. Use the provided stats and apply 
the tactical rules from your references.

MATCH: {home_team} vs {away_team}

HOME TEAM ({home_team}) STATS:
{json.dumps(home_stats, indent=2)}

AWAY TEAM ({away_team}) STATS:
{json.dumps(away_stats, indent=2)}

REQUIRED OUTPUT (valid JSON only):
{{
  "home_team_analysis": {{
    "team_name": "{home_team}",
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "wing_usage": 0.0-1.0,
    "tactical_notes": "Brief tactical justification based on stats"
  }},
  "away_team_analysis": {{
    "team_name": "{away_team}",
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "wing_usage": 0.0-1.0,
    "tactical_notes": "Brief tactical justification based on stats"
  }}
}}
"""

    @staticmethod
    def step2_scenario_simulation(
        home_team: str,
        away_team: str,
        home_analysis: Dict,
        away_analysis: Dict,
        league_priors: Dict
    ) -> str:
        """Step 2: Scenario simulation prompt (Tree-of-Thought)."""
        return f"""
STEP 2: SCENARIO SIMULATION (Tree-of-Thought)

Based on the tactical analysis, simulate 3 plausible match scenarios.
Consider: scoring sequences, open system vulnerabilities, counter-attacks,
pressing breakdowns, set pieces.

MATCH: {home_team} vs {away_team}

HOME TACTICAL PROFILE:
{json.dumps(home_analysis, indent=2)}

AWAY TACTICAL PROFILE:
{json.dumps(away_analysis, indent=2)}

LEAGUE CONTEXT:
- Average Goals: {league_priors.get('avg_goals', 2.7):.2f}
- BTTS Rate: {league_priors.get('btts_rate', 0.50):.0%}
- Open Play: {league_priors.get('open_play_tendency', 'medium')}

SIMULATION RULES:
1. Probabilities MUST sum to 1.0
2. Consider defensive_risk and attacking_strength matchups
3. If both transition_danger > 0.6, expect more goals
4. If pressing_intensity high for both, expect open game

REQUIRED OUTPUT (valid JSON only):
{{
  "scenarios": [
    {{
      "description": "Scenario 1 description",
      "home_goals": integer,
      "away_goals": integer,
      "probability": 0.0-1.0,
      "key_events": ["event1", "event2"]
    }},
    {{
      "description": "Scenario 2 description",
      "home_goals": integer,
      "away_goals": integer,
      "probability": 0.0-1.0,
      "key_events": ["event1", "event2"]
    }},
    {{
      "description": "Scenario 3 description",
      "home_goals": integer,
      "away_goals": integer,
      "probability": 0.0-1.0,
      "key_events": ["event1", "event2"]
    }}
  ]
}}
"""

    @staticmethod
    def step3_market_evaluation(
        home_team: str,
        away_team: str,
        scenarios: List[Dict],
        league_priors: Dict,
        market_rules: Dict
    ) -> str:
        """Step 3: Market evaluation prompt."""
        return f"""
STEP 3: MARKET EVALUATION

Evaluate betting markets using scenario probabilities and structural priors.
Apply the market rules to determine recommendations.

MATCH: {home_team} vs {away_team}

SIMULATED SCENARIOS:
{json.dumps(scenarios, indent=2)}

LEAGUE PRIORS:
{json.dumps(league_priors, indent=2)}

MARKET RULES:
- Single Leg Odds: {market_rules['min_single_odds']} - {market_rules['max_single_odds']}
- Min Confidence: {market_rules['min_confidence']:.0%}
- Min Edge Required: {market_rules['min_edge']:.0%}

EVALUATION RULES:
1. Calculate probability from scenario weights
2. Compare to league priors (blend 60% model / 40% prior)
3. Calculate edge = model_probability - implied_probability
4. Only recommend if confidence >= {market_rules['min_confidence']} AND edge >= {market_rules['min_edge']}

REQUIRED OUTPUT (valid JSON only):
{{
  "market_recommendations": {{
    "btts": {{
      "market": "btts",
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": -1.0 to 1.0,
      "reasoning": "Tactical/statistical justification",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Which structural prior supports this"
    }},
    "over_1_5": {{
      "market": "over_1_5",
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": -1.0 to 1.0,
      "reasoning": "Tactical/statistical justification",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Which structural prior supports this"
    }},
    "over_2_5": {{
      "market": "over_2_5",
      "tip": "Yes" or "No",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "edge": -1.0 to 1.0,
      "reasoning": "Tactical/statistical justification",
      "scenario_support": 0.0-1.0,
      "structural_justification": "Which structural prior supports this"
    }}
  }}
}}
"""


# =============================================================================
# GOAL-DIRECTED REASONING ENGINE
# =============================================================================

class GoalDirectedReasoningEngine:
    """
    🎯 Goal-Directed LLM Reasoning Engine
    
    Implements the complete 4-step reasoning pipeline with hardcoded goals.
    
    Features:
    - Step 0: System goal injection (every prompt)
    - Step 1: Team tactical analysis
    - Step 2: Scenario simulation (Tree-of-Thought)
    - Step 3: Market evaluation with confidence
    - Step 4: Multi-bet builder integration
    - Caching to reduce inference costs
    - Statistical fallback when LLM unavailable
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        llm_backend: str = 'ollama',
        model_name: str = 'deepseek-llm:7b-chat'
    ):
        self.cache_dir = Path(cache_dir or 'data/goal_reasoning_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = self.cache_dir / 'goal_reasoning.db'
        self.llm_backend = llm_backend
        self.model_name = model_name
        
        # Initialize advanced prompt builder
        self.prompt_builder = AdvancedPromptBuilder()
        
        self._llm_available = False
        self._init_cache()
        self._init_llm()
        
        logger.info(f"🎯 GoalDirectedReasoningEngine initialized")
        logger.info(f"   LLM: {llm_backend}/{model_name}")
        logger.info(f"   Available: {self._llm_available}")
        logger.info(f"   Advanced Prompt Template: Enabled")
    
    def _init_cache(self):
        """Initialize SQLite cache."""
        conn = sqlite3.connect(str(self.cache_db))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                match_hash TEXT PRIMARY KEY,
                match_id TEXT,
                result_json TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _init_llm(self):
        """Check LLM availability."""
        if self.llm_backend == 'ollama':
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                self._llm_available = response.status_code == 200
            except:
                self._llm_available = False
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call LLM with system and user prompts."""
        if not self._llm_available:
            return None
        
        try:
            import requests
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': full_prompt,
                    'stream': False,
                    'options': {'temperature': 0.3, 'num_predict': 2048}
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
        
        return None
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        if not text:
            return None
        
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Find JSON in text
        patterns = [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{[\s\S]*\}']
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        return None
    
    def _get_league_priors(self, league: str) -> Dict:
        """Get league priors from references."""
        key = league.lower().replace(' ', '_').replace('-', '_')
        return DOMAIN_REFERENCES['league_priors'].get(
            key,
            DOMAIN_REFERENCES['league_priors']['premier_league']
        )
    
    def _statistical_team_analysis(
        self,
        team_name: str,
        stats: Dict
    ) -> TeamTacticalAnalysis:
        """Fallback: Statistical team analysis."""
        goals_avg = stats.get('goals_avg', 1.4)
        conceded_avg = stats.get('conceded_avg', 1.3)
        elo = stats.get('elo', 1500)
        form = stats.get('form', 0.5)
        
        attacking = min(1.0, goals_avg / 2.5)
        defensive_risk = min(1.0, conceded_avg / 2.0)
        transition = min(1.0, (attacking + form) / 2)
        pressing = 0.5 + (elo - 1500) / 1000
        
        return TeamTacticalAnalysis(
            team_name=team_name,
            attacking_strength=round(attacking, 2),
            defensive_risk=round(defensive_risk, 2),
            transition_danger=round(transition, 2),
            pressing_intensity=round(max(0, min(1, pressing)), 2),
            set_piece_threat=0.4,
            wing_usage=0.5,
            tactical_notes=f"Statistical profile: {goals_avg:.1f} GF, {conceded_avg:.1f} GA"
        )
    
    def _statistical_scenarios(
        self,
        home: TeamTacticalAnalysis,
        away: TeamTacticalAnalysis,
        priors: Dict
    ) -> List[MatchScenario]:
        """Fallback: Statistical scenario generation."""
        home_xg = home.attacking_strength * away.defensive_risk * 2.5 + 0.5
        away_xg = away.attacking_strength * home.defensive_risk * 2.0 + 0.3
        
        scenarios = [
            MatchScenario(
                description="Expected outcome based on tactical profiles",
                home_goals=round(home_xg),
                away_goals=round(away_xg),
                probability=0.50,
                key_events=["Normal game flow", "Tactical battle"]
            ),
            MatchScenario(
                description="Home dominance scenario",
                home_goals=round(home_xg + 1),
                away_goals=max(0, round(away_xg - 0.5)),
                probability=0.30,
                key_events=["Early home goal", "Away struggles to create"]
            ),
            MatchScenario(
                description="Away surprise / counter-attack success",
                home_goals=max(0, round(home_xg - 0.5)),
                away_goals=round(away_xg + 1),
                probability=0.20,
                key_events=["Counter-attack goals", "Home over-commits"]
            )
        ]
        
        return scenarios
    
    def _statistical_market_evaluation(
        self,
        scenarios: List[MatchScenario],
        priors: Dict,
        home: TeamTacticalAnalysis,
        away: TeamTacticalAnalysis
    ) -> Dict[str, MarketRecommendation]:
        """Fallback: Statistical market evaluation."""
        rules = DOMAIN_REFERENCES['market_rules']
        tactical = DOMAIN_REFERENCES['tactical_rules']
        
        # Calculate from scenarios
        btts_scenario = sum(s.probability for s in scenarios if s.btts)
        over_1_5_scenario = sum(s.probability for s in scenarios if s.over_1_5)
        over_2_5_scenario = sum(s.probability for s in scenarios if s.over_2_5)
        
        # Apply tactical boosts
        btts_boost = 0
        over_boost = 0
        
        if home.transition_danger > 0.6 and away.transition_danger > 0.6:
            btts_boost += tactical['high_pressing_btts_boost']
        if home.pressing_intensity > 0.65 and away.pressing_intensity > 0.65:
            over_boost += tactical['high_transition_over_boost']
        if home.defensive_risk > 0.7 and away.defensive_risk > 0.7:
            btts_boost += 0.1
        
        # Blend with priors (60% model, 40% prior)
        btts_final = (btts_scenario + btts_boost) * 0.6 + priors['btts_rate'] * 0.4
        over_1_5_final = (over_1_5_scenario + over_boost) * 0.6 + priors['over_1_5_rate'] * 0.4
        over_2_5_final = (over_2_5_scenario + over_boost * 0.5) * 0.6 + priors['over_2_5_rate'] * 0.4
        
        # Calculate edges (assuming market odds from priors)
        btts_edge = btts_final - priors['btts_rate']
        over_1_5_edge = over_1_5_final - priors['over_1_5_rate']
        over_2_5_edge = over_2_5_final - priors['over_2_5_rate']
        
        return {
            'btts': MarketRecommendation(
                market='btts',
                tip='Yes' if btts_final > 0.55 else 'No',
                probability=round(btts_final, 3),
                confidence=round(abs(btts_final - 0.5) * 2, 3),
                edge=round(btts_edge, 3),
                reasoning=f"Scenario: {btts_scenario:.0%}, Prior: {priors['btts_rate']:.0%}, Boost: {btts_boost:.0%}",
                scenario_support=round(btts_scenario, 3),
                structural_justification="transition_danger + defensive_risk analysis"
            ),
            'over_1_5': MarketRecommendation(
                market='over_1_5',
                tip='Yes' if over_1_5_final > 0.55 else 'No',
                probability=round(over_1_5_final, 3),
                confidence=round(abs(over_1_5_final - 0.5) * 2, 3),
                edge=round(over_1_5_edge, 3),
                reasoning=f"Scenario: {over_1_5_scenario:.0%}, Prior: {priors['over_1_5_rate']:.0%}",
                scenario_support=round(over_1_5_scenario, 3),
                structural_justification="attacking_strength + pressing_intensity analysis"
            ),
            'over_2_5': MarketRecommendation(
                market='over_2_5',
                tip='Yes' if over_2_5_final > 0.55 else 'No',
                probability=round(over_2_5_final, 3),
                confidence=round(abs(over_2_5_final - 0.5) * 2, 3),
                edge=round(over_2_5_edge, 3),
                reasoning=f"Scenario: {over_2_5_scenario:.0%}, Prior: {priors['over_2_5_rate']:.0%}",
                scenario_support=round(over_2_5_scenario, 3),
                structural_justification="high_scoring_potential analysis"
            )
        }
    
    def analyze_match(
        self,
        match_data: Dict,
        home_stats: Dict = None,
        away_stats: Dict = None,
        use_llm: bool = True
    ) -> GoalDirectedAnalysis:
        """
        🎯 Run complete goal-directed analysis pipeline.
        
        Steps:
        0. Inject system goal + references
        1. Team tactical analysis
        2. Scenario simulation
        3. Market evaluation
        """
        import time
        start_time = time.time()
        
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        league = match_data.get('league', 'Unknown')
        date = match_data.get('date', 'Unknown')
        match_id = f"{home_team}_vs_{away_team}_{date}"
        
        home_stats = home_stats or {}
        away_stats = away_stats or {}
        priors = self._get_league_priors(league)
        reasoning_chain = []
        references_used = []
        
        # Step 0: System goal
        system_prompt = PromptTemplates.step0_system_instruction(league)
        reasoning_chain.append("Step 0: Loaded system goal and domain references")
        references_used.append(f"league_priors:{league}")
        references_used.append("market_rules")
        references_used.append("tactical_rules")
        
        # Step 1: Team Analysis
        reasoning_chain.append("Step 1: Analyzing team tactical profiles")
        
        if use_llm and self._llm_available:
            step1_prompt = PromptTemplates.step1_team_analysis(
                home_team, away_team, home_stats, away_stats
            )
            response = self._call_llm(system_prompt, step1_prompt)
            parsed = self._parse_json(response)
            
            if parsed and 'home_team_analysis' in parsed:
                home_analysis = TeamTacticalAnalysis.from_dict(parsed['home_team_analysis'])
                away_analysis = TeamTacticalAnalysis.from_dict(parsed['away_team_analysis'])
                reasoning_chain.append("  → LLM analysis successful")
            else:
                home_analysis = self._statistical_team_analysis(home_team, home_stats)
                away_analysis = self._statistical_team_analysis(away_team, away_stats)
                reasoning_chain.append("  → Fallback to statistical analysis")
        else:
            home_analysis = self._statistical_team_analysis(home_team, home_stats)
            away_analysis = self._statistical_team_analysis(away_team, away_stats)
            reasoning_chain.append("  → Statistical analysis (LLM not available)")
        
        # Step 2: Scenario Simulation
        reasoning_chain.append("Step 2: Simulating match scenarios (Tree-of-Thought)")
        
        if use_llm and self._llm_available:
            step2_prompt = PromptTemplates.step2_scenario_simulation(
                home_team, away_team,
                home_analysis.to_dict(), away_analysis.to_dict(),
                priors
            )
            response = self._call_llm(system_prompt, step2_prompt)
            parsed = self._parse_json(response)
            
            if parsed and 'scenarios' in parsed:
                scenarios = [
                    MatchScenario(
                        description=s.get('description', ''),
                        home_goals=s.get('home_goals', 1),
                        away_goals=s.get('away_goals', 1),
                        probability=s.get('probability', 0.33),
                        key_events=s.get('key_events', [])
                    )
                    for s in parsed['scenarios']
                ]
                reasoning_chain.append(f"  → Generated {len(scenarios)} scenarios via LLM")
            else:
                scenarios = self._statistical_scenarios(home_analysis, away_analysis, priors)
                reasoning_chain.append("  → Fallback to statistical scenarios")
        else:
            scenarios = self._statistical_scenarios(home_analysis, away_analysis, priors)
            reasoning_chain.append("  → Statistical scenarios generated")
        
        # Step 3: Market Evaluation
        reasoning_chain.append("Step 3: Evaluating markets with confidence scoring")
        
        if use_llm and self._llm_available:
            step3_prompt = PromptTemplates.step3_market_evaluation(
                home_team, away_team,
                [s.to_dict() for s in scenarios],
                priors,
                DOMAIN_REFERENCES['market_rules']
            )
            response = self._call_llm(system_prompt, step3_prompt)
            parsed = self._parse_json(response)
            
            if parsed and 'market_recommendations' in parsed:
                market_recs = {}
                for name, data in parsed['market_recommendations'].items():
                    market_recs[name] = MarketRecommendation(
                        market=name,
                        tip=data.get('tip', 'No'),
                        probability=data.get('probability', 0.5),
                        confidence=data.get('confidence', 0.5),
                        edge=data.get('edge', 0),
                        reasoning=data.get('reasoning', ''),
                        scenario_support=data.get('scenario_support', 0.5),
                        structural_justification=data.get('structural_justification', '')
                    )
                reasoning_chain.append("  → LLM market evaluation successful")
            else:
                market_recs = self._statistical_market_evaluation(
                    scenarios, priors, home_analysis, away_analysis
                )
                reasoning_chain.append("  → Fallback to statistical evaluation")
        else:
            market_recs = self._statistical_market_evaluation(
                scenarios, priors, home_analysis, away_analysis
            )
            reasoning_chain.append("  → Statistical market evaluation")
        
        # Finalize
        processing_time = int((time.time() - start_time) * 1000)
        
        result = GoalDirectedAnalysis(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            date=date,
            home_analysis=home_analysis,
            away_analysis=away_analysis,
            scenarios=scenarios,
            market_recommendations=market_recs,
            system_goal=SYSTEM_GOAL,
            references_used=references_used,
            reasoning_chain=reasoning_chain,
            processing_time_ms=processing_time,
            model_used='llm' if (use_llm and self._llm_available) else 'statistical'
        )
        
        return result
    
    def analyze_match_advanced(
        self,
        match_data: Dict,
        historical_context: Dict = None,
        market_odds: Dict[str, float] = None,
        use_llm: bool = True
    ) -> GoalDirectedAnalysis:
        """
        🎯 ADVANCED: Run complete pipeline using AdvancedPromptBuilder.
        
        This method uses the new advanced prompt templates with:
        - Hardcoded system objectives
        - Embedded domain references
        - Structured JSON outputs
        - Step-by-step reasoning with validation
        
        Args:
            match_data: Basic match info (home, away, league, date)
            historical_context: H2H, form, injuries, etc.
            market_odds: Current odds for markets
            use_llm: Whether to use LLM (fallback to statistical if False)
        
        Returns:
            GoalDirectedAnalysis with complete reasoning chain
        """
        import time
        start_time = time.time()
        
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        league = match_data.get('league', 'Unknown')
        date = match_data.get('date', 'Unknown')
        match_id = f"{home_team}_vs_{away_team}_{date}"
        
        historical_context = historical_context or {}
        market_odds = market_odds or {'btts': 1.55, 'over_1_5': 1.50, 'over_2_5': 1.65}
        
        priors = self._get_league_priors(league)
        reasoning_chain = []
        references_used = ['league_priors', 'market_rules', 'tactical_rules', 'structural_priors']
        
        reasoning_chain.append("🎯 ADVANCED PIPELINE: Using AdvancedPromptBuilder")
        
        # Step 1: Team Tactical Analysis
        reasoning_chain.append("Step 1: Team Tactical Analysis (Advanced Template)")
        
        if use_llm and self._llm_available:
            step1_prompt = self.prompt_builder.build_step_1_prompt(
                match_data=match_data,
                historical_context=historical_context,
                league_priors=priors
            )
            
            response = self._call_llm("", step1_prompt)  # System goal embedded in prompt
            parsed = extract_json_from_response(response)
            
            if parsed and validate_step_1_output(parsed):
                home_analysis = TeamTacticalAnalysis.from_dict(parsed['home_team_analysis'])
                away_analysis = TeamTacticalAnalysis.from_dict(parsed['away_team_analysis'])
                reasoning_chain.append("  ✓ LLM analysis validated and parsed")
            else:
                home_analysis = self._statistical_team_analysis(home_team, {})
                away_analysis = self._statistical_team_analysis(away_team, {})
                reasoning_chain.append("  ✗ LLM failed validation → statistical fallback")
        else:
            home_analysis = self._statistical_team_analysis(home_team, {})
            away_analysis = self._statistical_team_analysis(away_team, {})
            reasoning_chain.append("  → Statistical analysis (LLM not available)")
        
        # Step 2: Scenario Simulation
        reasoning_chain.append("Step 2: Scenario Simulation (Tree-of-Thought)")
        
        if use_llm and self._llm_available:
            step2_prompt = self.prompt_builder.build_step_2_prompt(
                team_analysis={
                    'home': home_analysis.to_dict(),
                    'away': away_analysis.to_dict()
                },
                h2h_context=historical_context.get('h2h', {}),
                form_context=historical_context.get('form', {})
            )
            
            response = self._call_llm("", step2_prompt)
            parsed = extract_json_from_response(response)
            
            if parsed and validate_step_2_output(parsed):
                scenarios = [
                    MatchScenario(
                        description=s['description'],
                        home_goals=s['home_goals'],
                        away_goals=s['away_goals'],
                        probability=s['probability'],
                        key_events=s.get('key_events', [])
                    )
                    for s in parsed['scenarios']
                ]
                reasoning_chain.append(f"  ✓ Generated {len(scenarios)} validated scenarios")
            else:
                scenarios = self._statistical_scenarios(home_analysis, away_analysis, priors)
                reasoning_chain.append("  ✗ Scenario validation failed → statistical fallback")
        else:
            scenarios = self._statistical_scenarios(home_analysis, away_analysis, priors)
            reasoning_chain.append("  → Statistical scenarios generated")
        
        # Step 3: Market Evaluation
        reasoning_chain.append("Step 3: Market Evaluation with Confidence")
        
        if use_llm and self._llm_available:
            step3_prompt = self.prompt_builder.build_step_3_prompt(
                scenarios=[s.to_dict() for s in scenarios],
                market_odds=market_odds,
                structural_priors=priors,
                tactical_rules=DOMAIN_REFERENCES['tactical_rules']
            )
            
            response = self._call_llm("", step3_prompt)
            parsed = extract_json_from_response(response)
            
            if parsed and validate_step_3_output(parsed):
                market_recs = {}
                for market_name, data in parsed['market_recommendations'].items():
                    market_recs[market_name] = MarketRecommendation(
                        market=market_name,
                        tip=data['tip'],
                        probability=data['probability'],
                        confidence=data['confidence'],
                        edge=data['edge'],
                        reasoning=data['reasoning'],
                        scenario_support=data.get('scenario_support', 0.5),
                        structural_justification=data.get('structural_justification', '')
                    )
                reasoning_chain.append("  ✓ Market evaluation validated")
            else:
                market_recs = self._statistical_market_evaluation(
                    scenarios, priors, home_analysis, away_analysis
                )
                reasoning_chain.append("  ✗ Market validation failed → statistical fallback")
        else:
            market_recs = self._statistical_market_evaluation(
                scenarios, priors, home_analysis, away_analysis
            )
            reasoning_chain.append("  → Statistical market evaluation")
        
        # Finalize
        processing_time = int((time.time() - start_time) * 1000)
        
        result = GoalDirectedAnalysis(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            date=date,
            home_analysis=home_analysis,
            away_analysis=away_analysis,
            scenarios=scenarios,
            market_recommendations=market_recs,
            system_goal=self.prompt_builder.system_objective,
            references_used=references_used,
            reasoning_chain=reasoning_chain,
            processing_time_ms=processing_time,
            model_used='advanced_llm' if (use_llm and self._llm_available) else 'statistical'
        )
        
        return result
    
    def build_multi_bet_ticket(
        self,
        analyses: List[GoalDirectedAnalysis],
        stake: float = 50.0
    ) -> Optional[MultiBeTicket]:
        """
        Step 4: Build multi-bet ticket from analyses.
        
        Selects best legs meeting market rules.
        """
        rules = DOMAIN_REFERENCES['market_rules']
        candidates = []
        
        for analysis in analyses:
            for market_name, rec in analysis.market_recommendations.items():
                if not rec.is_actionable:
                    continue
                
                # Calculate odds from probability
                if rec.probability > 0 and rec.probability < 1:
                    fair_odds = 1 / rec.probability
                    market_odds = fair_odds * 0.95  # 5% margin
                else:
                    continue
                
                # Check odds range
                if market_odds < rules['min_single_odds'] or market_odds > rules['max_single_odds']:
                    continue
                
                candidates.append(BetLeg(
                    match_id=analysis.match_id,
                    home_team=analysis.home_team,
                    away_team=analysis.away_team,
                    league=analysis.league,
                    market=market_name,
                    tip=rec.tip,
                    odds=round(market_odds, 2),
                    probability=rec.probability,
                    confidence=rec.confidence,
                    edge=rec.edge,
                    reasoning=rec.reasoning
                ))
        
        if len(candidates) < rules['min_legs']:
            logger.warning(f"Not enough candidates: {len(candidates)} < {rules['min_legs']}")
            return None
        
        # Sort by edge * confidence (best value first)
        candidates.sort(key=lambda x: x.edge * x.confidence, reverse=True)
        
        # Greedy selection to hit target odds
        selected = []
        current_odds = 1.0
        
        for leg in candidates:
            if len(selected) >= rules['max_legs']:
                break
            
            potential_odds = current_odds * leg.odds
            
            if potential_odds <= rules['target_total_odds'] * 1.5:  # Allow 50% overshoot
                selected.append(leg)
                current_odds = potential_odds
        
        if len(selected) < rules['min_legs']:
            return None
        
        return MultiBeTicket.from_legs(selected, stake)


# =============================================================================
# INTEGRATION WITH ORCHESTRATOR
# =============================================================================

class GoalDirectedOrchestrator:
    """
    🎯 Complete Goal-Directed Pipeline Orchestrator
    
    Integrates:
    - Goal-directed reasoning engine
    - ML model predictions
    - Multi-bet builder
    - Walk-forward backtest
    """
    
    def __init__(self, data_path: str = None):
        from pathlib import Path
        import pandas as pd
        
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = data_path or str(
            self.project_root / 'data/historical/massive_training_data.csv'
        )
        
        self.reasoning_engine = GoalDirectedReasoningEngine()
        self._data = None
        
        logger.info("🎯 GoalDirectedOrchestrator initialized")
    
    def load_data(self):
        """Load historical data."""
        import pandas as pd
        
        if self._data is None:
            self._data = pd.read_csv(self.data_path)
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data = self._data.sort_values('date').reset_index(drop=True)
        
        return self._data
    
    def analyze_matches(
        self,
        matches: List[Dict],
        use_llm: bool = False
    ) -> List[GoalDirectedAnalysis]:
        """Analyze multiple matches."""
        analyses = []
        
        for match in matches:
            home_stats = {
                'goals_avg': match.get('home_goals_avg', 1.5),
                'conceded_avg': match.get('home_conceded_avg', 1.2),
                'elo': match.get('home_elo', 1500),
                'form': match.get('home_form', 0.5)
            }
            away_stats = {
                'goals_avg': match.get('away_goals_avg', 1.3),
                'conceded_avg': match.get('away_conceded_avg', 1.4),
                'elo': match.get('away_elo', 1500),
                'form': match.get('away_form', 0.5)
            }
            
            analysis = self.reasoning_engine.analyze_match(
                match, home_stats, away_stats, use_llm=use_llm
            )
            analyses.append(analysis)
        
        return analyses
    
    def generate_daily_ticket(
        self,
        date: str = None,
        stake: float = 50.0
    ) -> Optional[MultiBeTicket]:
        """Generate daily multi-bet ticket."""
        import pandas as pd
        
        df = self.load_data()
        
        if date:
            date = pd.to_datetime(date)
            matches = df[df['date'] == date].to_dict('records')
        else:
            # Use most recent matches
            latest_date = df['date'].max()
            matches = df[df['date'] == latest_date].to_dict('records')
        
        if not matches:
            logger.warning("No matches found for date")
            return None
        
        # Format for analysis
        formatted = []
        for m in matches[:10]:  # Limit to 10
            formatted.append({
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'league': m.get('league', 'Unknown'),
                'date': str(m['date']),
                'home_elo': m.get('home_elo', 1500),
                'away_elo': m.get('away_elo', 1500),
                'home_form': m.get('home_form', 0.5),
                'away_form': m.get('away_form', 0.5)
            })
        
        analyses = self.analyze_matches(formatted, use_llm=False)
        ticket = self.reasoning_engine.build_multi_bet_ticket(analyses, stake)
        
        return ticket


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SYSTEM_GOAL',
    'DOMAIN_REFERENCES',
    'TeamTacticalAnalysis',
    'MatchScenario',
    'MarketRecommendation',
    'BetLeg',
    'MultiBeTicket',
    'GoalDirectedAnalysis',
    'PromptTemplates',
    'GoalDirectedReasoningEngine',
    'GoalDirectedOrchestrator'
]

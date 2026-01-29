"""
🧠 Deep Reasoning Agent - Battle-Tested Multi-Step LLM Reasoning
================================================================
Implements Chain-of-Thought, Scenario Simulation, and Market Evaluation
following Tree-of-Thoughts patterns (Microsoft Research 2023).

Architecture:
    Step 1: Team Analysis (tactical strengths/weaknesses)
    Step 2: Scenario Simulation (3 plausible outcomes)
    Step 3: Market Evaluation (BTTS, Over 1.5, Over 2.5)
    Step 4: Confidence Scoring with Priors

This is NOT a simple stat summarizer - it's a structured reasoning engine.
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

logger = logging.getLogger(__name__)


@dataclass
class TeamAnalysis:
    """Structured team analysis output."""
    team_name: str
    attacking_strength: float  # 0-1
    defensive_risk: float      # 0-1
    transition_danger: float   # 0-1
    pressing_intensity: float  # 0-1
    set_piece_threat: float    # 0-1
    tactical_summary: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MatchScenario:
    """A plausible match outcome scenario."""
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
    
    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'home_goals': self.home_goals,
            'away_goals': self.away_goals,
            'total_goals': self.total_goals,
            'probability': self.probability,
            'btts': self.btts,
            'key_events': self.key_events
        }


@dataclass
class MarketRecommendation:
    """Market recommendation with confidence."""
    market: str
    tip: str
    confidence: float
    probability: float
    reasoning: str
    scenario_support: float  # % of scenarios supporting this
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class DeepAnalysisResult:
    """Complete deep analysis result for a match."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    
    # Analysis outputs
    home_analysis: TeamAnalysis
    away_analysis: TeamAnalysis
    scenarios: List[MatchScenario]
    market_recommendations: Dict[str, MarketRecommendation]
    
    # Meta
    reasoning_steps: List[str]
    processing_time_ms: int
    model_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
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
            'reasoning_steps': self.reasoning_steps,
            'processing_time_ms': self.processing_time_ms,
            'model_used': self.model_used,
            'timestamp': self.timestamp
        }


class DeepReasoningAgent:
    """
    🧠 Battle-Tested Deep Reasoning Agent
    
    Implements multi-step reasoning with:
    1. Chain-of-Thought prompting
    2. Scenario simulation (Tree-of-Thoughts)
    3. Structured JSON outputs
    4. Confidence scoring with priors
    5. Caching to avoid re-computation
    
    Usage:
        agent = DeepReasoningAgent()
        result = agent.analyze_match(match_data, team_stats, context)
    """
    
    # Historical priors by league (battle-tested baselines)
    LEAGUE_PRIORS = {
        'bundesliga': {'btts_rate': 0.53, 'over_1_5_rate': 0.76, 'over_2_5_rate': 0.52, 'avg_goals': 2.95},
        'premier_league': {'btts_rate': 0.52, 'over_1_5_rate': 0.75, 'over_2_5_rate': 0.51, 'avg_goals': 2.85},
        'la_liga': {'btts_rate': 0.48, 'over_1_5_rate': 0.72, 'over_2_5_rate': 0.47, 'avg_goals': 2.65},
        'serie_a': {'btts_rate': 0.49, 'over_1_5_rate': 0.73, 'over_2_5_rate': 0.49, 'avg_goals': 2.75},
        'ligue_1': {'btts_rate': 0.47, 'over_1_5_rate': 0.71, 'over_2_5_rate': 0.46, 'avg_goals': 2.60},
        'eredivisie': {'btts_rate': 0.58, 'over_1_5_rate': 0.82, 'over_2_5_rate': 0.60, 'avg_goals': 3.20},
        'championship': {'btts_rate': 0.51, 'over_1_5_rate': 0.74, 'over_2_5_rate': 0.50, 'avg_goals': 2.80},
    }
    
    DEFAULT_PRIORS = {'btts_rate': 0.50, 'over_1_5_rate': 0.74, 'over_2_5_rate': 0.50, 'avg_goals': 2.70}
    
    def __init__(
        self,
        cache_dir: str = None,
        llm_backend: str = 'ollama',
        model_name: str = 'deepseek-llm:7b-chat',
        use_cache: bool = True
    ):
        self.cache_dir = Path(cache_dir or 'data/reasoning_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = self.cache_dir / 'reasoning_cache.db'
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.use_cache = use_cache
        
        self._llm_client = None
        self._init_cache_db()
        self._init_llm()
        
        logger.info(f"🧠 DeepReasoningAgent initialized (backend={llm_backend})")
    
    def _init_cache_db(self):
        """Initialize SQLite cache."""
        conn = sqlite3.connect(str(self.cache_db))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                match_hash TEXT PRIMARY KEY,
                match_id TEXT,
                result_json TEXT,
                created_at TEXT,
                ttl_hours INTEGER DEFAULT 168
            )
        ''')
        conn.commit()
        conn.close()
    
    def _init_llm(self):
        """Initialize LLM client."""
        if self.llm_backend == 'ollama':
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code == 200:
                    self._llm_client = 'ollama'
                    logger.info("✅ Ollama backend connected")
                else:
                    self._llm_client = None
            except:
                self._llm_client = None
                logger.warning("⚠️ Ollama not available, using statistical fallback")
    
    def _get_cache_key(self, match_data: Dict) -> str:
        """Generate cache key from match data."""
        key_data = f"{match_data.get('home_team')}_{match_data.get('away_team')}_{match_data.get('date')}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[DeepAnalysisResult]:
        """Check cache for existing analysis."""
        if not self.use_cache:
            return None
            
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.execute(
            'SELECT result_json FROM reasoning_cache WHERE match_hash = ?',
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                return json.loads(row[0])
            except:
                return None
        return None
    
    def _save_cache(self, cache_key: str, match_id: str, result: Dict):
        """Save analysis to cache."""
        conn = sqlite3.connect(str(self.cache_db))
        conn.execute('''
            INSERT OR REPLACE INTO reasoning_cache 
            (match_hash, match_id, result_json, created_at)
            VALUES (?, ?, ?, ?)
        ''', (cache_key, match_id, json.dumps(result), datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def _get_league_priors(self, league: str) -> Dict:
        """Get historical priors for league."""
        league_key = league.lower().replace(' ', '_').replace('-', '_')
        return self.LEAGUE_PRIORS.get(league_key, self.DEFAULT_PRIORS)
    
    def _build_step1_prompt(self, match_data: Dict, team_stats: Dict) -> str:
        """Build Step 1: Team Analysis prompt."""
        return f"""You are an expert football tactical analyst. Analyze the tactical profile of each team.

MATCH: {match_data['home_team']} vs {match_data['away_team']}
LEAGUE: {match_data.get('league', 'Unknown')}
DATE: {match_data.get('date', 'Unknown')}

HOME TEAM STATS ({match_data['home_team']}):
- Goals Scored (avg): {team_stats.get('home_goals_avg', 1.5):.2f}
- Goals Conceded (avg): {team_stats.get('home_conceded_avg', 1.2):.2f}
- Home Elo Rating: {team_stats.get('home_elo', 1500)}
- Recent Form: {team_stats.get('home_form', 0.5):.2f}
- Transition Speed Index: {team_stats.get('home_transition', 0.5):.2f}

AWAY TEAM STATS ({match_data['away_team']}):
- Goals Scored (avg): {team_stats.get('away_goals_avg', 1.3):.2f}
- Goals Conceded (avg): {team_stats.get('away_conceded_avg', 1.4):.2f}
- Away Elo Rating: {team_stats.get('away_elo', 1500)}
- Recent Form: {team_stats.get('away_form', 0.5):.2f}
- Transition Speed Index: {team_stats.get('away_transition', 0.5):.2f}

Analyze each team and output ONLY valid JSON:
{{
  "home_analysis": {{
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "tactical_summary": "brief tactical description"
  }},
  "away_analysis": {{
    "attacking_strength": 0.0-1.0,
    "defensive_risk": 0.0-1.0,
    "transition_danger": 0.0-1.0,
    "pressing_intensity": 0.0-1.0,
    "set_piece_threat": 0.0-1.0,
    "tactical_summary": "brief tactical description"
  }}
}}"""

    def _build_step2_prompt(self, match_data: Dict, team_analysis: Dict) -> str:
        """Build Step 2: Scenario Simulation prompt."""
        return f"""Based on the tactical analysis, simulate 3 plausible match scenarios.

MATCH: {match_data['home_team']} vs {match_data['away_team']}

TACTICAL ANALYSIS:
Home ({match_data['home_team']}):
- Attacking: {team_analysis['home_analysis']['attacking_strength']:.2f}
- Defensive Risk: {team_analysis['home_analysis']['defensive_risk']:.2f}
- Transition: {team_analysis['home_analysis']['transition_danger']:.2f}

Away ({match_data['away_team']}):
- Attacking: {team_analysis['away_analysis']['attacking_strength']:.2f}
- Defensive Risk: {team_analysis['away_analysis']['defensive_risk']:.2f}
- Transition: {team_analysis['away_analysis']['transition_danger']:.2f}

Generate 3 scenarios with probabilities that sum to 1.0. Output ONLY valid JSON:
{{
  "scenarios": [
    {{
      "description": "scenario description",
      "home_goals": integer,
      "away_goals": integer,
      "probability": 0.0-1.0,
      "key_events": ["event1", "event2"]
    }},
    {{...}},
    {{...}}
  ]
}}"""

    def _build_step3_prompt(self, match_data: Dict, scenarios: List[Dict], priors: Dict) -> str:
        """Build Step 3: Market Evaluation prompt."""
        return f"""Evaluate betting markets based on scenarios and historical priors.

MATCH: {match_data['home_team']} vs {match_data['away_team']}

SCENARIOS:
{json.dumps(scenarios, indent=2)}

LEAGUE HISTORICAL PRIORS:
- BTTS Rate: {priors['btts_rate']:.0%}
- Over 1.5 Rate: {priors['over_1_5_rate']:.0%}
- Over 2.5 Rate: {priors['over_2_5_rate']:.0%}
- Average Goals: {priors['avg_goals']:.2f}

Calculate probability for each market by weighting scenarios + priors.
Output ONLY valid JSON:
{{
  "market_recommendations": {{
    "btts": {{
      "tip": "Yes" or "No",
      "confidence": 0.0-1.0,
      "probability": 0.0-1.0,
      "reasoning": "brief explanation",
      "scenario_support": 0.0-1.0
    }},
    "over_1_5": {{...}},
    "over_2_5": {{...}}
  }}
}}"""

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM backend."""
        if self._llm_client == 'ollama':
            try:
                import requests
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': self.model_name,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.3,
                            'num_predict': 2048
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    return response.json().get('response', '')
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
        return None
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        if not text:
            return None
            
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON block
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match if isinstance(match, str) else match)
                except:
                    continue
        
        return None
    
    def _statistical_team_analysis(self, match_data: Dict, team_stats: Dict) -> Dict:
        """Fallback: Statistical team analysis without LLM."""
        home_attack = min(1.0, team_stats.get('home_goals_avg', 1.5) / 2.5)
        home_defense = min(1.0, team_stats.get('home_conceded_avg', 1.2) / 2.0)
        away_attack = min(1.0, team_stats.get('away_goals_avg', 1.3) / 2.5)
        away_defense = min(1.0, team_stats.get('away_conceded_avg', 1.4) / 2.0)
        
        home_elo = team_stats.get('home_elo', 1500)
        away_elo = team_stats.get('away_elo', 1500)
        elo_factor = (home_elo - away_elo) / 400
        
        return {
            'home_analysis': {
                'attacking_strength': home_attack,
                'defensive_risk': home_defense,
                'transition_danger': min(1.0, home_attack * 0.8 + 0.2),
                'pressing_intensity': 0.5 + elo_factor * 0.1,
                'set_piece_threat': 0.4,
                'tactical_summary': f"Home side with {home_attack:.0%} attacking strength"
            },
            'away_analysis': {
                'attacking_strength': away_attack,
                'defensive_risk': away_defense,
                'transition_danger': min(1.0, away_attack * 0.7 + 0.1),
                'pressing_intensity': 0.5 - elo_factor * 0.1,
                'set_piece_threat': 0.35,
                'tactical_summary': f"Away side with {away_attack:.0%} attacking strength"
            }
        }
    
    def _statistical_scenarios(self, team_analysis: Dict, team_stats: Dict) -> List[Dict]:
        """Fallback: Statistical scenario generation."""
        home_attack = team_analysis['home_analysis']['attacking_strength']
        away_attack = team_analysis['away_analysis']['attacking_strength']
        home_defense = team_analysis['home_analysis']['defensive_risk']
        away_defense = team_analysis['away_analysis']['defensive_risk']
        
        # Expected goals based on attack vs defense
        home_xg = home_attack * away_defense * 2.5 + 0.5
        away_xg = away_attack * home_defense * 2.0 + 0.3
        
        # Generate 3 scenarios
        scenarios = []
        
        # Scenario 1: Expected outcome
        scenarios.append({
            'description': 'Expected outcome based on team strengths',
            'home_goals': round(home_xg),
            'away_goals': round(away_xg),
            'probability': 0.50,
            'key_events': ['Normal game flow', 'Expected possession patterns']
        })
        
        # Scenario 2: Home dominance
        scenarios.append({
            'description': 'Home team dominates',
            'home_goals': round(home_xg + 1),
            'away_goals': max(0, round(away_xg - 0.5)),
            'probability': 0.30,
            'key_events': ['Early home goal', 'Away team struggles']
        })
        
        # Scenario 3: Away surprise
        scenarios.append({
            'description': 'Away team surprises',
            'home_goals': max(0, round(home_xg - 0.5)),
            'away_goals': round(away_xg + 1),
            'probability': 0.20,
            'key_events': ['Counter-attack goals', 'Home defense errors']
        })
        
        return scenarios
    
    def _statistical_market_evaluation(
        self, 
        scenarios: List[Dict], 
        priors: Dict,
        team_stats: Dict
    ) -> Dict:
        """Fallback: Statistical market evaluation."""
        
        # Calculate scenario-based probabilities
        btts_prob = sum(
            s['probability'] for s in scenarios 
            if s['home_goals'] > 0 and s['away_goals'] > 0
        )
        
        over_1_5_prob = sum(
            s['probability'] for s in scenarios 
            if s['home_goals'] + s['away_goals'] > 1.5
        )
        
        over_2_5_prob = sum(
            s['probability'] for s in scenarios 
            if s['home_goals'] + s['away_goals'] > 2.5
        )
        
        # Blend with priors (60% model, 40% priors)
        btts_final = btts_prob * 0.6 + priors['btts_rate'] * 0.4
        over_1_5_final = over_1_5_prob * 0.6 + priors['over_1_5_rate'] * 0.4
        over_2_5_final = over_2_5_prob * 0.6 + priors['over_2_5_rate'] * 0.4
        
        return {
            'btts': {
                'tip': 'Yes' if btts_final > 0.55 else 'No',
                'confidence': abs(btts_final - 0.5) * 2,
                'probability': btts_final,
                'reasoning': f"Scenario support: {btts_prob:.0%}, Prior: {priors['btts_rate']:.0%}",
                'scenario_support': btts_prob
            },
            'over_1_5': {
                'tip': 'Yes' if over_1_5_final > 0.55 else 'No',
                'confidence': abs(over_1_5_final - 0.5) * 2,
                'probability': over_1_5_final,
                'reasoning': f"Scenario support: {over_1_5_prob:.0%}, Prior: {priors['over_1_5_rate']:.0%}",
                'scenario_support': over_1_5_prob
            },
            'over_2_5': {
                'tip': 'Yes' if over_2_5_final > 0.55 else 'No',
                'confidence': abs(over_2_5_final - 0.5) * 2,
                'probability': over_2_5_final,
                'reasoning': f"Scenario support: {over_2_5_prob:.0%}, Prior: {priors['over_2_5_rate']:.0%}",
                'scenario_support': over_2_5_prob
            }
        }
    
    def analyze_match(
        self,
        match_data: Dict,
        team_stats: Dict = None,
        use_llm: bool = True
    ) -> DeepAnalysisResult:
        """
        🎯 Main entry point: Perform deep multi-step analysis.
        
        Args:
            match_data: Match info (home_team, away_team, league, date)
            team_stats: Team statistics dictionary
            use_llm: Whether to use LLM (True) or statistical fallback (False)
        
        Returns:
            DeepAnalysisResult with complete analysis
        """
        import time
        start_time = time.time()
        
        match_id = f"{match_data['home_team']}_vs_{match_data['away_team']}_{match_data.get('date', 'unknown')}"
        cache_key = self._get_cache_key(match_data)
        
        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for {match_id}")
            return cached
        
        team_stats = team_stats or {}
        priors = self._get_league_priors(match_data.get('league', ''))
        reasoning_steps = []
        
        # Step 1: Team Analysis
        reasoning_steps.append("Step 1: Analyzing team tactical profiles")
        
        if use_llm and self._llm_client:
            prompt1 = self._build_step1_prompt(match_data, team_stats)
            response1 = self._call_llm(prompt1)
            team_analysis = self._parse_json_response(response1)
            
            if not team_analysis:
                team_analysis = self._statistical_team_analysis(match_data, team_stats)
                reasoning_steps.append("  → LLM parse failed, using statistical fallback")
        else:
            team_analysis = self._statistical_team_analysis(match_data, team_stats)
            reasoning_steps.append("  → Using statistical analysis (LLM not available)")
        
        # Step 2: Scenario Simulation
        reasoning_steps.append("Step 2: Simulating match scenarios (Tree-of-Thoughts)")
        
        if use_llm and self._llm_client:
            prompt2 = self._build_step2_prompt(match_data, team_analysis)
            response2 = self._call_llm(prompt2)
            scenarios_data = self._parse_json_response(response2)
            
            if scenarios_data and 'scenarios' in scenarios_data:
                scenarios = scenarios_data['scenarios']
            else:
                scenarios = self._statistical_scenarios(team_analysis, team_stats)
                reasoning_steps.append("  → LLM parse failed, using statistical scenarios")
        else:
            scenarios = self._statistical_scenarios(team_analysis, team_stats)
        
        reasoning_steps.append(f"  → Generated {len(scenarios)} scenarios")
        
        # Step 3: Market Evaluation
        reasoning_steps.append("Step 3: Evaluating markets with confidence scoring")
        
        if use_llm and self._llm_client:
            prompt3 = self._build_step3_prompt(match_data, scenarios, priors)
            response3 = self._call_llm(prompt3)
            markets_data = self._parse_json_response(response3)
            
            if markets_data and 'market_recommendations' in markets_data:
                market_recs = markets_data['market_recommendations']
            else:
                market_recs = self._statistical_market_evaluation(scenarios, priors, team_stats)
                reasoning_steps.append("  → LLM parse failed, using statistical evaluation")
        else:
            market_recs = self._statistical_market_evaluation(scenarios, priors, team_stats)
        
        # Build result objects
        home_analysis = TeamAnalysis(
            team_name=match_data['home_team'],
            **team_analysis.get('home_analysis', {})
        )
        
        away_analysis = TeamAnalysis(
            team_name=match_data['away_team'],
            **team_analysis.get('away_analysis', {})
        )
        
        scenario_objs = [
            MatchScenario(
                description=s.get('description', ''),
                home_goals=s.get('home_goals', 1),
                away_goals=s.get('away_goals', 1),
                probability=s.get('probability', 0.33),
                key_events=s.get('key_events', [])
            )
            for s in scenarios
        ]
        
        market_objs = {
            name: MarketRecommendation(
                market=name,
                tip=data.get('tip', 'No'),
                confidence=data.get('confidence', 0.5),
                probability=data.get('probability', 0.5),
                reasoning=data.get('reasoning', ''),
                scenario_support=data.get('scenario_support', 0.5)
            )
            for name, data in market_recs.items()
        }
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = DeepAnalysisResult(
            match_id=match_id,
            home_team=match_data['home_team'],
            away_team=match_data['away_team'],
            league=match_data.get('league', 'Unknown'),
            date=match_data.get('date', 'Unknown'),
            home_analysis=home_analysis,
            away_analysis=away_analysis,
            scenarios=scenario_objs,
            market_recommendations=market_objs,
            reasoning_steps=reasoning_steps,
            processing_time_ms=processing_time,
            model_used='llm' if (use_llm and self._llm_client) else 'statistical'
        )
        
        # Cache result
        self._save_cache(cache_key, match_id, result.to_dict())
        
        return result
    
    def get_market_probability(
        self,
        match_data: Dict,
        team_stats: Dict,
        market: str
    ) -> Tuple[float, float]:
        """
        Quick method to get probability and confidence for a specific market.
        
        Returns:
            (probability, confidence)
        """
        result = self.analyze_match(match_data, team_stats, use_llm=False)
        
        market_key = market.lower().replace(' ', '_').replace('.', '_')
        
        if market_key in result.market_recommendations:
            rec = result.market_recommendations[market_key]
            return rec.probability, rec.confidence
        
        return 0.5, 0.0


# Singleton for efficiency
_agent_instance = None

def get_reasoning_agent() -> DeepReasoningAgent:
    """Get or create singleton reasoning agent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DeepReasoningAgent()
    return _agent_instance

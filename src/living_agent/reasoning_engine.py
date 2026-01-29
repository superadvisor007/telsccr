"""
🧠 STRUCTURAL REASONING ENGINE - Multi-Step Chain-of-Thought
============================================================
DeepSeek 7B powered reasoning with simulation and curiosity.

This is the "brain" of the Living Agent:
1. Multi-step reasoning chains (not single-pass stats)
2. Scenario simulation integration
3. Curiosity prompts for hidden edges
4. Reflection and self-correction

Prompt Engineering:
- Long-form prompts with embedded knowledge
- Chunked analysis for complex matches
- Chain-of-thought with explicit reasoning steps
- Contrarian checks (devil's advocate)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from living_agent.knowledge_cache import KnowledgeCache, CachedAnalysis
from living_agent.scenario_simulator import (
    ScenarioSimulator, TeamProfile, SimulationResult,
    create_team_profile_from_stats
)


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    step_type: str  # 'analysis', 'simulation', 'curiosity', 'reflection', 'decision'
    thought: str
    evidence: List[str]
    confidence: float
    time_ms: int


@dataclass
class MarketAnalysis:
    """Analysis for a single betting market."""
    market: str  # 'btts', 'over_1_5', 'over_2_5'
    probability: float
    confidence: float
    reasoning: str
    key_factors: List[str]
    contrarian_view: str
    recommendation: str  # 'BET', 'AVOID', 'SKIP'
    suggested_odds_range: Tuple[float, float]


@dataclass
class MatchReasoning:
    """Complete reasoning chain for a match."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    reasoning_steps: List[ReasoningStep]
    market_analyses: List[MarketAnalysis]
    simulation_result: Optional[SimulationResult]
    overall_confidence: float
    key_insights: List[str]
    curiosity_findings: List[str]
    total_reasoning_time_ms: int
    timestamp: str


class StructuralReasoningEngine:
    """
    🧠 Multi-Step Reasoning Engine with DeepSeek 7B
    
    Implements:
    1. Chain-of-Thought prompting
    2. Scenario simulation integration
    3. Curiosity-driven exploration
    4. Self-reflection and correction
    5. Caching for efficiency
    
    Can run "as long as it needs" because results are cached.
    No rush - deep reasoning for quality predictions.
    """
    
    def __init__(
        self,
        cache: KnowledgeCache = None,
        simulator: ScenarioSimulator = None,
        use_llm: bool = True,
        verbose: bool = True
    ):
        self.cache = cache or KnowledgeCache()
        self.simulator = simulator or ScenarioSimulator()
        self.use_llm = use_llm
        self.verbose = verbose
        
        # Try to load DeepSeek LLM
        self.llm = None
        if use_llm:
            try:
                from llm.deepseek_client import DeepSeekLLM
                self.llm = DeepSeekLLM()
                if self.verbose:
                    print("✅ DeepSeek 7B LLM loaded")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ LLM not available: {e} - using statistical fallback")
        
        # Reasoning templates
        self.reasoning_templates = self._load_reasoning_templates()
        
        # Market characteristics - adjusted for realistic edge finding
        self.market_info = {
            'btts': {
                'name': 'Both Teams to Score',
                'typical_odds': (1.70, 2.10),  # Higher odds = lower implied prob
                'golden_range': (1.75, 1.95),
                'key_factors': ['attack_strength', 'defense_weakness', 'open_play_tendency'],
            },
            'over_1_5': {
                'name': 'Over 1.5 Goals',
                'typical_odds': (1.25, 1.55),  # Adjusted for realistic edge
                'golden_range': (1.30, 1.50),
                'key_factors': ['total_xg', 'attack_combined', 'league_goals'],
            },
            'over_2_5': {
                'name': 'Over 2.5 Goals',
                'typical_odds': (1.80, 2.50),  # Higher range
                'golden_range': (1.85, 2.20),
                'key_factors': ['attacking_teams', 'defensive_weakness', 'high_line'],
            },
        }
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load prompt templates for different reasoning steps."""
        return {
            'team_analysis': """
Analyze {team_name} tactical profile:

Current Form: {form_description}
Attack Metrics: {attack_metrics}
Defense Metrics: {defense_metrics}
League Position: {league_position}

Consider:
1. How do they typically set up tactically?
2. What are their main attacking threats?
3. Where are their defensive vulnerabilities?
4. How does current form affect confidence?

Provide structured analysis with confidence scores.
""",
            
            'match_interaction': """
Analyze the structural interaction between {home_team} and {away_team}:

Home Team Profile:
{home_profile}

Away Team Profile:
{away_profile}

League Tendencies:
{league_tendencies}

Consider:
1. How do their tactical styles interact?
2. Where are the potential mismatches?
3. What scenarios could unfold?
4. What chaos factors could influence the outcome?

Think through multiple scenarios, not just the most likely one.
""",
            
            'market_evaluation': """
Evaluate {market_name} for {home_team} vs {away_team}:

Simulation Results:
{simulation_summary}

League Prior for this market: {league_prior:.1%}
Historical Hit Rate: {historical_rate:.1%}

Consider:
1. Does the probability justify the typical odds range?
2. What factors strengthen this bet?
3. What factors weaken this bet?
4. What would make you AVOID this market?

Provide:
- Probability estimate (0-100%)
- Confidence in estimate (0-100%)
- Clear recommendation (BET/AVOID/SKIP)
- Key reasoning points
""",
            
            'curiosity_exploration': """
Look for HIDDEN EDGES in {home_team} vs {away_team}:

Standard Analysis Says:
{standard_analysis}

NOW, think like a contrarian:
1. What unusual factors could flip the prediction?
2. Are there any overlooked tactical elements?
3. Is there counter-intuitive evidence?
4. What do the stats NOT capture?

Find insights that others might miss.
""",
            
            'self_reflection': """
Review your analysis for {home_team} vs {away_team}:

Your Predictions:
{predictions}

Your Reasoning:
{reasoning_summary}

Self-Check:
1. Is your confidence calibrated properly?
2. Are you overweighting recent form?
3. Are you missing any obvious factors?
4. Would a devil's advocate disagree?

Adjust if needed. Explain any changes.
""",
        }
    
    def analyze_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_date: str = None,
        team_stats: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> MatchReasoning:
        """
        Complete match analysis with multi-step reasoning.
        
        Steps:
        1. Check cache for existing analysis
        2. Gather team profiles
        3. Run scenario simulation
        4. Deep LLM reasoning (if available)
        5. Curiosity exploration
        6. Self-reflection
        7. Final market recommendations
        8. Cache results
        
        Returns:
            MatchReasoning with complete analysis
        """
        start_time = time.time()
        match_date = match_date or datetime.now().strftime('%Y-%m-%d')
        match_id = f"{home_team}_{away_team}_{match_date}".replace(" ", "_").lower()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🧠 STRUCTURAL REASONING: {home_team} vs {away_team}")
            print(f"{'='*60}")
        
        # Step 1: Check cache
        cached = self.cache.get_cached_analysis(home_team, away_team, match_date)
        if cached:
            if self.verbose:
                print(f"✅ Found cached analysis from {cached.analysis_date}")
            return self._cached_to_reasoning(cached)
        
        reasoning_steps = []
        step_num = 0
        
        # Step 2: Build team profiles
        step_num += 1
        if self.verbose:
            print(f"\n📊 Step {step_num}: Building team profiles...")
        
        step_start = time.time()
        home_profile, away_profile = self._build_team_profiles(
            home_team, away_team, league, team_stats
        )
        reasoning_steps.append(ReasoningStep(
            step_number=step_num,
            step_type='analysis',
            thought=f"Built tactical profiles for {home_team} and {away_team}",
            evidence=[
                f"Home attack: {home_profile.attack_strength:.0%}",
                f"Away defense: {away_profile.defense_strength:.0%}",
            ],
            confidence=0.75,
            time_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 3: Run scenario simulation
        step_num += 1
        if self.verbose:
            print(f"\n🎯 Step {step_num}: Running scenario simulation...")
        
        step_start = time.time()
        league_priors = self.cache.get_all_league_insights(league) or {
            'btts_rate': 0.50, 'over_1_5_rate': 0.75, 'over_2_5_rate': 0.50
        }
        
        simulation = self.simulator.simulate_match(
            home_profile, away_profile, league_priors, context or {}
        )
        
        reasoning_steps.append(ReasoningStep(
            step_number=step_num,
            step_type='simulation',
            thought=f"Simulated {len(simulation.scenarios)} scenarios",
            evidence=simulation.key_insights[:3],
            confidence=0.70,
            time_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 4: Deep LLM reasoning (if available)
        step_num += 1
        if self.llm and self.use_llm:
            if self.verbose:
                print(f"\n🤖 Step {step_num}: Deep LLM reasoning...")
            
            step_start = time.time()
            llm_insights = self._deep_llm_reasoning(
                home_team, away_team, home_profile, away_profile,
                simulation, league_priors
            )
            reasoning_steps.append(ReasoningStep(
                step_number=step_num,
                step_type='analysis',
                thought="Deep structural reasoning with DeepSeek 7B",
                evidence=llm_insights.get('key_points', [])[:3],
                confidence=llm_insights.get('confidence', 0.65),
                time_ms=int((time.time() - step_start) * 1000)
            ))
        else:
            llm_insights = {'key_points': [], 'confidence': 0.5}
        
        # Step 5: Curiosity exploration
        step_num += 1
        if self.verbose:
            print(f"\n🔍 Step {step_num}: Curiosity exploration...")
        
        step_start = time.time()
        curiosity_findings = self._explore_curiosity(
            home_team, away_team, simulation, llm_insights
        )
        reasoning_steps.append(ReasoningStep(
            step_number=step_num,
            step_type='curiosity',
            thought="Explored hidden edges and contrarian views",
            evidence=curiosity_findings[:3],
            confidence=0.55,
            time_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 6: Market analysis
        step_num += 1
        if self.verbose:
            print(f"\n💰 Step {step_num}: Market evaluation...")
        
        step_start = time.time()
        market_analyses = self._evaluate_markets(
            home_team, away_team, simulation, league_priors,
            llm_insights, curiosity_findings
        )
        reasoning_steps.append(ReasoningStep(
            step_number=step_num,
            step_type='decision',
            thought="Evaluated all target markets",
            evidence=[f"{m.market}: {m.probability:.0%} conf={m.confidence:.0%}" for m in market_analyses],
            confidence=0.75,
            time_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 7: Self-reflection
        step_num += 1
        if self.verbose:
            print(f"\n🔄 Step {step_num}: Self-reflection...")
        
        step_start = time.time()
        market_analyses = self._self_reflect(
            home_team, away_team, market_analyses, simulation
        )
        reasoning_steps.append(ReasoningStep(
            step_number=step_num,
            step_type='reflection',
            thought="Self-checked predictions and adjusted confidence",
            evidence=["Calibrated confidence scores", "Checked for biases"],
            confidence=0.80,
            time_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Calculate overall confidence
        overall_confidence = sum(m.confidence for m in market_analyses) / len(market_analyses) if market_analyses else 0.5
        
        # Compile key insights
        key_insights = simulation.key_insights[:3]
        key_insights.extend(llm_insights.get('key_points', [])[:2])
        
        total_time = int((time.time() - start_time) * 1000)
        
        # Create result
        result = MatchReasoning(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date,
            reasoning_steps=reasoning_steps,
            market_analyses=market_analyses,
            simulation_result=simulation,
            overall_confidence=overall_confidence,
            key_insights=key_insights[:5],
            curiosity_findings=curiosity_findings[:3],
            total_reasoning_time_ms=total_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache the result
        self._cache_reasoning(result)
        
        if self.verbose:
            print(f"\n✅ Analysis complete in {total_time}ms")
            print(f"📊 Overall confidence: {overall_confidence:.0%}")
            for m in market_analyses:
                emoji = "✅" if m.recommendation == 'BET' else "❌" if m.recommendation == 'AVOID' else "⏭️"
                print(f"   {emoji} {m.market}: {m.probability:.0%} ({m.recommendation})")
        
        return result
    
    def _build_team_profiles(
        self,
        home_team: str,
        away_team: str,
        league: str,
        stats: Dict[str, Any] = None
    ) -> Tuple[TeamProfile, TeamProfile]:
        """Build tactical profiles from available data."""
        
        stats = stats or {}
        league_avg = 2.8  # Default league average
        
        # Home team stats (use defaults if not provided)
        home_stats = stats.get(home_team, {})
        home_profile = create_team_profile_from_stats(
            team_name=home_team,
            goals_scored=home_stats.get('goals_scored', 1.5),
            goals_conceded=home_stats.get('goals_conceded', 1.2),
            form_points=home_stats.get('form_points', 7),
            league_avg_goals=league_avg
        )
        
        # Away team stats
        away_stats = stats.get(away_team, {})
        away_profile = create_team_profile_from_stats(
            team_name=away_team,
            goals_scored=away_stats.get('goals_scored', 1.3),
            goals_conceded=away_stats.get('goals_conceded', 1.4),
            form_points=away_stats.get('form_points', 6),
            league_avg_goals=league_avg
        )
        
        return home_profile, away_profile
    
    def _deep_llm_reasoning(
        self,
        home_team: str,
        away_team: str,
        home_profile: TeamProfile,
        away_profile: TeamProfile,
        simulation: SimulationResult,
        league_priors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run deep LLM reasoning with chain-of-thought."""
        
        if not self.llm:
            return {'key_points': [], 'confidence': 0.5}
        
        prompt = self.reasoning_templates['match_interaction'].format(
            home_team=home_team,
            away_team=away_team,
            home_profile=f"Attack: {home_profile.attack_strength:.0%}, Defense: {home_profile.defense_strength:.0%}, Form: {home_profile.form_momentum:+.1f}",
            away_profile=f"Attack: {away_profile.attack_strength:.0%}, Defense: {away_profile.defense_strength:.0%}, Form: {away_profile.form_momentum:+.1f}",
            league_tendencies=f"Avg goals: {league_priors.get('avg_goals', 2.8):.1f}, BTTS rate: {league_priors.get('btts_rate', 0.5):.0%}"
        )
        
        try:
            response = self.llm.generate(prompt)
            
            # Parse key points from response
            key_points = []
            if 'scenario' in response.lower():
                key_points.append("Multiple scenarios considered")
            if 'tactical' in response.lower():
                key_points.append("Tactical interaction analyzed")
            if 'mismatch' in response.lower():
                key_points.append("Potential mismatches identified")
            
            return {
                'key_points': key_points,
                'raw_response': response,
                'confidence': 0.70
            }
        except Exception as e:
            return {'key_points': [], 'confidence': 0.5, 'error': str(e)}
    
    def _explore_curiosity(
        self,
        home_team: str,
        away_team: str,
        simulation: SimulationResult,
        llm_insights: Dict[str, Any]
    ) -> List[str]:
        """Explore hidden edges through curiosity prompts."""
        
        findings = []
        
        # Check for high variance across scenarios
        predictions = simulation.weighted_predictions
        confidences = simulation.confidence_scores
        
        for market, prob in predictions.items():
            if market in confidences:
                conf = confidences.get(market, 0.5)
                if conf < 0.5:
                    findings.append(f"⚡ {market} has high uncertainty - scenarios diverge significantly")
        
        # Check for chaos potential
        chaos_scenario = next(
            (s for s in simulation.scenarios if s.scenario_type == 'chaos'),
            None
        )
        if chaos_scenario and len(chaos_scenario.chaos_events) > 3:
            findings.append(f"🎲 High chaos potential: {', '.join(chaos_scenario.chaos_events[:3])}")
        
        # Contrarian check
        if predictions.get('btts', 0) > 0.60 and predictions.get('over_2_5', 0) < 0.40:
            findings.append("🤔 Contrarian: High BTTS but low Over 2.5 - expect 1-1 or 2-1 games")
        
        # Form vs fundamentals
        base = next(s for s in simulation.scenarios if s.scenario_type == 'base')
        high_scoring = next(s for s in simulation.scenarios if s.scenario_type == 'high_scoring')
        if abs(base.btts_probability - high_scoring.btts_probability) > 0.20:
            findings.append("📊 Scenario dependent: Outcome varies significantly by match flow")
        
        # Check cached curiosity findings for relevance
        relevant_findings = self.cache.get_relevant_findings(min_confidence=0.6)
        for finding in relevant_findings[:2]:
            if finding['validations'] > 0:
                success_rate = finding['successes'] / finding['validations']
                if success_rate > 0.6:
                    findings.append(f"📚 Historical insight: {finding['description']}")
        
        return findings
    
    def _evaluate_markets(
        self,
        home_team: str,
        away_team: str,
        simulation: SimulationResult,
        league_priors: Dict[str, float],
        llm_insights: Dict[str, Any],
        curiosity: List[str]
    ) -> List[MarketAnalysis]:
        """Evaluate each target market with full reasoning."""
        
        analyses = []
        predictions = simulation.weighted_predictions
        confidences = simulation.confidence_scores
        
        for market_key, market_info in self.market_info.items():
            sim_prob = predictions.get(market_key, 0.5)
            league_prior = league_priors.get(f'{market_key}_rate', 0.5)
            
            # Blend simulation with league prior
            probability = 0.7 * sim_prob + 0.3 * league_prior
            
            # Confidence from simulation
            confidence = confidences.get(market_key, 0.5)
            
            # Adjust for curiosity findings (uncertainty)
            if any('uncertainty' in f.lower() for f in curiosity):
                confidence *= 0.9
            if any('chaos' in f.lower() for f in curiosity):
                confidence *= 0.85
            
            # Key factors from simulation
            base_scenario = next(s for s in simulation.scenarios if s.scenario_type == 'base')
            key_factors = base_scenario.key_factors[:3]
            
            # Generate recommendation
            typical_odds = market_info['typical_odds']
            golden_range = market_info['golden_range']
            
            # Expected value calculation
            mid_odds = (typical_odds[0] + typical_odds[1]) / 2
            implied_prob = 1 / mid_odds
            edge = probability - implied_prob
            
            if edge > 0.05 and confidence > 0.50:
                recommendation = 'BET'
                reasoning = f"Strong edge ({edge:.0%}) with good confidence"
            elif edge > 0.02 and confidence > 0.45:
                recommendation = 'BET' if edge > 0.03 else 'SKIP'
                reasoning = f"Marginal edge ({edge:.0%}) - consider carefully"
            elif edge < -0.10:
                recommendation = 'AVOID'
                reasoning = f"Negative edge ({edge:.0%}) - value not present"
            else:
                recommendation = 'SKIP'
                reasoning = f"Insufficient edge ({edge:.0%}) for confident bet"
            
            # Contrarian view
            if probability > 0.65:
                contrarian = f"Public likely heavy on this - check for line movement"
            elif probability < 0.40:
                contrarian = f"Low probability - could be value on the opposite"
            else:
                contrarian = f"Balanced market - no strong contrarian signal"
            
            analyses.append(MarketAnalysis(
                market=market_key,
                probability=probability,
                confidence=confidence,
                reasoning=reasoning,
                key_factors=key_factors,
                contrarian_view=contrarian,
                recommendation=recommendation,
                suggested_odds_range=golden_range
            ))
        
        return analyses
    
    def _self_reflect(
        self,
        home_team: str,
        away_team: str,
        analyses: List[MarketAnalysis],
        simulation: SimulationResult
    ) -> List[MarketAnalysis]:
        """Self-reflect and adjust predictions if needed."""
        
        adjusted = []
        
        for analysis in analyses:
            # Check for overconfidence
            if analysis.probability > 0.75 and analysis.confidence > 0.80:
                # Calibrate down slightly
                analysis.confidence = min(analysis.confidence, 0.75)
                analysis.key_factors.append("⚠️ Adjusted for potential overconfidence")
            
            # Check for extreme predictions
            if analysis.probability > 0.85 or analysis.probability < 0.20:
                analysis.confidence *= 0.9
                analysis.key_factors.append("⚠️ Extreme prediction - lower confidence")
            
            # Check scenario divergence
            probs = [s.btts_probability if analysis.market == 'btts' else 
                     s.over_1_5_probability if analysis.market == 'over_1_5' else
                     s.over_2_5_probability for s in simulation.scenarios]
            divergence = max(probs) - min(probs)
            
            if divergence > 0.25:
                analysis.confidence *= 0.85
                analysis.key_factors.append(f"⚠️ High scenario divergence ({divergence:.0%})")
            
            adjusted.append(analysis)
        
        return adjusted
    
    def _cache_reasoning(self, result: MatchReasoning):
        """Cache the reasoning result."""
        
        analysis = CachedAnalysis(
            match_id=result.match_id,
            home_team=result.home_team,
            away_team=result.away_team,
            league=result.league,
            match_date=result.match_date,
            analysis_date=result.timestamp,
            reasoning_chain={
                'steps': [asdict(s) for s in result.reasoning_steps],
                'total_time_ms': result.total_reasoning_time_ms
            },
            market_predictions={
                m.market: m.probability for m in result.market_analyses
            },
            confidence_scores={
                m.market: m.confidence for m in result.market_analyses
            },
            scenarios=[asdict(s) for s in result.simulation_result.scenarios] if result.simulation_result else [],
            metadata={
                'key_insights': result.key_insights,
                'curiosity_findings': result.curiosity_findings,
                'overall_confidence': result.overall_confidence
            }
        )
        
        self.cache.cache_analysis(analysis)
    
    def _cached_to_reasoning(self, cached: CachedAnalysis) -> MatchReasoning:
        """Convert cached analysis back to MatchReasoning."""
        
        # Reconstruct market analyses
        market_analyses = []
        for market, prob in cached.market_predictions.items():
            conf = cached.confidence_scores.get(market, 0.5)
            market_analyses.append(MarketAnalysis(
                market=market,
                probability=prob,
                confidence=conf,
                reasoning="From cached analysis",
                key_factors=[],
                contrarian_view="",
                recommendation='BET' if prob > 0.55 and conf > 0.55 else 'SKIP',
                suggested_odds_range=(1.40, 1.70)
            ))
        
        # Reconstruct steps (simplified)
        steps = []
        for i, step_data in enumerate(cached.reasoning_chain.get('steps', [])):
            steps.append(ReasoningStep(
                step_number=i + 1,
                step_type=step_data.get('step_type', 'analysis'),
                thought=step_data.get('thought', ''),
                evidence=step_data.get('evidence', []),
                confidence=step_data.get('confidence', 0.5),
                time_ms=step_data.get('time_ms', 0)
            ))
        
        return MatchReasoning(
            match_id=cached.match_id,
            home_team=cached.home_team,
            away_team=cached.away_team,
            league=cached.league,
            match_date=cached.match_date,
            reasoning_steps=steps,
            market_analyses=market_analyses,
            simulation_result=None,  # Not cached fully
            overall_confidence=cached.metadata.get('overall_confidence', 0.5),
            key_insights=cached.metadata.get('key_insights', []),
            curiosity_findings=cached.metadata.get('curiosity_findings', []),
            total_reasoning_time_ms=cached.reasoning_chain.get('total_time_ms', 0),
            timestamp=cached.analysis_date
        )

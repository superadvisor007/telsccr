#!/usr/bin/env python3
"""
🤖 REASONING AGENT WITH CHAIN-OF-THOUGHT
========================================
ReAct (Reasoning + Acting) pattern for betting analysis.

Based on battle-tested patterns from:
- LangChain ReAct agents
- Microsoft AutoGen MagenticOne  
- Professional betting systems

The agent follows this reasoning cycle:
1. Thought: What do I need to figure out?
2. Action: What tool should I use?
3. Observation: What did I learn?
4. ... repeat until Final Answer

Chain-of-Thought Structure:
1. Statistical Analysis → Raw numbers
2. Contextual Factors → Situational modifiers
3. Value Assessment → EV calculation
4. Contrarian Check → Devil's advocate
5. Final Decision → BET/AVOID with reasoning
"""

import os
import sys
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning.match_context_collector import MatchContextCollector, MatchContext
from reasoning.schemas import (
    BettingDecision, ChainOfThought, ReasoningStep, ValueAnalysis,
    ContrarianCheck, BettingMarket, ActionType, Recommendation
)


class ReasoningAgent:
    """
    🤖 ReAct Reasoning Agent for Betting Decisions
    
    Uses Chain-of-Thought reasoning to:
    1. Gather all relevant match context
    2. Analyze from multiple angles
    3. Calculate true value
    4. Check for blindspots
    5. Make explainable decision
    
    "Show your work" - every decision is traceable.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.context_collector = MatchContextCollector()
        
        # Tool registry - what the agent can do
        self.tools = {
            ActionType.SEARCH_STATS: self._tool_search_stats,
            ActionType.CHECK_FORM: self._tool_check_form,
            ActionType.GET_H2H: self._tool_get_h2h,
            ActionType.GET_PSYCHOLOGY: self._tool_get_psychology,
            ActionType.CALCULATE_VALUE: self._tool_calculate_value,
            ActionType.CHECK_WEATHER: self._tool_check_weather,
            ActionType.GET_NEWS: self._tool_get_news,
            ActionType.FINAL_DECISION: self._tool_final_decision,
        }
        
        # Reasoning knowledge base
        self.betting_principles = {
            "value_threshold": 0.02,  # 2% minimum EV
            "confidence_threshold": 60,  # Minimum confidence to bet
            "kelly_fraction": 0.25,  # Quarter Kelly for safety
            "min_odds": 1.20,
            "max_odds": 3.00,
            "golden_range": (1.40, 1.70),
        }
        
        # League characteristics
        self.league_profiles = {
            'Bundesliga': {'avg_goals': 3.15, 'over_2_5_rate': 0.55, 'btts_rate': 0.53},
            'Premier League': {'avg_goals': 2.85, 'over_2_5_rate': 0.50, 'btts_rate': 0.51},
            'La Liga': {'avg_goals': 2.65, 'over_2_5_rate': 0.46, 'btts_rate': 0.46},
            'Serie A': {'avg_goals': 2.75, 'over_2_5_rate': 0.48, 'btts_rate': 0.50},
            'Ligue 1': {'avg_goals': 2.80, 'over_2_5_rate': 0.48, 'btts_rate': 0.48},
            'Eredivisie': {'avg_goals': 3.35, 'over_2_5_rate': 0.58, 'btts_rate': 0.56},
        }
    
    def analyze_match(self, home_team: str, away_team: str, league: str,
                      match_date: str = None, target_markets: List[str] = None) -> List[BettingDecision]:
        """
        Complete match analysis with Chain-of-Thought reasoning.
        
        Returns list of BettingDecision for each analyzed market.
        """
        start_time = time.time()
        
        if target_markets is None:
            target_markets = ['over_1_5', 'over_2_5', 'btts_yes', 'btts_no']
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 REASONING AGENT: {home_team} vs {away_team}")
            print(f"{'='*60}\n")
        
        # Phase 1: Collect comprehensive context
        context = self.context_collector.collect_full_context(
            home_team, away_team, league, match_date
        )
        
        # Phase 2: Analyze each target market
        decisions = []
        for market in target_markets:
            decision = self._analyze_market(context, market)
            if decision:
                decisions.append(decision)
        
        # Sort by expected value
        decisions.sort(key=lambda x: x.expected_value, reverse=True)
        
        total_time = int((time.time() - start_time) * 1000)
        
        if self.verbose:
            print(f"\n⏱️ Total reasoning time: {total_time}ms")
            print(f"📊 Analyzed {len(decisions)} markets")
            
            for d in decisions:
                emoji = "✅" if d.recommendation in [Recommendation.BET, Recommendation.STRONG_BET] else "❌"
                print(f"   {emoji} {d.market.value}: EV={d.expected_value:+.1%}, Confidence={d.confidence_score:.0f}")
        
        return decisions
    
    def _analyze_market(self, context: MatchContext, market: str) -> Optional[BettingDecision]:
        """
        Analyze a single market using Chain-of-Thought.
        """
        start_time = time.time()
        
        # Initialize chain
        chain = ChainOfThought(
            match_id=context.match_id,
            home_team=context.home_team,
            away_team=context.away_team
        )
        
        try:
            market_enum = BettingMarket(market)
        except:
            if self.verbose:
                print(f"   ⚠️ Unknown market: {market}")
            return None
        
        if self.verbose:
            print(f"\n📊 Analyzing: {market.upper()}")
            print("-" * 40)
        
        # =================================================================
        # STEP 1: Statistical Analysis
        # =================================================================
        chain.add_step(
            thought=f"I need to understand the statistical baseline for {market} in this match.",
            action=ActionType.SEARCH_STATS,
            action_input={"market": market, "context": "team_stats"},
            observation=self._tool_search_stats(context, market)
        )
        
        # =================================================================
        # STEP 2: Form Analysis
        # =================================================================
        chain.add_step(
            thought="Recent form can significantly impact goal scoring. Let me check both teams' momentum.",
            action=ActionType.CHECK_FORM,
            action_input={"home_team": context.home_team, "away_team": context.away_team},
            observation=self._tool_check_form(context)
        )
        
        # =================================================================
        # STEP 3: H2H Analysis
        # =================================================================
        chain.add_step(
            thought="Historical head-to-head results often reveal patterns not visible in raw stats.",
            action=ActionType.GET_H2H,
            action_input={"matchup": f"{context.home_team} vs {context.away_team}"},
            observation=self._tool_get_h2h(context)
        )
        
        # =================================================================
        # STEP 4: Psychological Factors
        # =================================================================
        chain.add_step(
            thought="Psychological factors like derbies, relegation battles, or title races can swing outcomes by 10-15%.",
            action=ActionType.GET_PSYCHOLOGY,
            action_input={"match": context.match_id},
            observation=self._tool_get_psychology(context)
        )
        
        # =================================================================
        # STEP 5: Calculate Value
        # =================================================================
        value_result = self._tool_calculate_value(context, market)
        chain.add_step(
            thought="Now I need to calculate if there's genuine betting value - where our probability exceeds implied market probability.",
            action=ActionType.CALCULATE_VALUE,
            action_input={"market": market, "methodology": "poisson_adjusted"},
            observation=value_result['observation']
        )
        
        # =================================================================
        # STEP 6: Contrarian Check
        # =================================================================
        contrarian = self._tool_contrarian_check(context, market, value_result)
        chain.add_step(
            thought="Let me play devil's advocate - what could go wrong with this bet?",
            action=ActionType.FINAL_DECISION,
            action_input={"check_type": "contrarian"},
            observation=contrarian['observation']
        )
        
        # Calculate timing
        chain.reasoning_time_ms = int((time.time() - start_time) * 1000)
        
        # =================================================================
        # BUILD FINAL DECISION
        # =================================================================
        decision = self._build_decision(context, market_enum, chain, value_result, contrarian)
        
        if self.verbose:
            print(f"   💭 Steps: {chain.total_steps}")
            print(f"   ⏱️ Time: {chain.reasoning_time_ms}ms")
            print(f"   📊 Probability: {decision.our_probability:.1%}")
            print(f"   💰 Odds: {decision.market_odds:.2f}")
            print(f"   📈 EV: {decision.expected_value:+.1%}")
            print(f"   🎯 Recommendation: {decision.recommendation.value}")
        
        return decision
    
    # =====================================================================
    # TOOLS - What the agent can do
    # =====================================================================
    
    def _tool_search_stats(self, context: MatchContext, market: str) -> str:
        """Search and analyze team statistics for given market."""
        home = context.home_stats
        away = context.away_stats
        league_profile = self.league_profiles.get(context.league, {})
        
        if 'over' in market or 'under' in market:
            combined_gpg = home.goals_per_game + away.goals_conceded_per_game + \
                          away.goals_per_game + home.goals_conceded_per_game
            avg_expected = combined_gpg / 2
            
            return f"""
STATISTICAL ANALYSIS:
- Home ({context.home_team}): {home.goals_per_game:.2f} gpg scored, {home.goals_conceded_per_game:.2f} gpg conceded
- Away ({context.away_team}): {away.goals_per_game:.2f} gpg scored, {away.goals_conceded_per_game:.2f} gpg conceded
- Combined Expected Goals: {avg_expected:.2f}
- Home Over 2.5 Rate: {home.over_2_5_rate:.0%}
- Away Over 2.5 Rate: {away.over_2_5_rate:.0%}
- League Average: {league_profile.get('avg_goals', 2.8):.2f} goals/game
- League Over 2.5 Rate: {league_profile.get('over_2_5_rate', 0.50):.0%}
"""
        elif 'btts' in market:
            home_scores = 1 - home.failed_to_score_rate
            away_scores = 1 - away.failed_to_score_rate
            
            return f"""
BTTS ANALYSIS:
- Home scores probability: {home_scores:.0%} (failed to score: {home.failed_to_score_rate:.0%})
- Away scores probability: {away_scores:.0%} (failed to score: {away.failed_to_score_rate:.0%})
- Home BTTS rate: {home.btts_rate:.0%}
- Away BTTS rate: {away.btts_rate:.0%}
- Home clean sheet rate: {home.clean_sheet_rate:.0%}
- Away clean sheet rate: {away.clean_sheet_rate:.0%}
- League BTTS rate: {league_profile.get('btts_rate', 0.50):.0%}
"""
        else:
            return f"Statistics gathered for {market}"
    
    def _tool_check_form(self, context: MatchContext) -> str:
        """Check recent form of both teams."""
        home = context.home_stats
        away = context.away_stats
        
        return f"""
FORM ANALYSIS:
- Home ({context.home_team}): {home.recent_form} ({home.form_points} pts from last 5)
- Away ({context.away_team}): {away.recent_form} ({away.form_points} pts from last 5)
- Home PPG: {home.home_ppg:.2f} (at home)
- Away PPG: {away.away_ppg:.2f} (away from home)
- Form Momentum: {'HOME ADVANTAGE' if home.form_points > away.form_points else 'AWAY IN FORM' if away.form_points > home.form_points else 'BALANCED'}
"""
    
    def _tool_get_h2h(self, context: MatchContext) -> str:
        """Get head-to-head history."""
        h2h = context.h2h
        
        return f"""
HEAD-TO-HEAD ({h2h.total_matches} meetings):
- Record: {context.home_team} {h2h.home_wins}W - {h2h.draws}D - {h2h.away_wins}W {context.away_team}
- Average Goals: {h2h.avg_goals_per_match:.1f} per match
- Over 2.5 Rate: {h2h.over_2_5_rate:.0%}
- BTTS Rate: {h2h.btts_rate:.0%}
- Pattern: {'HIGH-SCORING FIXTURE' if h2h.avg_goals_per_match > 3.0 else 'AVERAGE SCORING' if h2h.avg_goals_per_match > 2.5 else 'LOW-SCORING FIXTURE'}
"""
    
    def _tool_get_psychology(self, context: MatchContext) -> str:
        """Analyze psychological factors."""
        m = context.motivation
        
        factors = []
        if m.is_derby:
            factors.append(f"⚡ DERBY MATCH (Intensity: {m.rivalry_intensity}/10)")
        if m.home_motivation == 'title_race':
            factors.append(f"🏆 Home team in TITLE RACE")
        if m.away_motivation == 'title_race':
            factors.append(f"🏆 Away team in TITLE RACE")
        if m.home_motivation == 'relegation_battle':
            factors.append(f"⚠️ Home team in RELEGATION BATTLE")
        if m.away_motivation == 'relegation_battle':
            factors.append(f"⚠️ Away team in RELEGATION BATTLE")
        
        pressure_diff = m.home_pressure_level - m.away_pressure_level
        
        return f"""
PSYCHOLOGICAL ANALYSIS:
{chr(10).join(factors) if factors else '- No special psychological factors'}
- Home Pressure: {m.home_pressure_level}/10 ({m.home_motivation})
- Away Pressure: {m.away_pressure_level}/10 ({m.away_motivation})
- Pressure Difference: {abs(pressure_diff)} ({'favors HOME' if pressure_diff < 0 else 'favors AWAY' if pressure_diff > 0 else 'balanced'})
- Rest: Home {m.days_since_last_home}d, Away {m.days_since_last_away}d
"""
    
    def _tool_check_weather(self, context: MatchContext) -> str:
        """Check weather conditions."""
        w = context.weather
        return f"""
WEATHER CONDITIONS:
- Temperature: {w.temperature}°C (feels like {w.feels_like}°C)
- Wind: {w.wind_speed} km/h {w.wind_direction}
- Humidity: {w.humidity}%
- Precipitation: {w.precipitation_prob}%
- Impact: {w.weather_impact} (Score: {w.impact_score}/10)
- Assessment: {'Good conditions for football' if w.impact_score >= 6 else 'Conditions may affect play'}
"""
    
    def _tool_get_news(self, context: MatchContext) -> str:
        """Get team news."""
        home_news = context.home_news
        away_news = context.away_news
        return f"""
TEAM NEWS:
Home ({context.home_team}):
- Injuries: {len(home_news.injuries)}
- Key Players Out: {home_news.key_players_out}
- Top Scorer Available: {'Yes' if home_news.top_scorer_available else 'No'}
- Squad Depth: {home_news.squad_depth_rating}/10

Away ({context.away_team}):
- Injuries: {len(away_news.injuries)}
- Key Players Out: {away_news.key_players_out}
- Top Scorer Available: {'Yes' if away_news.top_scorer_available else 'No'}
- Squad Depth: {away_news.squad_depth_rating}/10
"""
    
    def _tool_final_decision(self, *args, **kwargs) -> str:
        """Final decision helper (used for contrarian check step)."""
        return "Contrarian analysis complete."
    
    def _tool_calculate_value(self, context: MatchContext, market: str) -> Dict:
        """Calculate betting value using Poisson distribution."""
        home = context.home_stats
        away = context.away_stats
        odds = context.odds
        
        # Get odds for market
        odds_map = {
            'over_0_5': odds.over_0_5,
            'over_1_5': odds.over_1_5,
            'over_2_5': odds.over_2_5,
            'over_3_5': odds.over_3_5,
            'under_1_5': odds.under_1_5,
            'under_2_5': odds.under_2_5,
            'btts_yes': odds.btts_yes,
            'btts_no': odds.btts_no,
        }
        
        market_odds = odds_map.get(market, 1.80)
        if market_odds <= 0:
            market_odds = 1.80  # Default
        
        implied_prob = 1 / market_odds
        
        # Calculate probability using Poisson
        home_lambda = (home.goals_per_game + away.goals_conceded_per_game) / 2
        away_lambda = (away.goals_per_game + home.goals_conceded_per_game) / 2
        
        # Adjust for league
        league_profile = self.league_profiles.get(context.league, {})
        league_factor = league_profile.get('avg_goals', 2.8) / 2.8
        home_lambda *= league_factor
        away_lambda *= league_factor
        
        # Calculate market probability
        if 'over' in market or 'under' in market:
            total_lambda = home_lambda + away_lambda
            
            # Poisson probability
            def poisson_prob(k, lam):
                return (lam ** k) * math.exp(-lam) / math.factorial(k)
            
            if market == 'over_0_5':
                our_prob = 1 - poisson_prob(0, total_lambda)
            elif market == 'over_1_5':
                our_prob = 1 - sum(poisson_prob(k, total_lambda) for k in range(2))
            elif market == 'over_2_5':
                our_prob = 1 - sum(poisson_prob(k, total_lambda) for k in range(3))
            elif market == 'over_3_5':
                our_prob = 1 - sum(poisson_prob(k, total_lambda) for k in range(4))
            elif market == 'under_1_5':
                our_prob = sum(poisson_prob(k, total_lambda) for k in range(2))
            elif market == 'under_2_5':
                our_prob = sum(poisson_prob(k, total_lambda) for k in range(3))
            else:
                our_prob = 0.5
        
        elif 'btts' in market:
            home_scores = 1 - math.exp(-home_lambda)
            away_scores = 1 - math.exp(-away_lambda)
            btts_prob = home_scores * away_scores
            
            if market == 'btts_yes':
                our_prob = btts_prob
            else:
                our_prob = 1 - btts_prob
        else:
            our_prob = 0.5
        
        # Apply psychological adjustments
        if context.motivation.is_derby:
            # Derbies tend to be tighter
            if 'over' in market:
                our_prob *= 0.95  # 5% reduction for over markets
            elif 'btts' in market and market == 'btts_yes':
                our_prob *= 1.05  # 5% boost for BTTS in derbies
        
        # Calculate value metrics
        ev = our_prob * market_odds - 1
        edge = our_prob - implied_prob
        
        # Kelly criterion (quarter Kelly for safety)
        if ev > 0:
            kelly = ((our_prob * market_odds - 1) / (market_odds - 1)) * self.betting_principles['kelly_fraction']
        else:
            kelly = 0
        
        result = {
            'our_probability': our_prob,
            'market_odds': market_odds,
            'implied_probability': implied_prob,
            'expected_value': ev,
            'edge': edge,
            'kelly_stake': kelly,
            'is_value': ev > self.betting_principles['value_threshold'],
            'observation': f"""
VALUE CALCULATION:
- Expected Goals (Home): {home_lambda:.2f}
- Expected Goals (Away): {away_lambda:.2f}
- Total Expected: {home_lambda + away_lambda:.2f}
- Our Probability: {our_prob:.1%}
- Market Odds: {market_odds:.2f}
- Implied Probability: {implied_prob:.1%}
- Edge: {edge:+.1%}
- Expected Value: {ev:+.1%}
- Kelly Stake: {kelly:.1%}
- VALUE BET: {'✅ YES' if ev > self.betting_principles['value_threshold'] else '❌ NO'}
"""
        }
        
        return result
    
    def _tool_contrarian_check(self, context: MatchContext, market: str, 
                               value_result: Dict) -> Dict:
        """Devil's advocate - what could go wrong?"""
        concerns = []
        risk_factors = []
        blindspots = []
        
        # Check market-specific risks
        if 'over' in market:
            if context.motivation.is_derby:
                concerns.append("Derby matches are often tighter than stats suggest (tactical caution)")
            if context.weather.wind_speed > 25:
                concerns.append(f"High wind ({context.weather.wind_speed}km/h) can reduce scoring")
            if context.home_stats.clean_sheet_rate > 0.30:
                concerns.append(f"Home team has good defensive record ({context.home_stats.clean_sheet_rate:.0%} clean sheets)")
            if context.away_stats.clean_sheet_rate > 0.30:
                concerns.append(f"Away team has good defensive record ({context.away_stats.clean_sheet_rate:.0%} clean sheets)")
        
        elif 'btts' in market and market == 'btts_yes':
            if context.home_stats.failed_to_score_rate > 0.20:
                concerns.append(f"Home team fails to score in {context.home_stats.failed_to_score_rate:.0%} of games")
            if context.away_stats.failed_to_score_rate > 0.25:
                concerns.append(f"Away team fails to score in {context.away_stats.failed_to_score_rate:.0%} of games")
        
        # General risk factors
        if value_result['edge'] < 0.03:
            risk_factors.append("Edge is thin (<3%) - variance could wipe out value")
        
        if context.data_quality < 70:
            risk_factors.append(f"Data quality is limited ({context.data_quality}%) - predictions less reliable")
        
        # Potential blindspots
        blindspots.append("Injuries/suspensions may not be fully captured")
        blindspots.append("Market odds may already incorporate news we don't have")
        
        # Calculate confidence adjustment
        confidence_adj = 0
        if len(concerns) >= 2:
            confidence_adj = -0.05
        if len(risk_factors) >= 2:
            confidence_adj -= 0.05
        
        should_proceed = value_result['is_value'] and len(concerns) < 3
        
        return {
            'concerns': concerns,
            'risk_factors': risk_factors,
            'blindspots': blindspots,
            'confidence_adjustment': confidence_adj,
            'should_proceed': should_proceed,
            'observation': f"""
CONTRARIAN CHECK (Devil's Advocate):
⚠️ Concerns ({len(concerns)}):
{chr(10).join(f'  - {c}' for c in concerns) if concerns else '  - None identified'}

🚨 Risk Factors ({len(risk_factors)}):
{chr(10).join(f'  - {r}' for r in risk_factors) if risk_factors else '  - None identified'}

🔍 Blindspots:
{chr(10).join(f'  - {b}' for b in blindspots)}

Confidence Adjustment: {confidence_adj:+.0%}
Proceed with Bet: {'✅ Yes' if should_proceed else '❌ No'}
"""
        }
    
    def _build_decision(self, context: MatchContext, market: BettingMarket,
                        chain: ChainOfThought, value_result: Dict, 
                        contrarian: Dict) -> BettingDecision:
        """Build final betting decision."""
        
        # Calculate confidence score (0-100)
        base_confidence = 50
        
        # Boost for positive EV
        if value_result['is_value']:
            base_confidence += 15
        if value_result['edge'] > 0.05:
            base_confidence += 10
        if value_result['edge'] > 0.10:
            base_confidence += 5
        
        # Boost for data quality
        base_confidence += (context.data_quality - 50) / 5
        
        # Apply contrarian adjustment
        base_confidence += contrarian['confidence_adjustment'] * 100
        
        # Penalties for concerns
        base_confidence -= len(contrarian['concerns']) * 3
        base_confidence -= len(contrarian['risk_factors']) * 2
        
        confidence = max(0, min(100, base_confidence))
        
        # Determine recommendation
        if value_result['expected_value'] > 0.08 and confidence >= 75:
            recommendation = Recommendation.STRONG_BET
        elif value_result['expected_value'] > 0.02 and confidence >= 60:
            recommendation = Recommendation.BET
        elif value_result['expected_value'] > 0 and confidence >= 50:
            recommendation = Recommendation.MONITOR
        elif value_result['expected_value'] < -0.05:
            recommendation = Recommendation.STRONG_AVOID
        else:
            recommendation = Recommendation.AVOID
        
        # Build "why this bet" reasons
        why_this_bet = []
        if value_result['is_value']:
            why_this_bet.append(f"Positive expected value of {value_result['expected_value']:.1%}")
        if value_result['edge'] > 0.03:
            why_this_bet.append(f"Edge of {value_result['edge']:.1%} vs market")
        
        # Add league context
        league_profile = self.league_profiles.get(context.league, {})
        if 'over' in market.value and league_profile.get('over_2_5_rate', 0.5) > 0.52:
            why_this_bet.append(f"{context.league} is high-scoring league ({league_profile['avg_goals']:.2f} avg goals)")
        
        # Add H2H context
        if context.h2h.avg_goals_per_match > 3.0:
            why_this_bet.append(f"H2H average is {context.h2h.avg_goals_per_match:.1f} goals per match")
        
        # Build "why might fail" reasons
        why_might_fail = contrarian['concerns'][:3]
        if not why_might_fail:
            why_might_fail = ["Standard variance in football outcomes"]
        
        # Confidence stars
        confidence_stars = min(5, max(1, int(confidence / 20)))
        
        # Create value analysis
        value_analysis = ValueAnalysis(
            market=market,
            our_probability=value_result['our_probability'],
            market_odds=value_result['market_odds'],
            implied_probability=value_result['implied_probability'],
            expected_value=value_result['expected_value'],
            edge_percentage=value_result['edge'] * 100,
            kelly_stake=value_result['kelly_stake'],
            is_value=value_result['is_value']
        )
        
        # Create contrarian check
        contrarian_check = ContrarianCheck(
            concerns=contrarian['concerns'],
            risk_factors=contrarian['risk_factors'],
            potential_blindspots=contrarian['blindspots'],
            confidence_adjustment=contrarian['confidence_adjustment'],
            should_proceed=contrarian['should_proceed'],
            reasoning="Analysis complete"
        )
        
        return BettingDecision(
            match_id=context.match_id,
            home_team=context.home_team,
            away_team=context.away_team,
            league=context.league,
            match_date=context.date,
            market=market,
            recommendation=recommendation,
            confidence_score=confidence,
            confidence_stars=confidence_stars,
            our_probability=value_result['our_probability'],
            market_odds=value_result['market_odds'],
            implied_probability=value_result['implied_probability'],
            expected_value=value_result['expected_value'],
            edge_percentage=value_result['edge'] * 100,
            kelly_stake=value_result['kelly_stake'],
            chain_of_thought=chain,
            value_analysis=value_analysis,
            contrarian_check=contrarian_check,
            why_this_bet=why_this_bet,
            why_might_fail=why_might_fail,
            key_factors=[f"League: {context.league}", f"Form: {context.home_stats.recent_form} vs {context.away_stats.recent_form}"]
        )
    
    def find_best_bets(self, matches: List[Dict], min_ev: float = 0.02, 
                       max_bets: int = 5) -> List[BettingDecision]:
        """
        Analyze multiple matches and return the best value bets.
        """
        all_decisions = []
        
        for match in matches:
            decisions = self.analyze_match(
                home_team=match['home_team'],
                away_team=match['away_team'],
                league=match['league'],
                match_date=match.get('date')
            )
            all_decisions.extend(decisions)
        
        # Filter by minimum EV and sort
        value_bets = [d for d in all_decisions if d.expected_value >= min_ev]
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)
        
        return value_bets[:max_bets]


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 REASONING AGENT TEST")
    print("="*60)
    
    agent = ReasoningAgent(verbose=True)
    
    # Test single match
    decisions = agent.analyze_match(
        home_team="Bayern Munich",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        match_date="2025-01-28"
    )
    
    print("\n" + "="*60)
    print("📊 DECISIONS")
    print("="*60)
    
    for decision in decisions:
        print(decision.to_telegram_message())
        print("-" * 40)
    
    # Print full report for best decision
    if decisions:
        best = max(decisions, key=lambda x: x.expected_value)
        print("\n" + "="*60)
        print("📋 FULL REPORT (BEST BET)")
        print("="*60)
        print(best.to_full_report())

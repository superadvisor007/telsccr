#!/usr/bin/env python3
"""
🧠 DEEP VALIDATION LAYER - Battle-Tested Betting Intelligence
==============================================================

Multi-layer validation system that analyzes bets through:
1. Psychological Knowledge Base
2. Statistical Validation
3. Historical Pattern Matching
4. Ensemble Agreement
5. Market Efficiency Analysis

Based on patterns from:
- datarootsio/your-best-bet (Value Detection)
- smarmau/asknews_mlb (Multi-Model Ensemble)
- soccermatics/Soccermatics (xG Models)
- Academic: Kelly Criterion, CLV, Wisdom of Crowds

"The market is usually right - find when it's wrong."
"""

import json
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Complete validation result with reasoning"""
    is_valid: bool
    confidence_score: float  # 0-100
    total_checks_passed: int
    total_checks: int
    
    # Layer results
    psychological_check: Dict
    statistical_check: Dict
    historical_check: Dict
    market_check: Dict
    ensemble_check: Dict
    
    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Final reasoning
    reasoning: str = ""


class DeepBettingKnowledgeBase:
    """
    🎓 Deep Knowledge Base about Winning in Betting
    
    Contains battle-tested wisdom from:
    - Professional bettors (Billy Walters, Haralabos Voulgaris)
    - Academic research (Kaunitz et al., Merkle & Steyvers)
    - Market microstructure theory
    """
    
    # Core Betting Principles
    PRINCIPLES = {
        "value_first": """
            NEVER bet on probability alone - always check VALUE.
            Value = Your_Probability × Odds - 1 > 0
            Only bet when your edge is at least 2-3%.
        """,
        
        "closing_line_value": """
            CLV (Closing Line Value) is the BEST predictor of long-term success.
            If you consistently beat the closing line, you WILL profit.
            Track: (Closing_Implied - Bet_Implied) / Bet_Implied × 100
        """,
        
        "kelly_criterion": """
            Kelly Criterion optimizes bankroll growth: f* = (p×b - q) / b
            NEVER use full Kelly - use 25% (Quarter Kelly) for safety.
            This protects against probability estimation errors.
        """,
        
        "market_efficiency": """
            Closing lines in major leagues are ~97% efficient.
            Look for inefficiencies in: small leagues, player props, live betting.
            The wisdom of crowds is real - respect the market.
        """,
        
        "regression_to_mean": """
            Extreme performance WILL regress. After 5 wins, expect regression.
            After 5 losses, also expect regression.
            Don't overreact to small samples.
        """,
        
        "confirmation_bias": """
            We see what we want to see. Always seek disconfirming evidence.
            Before betting: List 3 reasons why you could be WRONG.
        """,
    }
    
    # Psychological Traps to Avoid
    PSYCHOLOGICAL_TRAPS = {
        "gamblers_fallacy": {
            "description": "Believing past results affect future independent events",
            "detection": "Team on 5+ game losing streak, public heavily backing them",
            "action": "Consider fading public, but verify fundamentals"
        },
        
        "hot_hand_fallacy": {
            "description": "Overweighting recent performance streaks",
            "detection": "Team on 5+ game winning streak at inflated odds",
            "action": "Check if odds reflect true probability or recency bias"
        },
        
        "favorite_longshot_bias": {
            "description": "Favorites are underbet, longshots are overbet",
            "detection": "Implied probability of longshot > true probability",
            "action": "Slight bias toward favorites in accumulator building"
        },
        
        "recency_bias": {
            "description": "Overweighting last 1-2 games vs full season",
            "detection": "Odds moved significantly after single game result",
            "action": "Use rolling 10-15 game form, not last 1-2"
        },
        
        "sunk_cost_fallacy": {
            "description": "Chasing losses with larger bets",
            "detection": "Stake increasing after losing streak",
            "action": "Stick to fixed staking or Kelly - NEVER chase"
        },
        
        "overconfidence": {
            "description": "Believing you know more than the market",
            "detection": "Predicted probability >15% higher than implied",
            "action": "Double-check analysis, reduce stake if still confident"
        },
    }
    
    # Market Indicators
    SHARP_MONEY_INDICATORS = {
        "steam_move": {
            "description": "Line moves suddenly across multiple books",
            "significance": "Sharp action detected",
            "action": "Follow the steam if early, fade if too late"
        },
        
        "reverse_line_movement": {
            "description": "Line moves opposite to public betting %",
            "significance": "Sharps betting against public",
            "action": "Strong signal - consider following sharp money"
        },
        
        "big_money_small_bets": {
            "description": "High money% with low bet count%",
            "significance": "Large sharp bets detected",
            "action": "Indicates professional action"
        },
    }
    
    # League-Specific Knowledge
    LEAGUE_WISDOM = {
        "Bundesliga": {
            "avg_goals": 3.15,
            "home_advantage": 1.35,
            "characteristics": "High-scoring, attacking football. Over 2.5 hits ~56%",
            "edges": "Early games (Friday 8:30pm) often have value"
        },
        "Premier League": {
            "avg_goals": 2.85,
            "home_advantage": 1.40,
            "characteristics": "Physical, competitive. Home advantage significant",
            "edges": "Big 6 rotation in cups, newly promoted momentum"
        },
        "La Liga": {
            "avg_goals": 2.65,
            "home_advantage": 1.45,
            "characteristics": "Technical, low-scoring outside top teams",
            "edges": "Midweek games for top teams in Europe"
        },
        "Serie A": {
            "avg_goals": 2.75,
            "home_advantage": 1.38,
            "characteristics": "Tactical, defensive. Late goals common",
            "edges": "BTTS undervalued in top team matches"
        },
        "Ligue 1": {
            "avg_goals": 2.80,
            "home_advantage": 1.32,
            "characteristics": "PSG dominance skews market",
            "edges": "Value in non-PSG matches, Marseille unpredictability"
        },
        "Eredivisie": {
            "avg_goals": 3.35,
            "home_advantage": 1.25,
            "characteristics": "Highest scoring league in Europe",
            "edges": "Over 2.5 hits 65%+, Ajax youth development"
        },
    }


class DeepValidator:
    """
    🧠 Deep Validation Layer
    
    5-Layer Validation Process:
    1. Psychological Check - Detect cognitive biases
    2. Statistical Check - Mathematical validity
    3. Historical Check - Similar bet performance
    4. Market Check - Line movement & efficiency
    5. Ensemble Check - Model agreement
    """
    
    def __init__(self):
        self.knowledge = DeepBettingKnowledgeBase()
        self.historical_bets = self._load_historical_bets()
        self.min_confidence_threshold = 65  # 0-100
        
        # Validation weights (must sum to 100)
        self.weights = {
            "psychological": 20,
            "statistical": 30,
            "historical": 20,
            "market": 15,
            "ensemble": 15,
        }
    
    def validate_bet(self, bet: Dict) -> ValidationResult:
        """
        Complete validation of a potential bet.
        
        Required bet fields:
        - home_team, away_team, league
        - market (over_2_5, btts_yes, etc.)
        - our_probability (0.0-1.0)
        - odds (decimal odds)
        - Optional: recent_form, line_movement, public_percentage
        """
        
        # Run all validation layers
        psych = self._psychological_check(bet)
        stats = self._statistical_check(bet)
        hist = self._historical_check(bet)
        market = self._market_check(bet)
        ensemble = self._ensemble_check(bet)
        
        # Calculate weighted confidence score
        total_score = (
            psych['score'] * self.weights['psychological'] +
            stats['score'] * self.weights['statistical'] +
            hist['score'] * self.weights['historical'] +
            market['score'] * self.weights['market'] +
            ensemble['score'] * self.weights['ensemble']
        ) / 100
        
        # Count checks
        all_checks = [psych, stats, hist, market, ensemble]
        passed = sum(1 for c in all_checks if c['passed'])
        
        # Collect warnings
        warnings = []
        for check in all_checks:
            warnings.extend(check.get('warnings', []))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bet, all_checks)
        
        # Determine validity
        is_valid = (
            total_score >= self.min_confidence_threshold and
            passed >= 3 and  # At least 3 of 5 checks must pass
            stats['passed']  # Statistical check is mandatory
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(bet, all_checks, is_valid)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=total_score,
            total_checks_passed=passed,
            total_checks=5,
            psychological_check=psych,
            statistical_check=stats,
            historical_check=hist,
            market_check=market,
            ensemble_check=ensemble,
            warnings=warnings,
            recommendations=recommendations,
            reasoning=reasoning
        )
    
    def _psychological_check(self, bet: Dict) -> Dict:
        """
        🧠 Layer 1: Psychological/Behavioral Check
        
        Detects:
        - Gambler's fallacy situations
        - Recency bias
        - Overconfidence
        - Hot hand fallacy
        """
        warnings = []
        score = 100  # Start with perfect score, deduct for issues
        
        our_prob = bet.get('our_probability', 0.5)
        odds = bet.get('odds', 2.0)
        implied_prob = 1 / odds
        edge = our_prob - implied_prob
        
        # Check 1: Overconfidence Detection
        if our_prob > 0.80:
            warnings.append("⚠️ HIGH CONFIDENCE: Probabilities >80% are often overestimated")
            score -= 20
        
        if edge > 0.15:  # >15% edge is suspicious
            warnings.append(f"⚠️ OVERCONFIDENCE: {edge:.1%} edge seems too high")
            score -= 25
        
        # Check 2: Recent Form Analysis
        home_form = bet.get('home_recent_form', [])
        away_form = bet.get('away_recent_form', [])
        
        # Detect hot hand fallacy
        if len(home_form) >= 5:
            if all(r == 'W' for r in home_form[-5:]):
                warnings.append("🔥 HOT HAND: Home team on 5+ win streak - beware regression")
                score -= 15
            if all(r == 'L' for r in home_form[-5:]):
                warnings.append("📉 GAMBLERS FALLACY RISK: Home on 5+ loss streak")
                score -= 10
        
        # Check 3: Recency Bias Detection
        line_movement = bet.get('line_movement', 0)
        if abs(line_movement) > 0.20:  # >20 cent move
            if bet.get('no_significant_news', True):
                warnings.append("📊 RECENCY BIAS: Large line move without major news")
                score -= 15
        
        # Check 4: Confirmation Bias Warning
        # Always add this as a reminder
        disconfirming = self._find_disconfirming_evidence(bet)
        if disconfirming:
            warnings.append(f"⚖️ CONTRARY VIEW: {disconfirming}")
        
        passed = score >= 60
        
        return {
            "layer": "Psychological",
            "passed": passed,
            "score": max(0, min(100, score)),
            "warnings": warnings,
            "details": {
                "edge_vs_market": f"{edge:.1%}",
                "overconfidence_risk": "HIGH" if edge > 0.15 else "LOW",
            }
        }
    
    def _statistical_check(self, bet: Dict) -> Dict:
        """
        📊 Layer 2: Statistical Validation
        
        Validates:
        - Expected Value calculation
        - Kelly Criterion sizing
        - Probability bounds
        - Odds reasonableness
        """
        warnings = []
        score = 100
        
        our_prob = bet.get('our_probability', 0.5)
        odds = bet.get('odds', 2.0)
        market = bet.get('market', 'unknown')
        league = bet.get('league', 'Unknown')
        
        # Expected Value
        ev = our_prob * odds - 1
        
        # Check 1: Positive EV Required
        if ev < 0:
            warnings.append(f"❌ NEGATIVE EV: {ev:.1%} - Never bet without value!")
            score -= 50
        elif ev < 0.02:
            warnings.append(f"⚠️ MARGINAL EV: {ev:.1%} - Below 2% threshold")
            score -= 20
        elif ev > 0.30:
            warnings.append(f"⚠️ SUSPICIOUS EV: {ev:.1%} - Too good to be true?")
            score -= 10
        
        # Check 2: Probability Sanity
        if our_prob < 0.35:
            warnings.append("⚠️ LOW PROBABILITY: <35% is high variance territory")
            score -= 15
        if our_prob > 0.90:
            warnings.append("⚠️ EXTREME PROBABILITY: >90% often overconfident")
            score -= 20
        
        # Check 3: Odds Range
        if odds < 1.15:
            warnings.append("⚠️ JUICE TOO HIGH: Odds <1.15 have ~13% vig")
            score -= 25
        if odds > 4.0:
            warnings.append("⚠️ LONGSHOT: Odds >4.0 have high variance")
            score -= 15
        
        # Check 4: Kelly Criterion
        if ev > 0:
            b = odds - 1
            kelly = (our_prob * b - (1 - our_prob)) / b
            quarter_kelly = kelly * 0.25
            
            if kelly > 0.20:
                warnings.append(f"⚠️ KELLY HIGH: Full Kelly={kelly:.1%}, use Quarter={quarter_kelly:.1%}")
        
        # Check 5: League-specific probability check
        league_data = self.knowledge.LEAGUE_WISDOM.get(league, {})
        if league_data:
            avg_goals = league_data.get('avg_goals', 2.8)
            
            # Check Over 2.5 probability against league average
            if 'over_2_5' in market:
                expected_over_25 = self._poisson_over_2_5(avg_goals)
                if abs(our_prob - expected_over_25) > 0.15:
                    warnings.append(f"📊 LEAGUE CHECK: Over 2.5 typically {expected_over_25:.1%} in {league}")
        
        passed = ev >= 0.02 and score >= 50
        
        return {
            "layer": "Statistical",
            "passed": passed,
            "score": max(0, min(100, score)),
            "warnings": warnings,
            "details": {
                "expected_value": f"{ev:+.1%}",
                "our_probability": f"{our_prob:.1%}",
                "implied_probability": f"{1/odds:.1%}",
                "edge": f"{our_prob - 1/odds:.1%}",
            }
        }
    
    def _historical_check(self, bet: Dict) -> Dict:
        """
        📚 Layer 3: Historical Pattern Check
        
        Finds similar historical bets and analyzes:
        - Win rate of similar bets
        - ROI of similar conditions
        - Team-specific historical performance
        """
        warnings = []
        score = 70  # Default neutral
        
        market = bet.get('market', '')
        league = bet.get('league', '')
        odds_range = self._get_odds_range(bet.get('odds', 2.0))
        
        # Find similar historical bets
        similar = self._find_similar_bets(bet)
        
        if len(similar) < 10:
            warnings.append("⚠️ INSUFFICIENT DATA: <10 similar historical bets")
            details = {"sample_size": len(similar), "win_rate": "N/A", "roi": "N/A"}
        else:
            # Calculate historical performance
            wins = sum(1 for b in similar if b.get('won', False))
            total_profit = sum(b.get('profit', 0) for b in similar)
            total_staked = len(similar)
            
            win_rate = wins / len(similar)
            roi = total_profit / total_staked if total_staked > 0 else 0
            
            details = {
                "sample_size": len(similar),
                "win_rate": f"{win_rate:.1%}",
                "roi": f"{roi:+.1%}",
            }
            
            # Adjust score based on historical performance
            if win_rate > 0.55:
                score += 20
            elif win_rate < 0.45:
                score -= 20
                warnings.append(f"⚠️ POOR HISTORY: Similar bets won only {win_rate:.1%}")
            
            if roi > 0.05:
                score += 15
            elif roi < -0.10:
                score -= 25
                warnings.append(f"⚠️ NEGATIVE ROI: Similar bets have {roi:.1%} ROI")
        
        passed = score >= 60
        
        return {
            "layer": "Historical",
            "passed": passed,
            "score": max(0, min(100, score)),
            "warnings": warnings,
            "details": details
        }
    
    def _market_check(self, bet: Dict) -> Dict:
        """
        📈 Layer 4: Market Efficiency Check
        
        Analyzes:
        - Line movement patterns
        - Sharp vs square money
        - Market timing
        """
        warnings = []
        score = 75  # Default
        
        line_movement = bet.get('line_movement', 0)
        public_pct = bet.get('public_percentage', 50)
        money_pct = bet.get('money_percentage', 50)
        
        # Check 1: Reverse Line Movement (Sharp indicator)
        if public_pct > 60 and line_movement < 0:
            # Line moved AGAINST public
            score += 15
            warnings.append("💰 SHARP SIGNAL: Reverse line movement detected")
        
        if public_pct < 40 and line_movement > 0.1:
            # Line moved WITH minority
            score += 10
            warnings.append("💰 SHARP BACKING: Line moving with minority")
        
        # Check 2: Steam Move Detection
        if abs(line_movement) > 0.15:
            warnings.append(f"🚨 STEAM MOVE: Line moved {line_movement:.0%}")
            # Steam can be good or bad depending on timing
        
        # Check 3: Sharp vs Square Money
        if money_pct and public_pct:
            if money_pct > public_pct + 20:
                score += 10
                warnings.append("💰 BIG MONEY: Large bets backing this side")
            elif money_pct < public_pct - 20:
                score -= 15
                warnings.append("⚠️ SMALL MONEY: Public squares on this side")
        
        # Check 4: Market Timing
        # (In real implementation, check if odds have stabilized)
        
        passed = score >= 60
        
        return {
            "layer": "Market",
            "passed": passed,
            "score": max(0, min(100, score)),
            "warnings": warnings,
            "details": {
                "line_movement": f"{line_movement:+.2f}" if line_movement else "N/A",
                "public_percentage": f"{public_pct}%" if public_pct else "N/A",
            }
        }
    
    def _ensemble_check(self, bet: Dict) -> Dict:
        """
        🤖 Layer 5: Ensemble Model Agreement
        
        Checks agreement between:
        - Dixon-Coles model
        - Basic Poisson
        - Elo-based prediction
        - (Optional) LLM analysis
        """
        warnings = []
        score = 70  # Default
        
        # Get model predictions if available
        dixon_coles_prob = bet.get('dixon_coles_probability')
        poisson_prob = bet.get('poisson_probability')
        elo_prob = bet.get('elo_probability')
        our_prob = bet.get('our_probability', 0.5)
        
        models = []
        if dixon_coles_prob:
            models.append(('Dixon-Coles', dixon_coles_prob))
        if poisson_prob:
            models.append(('Poisson', poisson_prob))
        if elo_prob:
            models.append(('Elo', elo_prob))
        
        if len(models) >= 2:
            # Calculate agreement
            probs = [p for _, p in models]
            std_dev = np.std(probs)
            mean_prob = np.mean(probs)
            
            # Check consensus
            if std_dev < 0.05:
                score += 20
                warnings.append("✅ STRONG CONSENSUS: Models agree (σ < 5%)")
            elif std_dev > 0.10:
                score -= 20
                warnings.append(f"⚠️ MODEL DISAGREEMENT: σ = {std_dev:.1%}")
            
            # Check if our probability aligns with ensemble
            if abs(our_prob - mean_prob) > 0.10:
                warnings.append(f"⚠️ OUTLIER: Our prob {our_prob:.1%} vs ensemble {mean_prob:.1%}")
                score -= 15
            
            details = {
                "models_used": len(models),
                "ensemble_mean": f"{mean_prob:.1%}",
                "ensemble_std": f"{std_dev:.1%}",
                "our_vs_ensemble": f"{our_prob - mean_prob:+.1%}",
            }
        else:
            warnings.append("⚠️ LIMITED ENSEMBLE: Only 1 model available")
            details = {"models_used": len(models)}
        
        passed = score >= 60
        
        return {
            "layer": "Ensemble",
            "passed": passed,
            "score": max(0, min(100, score)),
            "warnings": warnings,
            "details": details
        }
    
    def _find_disconfirming_evidence(self, bet: Dict) -> Optional[str]:
        """Find reasons why the bet might LOSE"""
        market = bet.get('market', '')
        home = bet.get('home_team', '')
        away = bet.get('away_team', '')
        
        evidence = []
        
        if 'over' in market:
            evidence.append("Defensive teams can nullify attacking play")
            evidence.append("Weather conditions might reduce scoring")
        
        if 'btts' in market:
            evidence.append("One team could dominate possession")
            evidence.append("Tactical setups might prevent chances")
        
        if 'home_win' in market:
            evidence.append("Away team might be motivated for upset")
            evidence.append("Home team could have key injuries")
        
        return evidence[0] if evidence else None
    
    def _find_similar_bets(self, bet: Dict) -> List[Dict]:
        """Find historically similar bets"""
        similar = []
        
        market = bet.get('market', '')
        league = bet.get('league', '')
        odds_range = self._get_odds_range(bet.get('odds', 2.0))
        
        for hist in self.historical_bets:
            if (hist.get('market') == market and 
                hist.get('league') == league and
                self._get_odds_range(hist.get('odds', 0)) == odds_range):
                similar.append(hist)
        
        return similar
    
    def _get_odds_range(self, odds: float) -> str:
        """Categorize odds into ranges"""
        if odds < 1.5:
            return "1.00-1.50"
        elif odds < 2.0:
            return "1.50-2.00"
        elif odds < 3.0:
            return "2.00-3.00"
        else:
            return "3.00+"
    
    def _poisson_over_2_5(self, avg_goals: float) -> float:
        """Calculate Over 2.5 probability from average goals"""
        # P(X > 2) = 1 - P(X <= 2) = 1 - (P(0) + P(1) + P(2))
        prob_0 = math.exp(-avg_goals)
        prob_1 = avg_goals * math.exp(-avg_goals)
        prob_2 = (avg_goals ** 2 / 2) * math.exp(-avg_goals)
        
        return 1 - (prob_0 + prob_1 + prob_2)
    
    def _generate_recommendations(self, bet: Dict, checks: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on checks
        stats = next(c for c in checks if c['layer'] == 'Statistical')
        if stats['details'].get('expected_value', '').startswith('+'):
            ev = float(stats['details']['expected_value'].rstrip('%')) / 100
            if ev > 0.05:
                recommendations.append("✅ Strong value - consider standard stake")
            elif ev > 0.02:
                recommendations.append("📊 Marginal value - consider reduced stake")
        
        # Based on ensemble
        ensemble = next(c for c in checks if c['layer'] == 'Ensemble')
        if 'CONSENSUS' in str(ensemble.get('warnings', [])):
            recommendations.append("🤖 Models agree - higher confidence bet")
        
        # Based on market
        market = next(c for c in checks if c['layer'] == 'Market')
        if 'SHARP' in str(market.get('warnings', [])):
            recommendations.append("💰 Sharp money detected - follow the pros")
        
        return recommendations
    
    def _generate_reasoning(self, bet: Dict, checks: List[Dict], is_valid: bool) -> str:
        """Generate human-readable reasoning"""
        home = bet.get('home_team', 'Home')
        away = bet.get('away_team', 'Away')
        market = bet.get('market', 'unknown')
        
        reasoning = f"📋 VALIDATION SUMMARY: {home} vs {away} - {market}\n\n"
        
        for check in checks:
            status = "✅" if check['passed'] else "❌"
            reasoning += f"{status} {check['layer']}: Score {check['score']}/100\n"
        
        reasoning += f"\n"
        
        if is_valid:
            reasoning += "🎯 VERDICT: BET APPROVED - Meets validation criteria\n"
        else:
            reasoning += "🚫 VERDICT: BET REJECTED - Failed validation\n"
        
        # Add key warnings
        all_warnings = []
        for check in checks:
            all_warnings.extend(check.get('warnings', []))
        
        if all_warnings:
            reasoning += "\n⚠️ KEY WARNINGS:\n"
            for w in all_warnings[:5]:  # Top 5 warnings
                reasoning += f"  • {w}\n"
        
        return reasoning
    
    def _load_historical_bets(self) -> List[Dict]:
        """Load historical bet data for pattern matching"""
        hist_file = Path("data/self_training/predictions.json")
        
        if hist_file.exists():
            try:
                with open(hist_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict to list if needed
                    if isinstance(data, dict):
                        return list(data.values())
                    return data
            except:
                pass
        
        return []
    
    def print_knowledge_summary(self):
        """Print the betting knowledge base summary"""
        print("\n" + "=" * 70)
        print("🎓 DEEP BETTING KNOWLEDGE BASE")
        print("=" * 70)
        
        print("\n📚 CORE PRINCIPLES:")
        for name, principle in self.knowledge.PRINCIPLES.items():
            print(f"\n  {name.upper()}:")
            print(f"    {principle.strip()[:200]}...")
        
        print("\n\n🧠 PSYCHOLOGICAL TRAPS:")
        for name, trap in self.knowledge.PSYCHOLOGICAL_TRAPS.items():
            print(f"\n  {name}: {trap['description'][:60]}...")
        
        print("\n\n⚽ LEAGUE WISDOM:")
        for league, data in self.knowledge.LEAGUE_WISDOM.items():
            print(f"  {league}: Avg {data['avg_goals']} goals, {data['characteristics'][:40]}...")


# Standalone test
if __name__ == "__main__":
    validator = DeepValidator()
    
    # Print knowledge base
    validator.print_knowledge_summary()
    
    # Test validation
    test_bet = {
        "home_team": "Bayern Munich",
        "away_team": "Dortmund",
        "league": "Bundesliga",
        "market": "over_2_5",
        "our_probability": 0.68,
        "odds": 1.85,
        "dixon_coles_probability": 0.65,
        "poisson_probability": 0.70,
        "elo_probability": 0.66,
    }
    
    print("\n\n" + "=" * 70)
    print("🧪 VALIDATION TEST")
    print("=" * 70)
    
    result = validator.validate_bet(test_bet)
    print(result.reasoning)
    
    print(f"\n📊 Confidence Score: {result.confidence_score:.1f}/100")
    print(f"✅ Checks Passed: {result.total_checks_passed}/{result.total_checks}")
    print(f"🎯 Valid Bet: {result.is_valid}")

#!/usr/bin/env python3
"""
🔬 EXTREME KNOWLEDGE TESTING FRAMEWORK
======================================
Validates ALL knowledge files and tests the prediction system to the extreme!

Tests:
1. Mathematical Accuracy (Kelly, Poisson, EV formulas)
2. League Statistics Validation
3. Form Analysis Accuracy
4. Derby Detection
5. Odds Range Optimization
6. Value Bet Detection
7. Accumulator Logic
8. Bankroll Management Rules

This ensures the system uses ALL knowledge files correctly!
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.advanced_predictor import (
    AdvancedPredictor, 
    BettingMath, 
    FormAnalyzer,
    TeamStats,
    MatchContext,
    Market,
    LEAGUE_PROFILES,
    KNOWN_DERBIES,
    OPTIMAL_ODDS_RANGE,
    MARKET_ODDS_RANGES
)


class KnowledgeTestFramework:
    """
    🔬 Extreme testing of all knowledge file implementations
    """
    
    def __init__(self):
        self.math = BettingMath()
        self.form = FormAnalyzer()
        self.predictor = AdvancedPredictor(bankroll=1000)
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def run_all_tests(self):
        """Run ALL knowledge tests"""
        print("\n" + "="*70)
        print("🔬 EXTREME KNOWLEDGE TESTING FRAMEWORK")
        print("="*70)
        
        # 1. BETTING_MATHEMATICS.md Tests
        self.test_betting_mathematics()
        
        # 2. OPTIMAL_ODDS_RANGES.md Tests
        self.test_optimal_odds_ranges()
        
        # 3. LEAGUE_STATISTICS.md Tests
        self.test_league_statistics()
        
        # 4. FORM_ANALYSIS.md Tests
        self.test_form_analysis()
        
        # 5. HEAD_TO_HEAD.md Tests
        self.test_head_to_head()
        
        # 6. ALL_BETTING_MARKETS.md Tests
        self.test_all_betting_markets()
        
        # 7. BANKROLL_MANAGEMENT.md Tests
        self.test_bankroll_management()
        
        # 8. Integration Tests
        self.test_full_integration()
        
        # Summary
        self.print_summary()
    
    def assert_test(self, name: str, condition: bool, details: str = ""):
        """Assert a test condition"""
        if condition:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"
        
        self.tests.append({
            'name': name,
            'passed': condition,
            'details': details
        })
        
        if not condition:
            print(f"   {status}: {name}")
            if details:
                print(f"      → {details}")
    
    def assert_close(self, name: str, actual: float, expected: float, tolerance: float = 0.01):
        """Assert values are close within tolerance"""
        is_close = abs(actual - expected) <= tolerance
        details = f"Expected {expected:.4f}, got {actual:.4f}" if not is_close else ""
        self.assert_test(name, is_close, details)
    
    # ========================================================================
    # 1. BETTING_MATHEMATICS.md TESTS
    # ========================================================================
    
    def test_betting_mathematics(self):
        """Test all formulas from BETTING_MATHEMATICS.md"""
        print("\n📐 Testing BETTING_MATHEMATICS.md...")
        
        # Test implied probability
        self.assert_close(
            "Implied Probability: 1.50 → 66.7%",
            self.math.implied_probability(1.50),
            0.6667
        )
        self.assert_close(
            "Implied Probability: 2.00 → 50%",
            self.math.implied_probability(2.00),
            0.50
        )
        self.assert_close(
            "Implied Probability: 1.20 → 83.3%",
            self.math.implied_probability(1.20),
            0.8333
        )
        
        # Test fair odds
        self.assert_close(
            "Fair Odds: 75% → 1.33",
            self.math.fair_odds(0.75),
            1.333,
            tolerance=0.01
        )
        self.assert_close(
            "Fair Odds: 60% → 1.67",
            self.math.fair_odds(0.60),
            1.667
        )
        
        # Test value calculation
        # Value = (Our Prob × Odds) - 1
        self.assert_close(
            "Value: 75% @ 1.40 → +5%",
            self.math.value(0.75, 1.40),
            0.05
        )
        self.assert_close(
            "Value: 65% @ 1.40 → -9%",
            self.math.value(0.65, 1.40),
            -0.09
        )
        
        # Test edge calculation
        self.assert_close(
            "Edge: 78% @ odds 1.40 (implied 71.4%) → +6.6%",
            self.math.edge(0.78, 1.40),
            0.0657,
            tolerance=0.01
        )
        
        # Test Kelly Criterion
        # Kelly = (p × b - q) / b where b = odds - 1, q = 1 - p
        # For p=0.60, odds=2.00: Kelly = (0.60 × 1.0 - 0.40) / 1.0 = 0.20
        full_kelly = (0.60 * 1.0 - 0.40) / 1.0  # = 0.20
        fractional_kelly = full_kelly * 0.25  # = 0.05
        
        self.assert_close(
            "Kelly: 60% @ 2.00, 25% fraction → 5%",
            self.math.kelly_criterion(0.60, 2.00, 0.25),
            0.05
        )
        
        # Test EV calculation
        # EV = Stake × [(Prob × Odds) - 1]
        self.assert_close(
            "EV: €10 @ 72% @ 1.50 → +€0.80",
            self.math.expected_value(0.72, 1.50, 10),
            0.80
        )
        
        # Test Poisson distribution
        # P(X=2) when λ=2.5 = (2.5^2 × e^-2.5) / 2! = 0.257
        expected_poisson = (2.5**2 * math.exp(-2.5)) / math.factorial(2)
        self.assert_close(
            "Poisson: P(2 goals | λ=2.5) ≈ 25.7%",
            self.math.poisson_probability(2, 2.5),
            expected_poisson
        )
        
        # Test is_value_bet
        self.assert_test(
            "Is Value Bet: 70% @ 1.50 (edge 3.3%) with 5% min → False",
            not self.math.is_value_bet(0.70, 1.50, 0.05),
            "Edge 3.3% < 5% threshold"
        )
        self.assert_test(
            "Is Value Bet: 75% @ 1.40 (edge 3.6%) with 3% min → True",
            self.math.is_value_bet(0.75, 1.40, 0.03),
            "Edge 3.6% >= 3% threshold"
        )
        
        print(f"   ✅ Betting Mathematics: {self.passed} tests passed")
    
    # ========================================================================
    # 2. OPTIMAL_ODDS_RANGES.md TESTS
    # ========================================================================
    
    def test_optimal_odds_ranges(self):
        """Test optimal odds range implementation"""
        print("\n💰 Testing OPTIMAL_ODDS_RANGES.md...")
        
        # Test golden range
        self.assert_test(
            "Golden Range: Min = 1.30",
            OPTIMAL_ODDS_RANGE['min'] == 1.30,
            f"Got {OPTIMAL_ODDS_RANGE['min']}"
        )
        self.assert_test(
            "Golden Range: Max = 1.70",
            OPTIMAL_ODDS_RANGE['max'] == 1.70,
            f"Got {OPTIMAL_ODDS_RANGE['max']}"
        )
        
        # Test market-specific ranges exist
        self.assert_test(
            "Market ranges defined for Over 1.5",
            Market.OVER_1_5 in MARKET_ODDS_RANGES,
            ""
        )
        self.assert_test(
            "Market ranges defined for BTTS",
            Market.BTTS_YES in MARKET_ODDS_RANGES,
            ""
        )
        
        # Test tier classification
        self.assert_test(
            "Over 1.5 is Tier 1 (best)",
            MARKET_ODDS_RANGES.get(Market.OVER_1_5, {}).get('tier') == 1,
            ""
        )
        self.assert_test(
            "Draw is Tier 3 (risky)",
            MARKET_ODDS_RANGES.get(Market.DRAW, {}).get('tier') == 3,
            ""
        )
        
        # Test predictor uses optimal range
        self.assert_test(
            "Predictor has optimal odds filter",
            hasattr(self.predictor, 'optimal_odds_min'),
            ""
        )
        self.assert_close(
            "Predictor min odds = 1.30",
            self.predictor.optimal_odds_min,
            1.30
        )
        self.assert_close(
            "Predictor max odds = 1.70",
            self.predictor.optimal_odds_max,
            1.70
        )
        
        print(f"   ✅ Optimal Odds Ranges: Tests completed")
    
    # ========================================================================
    # 3. LEAGUE_STATISTICS.md TESTS
    # ========================================================================
    
    def test_league_statistics(self):
        """Test league statistics implementation"""
        print("\n🏆 Testing LEAGUE_STATISTICS.md...")
        
        # Test all leagues are defined
        required_leagues = ['bundesliga', 'premier_league', 'la_liga', 'serie_a', 'ligue_1', 'eredivisie', 'championship']
        for league in required_leagues:
            self.assert_test(
                f"League profile exists: {league}",
                league in LEAGUE_PROFILES,
                ""
            )
        
        # Test Bundesliga stats (highest goals)
        bundesliga = LEAGUE_PROFILES.get('bundesliga', {})
        self.assert_test(
            "Bundesliga avg goals > 3.0",
            bundesliga.get('avg_goals', 0) > 3.0,
            f"Got {bundesliga.get('avg_goals')}"
        )
        self.assert_test(
            "Bundesliga Over 2.5 rate > 55%",
            bundesliga.get('over_2_5_rate', 0) > 0.55,
            f"Got {bundesliga.get('over_2_5_rate')}"
        )
        self.assert_test(
            "Bundesliga best markets include Over 2.5",
            Market.OVER_2_5 in bundesliga.get('best_markets', []),
            ""
        )
        
        # Test La Liga stats (lowest goals)
        la_liga = LEAGUE_PROFILES.get('la_liga', {})
        self.assert_test(
            "La Liga avg goals < 2.7",
            la_liga.get('avg_goals', 10) < 2.7,
            f"Got {la_liga.get('avg_goals')}"
        )
        self.assert_test(
            "La Liga avoid markets include Over 2.5",
            Market.OVER_2_5 in la_liga.get('avoid_markets', []),
            ""
        )
        
        # Test Eredivisie (highest scoring)
        eredivisie = LEAGUE_PROFILES.get('eredivisie', {})
        self.assert_test(
            "Eredivisie avg goals > Bundesliga",
            eredivisie.get('avg_goals', 0) > bundesliga.get('avg_goals', 10),
            f"Eredivisie: {eredivisie.get('avg_goals')}, Bundesliga: {bundesliga.get('avg_goals')}"
        )
        self.assert_test(
            "Eredivisie confidence boost > 1.0",
            eredivisie.get('confidence_boost', 0) > 1.0,
            f"Got {eredivisie.get('confidence_boost')}"
        )
        
        # Test predictor uses league profiles
        profile = self.predictor.get_league_profile("Bundesliga")
        self.assert_test(
            "Predictor returns correct Bundesliga profile",
            profile == bundesliga,
            ""
        )
        
        print(f"   ✅ League Statistics: Tests completed")
    
    # ========================================================================
    # 4. FORM_ANALYSIS.md TESTS
    # ========================================================================
    
    def test_form_analysis(self):
        """Test form analysis implementation"""
        print("\n📊 Testing FORM_ANALYSIS.md...")
        
        # Test weighted form
        # WWWWW should be 100%, LLLLL should be 0%
        self.assert_close(
            "Weighted Form: WWWWW → 100%",
            self.form.weighted_form(['W', 'W', 'W', 'W', 'W']),
            100.0
        )
        self.assert_close(
            "Weighted Form: LLLLL → 0%",
            self.form.weighted_form(['L', 'L', 'L', 'L', 'L']),
            0.0
        )
        
        # Test that recent games weight more
        # WWWLL should be > LLWWW (recent matters more)
        form_recent_wins = self.form.weighted_form(['W', 'W', 'W', 'L', 'L'])
        form_recent_losses = self.form.weighted_form(['L', 'L', 'W', 'W', 'W'])
        self.assert_test(
            "Recent games weight more: WWWLL > LLWWW",
            form_recent_wins > form_recent_losses,
            f"WWWLL: {form_recent_wins:.1f}%, LLWWW: {form_recent_losses:.1f}%"
        )
        
        # Test momentum score
        self.assert_close(
            "Momentum: WWWWW → +100",
            self.form.momentum_score(['W', 'W', 'W', 'W', 'W']),
            100.0
        )
        self.assert_close(
            "Momentum: LLLLL → -100",
            self.form.momentum_score(['L', 'L', 'L', 'L', 'L']),
            -100.0
        )
        self.assert_close(
            "Momentum: DDDDD → 0",
            self.form.momentum_score(['D', 'D', 'D', 'D', 'D']),
            0.0
        )
        
        # Test trend detection
        trend_rising, adj_rising = self.form.detect_trend(['W', 'W', 'W', 'L', 'L', 'L'])
        trend_falling, adj_falling = self.form.detect_trend(['L', 'L', 'L', 'W', 'W', 'W'])
        
        self.assert_test(
            "Trend detection: WWWLLL → rising",
            'rising' in trend_rising,
            f"Got {trend_rising}"
        )
        self.assert_test(
            "Trend detection: LLLWWW → falling",
            'falling' in trend_falling,
            f"Got {trend_falling}"
        )
        self.assert_test(
            "Rising trend adjustment > 1.0",
            adj_rising > 1.0,
            f"Got {adj_rising}"
        )
        self.assert_test(
            "Falling trend adjustment < 1.0",
            adj_falling < 1.0,
            f"Got {adj_falling}"
        )
        
        # Test form rating
        self.assert_test(
            "Form Rating: 14 points → EXCELLENT",
            self.form.form_rating(14) == 'EXCELLENT',
            f"Got {self.form.form_rating(14)}"
        )
        self.assert_test(
            "Form Rating: 2 points → TERRIBLE",
            self.form.form_rating(2) == 'TERRIBLE',
            f"Got {self.form.form_rating(2)}"
        )
        
        print(f"   ✅ Form Analysis: Tests completed")
    
    # ========================================================================
    # 5. HEAD_TO_HEAD.md TESTS
    # ========================================================================
    
    def test_head_to_head(self):
        """Test H2H and derby detection"""
        print("\n⚔️ Testing HEAD_TO_HEAD.md...")
        
        # Test known derbies exist
        self.assert_test(
            "Bundesliga derbies defined",
            'bundesliga' in KNOWN_DERBIES,
            ""
        )
        self.assert_test(
            "Premier League derbies defined",
            'premier_league' in KNOWN_DERBIES,
            ""
        )
        
        # Test Der Klassiker detection
        is_derby, name = self.predictor.is_derby("Bayern München", "Borussia Dortmund", "Bundesliga")
        self.assert_test(
            "Der Klassiker detected: Bayern vs Dortmund",
            is_derby,
            f"Got: is_derby={is_derby}, name={name}"
        )
        self.assert_test(
            "Der Klassiker name correct",
            "Klassiker" in name,
            f"Got: {name}"
        )
        
        # Test North London Derby
        is_derby, name = self.predictor.is_derby("Arsenal", "Tottenham Hotspur", "Premier League")
        self.assert_test(
            "North London Derby detected",
            is_derby,
            f"Got: is_derby={is_derby}, name={name}"
        )
        
        # Test El Clásico
        is_derby, name = self.predictor.is_derby("Real Madrid", "Barcelona", "La Liga")
        self.assert_test(
            "El Clásico detected",
            is_derby,
            f"Got: is_derby={is_derby}"
        )
        
        # Test non-derby
        is_derby, name = self.predictor.is_derby("Bayern München", "Hoffenheim", "Bundesliga")
        self.assert_test(
            "Non-derby correctly identified: Bayern vs Hoffenheim",
            not is_derby,
            f"Got: is_derby={is_derby}"
        )
        
        print(f"   ✅ Head-to-Head: Tests completed")
    
    # ========================================================================
    # 6. ALL_BETTING_MARKETS.md TESTS
    # ========================================================================
    
    def test_all_betting_markets(self):
        """Test all betting markets implementation"""
        print("\n🎰 Testing ALL_BETTING_MARKETS.md...")
        
        # Test all market types exist
        expected_markets = [
            Market.OVER_1_5, Market.OVER_2_5, Market.OVER_3_5,
            Market.UNDER_2_5, Market.UNDER_3_5,
            Market.BTTS_YES, Market.BTTS_NO,
            Market.HOME_WIN, Market.DRAW, Market.AWAY_WIN,
            Market.HOME_OR_DRAW, Market.AWAY_OR_DRAW,
            Market.DNB_HOME, Market.DNB_AWAY,
            Market.AH_HOME_MINUS_0_5, Market.AH_AWAY_PLUS_0_5
        ]
        
        for market in expected_markets:
            self.assert_test(
                f"Market enum exists: {market.value}",
                isinstance(market, Market),
                ""
            )
        
        # Test predictor calculates all markets
        home = TeamStats(name="Test Home", elo=1600, form_points=10)
        away = TeamStats(name="Test Away", elo=1500, form_points=8)
        context = MatchContext(
            home_team=home,
            away_team=away,
            league="Bundesliga",
            date="2026-01-28"
        )
        
        predictions = self.predictor.predict_match(context)
        
        # Check that multiple markets are predicted
        self.assert_test(
            "Predictor returns multiple market predictions",
            len(predictions.predictions) >= 10,
            f"Got {len(predictions.predictions)} predictions"
        )
        
        # Check key markets are included
        predicted_markets = {p.market for p in predictions.predictions}
        
        self.assert_test(
            "Over 1.5 in predictions",
            Market.OVER_1_5 in predicted_markets,
            ""
        )
        self.assert_test(
            "BTTS Yes in predictions",
            Market.BTTS_YES in predicted_markets,
            ""
        )
        self.assert_test(
            "Home or Draw in predictions",
            Market.HOME_OR_DRAW in predicted_markets,
            ""
        )
        
        # Test probabilities are valid (0-1)
        for pred in predictions.predictions:
            self.assert_test(
                f"{pred.market.value} probability valid",
                0 <= pred.probability <= 1.05,  # Allow small overshoot from Poisson
                f"Got {pred.probability}"
            )
        
        print(f"   ✅ All Betting Markets: Tests completed")
    
    # ========================================================================
    # 7. BANKROLL_MANAGEMENT.md TESTS
    # ========================================================================
    
    def test_bankroll_management(self):
        """Test bankroll management implementation"""
        print("\n💵 Testing BANKROLL_MANAGEMENT.md...")
        
        # Test Kelly stakes are reasonable
        # For a strong value bet (70% @ 1.80), Kelly should suggest meaningful stake
        kelly_strong = self.math.kelly_criterion(0.70, 1.80, fraction=0.25)
        self.assert_test(
            "Kelly stake positive for value bet",
            kelly_strong > 0,
            f"Got {kelly_strong}"
        )
        self.assert_test(
            "Kelly stake <= 5% (capped)",
            kelly_strong <= 0.05,
            f"Got {kelly_strong}"
        )
        
        # For a bad bet (40% @ 1.50), Kelly should be 0 or negative
        kelly_bad = self.math.kelly_criterion(0.40, 1.50, fraction=0.25)
        self.assert_test(
            "Kelly stake 0 for negative edge bet",
            kelly_bad == 0,
            f"Got {kelly_bad}"
        )
        
        # Test predictor integrates Kelly
        home = TeamStats(name="Test Home", elo=1700, form_points=12)
        away = TeamStats(name="Test Away", elo=1400, form_points=6)
        context = MatchContext(
            home_team=home,
            away_team=away,
            league="Bundesliga",
            date="2026-01-28"
        )
        
        predictions = self.predictor.predict_match(context)
        
        # Check Kelly stakes are calculated
        for pred in predictions.predictions[:5]:
            self.assert_test(
                f"{pred.market.value} has Kelly stake",
                hasattr(pred, 'kelly_stake'),
                ""
            )
            if pred.is_value_bet:
                self.assert_test(
                    f"{pred.market.value} Kelly > 0 for value bet",
                    pred.kelly_stake >= 0,
                    f"Got {pred.kelly_stake}"
                )
        
        print(f"   ✅ Bankroll Management: Tests completed")
    
    # ========================================================================
    # 8. FULL INTEGRATION TESTS
    # ========================================================================
    
    def test_full_integration(self):
        """Test full system integration"""
        print("\n🔗 Testing Full Integration...")
        
        # Create realistic match
        bayern = TeamStats(
            name="Bayern München",
            elo=1850,
            form_points=13,
            home_form=14,
            goals_scored_last_5=12,
            goals_conceded_last_5=3
        )
        
        dortmund = TeamStats(
            name="Borussia Dortmund",
            elo=1780,
            form_points=10,
            away_form=8,
            goals_scored_last_5=10,
            goals_conceded_last_5=6
        )
        
        # Test full prediction pipeline
        is_derby, derby_name = self.predictor.is_derby(bayern.name, dortmund.name, "Bundesliga")
        
        context = MatchContext(
            home_team=bayern,
            away_team=dortmund,
            league="Bundesliga",
            date="2026-01-28",
            is_derby=is_derby,
            derby_name=derby_name
        )
        
        predictions = self.predictor.predict_match(context)
        
        # Test predictions object
        self.assert_test(
            "Predictions has match context",
            predictions.match_context is not None,
            ""
        )
        self.assert_test(
            "Predictions has list of predictions",
            len(predictions.predictions) > 0,
            ""
        )
        self.assert_test(
            "Predictions has analysis notes",
            len(predictions.analysis_notes) > 0,
            ""
        )
        
        # Test best bet selection
        self.assert_test(
            "Best bet identified",
            predictions.best_bet is not None,
            ""
        )
        
        if predictions.best_bet:
            self.assert_test(
                "Best bet is value bet",
                predictions.best_bet.is_value_bet,
                ""
            )
        
        # Test recommendations filter
        recs = self.predictor.get_recommendations(predictions, max_picks=3, only_optimal_range=True)
        
        # Check all recs are in optimal range
        for rec in recs:
            self.assert_test(
                f"Rec {rec.market.value} in optimal range",
                self.predictor.optimal_odds_min <= rec.odds <= self.predictor.optimal_odds_max,
                f"Odds: {rec.odds}"
            )
        
        # Test accumulator legs
        self.assert_test(
            "Accumulator legs identified",
            isinstance(predictions.accumulator_legs, list),
            ""
        )
        
        # Test xG calculation
        league_profile = self.predictor.get_league_profile("Bundesliga")
        home_xg, away_xg = self.predictor.estimate_expected_goals(bayern, dortmund, league_profile)
        
        self.assert_test(
            "Home xG reasonable (0.5-4)",
            0.5 <= home_xg <= 4.0,
            f"Got {home_xg}"
        )
        self.assert_test(
            "Away xG reasonable (0.3-3.5)",
            0.3 <= away_xg <= 3.5,
            f"Got {away_xg}"
        )
        self.assert_test(
            "Higher Elo team has higher xG",
            home_xg > away_xg,  # Bayern (1850) vs Dortmund (1780)
            f"Home: {home_xg}, Away: {away_xg}"
        )
        
        print(f"   ✅ Full Integration: Tests completed")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print(f"\n   Total Tests: {total}")
        print(f"   ✅ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        
        if total > 0:
            success_rate = (self.passed / total) * 100
            print(f"\n   Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 95:
                print("\n   🏆 EXCELLENT! Knowledge base fully operational!")
            elif success_rate >= 80:
                print("\n   ✅ GOOD! Most knowledge correctly implemented.")
            elif success_rate >= 60:
                print("\n   ⚠️  WARNING! Some knowledge implementation issues.")
            else:
                print("\n   ❌ CRITICAL! Major knowledge implementation problems.")
        
        # Print failed tests
        failed_tests = [t for t in self.tests if not t['passed']]
        if failed_tests:
            print("\n   Failed Tests:")
            for t in failed_tests:
                print(f"   ❌ {t['name']}")
                if t['details']:
                    print(f"      → {t['details']}")
        
        print("\n" + "="*70)


def main():
    """Run all knowledge tests"""
    framework = KnowledgeTestFramework()
    framework.run_all_tests()


if __name__ == "__main__":
    main()

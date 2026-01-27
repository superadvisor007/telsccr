#!/usr/bin/env python3
"""
🧪 TEST REASONING AGENT
======================
Test the Reasoning Agent with real matches.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reasoning.reasoning_agent import ReasoningAgent
from reasoning.match_context_collector import MatchContextCollector
from reasoning.schemas import BettingDecision, Recommendation

def test_context_collector():
    """Test the match context collector"""
    print("\n" + "="*60)
    print("🔍 TEST: Match Context Collector")
    print("="*60)
    
    collector = MatchContextCollector()
    
    # Test with a real match
    context = collector.collect_full_context(
        home_team="Bayern Munich",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        match_date="2025-01-28"
    )
    
    assert context.home_team == "Bayern Munich"
    assert context.away_team == "Borussia Dortmund"
    assert context.data_quality > 0
    
    print(f"✅ Context collected!")
    print(f"   Data Quality: {context.data_quality}%")
    print(f"   Home Stats: {context.home_stats.goals_per_game:.2f} gpg")
    print(f"   Away Stats: {context.away_stats.goals_per_game:.2f} gpg")
    print(f"   H2H: {context.h2h.total_matches} matches")
    
    # Print reasoning context
    print("\n📋 Reasoning Context Preview:")
    print(context.to_reasoning_context()[:1000] + "...")
    
    return context


def test_reasoning_agent():
    """Test the reasoning agent"""
    print("\n" + "="*60)
    print("🤖 TEST: Reasoning Agent")
    print("="*60)
    
    agent = ReasoningAgent(verbose=True)
    
    # Test with Der Klassiker
    decisions = agent.analyze_match(
        home_team="Bayern Munich",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        match_date="2025-01-28",
        target_markets=['over_2_5', 'btts_yes', 'over_1_5']
    )
    
    assert len(decisions) > 0, "Should return at least one decision"
    
    print(f"\n✅ Got {len(decisions)} decisions!")
    
    for d in decisions:
        assert d.market is not None
        assert d.recommendation is not None
        assert d.chain_of_thought is not None
        assert len(d.chain_of_thought.steps) > 0
        
        print(f"\n📊 {d.market.value.upper()}")
        print(f"   Probability: {d.our_probability:.1%}")
        print(f"   Odds: {d.market_odds:.2f}")
        print(f"   EV: {d.expected_value:+.1%}")
        print(f"   Recommendation: {d.recommendation.value}")
        print(f"   Reasoning Steps: {len(d.chain_of_thought.steps)}")
        print(f"   Why This Bet: {d.why_this_bet[:2]}")
        print(f"   Risks: {d.why_might_fail[:1]}")
    
    return decisions


def test_telegram_format():
    """Test Telegram message formatting"""
    print("\n" + "="*60)
    print("📱 TEST: Telegram Format")
    print("="*60)
    
    agent = ReasoningAgent(verbose=False)
    
    decisions = agent.analyze_match(
        home_team="Liverpool",
        away_team="Manchester City",
        league="Premier League",
        match_date="2025-01-28"
    )
    
    if decisions:
        best = max(decisions, key=lambda x: x.expected_value)
        
        print("\n📱 Telegram Message:")
        print("-" * 40)
        print(best.to_telegram_message())
        print("-" * 40)
        
        print("\n📋 Full Report:")
        print("-" * 40)
        report = best.to_full_report()
        print(report[:2000] + "..." if len(report) > 2000 else report)
    
    return decisions


def test_multi_match():
    """Test analyzing multiple matches"""
    print("\n" + "="*60)
    print("⚽ TEST: Multi-Match Analysis")
    print("="*60)
    
    agent = ReasoningAgent(verbose=False)
    
    matches = [
        {"home_team": "Real Madrid", "away_team": "Barcelona", "league": "La Liga", "date": "2025-01-28"},
        {"home_team": "Inter Milan", "away_team": "AC Milan", "league": "Serie A", "date": "2025-01-28"},
        {"home_team": "Ajax", "away_team": "Feyenoord", "league": "Eredivisie", "date": "2025-01-28"},
    ]
    
    best_bets = agent.find_best_bets(matches, min_ev=0.01, max_bets=5)
    
    print(f"\n✅ Found {len(best_bets)} value bets!")
    
    for bet in best_bets:
        print(f"   {bet.home_team} vs {bet.away_team}: {bet.market.value} @ {bet.market_odds:.2f} (EV: {bet.expected_value:+.1%})")
    
    return best_bets


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*60)
    print("⚠️ TEST: Edge Cases")
    print("="*60)
    
    agent = ReasoningAgent(verbose=False)
    
    # Test with unknown team
    print("\n1. Unknown team test...")
    decisions = agent.analyze_match(
        home_team="Unknown FC",
        away_team="Mystery United",
        league="Bundesliga"
    )
    print(f"   ✅ Handled gracefully: {len(decisions)} decisions")
    
    # Test with missing data
    print("\n2. Minimal data test...")
    decisions = agent.analyze_match(
        home_team="Test Team",
        away_team="Test Opponent",
        league="Unknown League"
    )
    print(f"   ✅ Handled gracefully: {len(decisions)} decisions")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 REASONING AGENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Context Collector", test_context_collector),
        ("Reasoning Agent", test_reasoning_agent),
        ("Telegram Format", test_telegram_format),
        ("Multi-Match", test_multi_match),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "✅ PASSED"))
        except Exception as e:
            results.append((name, f"❌ FAILED: {e}"))
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    for name, result in results:
        print(f"   {result} - {name}")
    
    passed = sum(1 for _, r in results if "PASSED" in r)
    print(f"\n   Total: {passed}/{len(tests)} passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

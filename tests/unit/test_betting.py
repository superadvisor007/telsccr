"""Test suite for betting engine."""
import pytest
import numpy as np

from src.betting.engine import BettingEngine


@pytest.fixture
def betting_engine():
    return BettingEngine(
        initial_bankroll=1000.0,
        target_quote=1.40,
        min_probability=0.72,
        max_stake_percentage=2.0,
        stop_loss_percentage=15.0,
    )


@pytest.fixture
def sample_predictions():
    return [
        {
            "match_id": 1,
            "home_team": "Team A",
            "away_team": "Team B",
            "over_1_5_odds": 1.20,
            "over_1_5_probability": 0.85,
            "confidence_score": 0.8,
            "key_factors": ["High-scoring teams", "Good weather"],
        },
        {
            "match_id": 2,
            "home_team": "Team C",
            "away_team": "Team D",
            "over_1_5_odds": 1.18,
            "over_1_5_probability": 0.87,
            "confidence_score": 0.75,
            "key_factors": ["Strong attack", "Weak defense"],
        },
        {
            "match_id": 3,
            "home_team": "Team E",
            "away_team": "Team F",
            "btts_odds": 1.45,
            "btts_probability": 0.75,
            "confidence_score": 0.7,
            "key_factors": ["Open play style"],
        },
    ]


def test_find_value_bets(betting_engine, sample_predictions):
    """Test value bet detection."""
    value_bets = betting_engine.find_value_bets(sample_predictions)
    
    assert len(value_bets) > 0
    
    # Check first value bet
    value_bet = value_bets[0]
    assert "match_id" in value_bet
    assert "odds" in value_bet
    assert "researched_probability" in value_bet
    assert "expected_value" in value_bet
    
    # Verify value: researched prob > implied prob
    implied_prob = 1 / value_bet["odds"]
    assert value_bet["researched_probability"] > implied_prob


def test_build_accumulator(betting_engine):
    """Test accumulator building."""
    value_bets = [
        {
            "match_id": 1,
            "match_info": "Team A vs Team B",
            "market": "over_1_5",
            "odds": 1.20,
            "researched_probability": 0.85,
            "expected_value": 0.10,
            "confidence": 0.8,
        },
        {
            "match_id": 2,
            "match_info": "Team C vs Team D",
            "market": "over_1_5",
            "odds": 1.18,
            "researched_probability": 0.87,
            "expected_value": 0.12,
            "confidence": 0.75,
        },
    ]
    
    accumulator = betting_engine.build_accumulator(value_bets, num_selections=2)
    
    assert accumulator is not None
    assert accumulator["num_selections"] == 2
    assert len(accumulator["selections"]) == 2
    
    # Check odds calculation
    expected_odds = 1.20 * 1.18
    assert abs(accumulator["total_odds"] - expected_odds) < 0.01
    
    # Check probability
    expected_prob = 0.85 * 0.87
    assert abs(accumulator["combined_probability"] - expected_prob) < 0.01


def test_calculate_stake(betting_engine):
    """Test stake calculation."""
    accumulator = {
        "total_odds": 1.42,
        "combined_probability": 0.75,
        "num_selections": 2,
    }
    
    stake = betting_engine.calculate_stake(accumulator, use_kelly=False)
    
    # Should be 2% of bankroll (fixed staking)
    expected_stake = 1000.0 * 0.02
    assert abs(stake - expected_stake) < 0.01


def test_stop_loss_detection(betting_engine):
    """Test stop-loss mechanism."""
    # Initially no stop loss
    assert betting_engine.check_stop_loss() is False
    
    # Simulate losses
    betting_engine.update_bankroll(-200)  # Down to 800 (20% loss)
    
    assert betting_engine.check_stop_loss() is True


def test_betting_statistics(betting_engine):
    """Test betting statistics calculation."""
    # Place some test bets
    accumulator = {
        "total_odds": 1.40,
        "combined_probability": 0.75,
        "selections": [],
    }
    
    # Place 3 bets
    for _ in range(3):
        betting_engine.place_bet(accumulator, 20.0)
    
    # Mark results
    betting_engine.bet_history[0]["status"] = "won"
    betting_engine.bet_history[1]["status"] = "lost"
    betting_engine.bet_history[2]["status"] = "pending"
    
    stats = betting_engine.get_statistics()
    
    assert stats["total_bets"] == 3
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["pending"] == 1
    assert stats["win_rate"] == 50.0  # 1 win out of 2 settled bets


def test_value_check():
    """Test value checking logic."""
    result = BettingEngine._check_value(
        researched_prob=0.80,
        odds=1.20,
        min_prob=0.72
    )
    
    assert result["has_value"] is True
    assert result["implied_probability"] == 1 / 1.20
    assert result["probability_edge"] > 0
    assert result["expected_value"] > 0

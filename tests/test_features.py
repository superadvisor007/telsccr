"""Test suite for feature engineering."""
import pytest
import numpy as np

from src.features.feature_engineer import FeatureEngineer


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def sample_match_data():
    return {
        "league": "Bundesliga",
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund",
        "match_date": "2026-01-28",
        "odds": {
            "over_1_5": 1.25,
            "btts_yes": 1.50,
        }
    }


@pytest.fixture
def sample_stats():
    return {
        "matches_played": 5,
        "wins": 3,
        "draws": 1,
        "losses": 1,
        "goals_per_game": 2.4,
        "goals_conceded_per_game": 1.2,
        "over_1_5_percentage": 80.0,
        "btts_percentage": 60.0,
        "clean_sheet_percentage": 20.0,
        "ppg": 2.0,
    }


def test_engineer_features(feature_engineer, sample_match_data, sample_stats):
    """Test feature engineering pipeline."""
    features = feature_engineer.engineer_features(
        match_data=sample_match_data,
        home_stats=sample_stats,
        away_stats=sample_stats,
        h2h_stats={"matches": 5, "avg_goals": 3.2, "btts_rate": 70},
        weather=None
    )
    
    # Check that key features are present
    assert "home_goals_per_game" in features
    assert "away_goals_per_game" in features
    assert "total_expected_goals" in features
    assert "over_1_5_baseline_prob" in features
    assert "btts_baseline_prob" in features
    
    # Check feature values
    assert features["home_goals_per_game"] == 2.4
    assert features["total_expected_goals"] == 4.8  # 2.4 + 2.4
    assert 0 <= features["over_1_5_baseline_prob"] <= 1.0


def test_create_feature_vector(feature_engineer, sample_match_data, sample_stats):
    """Test feature vector creation."""
    features = feature_engineer.engineer_features(
        match_data=sample_match_data,
        home_stats=sample_stats,
        away_stats=sample_stats,
        h2h_stats={"matches": 5, "avg_goals": 3.2, "btts_rate": 70},
        weather=None
    )
    
    vector = feature_engineer.create_feature_vector(features)
    
    # Check vector properties
    assert isinstance(vector, np.ndarray)
    assert len(vector) == 24  # Expected number of features
    assert np.all(np.isfinite(vector))  # No NaN or inf


def test_high_scoring_league_detection(feature_engineer):
    """Test high-scoring league detection."""
    assert feature_engineer._is_high_scoring_league("Bundesliga") is True
    assert feature_engineer._is_high_scoring_league("Eredivisie") is True
    assert feature_engineer._is_high_scoring_league("Serie A") is False


def test_weather_features(feature_engineer):
    """Test weather feature extraction."""
    weather = {
        "impact_score": 5,
        "favorable_for_goals": False,
        "temperature": 10,
        "precipitation": 5.0,
        "wind_speed": 8.0,
    }
    
    weather_features = feature_engineer._weather_features(weather)
    
    assert weather_features["weather_impact"] == 0.5  # 5/10
    assert weather_features["weather_favorable"] == 0.0
    assert weather_features["temperature"] == 10

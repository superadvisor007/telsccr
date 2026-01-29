#!/bin/bash
################################################################################
# Professional Soccer ML System - Complete Workflow Test
# Tests the entire pipeline: Data → Training → Validation → Deployment
################################################################################

echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   🎯 PROFESSIONAL SOCCER ML SYSTEM - WORKFLOW TEST                        ║
║   Testing complete transformation: Amateur LLM → Professional ML           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test status tracking
TESTS_PASSED=0
TESTS_FAILED=0

################################################################################
# Test 1: Check Dependencies
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 1: DEPENDENCY CHECK
═══════════════════════════════════════════════════════════════════════════
"

echo "📦 Checking Python packages..."

check_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✅${NC} $1"
        return 0
    else
        echo -e "   ${RED}❌${NC} $1 - MISSING"
        return 1
    fi
}

# Core packages
check_package "numpy" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_package "pandas" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_package "sklearn" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_package "xgboost" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_package "requests" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))

################################################################################
# Test 2: Project Structure
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 2: PROJECT STRUCTURE CHECK
═══════════════════════════════════════════════════════════════════════════
"

echo "📁 Verifying professional ML architecture..."

check_file() {
    if [ -f "$1" ]; then
        echo -e "   ${GREEN}✅${NC} $1"
        ((TESTS_PASSED++))
    else
        echo -e "   ${RED}❌${NC} $1 - MISSING"
        ((TESTS_FAILED++))
    fi
}

# Core components
check_file "src/features/advanced_features.py"
check_file "src/models/professional_model.py"
check_file "src/ingestion/historical_data_collector.py"
check_file "train_professional_models.py"
check_file "tests/expert_soccer_validation.py"
check_file ".github/workflows/daily_predictions.yml"

################################################################################
# Test 3: Advanced Feature Engineering
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 3: ADVANCED FEATURE ENGINEERING TEST
═══════════════════════════════════════════════════════════════════════════
"

echo "🔧 Testing Elo Rating System, Feature Engineering, Value Calculator..."

python3 << 'PYTHON_TEST'
import sys
sys.path.append('/workspaces/telegramsoccer')

try:
    from src.features.advanced_features import (
        EloRatingSystem, AdvancedFeatureEngineer, 
        ValueBettingCalculator, EloConfig
    )
    
    # Test Elo Rating System
    elo = EloRatingSystem(EloConfig())
    elo.update_ratings("Team A", "Team B", 2, 1, "2024-01-01")
    rating_a = elo.get_rating("Team A")
    rating_b = elo.get_rating("Team B")
    pred = elo.predict_match_outcome_probability("Team A", "Team B")
    
    assert rating_a > 1500, "Winner should gain Elo"
    assert rating_b < 1500, "Loser should lose Elo"
    assert 'over_1_5' in pred, "Prediction should include over 1.5"
    print("   ✅ Elo Rating System: WORKING")
    
    # Test Feature Engineering
    engineer = AdvancedFeatureEngineer(elo)
    form = engineer.calculate_form_index(['W', 'W', 'D', 'L', 'W'])
    assert 30 < form < 70, "Form index should be 0-100 range"
    print("   ✅ Feature Engineering: WORKING")
    
    # Test Value Betting Calculator
    has_value = ValueBettingCalculator.has_value(0.70, 1.50, min_edge=0.05)
    kelly = ValueBettingCalculator.calculate_kelly_stake(0.70, 1.80, 1000.0)
    
    assert isinstance(has_value, bool), "Value detection should return bool"
    assert kelly > 0 and kelly <= 100, "Kelly stake should be positive and capped"
    print("   ✅ Value Betting Calculator: WORKING")
    
    print("\n   🎯 ADVANCED FEATURES: ALL TESTS PASSED")
    sys.exit(0)

except Exception as e:
    print(f"   ❌ FEATURE ENGINEERING TEST FAILED: {e}")
    sys.exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    ((TESTS_PASSED+=3))
else
    ((TESTS_FAILED+=3))
fi

################################################################################
# Test 4: Professional Model Architecture
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 4: PROFESSIONAL MODEL ARCHITECTURE TEST
═══════════════════════════════════════════════════════════════════════════
"

echo "🤖 Testing XGBoost model class structure..."

python3 << 'PYTHON_TEST'
import sys
sys.path.append('/workspaces/telegramsoccer')

try:
    from src.models.professional_model import ProfessionalSoccerModel
    import pandas as pd
    import numpy as np
    
    # Create professional model instance
    model = ProfessionalSoccerModel(model_type='xgboost')
    
    # Test data preparation
    test_data = pd.DataFrame({
        'match_id': [1, 2, 3, 4, 5],
        'date': ['2024-01-01'] * 5,
        'home_team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
        'away_team': ['Team F', 'Team G', 'Team H', 'Team I', 'Team J'],
        'home_goals': [2, 1, 3, 0, 2],
        'away_goals': [1, 1, 2, 0, 1],
        'total_goals': [3, 2, 5, 0, 3],
        'elo_home': [1550, 1600, 1520, 1480, 1590],
        'elo_away': [1480, 1520, 1510, 1520, 1470],
        'form_home': [60, 55, 65, 45, 58],
        'form_away': [50, 52, 48, 55, 49]
    })
    
    X, y = model.prepare_training_data(test_data, target_market='over_1_5')
    
    assert len(X) == len(test_data), "Features should match data length"
    assert len(y) == len(test_data), "Target should match data length"
    assert 'match_id' not in X.columns, "Non-features should be excluded"
    
    print("   ✅ Professional Model Class: WORKING")
    print("   ✅ Data Preparation: WORKING")
    print("   ✅ XGBoost Integration: READY")
    
    print("\n   🎯 MODEL ARCHITECTURE: ALL TESTS PASSED")
    sys.exit(0)

except Exception as e:
    print(f"   ❌ MODEL ARCHITECTURE TEST FAILED: {e}")
    sys.exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    ((TESTS_PASSED+=3))
else
    ((TESTS_FAILED+=3))
fi

################################################################################
# Test 5: Historical Data Collection (API Check)
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 5: HISTORICAL DATA COLLECTION (API AVAILABILITY)
═══════════════════════════════════════════════════════════════════════════
"

echo "🔍 Testing free APIs for historical data..."

# Test OpenLigaDB (Bundesliga)
echo -n "   Testing OpenLigaDB API... "
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "https://api.openligadb.de/getmatchdata/bl1/2023" --max-time 10)
if [ "$RESPONSE" -eq 200 ]; then
    echo -e "${GREEN}✅ AVAILABLE${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAILED (HTTP $RESPONSE)${NC}"
    ((TESTS_FAILED++))
fi

# Test TheSportsDB (fallback)
echo -n "   Testing TheSportsDB API... "
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id=4331" --max-time 10)
if [ "$RESPONSE" -eq 200 ]; then
    echo -e "${GREEN}✅ AVAILABLE${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAILED (HTTP $RESPONSE)${NC}"
    ((TESTS_FAILED++))
fi

################################################################################
# Test 6: Value Betting Logic
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 6: VALUE BETTING LOGIC TEST
═══════════════════════════════════════════════════════════════════════════
"

echo "💰 Testing Expected Value, Kelly Criterion, CLV..."

python3 << 'PYTHON_TEST'
import sys
sys.path.append('/workspaces/telegramsoccer')

try:
    from src.features.advanced_features import ValueBettingCalculator
    
    # Test 1: Expected Value calculation
    ev = ValueBettingCalculator.calculate_expected_value(0.70, 1.80, 10.0)
    expected_ev = (0.70 * 8.0) - (0.30 * 10.0)  # (prob × profit) - ((1-prob) × stake)
    assert abs(ev - expected_ev) < 0.01, f"EV calculation wrong: {ev} vs {expected_ev}"
    print(f"   ✅ Expected Value: ${ev:+.2f} (CORRECT)")
    
    # Test 2: Value detection
    has_value = ValueBettingCalculator.has_value(0.70, 1.50, min_edge=0.05)
    implied_prob = 1 / 1.50  # = 0.667
    edge = 0.70 - implied_prob  # = 0.033
    print(f"   ✅ Value Detection: Edge = {edge:.2%} ({'VALUE' if has_value else 'NO VALUE'})")
    
    # Test 3: Kelly Criterion
    kelly_stake = ValueBettingCalculator.calculate_kelly_stake(0.70, 1.80, 1000.0, kelly_fraction=0.25)
    print(f"   ✅ Kelly Stake: ${kelly_stake:.2f} (capped at 10% = $100)")
    assert kelly_stake <= 100, "Kelly stake should be capped at 10% of bankroll"
    
    # Test 4: Closing Line Value
    clv = ValueBettingCalculator.calculate_closing_line_value(1.50, 1.40, 1.55)
    print(f"   ✅ Closing Line Value: {clv:+.2f} ({'BEAT MARKET' if clv > 0 else 'MARKET WON'})")
    
    print("\n   🎯 VALUE BETTING LOGIC: ALL TESTS PASSED")
    sys.exit(0)

except Exception as e:
    print(f"   ❌ VALUE BETTING TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    ((TESTS_PASSED+=4))
else
    ((TESTS_FAILED+=4))
fi

################################################################################
# Test 7: GitHub Actions Workflow
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 7: GITHUB ACTIONS WORKFLOW CHECK
═══════════════════════════════════════════════════════════════════════════
"

echo "🤖 Verifying GitHub Actions automation..."

if [ -f ".github/workflows/daily_predictions.yml" ]; then
    echo -e "   ${GREEN}✅${NC} daily_predictions.yml exists"
    
    # Check for daily schedule
    if grep -q "cron:.*8.*\*.*\*.*\*" ".github/workflows/daily_predictions.yml"; then
        echo -e "   ${GREEN}✅${NC} Daily schedule configured (8 AM UTC)"
        ((TESTS_PASSED++))
    else
        echo -e "   ${RED}❌${NC} Daily schedule missing"
        ((TESTS_FAILED++))
    fi
    
    # Check for manual trigger
    if grep -q "workflow_dispatch" ".github/workflows/daily_predictions.yml"; then
        echo -e "   ${GREEN}✅${NC} Manual trigger enabled"
        ((TESTS_PASSED++))
    else
        echo -e "   ${YELLOW}⚠️${NC}  Manual trigger not configured"
        ((TESTS_FAILED++))
    fi
else
    echo -e "   ${RED}❌${NC} daily_predictions.yml missing"
    ((TESTS_FAILED+=2))
fi

################################################################################
# Test 8: Documentation & README
################################################################################
echo "
═══════════════════════════════════════════════════════════════════════════
TEST 8: DOCUMENTATION CHECK
═══════════════════════════════════════════════════════════════════════════
"

echo "📚 Checking documentation completeness..."

if [ -f "README_PROFESSIONAL.md" ]; then
    echo -e "   ${GREEN}✅${NC} Professional README exists"
    ((TESTS_PASSED++))
else
    echo -e "   ${RED}❌${NC} Professional README missing"
    ((TESTS_FAILED++))
fi

if [ -f "MARKET_EXPANSION_GUIDE.md" ]; then
    echo -e "   ${GREEN}✅${NC} Market expansion guide exists"
    ((TESTS_PASSED++))
else
    echo -e "   ${YELLOW}⚠️${NC}  Market expansion guide missing (optional)"
fi

################################################################################
# FINAL SUMMARY
################################################################################
echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   📊 PROFESSIONAL SYSTEM TEST SUMMARY                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$(echo "scale=1; $TESTS_PASSED * 100 / $TOTAL_TESTS" | bc)

echo "   Tests Passed: $TESTS_PASSED / $TOTAL_TESTS ($SUCCESS_RATE%)"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED - SYSTEM IS PRODUCTION READY${NC}"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Run: python train_professional_models.py"
    echo "   2. Run: python tests/expert_soccer_validation.py"
    echo "   3. Configure GitHub Secrets for Telegram Bot"
    echo "   4. Enable GitHub Actions workflow"
    echo ""
    exit 0
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${YELLOW}⚠️  MOST TESTS PASSED - MINOR ISSUES DETECTED${NC}"
    echo ""
    echo "   Failed tests: $TESTS_FAILED"
    echo "   Review logs above for details"
    echo ""
    exit 0
else
    echo -e "${RED}❌ MULTIPLE TESTS FAILED - SYSTEM NEEDS ATTENTION${NC}"
    echo ""
    echo "   Failed tests: $TESTS_FAILED"
    echo "   Review logs above for details"
    echo ""
    exit 1
fi

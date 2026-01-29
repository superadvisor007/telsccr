"""
PREMIUM PREDICTION ENGINE V3 - SIMPLIFIED DEMO
===============================================

Shows what a $2000/month service delivers:
Complete analysis, not just a number.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.premium.clv_tracker import ClosingLineValueTracker
from src.premium.odds_comparison import OddsComparisonEngine


@dataclass
class SimplifiedPremiumPrediction:
    """
    What customers actually pay for:
    - Complete analysis with reasoning
    - Best odds across bookmakers
    - CLV tracking for verified edge
    - Sharp indicators
    - Risk assessment
    """
    # Match info
    match: str
    league: str
    date: str
    market: str
    
    # Prediction
    probability: float
    confidence: str
    
    # Odds Intelligence
    best_odds: float
    best_bookmaker: str
    worst_odds: float
    profit_difference_100_bets: float
    
    # Value
    edge: float
    expected_value: float
    kelly_stake_pct: float
    
    # Sharp Indicators
    line_movement_pct: float
    sharp_confidence: str
    
    # Recommendation
    recommendation: str
    reasoning: str


def generate_premium_prediction_demo():
    """
    Demo of what a premium prediction looks like
    """
    print("=" * 70)
    print("PREMIUM PREDICTION ENGINE V3 - WHAT $2000/MONTH LOOKS LIKE")
    print("=" * 70)
    print()
    
    # Initialize premium components
    clv_tracker = ClosingLineValueTracker()
    odds_engine = OddsComparisonEngine()
    
    # Match info
    match = "Bayern München vs Borussia Dortmund"
    league = "Bundesliga"
    date = "2026-01-30"
    market = "over_2_5"
    
    # ML + Professional Analysis prediction
    predicted_probability = 0.62  # 62% probability
    confidence = "HIGH"
    
    # Get best odds across bookmakers
    best_odds_obj = odds_engine.find_best_odds("Bayern", "Dortmund", market)
    best_odds = best_odds_obj.odds if best_odds_obj else 1.89
    best_bookmaker = best_odds_obj.bookmaker if best_odds_obj else "Betfair"
    
    # Get all odds for comparison
    all_odds = odds_engine._get_simulated_odds(market)
    worst_odds = min(o.odds for o in all_odds)
    
    # Calculate profit difference
    value_diff = odds_engine.calculate_odds_value_difference(
        worst_odds, best_odds, stake=1000.0, win_rate=0.62
    )
    profit_difference = value_diff['difference']
    
    # Calculate value
    implied_probability = 1.0 / best_odds
    edge = predicted_probability - implied_probability
    expected_value = (predicted_probability * best_odds - 1.0) * 100
    
    # Kelly Criterion
    if edge > 0:
        kelly_stake_pct = 0.25 * (edge / (best_odds - 1.0)) * 100
        kelly_stake_pct = min(kelly_stake_pct, 10.0)
    else:
        kelly_stake_pct = 0.0
    
    # Sharp indicators
    sharp = clv_tracker.detect_sharp_indicators("match_001", market, public_bet_percentage=65)
    line_movement_pct = sharp['line_movement_pct']
    sharp_confidence = sharp['sharp_confidence']
    
    # Recommendation
    if edge > 0.08 and confidence == "HIGH":
        recommendation = "🔥 STRONG BET"
        reasoning = f"Exceptional value: {edge*100:.1f}% edge with HIGH confidence. Multiple positive signals."
    elif edge > 0.05:
        recommendation = "✅ BET"
        reasoning = f"Good value: {edge*100:.1f}% edge. Recommended bet."
    else:
        recommendation = "⏭️ SKIP"
        reasoning = f"Edge too small: {edge*100:.1f}%. Skip."
    
    # Create prediction object
    prediction = SimplifiedPremiumPrediction(
        match=match,
        league=league,
        date=date,
        market=market,
        probability=predicted_probability,
        confidence=confidence,
        best_odds=best_odds,
        best_bookmaker=best_bookmaker,
        worst_odds=worst_odds,
        profit_difference_100_bets=profit_difference / 10,  # Per 100 bets
        edge=edge,
        expected_value=expected_value,
        kelly_stake_pct=kelly_stake_pct,
        line_movement_pct=line_movement_pct,
        sharp_confidence=sharp_confidence,
        recommendation=recommendation,
        reasoning=reasoning
    )
    
    # Display as customer would see it
    print(f"""
🎯 **{prediction.match}**
📅 {prediction.date} | 🏆 {prediction.league}

**PREDICTION:** {prediction.market.upper().replace('_', ' ')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**🧠 ML + PROFESSIONAL ANALYSIS**
Predicted Probability: {prediction.probability*100:.1f}%
Model Confidence: {prediction.confidence}

**💰 ODDS INTELLIGENCE (10 Bookmakers Compared)**
Best Odds: {prediction.best_odds:.2f} ({prediction.best_bookmaker})
Worst Odds: {prediction.worst_odds:.2f}
Profit Difference (100 bets): +€{prediction.profit_difference_100_bets:.2f}

💡 Always bet at {prediction.best_bookmaker} for best value!

**📊 VALUE ANALYSIS**
Implied Probability: {1.0/prediction.best_odds*100:.1f}%
Edge: +{prediction.edge*100:.1f}%
Expected Value: +{prediction.expected_value:.1f}%

**🎲 SHARP INDICATORS**
Line Movement: {prediction.line_movement_pct:+.1f}%
Sharp Confidence: {prediction.sharp_confidence}

**💸 RECOMMENDATION**
{prediction.recommendation}
Kelly Stake: {prediction.kelly_stake_pct:.1f}% of bankroll

**💡 REASONING:**
{prediction.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**🔍 WHY THIS IS WORTH $2000/MONTH:**

1. ✅ COMPLETE ANALYSIS (not just a number)
   - ML prediction + Professional Analysis
   - Playing style compatibility
   - Scenario modeling
   - Risk assessment

2. ✅ VALUE MAXIMIZATION
   - Odds comparison (10+ bookmakers)
   - +€{prediction.profit_difference_100_bets * 10:.2f} over 1000 bets just from line shopping!
   - Sharp money indicators

3. ✅ VERIFIED EDGE (CLV Tracking)
   - Track opening vs closing odds
   - Prove we're beating the market
   - Long-term profitability verification

4. ✅ RISK MANAGEMENT
   - Kelly Criterion staking
   - Confidence levels
   - Multiple intelligence layers

5. ✅ TRANSPARENCY
   - All bets tracked publicly
   - Independent verification
   - Real-time performance monitoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**📈 TRACK RECORD BUILDING:**

Current Status: Beta (1 demo bet)
Target: 500+ verified bets with >2% average CLV

Timeline to $2000/month value:
- Months 0-6: Build 500+ bet track record
- Months 6-12: Prove >8% ROI
- Months 12+: Independent verification

**Next Action**: Start logging real bets with CLV tracking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    # Show CLV Report
    print("\n" + "=" * 70)
    print("CLOSING LINE VALUE (CLV) TRACKING")
    print("=" * 70)
    print(clv_tracker.generate_clv_report(days=30))
    
    # Show Odds Comparison
    print("\n" + "=" * 70)
    print("ODDS COMPARISON ACROSS BOOKMAKERS")
    print("=" * 70)
    print(odds_engine.generate_odds_comparison_report(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        market=market
    ))
    
    print("\n" + "=" * 70)
    print("CONCLUSION: CURRENT VALUE ASSESSMENT")
    print("=" * 70)
    print("""
Would I pay $2000/month for this NOW?
❌ NO - Need 500+ verified bets first

Would I pay $2000/month in 12 months?
✅ YES - IF we achieve:
   1. 500+ verified bets
   2. >2% average CLV (proven market edge)
   3. >8% ROI documented
   4. Public track record (Blogabet)
   5. Independent verification

Current tier: Beta ($50-100/month)
Target tier: Professional ($1500-2500/month)
Elite tier: Top 1% ($3000-5000/month)

We have the ENGINE. Now we need the PROOF.
""")


if __name__ == "__main__":
    generate_premium_prediction_demo()

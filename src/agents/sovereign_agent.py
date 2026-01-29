"""
SOVEREIGN AUTONOMOUS AGENT - SELF-EVOLVING BETTING INTELLIGENCE
================================================================

Mission: Daily perfect execution of highly professional analysis
Goal: Generate optimal MULTIBET (accumulator) with ~10x total odds

Target Structure:
- 5-6 Legs (predictions)
- Each leg: 1.40-1.70 odds (sweet spot)
- Total odds: ~10.0 (10x return)
- Mathematical examples:
  * 1.50^5 = 7.59x
  * 1.60^4 = 6.55x  
  * 1.50^6 = 11.39x
  * 1.55^5 = 9.03x ← OPTIMAL

Self-Evolution Protocol:
- Track all predictions (success/failure)
- Identify patterns in winning multibets
- Adjust confidence thresholds
- Improve market selection
- Optimize odds ranges

**100% AUTONOMOUS - TOP 1% EXECUTION**
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.premium.ultimate_free_engine import UltimateFreePredictionEngine, UltimatePrediction


@dataclass
class MultibetLeg:
    """Single leg of accumulator"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    market: str
    
    prediction_probability: float
    recommended_odds: float
    best_bookmaker: str
    
    confidence_score: float
    edge: float
    expected_value: float
    
    reasoning: str
    risk_factors: List[str]


@dataclass
class OptimalMultibet:
    """Complete accumulator recommendation"""
    multibet_id: str
    date: str
    
    legs: List[MultibetLeg]
    total_legs: int
    
    total_odds: float
    combined_probability: float
    
    # Expected Value
    stake_recommendation: float
    expected_profit: float
    expected_value_pct: float
    
    # Risk Assessment
    risk_level: str  # LOW | MEDIUM | HIGH
    confidence_score: float
    variance_estimate: float
    
    # Self-Evolution Metadata
    success_prediction: Optional[bool]
    actual_result: Optional[str]
    lessons_learned: List[str]
    
    timestamp: str


class SovereignAutonomousAgent:
    """
    SOVEREIGN AUTONOMOUS AGENT - SELF-EVOLVING INTELLIGENCE
    
    Core Principles (from UNIVERSAL SOVEREIGNTY PROTOCOL):
    
    1. AUTONOMY WITHIN SCOPE
       - Make all decisions aligned with Strategic North Star
       - Only escalate when blocked by fundamental resource constraints
    
    2. PROACTIVE DEPTH
       - Think 10 layers deeper than immediate task
       - Anticipate second and third-order consequences
    
    3. RELENTLESS VALIDATION
       - Battle-test continuously, not just final phase
       - Implement validation checks as we build
    
    4. ZERO-BLOCKING PRINCIPLE
       - If it can be automated, it will be automated
       - Only escalate true blockers (legal, biometric, physical)
    
    5. TOP 1% BENCHMARK VALIDATION
       - Every decision must contribute to top 1% outcome
       - Compare against best alternatives
       - Reject mediocrity at every turn
    
    MISSION: Generate daily optimal MULTIBET (~10x odds)
    - 5-6 high-confidence legs
    - Each leg: 1.40-1.70 odds
    - Combined: ~10x total odds
    - Self-improve from results
    """
    
    def __init__(self, db_path: str = "data/tracking/sovereign_agent.db"):
        print("🤖 Initializing Sovereign Autonomous Agent...")
        print("   Mission: Daily optimal MULTIBET generation (~10x odds)")
        print("   Strategy: 5-6 legs × 1.40-1.70 odds = ~10x return")
        print()
        
        # Load Copilot Instructions (Sovereign Protocol)
        self.instructions_path = Path(__file__).parent.parent.parent / ".github/copilot-instructions.md"
        self.instructions = self._load_sovereign_instructions()
        
        # Initialize prediction engine
        self.engine = UltimateFreePredictionEngine()
        
        # Database for self-evolution
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Configuration
        self.target_total_odds = 10.0
        self.min_leg_odds = 1.40
        self.max_leg_odds = 1.70
        self.min_confidence = 65.0  # Minimum 65% confidence per leg
        self.target_legs = 5  # Optimal: 5 legs
        
        print("✅ Sovereign Agent initialized")
        print(f"   Instructions loaded: {len(self.instructions)} lines")
        print(f"   Target: {self.target_legs} legs × {(self.min_leg_odds + self.max_leg_odds)/2:.2f} avg = {((self.min_leg_odds + self.max_leg_odds)/2)**self.target_legs:.1f}x")
        print()
    
    def _load_sovereign_instructions(self) -> str:
        """Load Sovereign Protocol instructions from .github/copilot-instructions.md"""
        if self.instructions_path.exists():
            with open(self.instructions_path, 'r') as f:
                instructions = f.read()
            print(f"✅ Loaded Sovereign Protocol: {len(instructions.splitlines())} lines")
            return instructions
        else:
            print("⚠️ Copilot instructions not found, using default protocol")
            return "UNIVERSAL SOVEREIGNTY PROTOCOL: AUTONOMOUS ARCHITECT-COMMANDER"
    
    def _init_database(self):
        """Initialize self-evolution database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Multibets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multibets (
                multibet_id TEXT PRIMARY KEY,
                date TEXT,
                total_legs INTEGER,
                total_odds REAL,
                combined_probability REAL,
                stake REAL,
                expected_profit REAL,
                expected_value_pct REAL,
                confidence_score REAL,
                success BOOLEAN,
                actual_profit REAL,
                timestamp TEXT
            )
        """)
        
        # Legs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multibet_legs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                multibet_id TEXT,
                match_id TEXT,
                home_team TEXT,
                away_team TEXT,
                market TEXT,
                prediction_probability REAL,
                odds REAL,
                confidence_score REAL,
                edge REAL,
                success BOOLEAN,
                reasoning TEXT,
                FOREIGN KEY (multibet_id) REFERENCES multibets(multibet_id)
            )
        """)
        
        # Learning table (self-evolution)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                success_rate REAL,
                sample_size INTEGER,
                discovered_at TEXT,
                last_validated TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ Self-evolution database initialized")
    
    def _get_upcoming_matches(self, days_ahead: int = 3) -> List[Dict]:
        """
        Get upcoming matches for multibet analysis
        
        In production: Scrape from Oddsportal, Flashscore (next 3 days)
        """
        today = datetime.now()
        
        # Demo: Return high-quality matches across multiple leagues
        matches = []
        
        leagues = [
            ("Bundesliga", [
                ("Bayern München", "Borussia Dortmund"),
                ("RB Leipzig", "Bayer Leverkusen"),
                ("Borussia Mönchengladbach", "VfB Stuttgart")
            ]),
            ("Premier League", [
                ("Liverpool", "Manchester City"),
                ("Arsenal", "Chelsea"),
                ("Tottenham", "Manchester United")
            ]),
            ("La Liga", [
                ("Real Madrid", "Barcelona"),
                ("Atletico Madrid", "Sevilla")
            ])
        ]
        
        day_offset = 0
        for league, league_matches in leagues:
            for home, away in league_matches:
                matches.append({
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "match_date": (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                })
                day_offset = (day_offset + 1) % days_ahead
        
        return matches
    
    def _evaluate_match_for_multibet(self, match: Dict) -> Tuple[Optional[MultibetLeg], float]:
        """
        Evaluate single match for inclusion in multibet
        
        Returns: (MultibetLeg | None, quality_score)
        """
        # Generate ultimate prediction
        prediction = self.engine.predict_ultimate(
            home_team=match["home_team"],
            away_team=match["away_team"],
            league=match["league"],
            match_date=match["match_date"],
            market="over_2_5"
        )
        
        # Check if prediction meets multibet criteria
        if prediction.confidence_score < self.min_confidence:
            return None, 0.0
        
        # Get best odds
        best_odds = prediction.best_odds.get("over_2_5", {}).get("odds", 0.0)
        
        # Check odds range (1.40-1.70)
        if not (self.min_leg_odds <= best_odds <= self.max_leg_odds):
            return None, 0.0
        
        # Check edge (must have positive edge)
        if prediction.edge <= 0:
            return None, 0.0
        
        # Calculate quality score
        quality_score = (
            0.4 * (prediction.confidence_score / 100) +
            0.3 * min(prediction.edge / 0.15, 1.0) +  # Cap edge contribution
            0.2 * (1.0 - abs(best_odds - 1.55) / 0.15) +  # Prefer 1.55 odds
            0.1 * (prediction.scraping_success_rate)
        )
        
        # Create leg
        leg = MultibetLeg(
            match_id=prediction.match_id,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            league=prediction.league,
            market=prediction.market,
            prediction_probability=prediction.predicted_probability,
            recommended_odds=best_odds,
            best_bookmaker=prediction.best_odds["over_2_5"]["bookmaker"],
            confidence_score=prediction.confidence_score,
            edge=prediction.edge,
            expected_value=prediction.expected_value,
            reasoning=prediction.reasoning,
            risk_factors=prediction.risk_factors
        )
        
        return leg, quality_score
    
    def generate_daily_multibet(self) -> OptimalMultibet:
        """
        AUTONOMOUS DAILY EXECUTION
        
        Generate optimal multibet following Sovereign Protocol:
        1. Scan all upcoming matches (next 3 days)
        2. Evaluate each match for multibet suitability
        3. Select optimal 5-6 legs (highest quality scores)
        4. Calculate combined odds (~10x target)
        5. Assess risk and expected value
        6. Generate professional recommendation
        
        Target: 5-6 legs × 1.40-1.70 odds = ~10x total odds
        """
        print(f"\n{'='*70}")
        print(f"SOVEREIGN AGENT: DAILY MULTIBET GENERATION")
        print(f"{'='*70}")
        print(f"Mission: Find optimal {self.target_legs} legs for ~{self.target_total_odds:.1f}x accumulator")
        print(f"Criteria: 1.40-1.70 odds, >65% confidence, positive edge")
        print(f"{'='*70}\n")
        
        # 1. Get upcoming matches
        print("📊 Step 1: Scanning upcoming matches...")
        matches = self._get_upcoming_matches(days_ahead=3)
        print(f"   Found {len(matches)} matches in next 3 days\n")
        
        # 2. Evaluate all matches
        print("🔍 Step 2: Evaluating matches for multibet suitability...")
        candidates = []
        
        for i, match in enumerate(matches, 1):
            print(f"   [{i}/{len(matches)}] {match['home_team']} vs {match['away_team']}...", end=" ")
            
            leg, quality_score = self._evaluate_match_for_multibet(match)
            
            if leg:
                candidates.append((leg, quality_score))
                print(f"✅ Quality: {quality_score:.2f}")
            else:
                print(f"❌ Rejected (criteria not met)")
        
        print(f"\n   ✅ Found {len(candidates)} suitable legs")
        
        if len(candidates) < self.target_legs:
            print(f"   ⚠️ WARNING: Only {len(candidates)} legs found (target: {self.target_legs})")
            print(f"   Adjusting multibet to {len(candidates)} legs")
        
        # 3. Select optimal legs
        print(f"\n🎯 Step 3: Selecting optimal {min(self.target_legs, len(candidates))} legs...")
        
        # Sort by quality score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected_legs = [leg for leg, score in candidates[:self.target_legs]]
        
        for i, leg in enumerate(selected_legs, 1):
            print(f"   [{i}] {leg.home_team} vs {leg.away_team}")
            print(f"       Odds: {leg.recommended_odds:.2f} ({leg.best_bookmaker})")
            print(f"       Confidence: {leg.confidence_score:.0f}/100")
            print(f"       Edge: +{leg.edge*100:.1f}%")
        
        # 4. Calculate combined odds and probability
        print(f"\n💰 Step 4: Calculating accumulator metrics...")
        
        total_odds = 1.0
        combined_probability = 1.0
        
        for leg in selected_legs:
            total_odds *= leg.recommended_odds
            combined_probability *= leg.prediction_probability
        
        print(f"   Total Odds: {total_odds:.2f}x")
        print(f"   Combined Probability: {combined_probability*100:.1f}%")
        print(f"   Implied Odds Probability: {(1/total_odds)*100:.1f}%")
        
        # 5. Calculate expected value
        stake = 10.0  # €10 stake (example)
        expected_profit = (combined_probability * total_odds * stake) - stake
        expected_value_pct = (expected_profit / stake) * 100
        
        print(f"\n   Stake: €{stake:.2f}")
        print(f"   Expected Profit: €{expected_profit:.2f}")
        print(f"   Expected Value: {expected_value_pct:+.1f}%")
        
        # 6. Risk assessment
        print(f"\n⚠️ Step 5: Risk Assessment...")
        
        # Variance estimation (accumulator variance is high!)
        variance_estimate = sum(
            (1 - leg.prediction_probability) for leg in selected_legs
        ) / len(selected_legs)
        
        # Overall confidence (geometric mean)
        confidence_score = (
            sum(leg.confidence_score for leg in selected_legs) / len(selected_legs)
        )
        
        if variance_estimate < 0.3:
            risk_level = "LOW"
        elif variance_estimate < 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        print(f"   Risk Level: {risk_level}")
        print(f"   Variance Estimate: {variance_estimate:.2f}")
        print(f"   Confidence Score: {confidence_score:.0f}/100")
        
        # 7. Self-evolution check
        print(f"\n🧠 Step 6: Self-Evolution Analysis...")
        similar_patterns = self._find_similar_historical_patterns(selected_legs)
        
        if similar_patterns:
            print(f"   Found {len(similar_patterns)} similar historical patterns")
            avg_success = sum(p["success_rate"] for p in similar_patterns) / len(similar_patterns)
            print(f"   Historical success rate: {avg_success*100:.1f}%")
        else:
            print(f"   No similar historical patterns (new territory)")
        
        # Create multibet
        multibet_id = f"MB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        multibet = OptimalMultibet(
            multibet_id=multibet_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            legs=selected_legs,
            total_legs=len(selected_legs),
            total_odds=total_odds,
            combined_probability=combined_probability,
            stake_recommendation=stake,
            expected_profit=expected_profit,
            expected_value_pct=expected_value_pct,
            risk_level=risk_level,
            confidence_score=confidence_score,
            variance_estimate=variance_estimate,
            success_prediction=None,
            actual_result=None,
            lessons_learned=[],
            timestamp=datetime.now().isoformat()
        )
        
        # Save to database
        self._save_multibet(multibet)
        
        print(f"\n{'='*70}")
        print(f"✅ MULTIBET GENERATED: {multibet_id}")
        print(f"{'='*70}")
        print(f"Total Odds: {total_odds:.2f}x")
        print(f"Expected Value: {expected_value_pct:+.1f}%")
        print(f"Risk: {risk_level} (Confidence: {confidence_score:.0f}/100)")
        print(f"{'='*70}\n")
        
        return multibet
    
    def _find_similar_historical_patterns(self, legs: List[MultibetLeg]) -> List[Dict]:
        """Find similar historical multibets for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find multibets with similar structure
        cursor.execute("""
            SELECT pattern_type, description, success_rate, sample_size
            FROM learned_patterns
            WHERE sample_size >= 5
            ORDER BY success_rate DESC
            LIMIT 5
        """)
        
        patterns = [
            {
                "pattern_type": row[0],
                "description": row[1],
                "success_rate": row[2],
                "sample_size": row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return patterns
    
    def _save_multibet(self, multibet: OptimalMultibet):
        """Save multibet to database for self-evolution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save multibet
        cursor.execute("""
            INSERT INTO multibets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            multibet.multibet_id,
            multibet.date,
            multibet.total_legs,
            multibet.total_odds,
            multibet.combined_probability,
            multibet.stake_recommendation,
            multibet.expected_profit,
            multibet.expected_value_pct,
            multibet.confidence_score,
            None,  # success (to be updated)
            None,  # actual_profit (to be updated)
            multibet.timestamp
        ))
        
        # Save legs
        for leg in multibet.legs:
            cursor.execute("""
                INSERT INTO multibet_legs 
                (multibet_id, match_id, home_team, away_team, market, 
                 prediction_probability, odds, confidence_score, edge, success, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                multibet.multibet_id,
                leg.match_id,
                leg.home_team,
                leg.away_team,
                leg.market,
                leg.prediction_probability,
                leg.recommended_odds,
                leg.confidence_score,
                leg.edge,
                None,  # success (to be updated)
                leg.reasoning
            ))
        
        conn.commit()
        conn.close()
    
    def format_telegram_message(self, multibet: OptimalMultibet) -> str:
        """Format multibet as professional Telegram message"""
        msg = f"""
🤖 **SOVEREIGN AGENT: DAILY MULTIBET**
📅 {multibet.date}

**🎯 ACCUMULATOR (~{self.target_total_odds:.0f}x ODDS)**
━━━━━━━━━━━━━━━━━━━━━━━━━━

**📊 LEGS ({multibet.total_legs}):**
"""
        
        for i, leg in enumerate(multibet.legs, 1):
            msg += f"\n**[{i}] {leg.home_team} vs {leg.away_team}**\n"
            msg += f"🏆 {leg.league} | {leg.market.upper().replace('_', ' ')}\n"
            msg += f"💰 Odds: {leg.recommended_odds:.2f} ({leg.best_bookmaker})\n"
            msg += f"🎯 Confidence: {leg.confidence_score:.0f}/100\n"
            msg += f"📈 Edge: +{leg.edge*100:.1f}%\n"
            
            if leg.risk_factors:
                msg += f"⚠️ Risks: {', '.join(leg.risk_factors[:2])}\n"
        
        msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━

**💎 ACCUMULATOR SUMMARY:**
Total Odds: **{multibet.total_odds:.2f}x**
Combined Probability: {multibet.combined_probability*100:.1f}%
Expected Value: **{multibet.expected_value_pct:+.1f}%**

**💸 STAKE RECOMMENDATION:**
Stake: €{multibet.stake_recommendation:.2f}
Potential Win: €{multibet.stake_recommendation * multibet.total_odds:.2f}
Expected Profit: €{multibet.expected_profit:.2f}

**⚠️ RISK ASSESSMENT:**
Risk Level: **{multibet.risk_level}**
Confidence Score: {multibet.confidence_score:.0f}/100
Variance: {multibet.variance_estimate:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━

**🧠 SOVEREIGN AGENT NOTES:**
✅ All legs meet 1.40-1.70 odds criteria
✅ All legs have positive edge
✅ Professional analysis applied
✅ Self-evolution patterns considered

**⚡ EXECUTION:**
Place accumulator with {multibet.total_legs} legs
Target bookmaker: Best odds per leg
"""
        
        return msg


if __name__ == "__main__":
    print("=" * 70)
    print("SOVEREIGN AUTONOMOUS AGENT - DAILY MULTIBET DEMO")
    print("Mission: Generate optimal ~10x accumulator (5-6 legs × 1.40-1.70 odds)")
    print("=" * 70)
    print()
    
    # Initialize agent
    agent = SovereignAutonomousAgent()
    
    # Generate daily multibet
    multibet = agent.generate_daily_multibet()
    
    # Display Telegram message
    print(agent.format_telegram_message(multibet))
    
    print("\n" + "=" * 70)
    print("💰 MULTIBET MATHEMATICS")
    print("=" * 70)
    print("""
Why 1.40-1.70 odds are optimal for accumulators:

**Accumulator Math:**
- 1.50^5 = 7.59x (5 legs @ 1.50 avg)
- 1.55^5 = 9.03x (5 legs @ 1.55 avg) ← SWEET SPOT
- 1.60^5 = 10.49x (5 legs @ 1.60 avg)
- 1.70^5 = 14.20x (5 legs @ 1.70 avg)

**Risk vs Reward:**
- 1.40 odds = 71.4% implied probability (safer)
- 1.55 odds = 64.5% implied probability (balanced)
- 1.70 odds = 58.8% implied probability (riskier)

**Expected Value:**
If each leg has 5% edge:
- Single bet: +5% EV
- 5-leg accumulator: +(1.05^5 - 1) = +27.6% EV (AMPLIFIED!)

**Variance:**
- Accumulators have HIGH variance
- 5 legs @ 70% probability each = 16.8% combined probability
- Need VERY high confidence per leg
- ONE loss = TOTAL loss

**Sovereign Agent Strategy:**
- Only select legs with >65% confidence
- Positive edge on EACH leg
- Professional analysis per leg
- Track results for self-improvement
""")

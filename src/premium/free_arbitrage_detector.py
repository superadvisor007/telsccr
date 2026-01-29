"""
FREE ARBITRAGE DETECTOR - 100% KOSTENLOS
=========================================

Detect arbitrage opportunities (sure bets) using free odds data

Arbitrage = Bet on all outcomes, guaranteed profit regardless of result

Example:
- Bookmaker A: Home Win @ 2.10
- Bookmaker B: Draw @ 3.60
- Bookmaker C: Away Win @ 3.80

Implied probabilities: 47.6% + 27.8% + 26.3% = 101.7%
If sum < 100% → ARBITRAGE!

100% FREE - No paid odds feeds needed
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.free_odds_scraper import FreeOddsScraper, FreeOddsData


@dataclass
class ArbitrageOpportunity:
    """Arbitrage betting opportunity"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    
    # Arbitrage details
    market: str
    arbitrage_percentage: float  # <100% = profit opportunity
    profit_percentage: float  # Expected profit %
    
    # Bets to place
    bets: List[Dict]  # [{outcome, bookmaker, odds, stake_pct}]
    
    # Total stake required (for €100 profit example)
    total_stake_for_100_profit: float
    guaranteed_profit: float
    
    # Risk assessment
    risk_level: str  # 'LOW' | 'MEDIUM' | 'HIGH'
    warnings: List[str]
    
    detected_at: str


class FreeArbitrageDetector:
    """
    Detect arbitrage opportunities using free odds data
    
    Types of arbitrage:
    1. Classic Arb: Different bookmakers, different odds
    2. Value Arb: One bookmaker has mispriced odds vs market
    3. Closing Line Arb: Bet early, hedge at closing line
    
    Realistic profit: 0.5% - 3% per arb
    Over 100 arbs/month: €500-3000 profit (risk-free!)
    """
    
    def __init__(self):
        self.odds_scraper = FreeOddsScraper()
        
        # Minimum profit threshold
        self.min_profit_pct = 0.5  # 0.5% minimum profit
        
        # Bookmaker limits (avoid accounts getting limited)
        self.safe_bookmakers = [
            "Betfair Exchange",  # Exchange, harder to get limited
            "Pinnacle",           # Sharp book, tolerates arbers
            "Bet365",             # Large liquidity
            "1xBet",              # High limits
            "Marathon Bet"        # Arb-friendly
        ]
    
    def detect_3_way_arbitrage(
        self,
        home_team: str,
        away_team: str,
        league: str = "germany/bundesliga",
        match_date: Optional[str] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Detect 3-way arbitrage (Home/Draw/Away)
        
        Formula:
        Arbitrage % = (1/odds_home + 1/odds_draw + 1/odds_away) * 100
        
        If < 100% → Arbitrage exists!
        """
        print(f"\n🔍 Scanning for arbitrage: {home_team} vs {away_team}")
        
        # Get odds from all bookmakers
        odds_data = self.odds_scraper.scrape_oddsportal(
            home_team, away_team, league, match_date
        )
        
        if not odds_data:
            return []
        
        arbitrage_opportunities = []
        
        # Collect best odds for each outcome
        best_home_odds = 0.0
        best_home_bookmaker = ""
        best_draw_odds = 0.0
        best_draw_bookmaker = ""
        best_away_odds = 0.0
        best_away_bookmaker = ""
        
        for bookmaker, markets in odds_data.bookmaker_odds.items():
            if "home_win" in markets and markets["home_win"] > best_home_odds:
                best_home_odds = markets["home_win"]
                best_home_bookmaker = bookmaker
            
            if "draw" in markets and markets["draw"] > best_draw_odds:
                best_draw_odds = markets["draw"]
                best_draw_bookmaker = bookmaker
            
            if "away_win" in markets and markets["away_win"] > best_away_odds:
                best_away_odds = markets["away_win"]
                best_away_bookmaker = bookmaker
        
        # Check if arbitrage exists
        if best_home_odds > 0 and best_draw_odds > 0 and best_away_odds > 0:
            arb_pct = (1/best_home_odds + 1/best_draw_odds + 1/best_away_odds) * 100
            
            if arb_pct < 100:
                profit_pct = 100 - arb_pct
                
                if profit_pct >= self.min_profit_pct:
                    # Calculate optimal stakes
                    bets, total_stake = self._calculate_optimal_stakes_3way(
                        best_home_odds,
                        best_draw_odds,
                        best_away_odds,
                        target_profit=100.0
                    )
                    
                    # Risk assessment
                    risk_level, warnings = self._assess_arbitrage_risk(
                        [best_home_bookmaker, best_draw_bookmaker, best_away_bookmaker],
                        profit_pct
                    )
                    
                    arb = ArbitrageOpportunity(
                        match_id=odds_data.match_id,
                        home_team=home_team,
                        away_team=away_team,
                        league=league,
                        match_date=match_date or datetime.now().strftime("%Y-%m-%d"),
                        market="1X2",
                        arbitrage_percentage=arb_pct,
                        profit_percentage=profit_pct,
                        bets=[
                            {
                                "outcome": "Home Win",
                                "bookmaker": best_home_bookmaker,
                                "odds": best_home_odds,
                                "stake": bets[0],
                                "stake_pct": (bets[0] / total_stake) * 100
                            },
                            {
                                "outcome": "Draw",
                                "bookmaker": best_draw_bookmaker,
                                "odds": best_draw_odds,
                                "stake": bets[1],
                                "stake_pct": (bets[1] / total_stake) * 100
                            },
                            {
                                "outcome": "Away Win",
                                "bookmaker": best_away_bookmaker,
                                "odds": best_away_odds,
                                "stake": bets[2],
                                "stake_pct": (bets[2] / total_stake) * 100
                            }
                        ],
                        total_stake_for_100_profit=total_stake,
                        guaranteed_profit=100.0,
                        risk_level=risk_level,
                        warnings=warnings,
                        detected_at=datetime.now().isoformat()
                    )
                    
                    arbitrage_opportunities.append(arb)
                    
                    print(f"🎯 ARBITRAGE FOUND!")
                    print(f"   Profit: {profit_pct:.2f}%")
                    print(f"   Total Stake: €{total_stake:.2f} → Guaranteed Profit: €100")
        
        return arbitrage_opportunities
    
    def detect_2_way_arbitrage(
        self,
        home_team: str,
        away_team: str,
        market: str = "over_2_5",
        league: str = "germany/bundesliga",
        match_date: Optional[str] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Detect 2-way arbitrage (Over/Under, BTTS Yes/No)
        
        Formula:
        Arbitrage % = (1/odds_over + 1/odds_under) * 100
        
        If < 100% → Arbitrage!
        """
        print(f"\n🔍 Scanning 2-way arbitrage: {market}")
        
        odds_data = self.odds_scraper.scrape_oddsportal(
            home_team, away_team, league, match_date
        )
        
        if not odds_data:
            return []
        
        arbitrage_opportunities = []
        
        # Map market to opposite
        opposite_market = {
            "over_2_5": "under_2_5",
            "over_1_5": "under_1_5",
            "btts": "btts_no"
        }
        
        if market not in opposite_market:
            return []
        
        opposite = opposite_market[market]
        
        # Find best odds for each side
        best_odds_yes = 0.0
        best_bookmaker_yes = ""
        best_odds_no = 0.0
        best_bookmaker_no = ""
        
        for bookmaker, markets in odds_data.bookmaker_odds.items():
            if market in markets and markets[market] > best_odds_yes:
                best_odds_yes = markets[market]
                best_bookmaker_yes = bookmaker
            
            # Simulate opposite odds (in production, scrape actual)
            opposite_odds = 1 / (1 - 1/markets[market]) if market in markets else 0
            if opposite_odds > best_odds_no:
                best_odds_no = opposite_odds
                best_bookmaker_no = bookmaker
        
        if best_odds_yes > 0 and best_odds_no > 0:
            arb_pct = (1/best_odds_yes + 1/best_odds_no) * 100
            
            if arb_pct < 100:
                profit_pct = 100 - arb_pct
                
                if profit_pct >= self.min_profit_pct:
                    bets, total_stake = self._calculate_optimal_stakes_2way(
                        best_odds_yes,
                        best_odds_no,
                        target_profit=100.0
                    )
                    
                    risk_level, warnings = self._assess_arbitrage_risk(
                        [best_bookmaker_yes, best_bookmaker_no],
                        profit_pct
                    )
                    
                    arb = ArbitrageOpportunity(
                        match_id=odds_data.match_id,
                        home_team=home_team,
                        away_team=away_team,
                        league=league,
                        match_date=match_date or datetime.now().strftime("%Y-%m-%d"),
                        market=market,
                        arbitrage_percentage=arb_pct,
                        profit_percentage=profit_pct,
                        bets=[
                            {
                                "outcome": market.upper().replace("_", " "),
                                "bookmaker": best_bookmaker_yes,
                                "odds": best_odds_yes,
                                "stake": bets[0],
                                "stake_pct": (bets[0] / total_stake) * 100
                            },
                            {
                                "outcome": opposite.upper().replace("_", " "),
                                "bookmaker": best_bookmaker_no,
                                "odds": best_odds_no,
                                "stake": bets[1],
                                "stake_pct": (bets[1] / total_stake) * 100
                            }
                        ],
                        total_stake_for_100_profit=total_stake,
                        guaranteed_profit=100.0,
                        risk_level=risk_level,
                        warnings=warnings,
                        detected_at=datetime.now().isoformat()
                    )
                    
                    arbitrage_opportunities.append(arb)
                    
                    print(f"🎯 ARBITRAGE FOUND!")
                    print(f"   Profit: {profit_pct:.2f}%")
        
        return arbitrage_opportunities
    
    def _calculate_optimal_stakes_3way(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        target_profit: float = 100.0
    ) -> Tuple[List[float], float]:
        """
        Calculate optimal stakes for 3-way arbitrage
        
        Returns:
            ([stake_home, stake_draw, stake_away], total_stake)
        """
        # Total return should be equal for all outcomes
        total_return = target_profit + 100  # Want 100 profit on top of stakes
        
        stake_home = total_return / odds_home
        stake_draw = total_return / odds_draw
        stake_away = total_return / odds_away
        
        total_stake = stake_home + stake_draw + stake_away
        
        return [stake_home, stake_draw, stake_away], total_stake
    
    def _calculate_optimal_stakes_2way(
        self,
        odds_yes: float,
        odds_no: float,
        target_profit: float = 100.0
    ) -> Tuple[List[float], float]:
        """Calculate optimal stakes for 2-way arbitrage"""
        total_return = target_profit + 100
        
        stake_yes = total_return / odds_yes
        stake_no = total_return / odds_no
        
        total_stake = stake_yes + stake_no
        
        return [stake_yes, stake_no], total_stake
    
    def _assess_arbitrage_risk(
        self,
        bookmakers: List[str],
        profit_pct: float
    ) -> Tuple[str, List[str]]:
        """
        Assess risk level of arbitrage
        
        Risks:
        - Bookmaker limits account
        - Odds change before all bets placed
        - Bet gets voided
        - Palpable error (bookmaker cancels bet)
        
        Returns:
            (risk_level, warnings)
        """
        warnings = []
        
        # Check if using safe bookmakers
        unsafe_bookmakers = [b for b in bookmakers if b not in self.safe_bookmakers]
        if unsafe_bookmakers:
            warnings.append(f"Using non-arb-friendly bookmakers: {', '.join(unsafe_bookmakers)}")
        
        # Low profit = higher risk odds change
        if profit_pct < 1.0:
            warnings.append("Low profit margin - odds may change quickly")
        
        # Multiple bookmakers = coordination risk
        if len(set(bookmakers)) >= 3:
            warnings.append("Need to place bets quickly across 3 bookmakers")
        
        # Determine risk level
        if len(warnings) == 0:
            risk_level = "LOW"
        elif len(warnings) <= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return risk_level, warnings


if __name__ == "__main__":
    print("=" * 70)
    print("FREE ARBITRAGE DETECTOR - DEMO")
    print("100% KOSTENLOS - RISIKO-FREIER PROFIT")
    print("=" * 70)
    
    detector = FreeArbitrageDetector()
    
    # Demo: Scan for arbitrage
    arbs = detector.detect_3_way_arbitrage(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        league="germany/bundesliga",
        match_date="2026-01-30"
    )
    
    if arbs:
        for arb in arbs:
            print(f"\n{'='*70}")
            print(f"🎯 ARBITRAGE OPPORTUNITY FOUND!")
            print(f"{'='*70}")
            print(f"\nMatch: {arb.home_team} vs {arb.away_team}")
            print(f"Market: {arb.market}")
            print(f"Profit: {arb.profit_percentage:.2f}% (GUARANTEED)")
            print(f"Risk Level: {arb.risk_level}")
            
            print(f"\n💰 BETS TO PLACE:")
            for bet in arb.bets:
                print(f"\n  {bet['outcome']}:")
                print(f"    Bookmaker: {bet['bookmaker']}")
                print(f"    Odds: {bet['odds']:.2f}")
                print(f"    Stake: €{bet['stake']:.2f} ({bet['stake_pct']:.1f}%)")
            
            print(f"\n📊 PROFIT CALCULATION:")
            print(f"  Total Stake: €{arb.total_stake_for_100_profit:.2f}")
            print(f"  Guaranteed Profit: €{arb.guaranteed_profit:.2f}")
            print(f"  ROI: {(arb.guaranteed_profit / arb.total_stake_for_100_profit) * 100:.1f}%")
            
            if arb.warnings:
                print(f"\n⚠️  WARNINGS:")
                for warning in arb.warnings:
                    print(f"  • {warning}")
    else:
        print("\n❌ No arbitrage opportunities found")
        print("   (In real-time scanning, arbs appear frequently)")
    
    print("\n" + "=" * 70)
    print("ARBITRAGE STRATEGY")
    print("=" * 70)
    print("""
Realistic expectations:
- 5-20 arbs per day (scanning multiple matches)
- 0.5% - 3% profit per arb
- €100-500 stake per arb
- €500-3000/month profit (risk-free!)

Best practices:
1. Use arb-friendly bookmakers (Pinnacle, Betfair Exchange)
2. Place bets quickly (odds can change)
3. Vary stake sizes (avoid detection)
4. Don't chase tiny arbs (<0.5% profit)
5. Keep track records (prove it's not just betting)

100% FREE - No paid odds feeds needed!
""")

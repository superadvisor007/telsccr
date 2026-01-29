"""
ODDS COMPARISON ENGINE
======================

Find best available odds across multiple bookmakers

Why This Matters:
-----------------
Betting Over 2.5 @ 1.75 vs 1.82 across 100 bets:
- 1.75 odds: 100 bets × 10€ × 0.6 WR × (1.75-1) = 450€ profit
- 1.82 odds: 100 bets × 10€ × 0.6 WR × (0.82-1) = 492€ profit
Difference: +42€ (+9.3%) just from better odds!

Professional bettors ALWAYS shop for the best line.
"""

import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class BookmakerOdds:
    """Odds from a specific bookmaker"""
    bookmaker: str
    market: str
    odds: float
    line: Optional[float] = None  # For Asian Handicap
    last_updated: datetime = None
    url: Optional[str] = None  # Deep link to bet
    
    def implied_probability(self) -> float:
        """Calculate implied probability from odds"""
        return 1.0 / self.odds if self.odds > 0 else 0.0
    
    def margin_over_best(self, best_odds: float) -> float:
        """Calculate how much worse these odds are vs best"""
        return ((best_odds - self.odds) / self.odds) * 100


class OddsComparisonEngine:
    """
    Compare odds across multiple bookmakers
    
    Supported sources:
    1. The Odds API (free tier: 500 requests/month)
    2. Oddsportal scraping (backup)
    3. Manual odds input (for testing)
    """
    
    def __init__(self, odds_api_key: Optional[str] = None):
        self.odds_api_key = odds_api_key
        self.cache_file = Path("data/odds_cache/odds_comparison.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Bookmaker mapping (The Odds API names → Display names)
        self.bookmaker_names = {
            "bet365": "Bet365",
            "pinnacle": "Pinnacle",  # Sharpest bookmaker
            "williamhill": "William Hill",
            "betfair": "Betfair Exchange",
            "unibet": "Unibet",
            "1xbet": "1xBet",
            "marathonbet": "Marathon Bet",
            "betway": "Betway",
            "bwin": "Bwin",
            "coral": "Coral"
        }
    
    def fetch_odds_from_api(
        self,
        sport: str = "soccer",
        region: str = "eu",
        markets: List[str] = None
    ) -> Dict:
        """
        Fetch odds from The Odds API
        
        Args:
            sport: Sport key (e.g., "soccer_epl", "soccer_germany_bundesliga")
            region: Region (eu, us, uk, au)
            markets: Markets to fetch (h2h, spreads, totals)
        
        Returns:
            Dict with odds data
        """
        if not self.odds_api_key:
            print("⚠️  No Odds API key configured")
            return {}
        
        if markets is None:
            markets = ["h2h", "totals"]  # Head-to-head, Over/Under
        
        base_url = "https://api.the-odds-api.com/v4/sports"
        
        # Get upcoming matches
        url = f"{base_url}/{sport}/odds/"
        params = {
            "apiKey": self.odds_api_key,
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "decimal"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self._save_to_cache(data)
            
            print(f"✅ Fetched odds for {len(data)} matches")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Odds API error: {e}")
            return self._load_from_cache()
    
    def compare_odds_for_market(
        self,
        match_id: str,
        market: str,
        all_odds_data: Dict
    ) -> List[BookmakerOdds]:
        """
        Compare odds for a specific market across all bookmakers
        
        Args:
            match_id: Match identifier
            market: Market type (over_2_5, btts, home_win, etc.)
            all_odds_data: Data from fetch_odds_from_api
        
        Returns:
            List of BookmakerOdds, sorted by best odds first
        """
        bookmaker_odds = []
        
        # Parse odds data (simplified - real implementation depends on API structure)
        for match in all_odds_data:
            if match.get('id') != match_id:
                continue
            
            for bookmaker in match.get('bookmakers', []):
                bookmaker_name = bookmaker.get('key')
                
                for market_data in bookmaker.get('markets', []):
                    if market_data.get('key') == self._map_market_to_api(market):
                        for outcome in market_data.get('outcomes', []):
                            odds = outcome.get('price')
                            
                            bookmaker_odds.append(BookmakerOdds(
                                bookmaker=self.bookmaker_names.get(bookmaker_name, bookmaker_name),
                                market=market,
                                odds=odds,
                                last_updated=datetime.now()
                            ))
        
        # Sort by best odds (highest) first
        bookmaker_odds.sort(key=lambda x: x.odds, reverse=True)
        
        return bookmaker_odds
    
    def find_best_odds(
        self,
        home_team: str,
        away_team: str,
        market: str,
        min_bookmakers: int = 5
    ) -> Optional[BookmakerOdds]:
        """
        Find best available odds for a specific bet
        
        Args:
            home_team: Home team name
            away_team: Away team name
            market: Market (over_2_5, btts, etc.)
            min_bookmakers: Minimum bookmakers to compare
        
        Returns:
            BookmakerOdds with best odds, or None if not found
        """
        # In production, this would query The Odds API
        # For now, return simulated data
        
        simulated_odds = self._get_simulated_odds(market)
        
        if len(simulated_odds) < min_bookmakers:
            print(f"⚠️  Only {len(simulated_odds)} bookmakers available (need {min_bookmakers})")
        
        if not simulated_odds:
            return None
        
        best = simulated_odds[0]
        print(f"✅ Best odds: {best.bookmaker} @ {best.odds:.2f}")
        
        return best
    
    def _get_simulated_odds(self, market: str) -> List[BookmakerOdds]:
        """
        Simulated odds for testing (replace with real API in production)
        
        Real bookmaker odds for Over 2.5 typically range 1.70-1.95
        """
        # Realistic odds distribution
        bookmakers_odds = {
            "Pinnacle": 1.88,      # Sharpest (best odds, lowest margin)
            "Bet365": 1.85,        # Popular, good odds
            "1xBet": 1.87,         # High odds
            "Marathon Bet": 1.86,  # Sharp
            "Betfair Exchange": 1.89,  # Best (exchange, no margin)
            "William Hill": 1.82,  # Mainstream
            "Unibet": 1.83,
            "Betway": 1.84,
            "Bwin": 1.81,
            "Coral": 1.80          # Lowest
        }
        
        odds_list = [
            BookmakerOdds(
                bookmaker=bookmaker,
                market=market,
                odds=odds,
                last_updated=datetime.now()
            )
            for bookmaker, odds in bookmakers_odds.items()
        ]
        
        # Sort by best odds
        odds_list.sort(key=lambda x: x.odds, reverse=True)
        
        return odds_list
    
    def calculate_odds_value_difference(
        self,
        worst_odds: float,
        best_odds: float,
        stake: float = 100.0,
        win_rate: float = 0.60
    ) -> Dict:
        """
        Calculate profit difference between worst and best odds
        
        Example:
            worst: 1.75, best: 1.88
            100 bets × 10€ stake × 60% WR
            
        Returns:
            {
                "worst_profit": float,
                "best_profit": float,
                "difference": float,
                "difference_pct": float
            }
        """
        worst_profit = stake * win_rate * (worst_odds - 1.0)
        best_profit = stake * win_rate * (best_odds - 1.0)
        difference = best_profit - worst_profit
        difference_pct = (difference / worst_profit) * 100 if worst_profit > 0 else 0
        
        return {
            "worst_profit": worst_profit,
            "best_profit": best_profit,
            "difference": difference,
            "difference_pct": difference_pct
        }
    
    def generate_odds_comparison_report(
        self,
        home_team: str,
        away_team: str,
        market: str
    ) -> str:
        """Generate odds comparison report"""
        odds_list = self.find_best_odds(home_team, away_team, market)
        
        if not odds_list:
            all_odds = self._get_simulated_odds(market)
        else:
            all_odds = [odds_list]
        
        all_odds = self._get_simulated_odds(market)  # Get full list
        
        best_odds = all_odds[0].odds
        worst_odds = all_odds[-1].odds
        
        value_diff = self.calculate_odds_value_difference(
            worst_odds, best_odds, stake=1000.0, win_rate=0.60
        )
        
        report = f"""
{'='*70}
🔍 ODDS COMPARISON - {home_team} vs {away_team}
{'='*70}

MARKET: {market.upper()}

BOOKMAKER COMPARISON (sorted by best odds)
------------------------------------------
"""
        
        for i, odds in enumerate(all_odds, 1):
            margin = odds.margin_over_best(best_odds)
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            report += f"{emoji} {i}. {odds.bookmaker:<20} {odds.odds:.2f}"
            if margin > 0:
                report += f"  (-{margin:.1f}%)"
            report += "\n"
        
        report += f"""
VALUE ANALYSIS (100 bets × 10€ × 60% WR)
-----------------------------------------
Worst odds ({worst_odds:.2f}):  {value_diff['worst_profit']:.2f}€ profit
Best odds ({best_odds:.2f}):   {value_diff['best_profit']:.2f}€ profit

Difference: +{value_diff['difference']:.2f}€ (+{value_diff['difference_pct']:.1f}%)

{'='*70}
💡 KEY INSIGHT: Always bet at Pinnacle, Betfair, or 1xBet for best odds.
   Over 1000 bets, this difference is {value_diff['difference'] * 10:.2f}€!
{'='*70}
"""
        
        return report
    
    def _map_market_to_api(self, market: str) -> str:
        """Map our market names to The Odds API keys"""
        mapping = {
            "over_2_5": "totals",
            "over_1_5": "totals",
            "btts": "btts",
            "home_win": "h2h",
            "asian_handicap": "spreads"
        }
        return mapping.get(market, "totals")
    
    def _save_to_cache(self, data: Dict):
        """Save odds data to cache"""
        with open(self.cache_file, 'w') as f:
            json.dump({
                "data": data,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_from_cache(self) -> Dict:
        """Load odds data from cache"""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                cached = json.load(f)
                return cached.get('data', {})
        except:
            return {}


if __name__ == "__main__":
    # Demo: Odds Comparison in Action
    engine = OddsComparisonEngine()
    
    # Example: Compare odds for Over 2.5
    report = engine.generate_odds_comparison_report(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        market="over_2_5"
    )
    
    print(report)
    
    # Example: Find best odds
    best = engine.find_best_odds(
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        market="over_2_5",
        min_bookmakers=5
    )
    
    if best:
        print(f"\n✅ Recommendation: Bet at {best.bookmaker} @ {best.odds:.2f}")

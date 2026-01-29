"""
CLOSING LINE VALUE (CLV) TRACKER
=================================

THE #1 METRIC PROFESSIONAL BETTORS USE

What is CLV?
-----------
If you bet at odds 2.00 and the closing line (odds when market closes) is 1.85,
you have POSITIVE CLV. This means you got better odds than the "sharp" final price.

Why CLV Matters More Than Win Rate:
- Win rate can be luck (short-term variance)
- CLV proves you're beating the market's final consensus
- Positive CLV over 500+ bets = long-term profitability guaranteed

Formula:
CLV = (Your Odds / Closing Odds) - 1

Example:
- You bet at 2.00
- Closing line: 1.85
- CLV = (2.00 / 1.85) - 1 = +8.1%

Over 100 bets with +5% average CLV = significant edge

This is what separates amateurs from professionals.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import requests
from dataclasses import dataclass
import json


@dataclass
class OddsSnapshot:
    """Snapshot of odds at a specific time"""
    timestamp: datetime
    bookmaker: str
    market: str
    odds: float
    line: Optional[float] = None  # For Asian Handicap
    

class ClosingLineValueTracker:
    """
    Track Closing Line Value (CLV) for all bets
    
    This is THE metric that matters for professional betting.
    Positive CLV = Beating the market
    """
    
    def __init__(self, db_path: str = "data/tracking/clv_tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create CLV tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bets with opening and closing odds
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clv_bets (
                bet_id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                match_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                market TEXT NOT NULL,
                
                -- Opening odds (when we bet)
                opening_odds REAL NOT NULL,
                opening_bookmaker TEXT NOT NULL,
                opening_timestamp TEXT NOT NULL,
                
                -- Closing odds (market close)
                closing_odds REAL,
                closing_bookmaker TEXT,
                closing_timestamp TEXT,
                
                -- CLV calculation
                clv_percentage REAL,
                
                -- Result
                actual_result INTEGER,
                profit_loss REAL,
                
                -- Metadata
                predicted_probability REAL,
                edge REAL,
                stake REAL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Odds movements over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_movements (
                movement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                market TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                odds REAL NOT NULL,
                line REAL,  -- For Asian Handicap
                implied_probability REAL,
                
                -- Sharp indicators
                odds_change_pct REAL,
                volume_indicator TEXT,  -- 'sharp' | 'public' | 'neutral'
                
                UNIQUE(match_id, market, timestamp, bookmaker)
            )
        """)
        
        # Sharp money indicators
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sharp_indicators (
                indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                market TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                
                -- Sharp signals
                reverse_line_movement BOOLEAN,  -- Line moves opposite to public %
                steam_move BOOLEAN,             -- Sudden sharp drop (>5% in <5min)
                pinnacle_moved BOOLEAN,         -- Pinnacle (sharpest book) moved
                
                -- Data
                public_bet_percentage REAL,
                line_movement_pct REAL,
                sharp_bookmakers_moved INTEGER,
                
                created_at TEXT NOT NULL
            )
        """)
        
        # CLV performance summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clv_summary (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_bets INTEGER,
                avg_clv_percentage REAL,
                positive_clv_count INTEGER,
                negative_clv_count INTEGER,
                win_rate REAL,
                roi REAL,
                calculated_at TEXT NOT NULL,
                
                UNIQUE(date)
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ CLV Tracker initialized: {self.db_path}")
    
    def log_bet_with_opening_odds(
        self,
        bet_id: str,
        match_id: str,
        match_date: str,
        home_team: str,
        away_team: str,
        market: str,
        opening_odds: float,
        opening_bookmaker: str,
        predicted_probability: float,
        stake: float
    ):
        """
        Log a bet with opening odds (when we place the bet)
        
        This starts the CLV tracking process
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        edge = predicted_probability - (1.0 / opening_odds)
        
        cursor.execute("""
            INSERT INTO clv_bets (
                bet_id, match_id, match_date, home_team, away_team, market,
                opening_odds, opening_bookmaker, opening_timestamp,
                predicted_probability, edge, stake, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bet_id, match_id, match_date, home_team, away_team, market,
            opening_odds, opening_bookmaker, datetime.now().isoformat(),
            predicted_probability, edge, stake, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ CLV tracking started: {match_id} | {market} @ {opening_odds}")
    
    def update_closing_odds(
        self,
        bet_id: str,
        closing_odds: float,
        closing_bookmaker: str
    ):
        """
        Update with closing odds (right before match starts)
        
        This calculates the CLV
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get opening odds
        cursor.execute("""
            SELECT opening_odds FROM clv_bets WHERE bet_id = ?
        """, (bet_id,))
        
        result = cursor.fetchone()
        if not result:
            print(f"⚠️  Bet {bet_id} not found")
            conn.close()
            return
        
        opening_odds = result[0]
        
        # Calculate CLV
        clv_percentage = ((closing_odds / opening_odds) - 1.0) * 100
        
        # Update
        cursor.execute("""
            UPDATE clv_bets
            SET closing_odds = ?,
                closing_bookmaker = ?,
                closing_timestamp = ?,
                clv_percentage = ?
            WHERE bet_id = ?
        """, (
            closing_odds,
            closing_bookmaker,
            datetime.now().isoformat(),
            clv_percentage,
            bet_id
        ))
        
        conn.commit()
        conn.close()
        
        clv_emoji = "✅" if clv_percentage > 0 else "❌"
        print(f"{clv_emoji} CLV: {clv_percentage:+.1f}% | Opened {opening_odds:.2f} → Closed {closing_odds:.2f}")
    
    def log_odds_movement(
        self,
        match_id: str,
        market: str,
        bookmaker: str,
        odds: float,
        line: Optional[float] = None
    ):
        """
        Log odds movement for line tracking
        
        Call this periodically (every 15-30 min) to track line movements
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous odds for this market
        cursor.execute("""
            SELECT odds, timestamp FROM odds_movements
            WHERE match_id = ? AND market = ? AND bookmaker = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (match_id, market, bookmaker))
        
        previous = cursor.fetchone()
        
        implied_probability = 1.0 / odds
        
        if previous:
            prev_odds = previous[0]
            odds_change_pct = ((odds - prev_odds) / prev_odds) * 100
        else:
            odds_change_pct = 0.0
        
        # Detect sharp move (>5% change)
        volume_indicator = "neutral"
        if abs(odds_change_pct) > 5.0:
            volume_indicator = "sharp"
        elif abs(odds_change_pct) > 2.0:
            volume_indicator = "public"
        
        cursor.execute("""
            INSERT OR REPLACE INTO odds_movements (
                match_id, market, timestamp, bookmaker, odds, line,
                implied_probability, odds_change_pct, volume_indicator
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, market, datetime.now().isoformat(), bookmaker,
            odds, line, implied_probability, odds_change_pct, volume_indicator
        ))
        
        conn.commit()
        conn.close()
        
        if abs(odds_change_pct) > 3.0:
            print(f"📊 Line Move: {market} {odds_change_pct:+.1f}% ({prev_odds:.2f} → {odds:.2f}) [{bookmaker}]")
    
    def detect_sharp_indicators(
        self,
        match_id: str,
        market: str,
        public_bet_percentage: Optional[float] = None
    ) -> Dict:
        """
        Detect sharp money indicators
        
        Sharp indicators:
        1. Reverse Line Movement: Line moves opposite to public betting %
        2. Steam Move: Sudden sharp drop (>5% in <5 minutes)
        3. Pinnacle Moved: Pinnacle (sharpest bookmaker) moved first
        
        Returns:
            {
                "reverse_line_movement": bool,
                "steam_move": bool,
                "pinnacle_moved": bool,
                "line_movement_pct": float,
                "sharp_confidence": str  # 'HIGH' | 'MEDIUM' | 'LOW'
            }
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get recent movements (last 30 minutes)
        query = """
            SELECT timestamp, bookmaker, odds, odds_change_pct
            FROM odds_movements
            WHERE match_id = ? AND market = ?
              AND datetime(timestamp) >= datetime('now', '-30 minutes')
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(match_id, market))
        conn.close()
        
        if len(df) < 2:
            return {
                "reverse_line_movement": False,
                "steam_move": False,
                "pinnacle_moved": False,
                "line_movement_pct": 0.0,
                "sharp_confidence": "LOW"
            }
        
        # Calculate total line movement
        first_odds = df.iloc[-1]['odds']
        last_odds = df.iloc[0]['odds']
        line_movement_pct = ((last_odds - first_odds) / first_odds) * 100
        
        # 1. Steam Move Detection (>5% in <5 min)
        steam_move = False
        for i in range(len(df) - 1):
            if abs(df.iloc[i]['odds_change_pct']) > 5.0:
                steam_move = True
                break
        
        # 2. Pinnacle Moved (sharp bookmaker indicator)
        pinnacle_moved = 'Pinnacle' in df['bookmaker'].values
        
        # 3. Reverse Line Movement (needs public bet %)
        reverse_line_movement = False
        if public_bet_percentage is not None:
            # If public is betting over 60% on one side but line moves opposite
            if public_bet_percentage > 60 and line_movement_pct < -2.0:
                reverse_line_movement = True
            elif public_bet_percentage < 40 and line_movement_pct > 2.0:
                reverse_line_movement = True
        
        # Sharp Confidence
        sharp_signals = sum([steam_move, pinnacle_moved, reverse_line_movement])
        if sharp_signals >= 2:
            sharp_confidence = "HIGH"
        elif sharp_signals == 1:
            sharp_confidence = "MEDIUM"
        else:
            sharp_confidence = "LOW"
        
        return {
            "reverse_line_movement": reverse_line_movement,
            "steam_move": steam_move,
            "pinnacle_moved": pinnacle_moved,
            "line_movement_pct": line_movement_pct,
            "sharp_confidence": sharp_confidence
        }
    
    def get_clv_performance(self, days: int = 30) -> Dict:
        """
        Get CLV performance summary
        
        Returns:
            {
                "total_bets": int,
                "avg_clv": float,
                "positive_clv_pct": float,
                "win_rate": float,
                "roi": float,
                "clv_by_market": Dict
            }
        """
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                market,
                COUNT(*) as total_bets,
                AVG(clv_percentage) as avg_clv,
                SUM(CASE WHEN clv_percentage > 0 THEN 1 ELSE 0 END) as positive_clv,
                SUM(CASE WHEN actual_result = 1 THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as total_profit,
                SUM(stake) as total_stake
            FROM clv_bets
            WHERE created_at >= ?
              AND closing_odds IS NOT NULL
            GROUP BY market
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        if len(df) == 0:
            return {
                "total_bets": 0,
                "avg_clv": 0.0,
                "positive_clv_pct": 0.0,
                "win_rate": 0.0,
                "roi": 0.0,
                "clv_by_market": {}
            }
        
        total_bets = df['total_bets'].sum()
        positive_clv = df['positive_clv'].sum()
        wins = df['wins'].sum()
        total_profit = df['total_profit'].sum()
        total_stake = df['total_stake'].sum()
        
        avg_clv = df['avg_clv'].mean()
        positive_clv_pct = (positive_clv / total_bets) * 100 if total_bets > 0 else 0
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        
        clv_by_market = df.set_index('market')['avg_clv'].to_dict()
        
        return {
            "total_bets": int(total_bets),
            "avg_clv": float(avg_clv),
            "positive_clv_pct": float(positive_clv_pct),
            "win_rate": float(win_rate),
            "roi": float(roi),
            "clv_by_market": clv_by_market
        }
    
    def generate_clv_report(self, days: int = 30) -> str:
        """Generate CLV performance report"""
        perf = self.get_clv_performance(days)
        
        report = f"""
{'='*70}
📊 CLOSING LINE VALUE (CLV) REPORT - Last {days} Days
{'='*70}

OVERALL PERFORMANCE
-------------------
Total Bets:           {perf['total_bets']}
Average CLV:          {perf['avg_clv']:+.2f}%
Positive CLV Rate:    {perf['positive_clv_pct']:.1f}%
Win Rate:             {perf['win_rate']:.1f}%
ROI:                  {perf['roi']:+.1f}%

CLV BY MARKET
-------------
"""
        
        for market, clv in perf['clv_by_market'].items():
            clv_emoji = "✅" if clv > 0 else "❌"
            report += f"{clv_emoji} {market}: {clv:+.2f}%\n"
        
        report += f"""
{'='*70}
INTERPRETATION
--------------
"""
        
        if perf['total_bets'] < 50:
            report += "⚠️  Sample size too small (need 50+ bets for meaningful CLV)\n"
        elif perf['avg_clv'] > 2.0:
            report += "✅ EXCELLENT: Avg CLV >2% indicates strong market edge\n"
        elif perf['avg_clv'] > 0:
            report += "✅ POSITIVE: Avg CLV >0% shows you're beating closing lines\n"
        else:
            report += "❌ NEGATIVE: Avg CLV <0% means market is beating you\n"
        
        if perf['positive_clv_pct'] > 55:
            report += "✅ HIGH HIT RATE: >55% positive CLV is professional level\n"
        
        report += f"""
{'='*70}
KEY INSIGHT: CLV is THE metric that matters. Win rate can be luck,
but positive CLV over 500+ bets guarantees long-term profitability.
{'='*70}
"""
        
        return report


if __name__ == "__main__":
    # Demo: CLV Tracking in Action
    tracker = ClosingLineValueTracker()
    
    # Example 1: Log bet with opening odds
    tracker.log_bet_with_opening_odds(
        bet_id="bet_001",
        match_id="match_001",
        match_date="2026-01-30",
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        market="over_2_5",
        opening_odds=1.85,
        opening_bookmaker="Bet365",
        predicted_probability=0.62,
        stake=50.0
    )
    
    # Example 2: Update closing odds (right before match)
    tracker.update_closing_odds(
        bet_id="bet_001",
        closing_odds=1.72,  # Line moved down (sharp money on over)
        closing_bookmaker="Pinnacle"
    )
    
    # Example 3: Track odds movements
    tracker.log_odds_movement("match_001", "over_2_5", "Bet365", 1.85)
    tracker.log_odds_movement("match_001", "over_2_5", "Pinnacle", 1.80)
    tracker.log_odds_movement("match_001", "over_2_5", "1xBet", 1.75)
    
    # Example 4: Detect sharp indicators
    sharp = tracker.detect_sharp_indicators("match_001", "over_2_5", public_bet_percentage=65)
    print(f"\n🔍 Sharp Indicators: {sharp}")
    
    # Example 5: Get CLV report
    print(tracker.generate_clv_report(days=30))

"""
LIVE BET TRACKING SYSTEM - Production Performance Measurement
===========================================================

Tracks real betting performance to:
1. Measure actual ROI/win rate vs backtest predictions
2. Detect model degradation (concept drift)
3. Enable self-improvement loop
4. Calculate CLV (Closing Line Value)

Database Schema:
- bets: bet_id, match_id, market, predicted_prob, odds, stake, timestamp
- results: bet_id, actual_result, profit_loss, settled_timestamp
- daily_stats: date, total_bets, wins, roi, win_rate, avg_odds
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import json


class LiveBetTracker:
    """
    Live bet tracking system for production performance monitoring
    """
    
    def __init__(self, db_path: str = "data/tracking/live_bets.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database schema if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                bet_id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                match_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT NOT NULL,
                market TEXT NOT NULL,
                predicted_prob REAL NOT NULL,
                odds REAL NOT NULL,
                stake REAL NOT NULL,
                expected_value REAL,
                confidence TEXT,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                UNIQUE(match_id, market)
            )
        """)
        
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id TEXT NOT NULL,
                actual_result INTEGER NOT NULL,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                profit_loss REAL NOT NULL,
                settled_timestamp TEXT NOT NULL,
                FOREIGN KEY (bet_id) REFERENCES bets(bet_id)
            )
        """)
        
        # Daily stats table (aggregated)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_bets INTEGER NOT NULL,
                settled_bets INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                total_stake REAL NOT NULL,
                total_profit_loss REAL NOT NULL,
                roi REAL NOT NULL,
                win_rate REAL NOT NULL,
                avg_odds REAL NOT NULL,
                calculated_timestamp TEXT NOT NULL
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model_id TEXT,
                market TEXT,
                evaluation_date TEXT,
                roc_auc REAL,
                calibration_error REAL,
                brier_score REAL,
                win_rate REAL,
                roi REAL,
                total_bets INTEGER,
                PRIMARY KEY (model_id, market, evaluation_date)
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def log_bet(
        self,
        match_id: str,
        match_date: str,
        home_team: str,
        away_team: str,
        league: str,
        market: str,
        predicted_prob: float,
        odds: float,
        stake: float,
        confidence: str = "medium"
    ) -> str:
        """
        Log a new bet to the tracker
        
        Returns:
            bet_id: Unique identifier for the bet
        """
        bet_id = f"{match_id}_{market}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate expected value
        implied_prob = 1.0 / odds
        expected_value = (predicted_prob * odds - 1.0) * 100  # EV as percentage
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO bets (
                    bet_id, match_id, match_date, home_team, away_team, league,
                    market, predicted_prob, odds, stake, expected_value,
                    confidence, timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet_id, match_id, match_date, home_team, away_team, league,
                market, predicted_prob, odds, stake, expected_value,
                confidence, datetime.now().isoformat(), 'pending'
            ))
            
            conn.commit()
            print(f"✅ Bet logged: {home_team} vs {away_team} | {market} @ {odds:.2f} (EV: {expected_value:+.1f}%)")
            
        except sqlite3.IntegrityError:
            print(f"⚠️  Bet already exists: {match_id} - {market}")
            bet_id = None
        finally:
            conn.close()
        
        return bet_id
    
    def update_result(
        self,
        match_id: str,
        market: str,
        actual_result: bool,
        home_goals: int,
        away_goals: int
    ):
        """
        Update bet with actual match result
        
        Args:
            match_id: Match identifier
            market: Betting market (over_1_5, over_2_5, btts)
            actual_result: True if bet won, False if lost
            home_goals: Actual home team goals
            away_goals: Actual away team goals
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find matching bet
        cursor.execute("""
            SELECT bet_id, odds, stake, status FROM bets
            WHERE match_id = ? AND market = ?
        """, (match_id, market))
        
        result = cursor.fetchone()
        
        if not result:
            print(f"⚠️  No bet found for {match_id} - {market}")
            conn.close()
            return
        
        bet_id, odds, stake, status = result
        
        if status == 'settled':
            print(f"⚠️  Bet already settled: {bet_id}")
            conn.close()
            return
        
        # Calculate profit/loss
        if actual_result:
            profit_loss = stake * (odds - 1.0)
        else:
            profit_loss = -stake
        
        # Insert result
        cursor.execute("""
            INSERT INTO results (
                bet_id, actual_result, actual_home_goals, actual_away_goals,
                profit_loss, settled_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bet_id, int(actual_result), home_goals, away_goals,
            profit_loss, datetime.now().isoformat()
        ))
        
        # Update bet status
        cursor.execute("""
            UPDATE bets SET status = 'settled' WHERE bet_id = ?
        """, (bet_id,))
        
        conn.commit()
        conn.close()
        
        result_emoji = "✅" if actual_result else "❌"
        print(f"{result_emoji} Result updated: {match_id} | {market} | {home_goals}-{away_goals} | P/L: {profit_loss:+.2f} €")
    
    def get_stats(self, days: int = 30) -> Dict:
        """
        Get performance statistics for last N days
        
        Returns:
            Dictionary with win_rate, roi, total_bets, total_profit, etc.
        """
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                COUNT(DISTINCT b.bet_id) as total_bets,
                COUNT(DISTINCT r.bet_id) as settled_bets,
                SUM(CASE WHEN r.actual_result = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN r.actual_result = 0 THEN 1 ELSE 0 END) as losses,
                SUM(b.stake) as total_stake,
                SUM(r.profit_loss) as total_profit,
                AVG(b.odds) as avg_odds
            FROM bets b
            LEFT JOIN results r ON b.bet_id = r.bet_id
            WHERE b.timestamp >= ?
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        stats = df.iloc[0].to_dict()
        
        # Calculate derived metrics
        if stats['settled_bets'] > 0:
            stats['win_rate'] = stats['wins'] / stats['settled_bets']
            stats['roi'] = (stats['total_profit'] / stats['total_stake']) * 100 if stats['total_stake'] > 0 else 0
        else:
            stats['win_rate'] = 0
            stats['roi'] = 0
        
        stats['pending_bets'] = stats['total_bets'] - stats['settled_bets']
        
        return stats
    
    def get_market_breakdown(self, days: int = 30) -> pd.DataFrame:
        """Get performance breakdown by market"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                b.market,
                COUNT(DISTINCT b.bet_id) as total_bets,
                COUNT(DISTINCT r.bet_id) as settled_bets,
                SUM(CASE WHEN r.actual_result = 1 THEN 1 ELSE 0 END) as wins,
                SUM(b.stake) as total_stake,
                SUM(r.profit_loss) as total_profit,
                AVG(b.odds) as avg_odds,
                AVG(b.predicted_prob) as avg_predicted_prob
            FROM bets b
            LEFT JOIN results r ON b.bet_id = r.bet_id
            WHERE b.timestamp >= ?
            GROUP BY b.market
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        # Calculate metrics
        df['win_rate'] = df['wins'] / df['settled_bets']
        df['roi'] = (df['total_profit'] / df['total_stake']) * 100
        df['pending_bets'] = df['total_bets'] - df['settled_bets']
        
        return df
    
    def get_daily_performance(self, days: int = 30) -> pd.DataFrame:
        """Get daily performance time series"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        query = """
            SELECT 
                DATE(b.match_date) as date,
                COUNT(DISTINCT b.bet_id) as total_bets,
                COUNT(DISTINCT r.bet_id) as settled_bets,
                SUM(CASE WHEN r.actual_result = 1 THEN 1 ELSE 0 END) as wins,
                SUM(b.stake) as total_stake,
                SUM(r.profit_loss) as total_profit
            FROM bets b
            LEFT JOIN results r ON b.bet_id = r.bet_id
            WHERE DATE(b.match_date) >= ?
            GROUP BY DATE(b.match_date)
            ORDER BY date
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        # Calculate metrics
        df['win_rate'] = df['wins'] / df['settled_bets'].replace(0, 1)
        df['roi'] = (df['total_profit'] / df['total_stake'].replace(0, 1)) * 100
        df['cumulative_profit'] = df['total_profit'].cumsum()
        
        return df
    
    def calculate_daily_stats(self, date: str = None):
        """Calculate and store daily aggregated statistics"""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get stats for the date
        query = """
            SELECT 
                COUNT(DISTINCT b.bet_id) as total_bets,
                COUNT(DISTINCT r.bet_id) as settled_bets,
                SUM(CASE WHEN r.actual_result = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN r.actual_result = 0 THEN 1 ELSE 0 END) as losses,
                SUM(b.stake) as total_stake,
                SUM(r.profit_loss) as total_profit,
                AVG(b.odds) as avg_odds
            FROM bets b
            LEFT JOIN results r ON b.bet_id = r.bet_id
            WHERE DATE(b.match_date) = ?
        """
        
        df = pd.read_sql_query(query, conn, params=(date,))
        stats = df.iloc[0]
        
        if stats['settled_bets'] > 0:
            win_rate = stats['wins'] / stats['settled_bets']
            roi = (stats['total_profit'] / stats['total_stake']) * 100 if stats['total_stake'] > 0 else 0
        else:
            win_rate = 0
            roi = 0
        
        # Insert/replace daily stats
        cursor.execute("""
            INSERT OR REPLACE INTO daily_stats (
                date, total_bets, settled_bets, wins, losses,
                total_stake, total_profit_loss, roi, win_rate, avg_odds,
                calculated_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, int(stats['total_bets']), int(stats['settled_bets']),
            int(stats['wins']), int(stats['losses']),
            float(stats['total_stake']), float(stats['total_profit']),
            float(roi), float(win_rate), float(stats['avg_odds']),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Daily stats calculated for {date}: {stats['settled_bets']} bets, {win_rate:.1%} win rate, {roi:+.2f}% ROI")
    
    def generate_report(self, days: int = 30) -> str:
        """Generate comprehensive performance report"""
        stats = self.get_stats(days)
        market_breakdown = self.get_market_breakdown(days)
        
        report = f"""
{'='*70}
📊 LIVE BET TRACKING REPORT (Last {days} Days)
{'='*70}

OVERALL PERFORMANCE
-------------------
Total Bets:        {stats['total_bets']:.0f}
Settled Bets:      {stats['settled_bets']:.0f}
Pending Bets:      {stats['pending_bets']:.0f}
Wins:              {stats['wins']:.0f}
Losses:            {stats['losses']:.0f}
Win Rate:          {stats['win_rate']:.1%}
ROI:               {stats['roi']:+.2f}%
Total Stake:       {stats['total_stake']:.2f} €
Total Profit:      {stats['total_profit']:+.2f} €
Avg Odds:          {stats['avg_odds']:.2f}

MARKET BREAKDOWN
----------------
"""
        
        for _, row in market_breakdown.iterrows():
            report += f"\n{row['market'].upper()}\n"
            report += f"  Bets: {row['settled_bets']:.0f} | Win Rate: {row['win_rate']:.1%} | ROI: {row['roi']:+.2f}% | Avg Odds: {row['avg_odds']:.2f}\n"
        
        report += f"\n{'='*70}\n"
        
        # Check if meeting top 1% targets
        if stats['settled_bets'] >= 100:  # Only evaluate after sufficient sample size
            if stats['win_rate'] >= 0.56 and stats['roi'] >= 8.0:
                report += "✅ TOP 1% PERFORMANCE ACHIEVED (>56% WR, >8% ROI)\n"
            else:
                report += f"⚠️  Gap to top 1%: WR {max(0, 56 - stats['win_rate']*100):.1f}pp, ROI {max(0, 8 - stats['roi']):.1f}pp\n"
        else:
            report += f"⏳ Insufficient data for evaluation (need {100 - stats['settled_bets']:.0f} more settled bets)\n"
        
        report += f"{'='*70}\n"
        
        return report


# FastAPI Endpoints (for integration with monitoring system)
def create_tracking_api():
    """Create FastAPI app for bet tracking endpoints"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("⚠️  FastAPI not installed, skipping API creation")
        return None
    
    app = FastAPI(title="Live Bet Tracker API")
    tracker = LiveBetTracker()
    
    class BetRequest(BaseModel):
        match_id: str
        match_date: str
        home_team: str
        away_team: str
        league: str
        market: str
        predicted_prob: float
        odds: float
        stake: float
        confidence: str = "medium"
    
    class ResultRequest(BaseModel):
        match_id: str
        market: str
        actual_result: bool
        home_goals: int
        away_goals: int
    
    @app.post("/api/tracking/log_bet")
    async def log_bet(bet: BetRequest):
        bet_id = tracker.log_bet(
            bet.match_id, bet.match_date, bet.home_team, bet.away_team,
            bet.league, bet.market, bet.predicted_prob, bet.odds,
            bet.stake, bet.confidence
        )
        if bet_id:
            return {"status": "success", "bet_id": bet_id}
        else:
            raise HTTPException(status_code=400, detail="Bet already exists")
    
    @app.post("/api/tracking/update_result")
    async def update_result(result: ResultRequest):
        tracker.update_result(
            result.match_id, result.market, result.actual_result,
            result.home_goals, result.away_goals
        )
        return {"status": "success"}
    
    @app.get("/api/tracking/stats")
    async def get_stats(days: int = 30):
        return tracker.get_stats(days)
    
    @app.get("/api/tracking/report")
    async def get_report(days: int = 30):
        return {"report": tracker.generate_report(days)}
    
    return app


if __name__ == "__main__":
    # Demo usage
    tracker = LiveBetTracker()
    
    # Example: Log a bet
    tracker.log_bet(
        match_id="demo_match_001",
        match_date="2026-01-30",
        home_team="Bayern München",
        away_team="Borussia Dortmund",
        league="Bundesliga",
        market="over_2_5",
        predicted_prob=0.68,
        odds=1.55,
        stake=10.0,
        confidence="high"
    )
    
    # Example: Update result (Bayern 3-2 Dortmund = Over 2.5 wins)
    tracker.update_result(
        match_id="demo_match_001",
        market="over_2_5",
        actual_result=True,
        home_goals=3,
        away_goals=2
    )
    
    # Get stats
    print(tracker.generate_report(days=30))

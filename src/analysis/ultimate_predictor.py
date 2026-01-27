#!/usr/bin/env python3
"""
🚀 ULTIMATE MATCH PREDICTOR
============================
Production-ready system combining:
- Real match data from APIs
- Advanced Predictor with ALL markets
- Optimal odds filter (1.30-1.70)
- Kelly staking
- Telegram notifications

This replaces tomorrow_matches.py with a much more powerful system!
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import joblib
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.advanced_predictor import (
    AdvancedPredictor, 
    MatchContext, 
    TeamStats, 
    Market,
    TelegramFormatter,
    LEAGUE_PROFILES
)
from config.telegram_config import get_bot_token, get_chat_id


class UltimateMatchPredictor:
    """
    🧠 Ultimate Match Prediction System
    
    Features:
    - Fetches REAL matches from APIs
    - Predicts ALL betting markets
    - Applies optimal odds filter (1.30-1.70)
    - Uses Kelly Criterion for staking
    - Sends to Telegram
    """
    
    def __init__(self, bankroll: float = 1000.0):
        self.predictor = AdvancedPredictor(bankroll=bankroll)
        self.data_dir = Path("data/historical")
        self.historical_data = self._load_historical_data()
        self.bankroll = bankroll
        
        # Settings
        self.optimal_odds_min = 1.30
        self.optimal_odds_max = 1.70
        self.min_probability = 0.60
        self.min_edge = 0.05
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data for Elo/Form lookup"""
        data_path = self.data_dir / "massive_training_data.csv"
        if data_path.exists():
            print("✅ Loaded historical data for team stats")
            return pd.read_csv(data_path)
        print("⚠️  No historical data found - using defaults")
        return pd.DataFrame()
    
    def get_team_stats(self, team_name: str) -> TeamStats:
        """Get team statistics from historical data"""
        stats = TeamStats(name=team_name)
        
        if self.historical_data.empty:
            return stats
        
        # Check column names
        home_col = 'HomeTeam' if 'HomeTeam' in self.historical_data.columns else 'home_team'
        away_col = 'AwayTeam' if 'AwayTeam' in self.historical_data.columns else 'away_team'
        date_col = 'Date' if 'Date' in self.historical_data.columns else 'date'
        
        # Find team matches
        home_matches = self.historical_data[
            self.historical_data[home_col].str.contains(team_name, case=False, na=False)
        ].sort_values(date_col, ascending=False)
        
        away_matches = self.historical_data[
            self.historical_data[away_col].str.contains(team_name, case=False, na=False)
        ].sort_values(date_col, ascending=False)
        
        # Get Elo
        if not home_matches.empty and 'home_elo' in home_matches.columns:
            stats.elo = home_matches.iloc[0]['home_elo']
        elif not away_matches.empty and 'away_elo' in away_matches.columns:
            stats.elo = away_matches.iloc[0]['away_elo']
        
        # Get Form
        if not home_matches.empty and 'home_form' in home_matches.columns:
            stats.form_points = int(home_matches.iloc[0]['home_form'] / 100 * 15)  # Convert to 0-15 scale
        elif not away_matches.empty and 'away_form' in away_matches.columns:
            stats.form_points = int(away_matches.iloc[0]['away_form'] / 100 * 15)
        
        # Get goals from last 5 matches
        all_matches = pd.concat([home_matches.head(3), away_matches.head(2)]).sort_values(date_col, ascending=False)
        
        if not all_matches.empty:
            goals_for = 0
            goals_against = 0
            clean_sheets = 0
            failed_to_score = 0
            
            for _, match in all_matches.head(5).iterrows():
                if team_name.lower() in str(match.get(home_col, '')).lower():
                    gf = match.get('FTHG', match.get('home_goals', 0)) or 0
                    ga = match.get('FTAG', match.get('away_goals', 0)) or 0
                else:
                    gf = match.get('FTAG', match.get('away_goals', 0)) or 0
                    ga = match.get('FTHG', match.get('home_goals', 0)) or 0
                
                goals_for += gf
                goals_against += ga
                if ga == 0:
                    clean_sheets += 1
                if gf == 0:
                    failed_to_score += 1
            
            stats.goals_scored_last_5 = goals_for
            stats.goals_conceded_last_5 = goals_against
            stats.clean_sheet_rate = clean_sheets / 5
            stats.failed_to_score_rate = failed_to_score / 5
        
        return stats
    
    def fetch_upcoming_matches(self, days_ahead: int = 3) -> List[Dict]:
        """Fetch REAL upcoming matches from multiple APIs"""
        matches = []
        
        print(f"\n🔍 Searching for REAL matches (next {days_ahead} days)...")
        
        # 1. OpenLigaDB (Bundesliga)
        matches.extend(self._fetch_openligadb())
        
        # 2. TheSportsDB (Multiple leagues)
        matches.extend(self._fetch_thesportsdb(days_ahead))
        
        # Remove duplicates
        seen = set()
        unique_matches = []
        for m in matches:
            key = f"{m['home_team']}_{m['away_team']}_{m['date']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)
        
        if not unique_matches:
            print("❌ NO REAL MATCHES FOUND!")
            print("   The system will NOT generate fake data.")
            return []
        
        print(f"✅ Found {len(unique_matches)} unique upcoming matches")
        return unique_matches
    
    def _fetch_openligadb(self) -> List[Dict]:
        """Fetch from OpenLigaDB (Bundesliga)"""
        matches = []
        try:
            url = "https://api.openligadb.de/getmatchdata/bl1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                now = datetime.now()
                
                for match in data:
                    match_datetime_str = match.get('matchDateTime', '')
                    if match_datetime_str:
                        try:
                            match_dt = datetime.fromisoformat(match_datetime_str.replace('Z', '+00:00'))
                            
                            # Only future matches
                            if match_dt.replace(tzinfo=None) > now:
                                matches.append({
                                    'home_team': match['team1']['teamName'],
                                    'away_team': match['team2']['teamName'],
                                    'league': 'Bundesliga',
                                    'date': match_datetime_str[:10],
                                    'time': match_datetime_str[11:16] if len(match_datetime_str) > 11 else 'TBD',
                                    'source': 'OpenLigaDB'
                                })
                                
                                if len(matches) >= 10:
                                    break
                        except:
                            pass
                
                if matches:
                    print(f"   ✅ OpenLigaDB: {len(matches)} Bundesliga matches")
        except Exception as e:
            print(f"   ⚠️  OpenLigaDB error: {e}")
        
        return matches
    
    def _fetch_thesportsdb(self, days_ahead: int) -> List[Dict]:
        """Fetch from TheSportsDB (Multiple leagues)"""
        matches = []
        
        leagues = [
            (4328, 'Premier League'),
            (4335, 'La Liga'),
            (4332, 'Serie A'),
            (4334, 'Ligue 1'),
            (4337, 'Eredivisie')
        ]
        
        try:
            now = datetime.now()
            
            for days in range(1, days_ahead + 1):
                check_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')
                
                for league_id, league_name in leagues:
                    url = f"https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d={check_date}&l={league_id}"
                    
                    try:
                        response = requests.get(url, timeout=5)
                        
                        if response.status_code == 200:
                            data = response.json()
                            events = data.get('events') or []
                            
                            for event in events:
                                if event and event.get('strSport') == 'Soccer':
                                    matches.append({
                                        'home_team': event['strHomeTeam'],
                                        'away_team': event['strAwayTeam'],
                                        'league': league_name,
                                        'date': check_date,
                                        'time': event.get('strTime', 'TBD'),
                                        'source': 'TheSportsDB'
                                    })
                            
                            if events:
                                print(f"   ✅ TheSportsDB: {len(events)} {league_name} matches on {check_date}")
                    except:
                        pass
                    
                    time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"   ⚠️  TheSportsDB error: {e}")
        
        return matches
    
    def analyze_match(self, match: Dict) -> Optional[Dict]:
        """Analyze a single match with full predictions"""
        home_team = match['home_team']
        away_team = match['away_team']
        league = match['league']
        
        # Get team stats
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)
        
        # Check if derby
        is_derby, derby_name = self.predictor.is_derby(home_team, away_team, league)
        
        # Create context
        context = MatchContext(
            home_team=home_stats,
            away_team=away_stats,
            league=league,
            date=match['date'],
            is_derby=is_derby,
            derby_name=derby_name
        )
        
        # Generate predictions
        match_predictions = self.predictor.predict_match(context)
        
        # Get recommendations (filtered by optimal range)
        recommendations = self.predictor.get_recommendations(
            match_predictions, 
            max_picks=5,
            only_optimal_range=True
        )
        
        if not recommendations:
            # Try without optimal range filter
            recommendations = self.predictor.get_recommendations(
                match_predictions, 
                max_picks=3,
                only_optimal_range=False
            )
        
        return {
            'match': match,
            'context': context,
            'predictions': match_predictions,
            'recommendations': recommendations
        }
    
    def analyze_all_matches(self, days_ahead: int = 3) -> List[Dict]:
        """Analyze all upcoming matches"""
        print("\n" + "="*70)
        print("🧠 ULTIMATE MATCH PREDICTOR - ANALYZING ALL MARKETS")
        print("="*70)
        
        matches = self.fetch_upcoming_matches(days_ahead)
        
        if not matches:
            return []
        
        all_analyses = []
        
        for i, match in enumerate(matches, 1):
            print(f"\n{'─'*70}")
            print(f"📊 Match {i}/{len(matches)}: {match['home_team']} vs {match['away_team']}")
            print(f"   🏆 {match['league']} | 📅 {match['date']}")
            print(f"{'─'*70}")
            
            analysis = self.analyze_match(match)
            
            if analysis and analysis['recommendations']:
                print(f"\n   🎯 RECOMMENDATIONS (Optimal Range 1.30-1.70):")
                for rec in analysis['recommendations'][:3]:
                    print(f"      ✅ {rec.market.value:20} @ {rec.odds:5.2f}")
                    print(f"         Prob: {rec.probability:.1%} | Edge: {rec.edge:+.1%} | {rec.confidence_tier}")
                
                all_analyses.append(analysis)
            else:
                print("   ⚠️  No value bets found for this match")
        
        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE: {len(all_analyses)} matches with recommendations")
        print(f"{'='*70}\n")
        
        return all_analyses
    
    def send_to_telegram(self, analyses: List[Dict]) -> None:
        """Send recommendations to Telegram"""
        bot_token = get_bot_token()
        chat_id = get_chat_id()
        
        if not bot_token or not chat_id:
            print("⚠️  Telegram not configured!")
            return
        
        if not analyses:
            print("⚠️  No analyses to send")
            return
        
        # Summary header
        total_picks = sum(len(a['recommendations']) for a in analyses)
        
        header = f"""
🧠 **ULTIMATE PREDICTOR - BETTING RECOMMENDATIONS**
{'='*50}

📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
⚽ Matches Analyzed: {len(analyses)}
🎯 Total Picks: {total_picks}

💰 Optimal Odds Range: 1.30 - 1.70
📊 Min Edge: 5%+
🔬 Powered by Knowledge-Enhanced ML System

{'='*50}
"""
        self._send_message(bot_token, chat_id, header)
        
        # Send each match analysis
        for i, analysis in enumerate(analyses, 1):
            match = analysis['match']
            context = analysis['context']
            recs = analysis['recommendations']
            
            if not recs:
                continue
            
            msg = f"""
🏟️ **MATCH {i}: {match['home_team']} vs {match['away_team']}**
🏆 {match['league']} | 📅 {match['date']}
"""
            
            if context.is_derby:
                msg += f"⚔️ *DERBY: {context.derby_name}*\n"
            
            msg += "\n🎯 **PICKS:**\n"
            
            for rec in recs[:3]:
                market_name = self._get_market_display_name(rec.market, context)
                kelly_stake = rec.kelly_stake * self.bankroll
                
                msg += f"""
✅ **{market_name}**
   📈 Probability: {rec.probability:.1%}
   💰 Odds: {rec.odds}
   📊 Edge: {rec.edge:+.1%}
   🎚️ Confidence: {rec.confidence_tier}
   💵 Kelly Stake: €{kelly_stake:.2f}
"""
            
            self._send_message(bot_token, chat_id, msg)
            time.sleep(1)
        
        # Footer with accumulator suggestion
        if len(analyses) >= 2:
            acca_msg = self._build_accumulator_message(analyses)
            self._send_message(bot_token, chat_id, acca_msg)
        
        print(f"✅ Sent {len(analyses)} match analyses to Telegram!")
    
    def _get_market_display_name(self, market: Market, context: MatchContext) -> str:
        """Get display name for market"""
        names = {
            Market.OVER_0_5: 'Over 0.5 Goals',
            Market.OVER_1_5: 'Over 1.5 Goals',
            Market.OVER_2_5: 'Over 2.5 Goals',
            Market.OVER_3_5: 'Over 3.5 Goals',
            Market.UNDER_2_5: 'Under 2.5 Goals',
            Market.UNDER_3_5: 'Under 3.5 Goals',
            Market.BTTS_YES: 'Both Teams to Score: YES',
            Market.BTTS_NO: 'Both Teams to Score: NO',
            Market.HOME_WIN: f'{context.home_team.name} Win',
            Market.AWAY_WIN: f'{context.away_team.name} Win',
            Market.DRAW: 'Draw',
            Market.HOME_OR_DRAW: f'{context.home_team.name} or Draw (1X)',
            Market.AWAY_OR_DRAW: f'{context.away_team.name} or Draw (X2)',
            Market.DNB_HOME: f'{context.home_team.name} Draw No Bet',
            Market.DNB_AWAY: f'{context.away_team.name} Draw No Bet',
        }
        return names.get(market, market.value)
    
    def _build_accumulator_message(self, analyses: List[Dict]) -> str:
        """Build accumulator suggestion from best picks"""
        acca_legs = []
        
        for analysis in analyses[:4]:  # Max 4 matches
            recs = analysis['recommendations']
            if recs:
                # Pick highest probability recommendation
                best = max(recs, key=lambda r: r.probability)
                if best.probability >= 0.65 and best.odds >= 1.20:
                    acca_legs.append({
                        'match': f"{analysis['match']['home_team']} vs {analysis['match']['away_team']}",
                        'pick': best.market.value,
                        'odds': best.odds,
                        'prob': best.probability
                    })
        
        if len(acca_legs) < 2:
            return ""
        
        combined_odds = 1.0
        combined_prob = 1.0
        for leg in acca_legs:
            combined_odds *= leg['odds']
            combined_prob *= leg['prob']
        
        msg = f"""
🔗 **ACCUMULATOR SUGGESTION**
{'─'*40}
"""
        
        for i, leg in enumerate(acca_legs, 1):
            msg += f"{i}. {leg['match']}\n   → {leg['pick']} @ {leg['odds']}\n"
        
        msg += f"""
{'─'*40}
📊 Combined Odds: **{combined_odds:.2f}**
📈 Combined Probability: {combined_prob:.1%}
💰 €10 → €{10 * combined_odds:.2f} potential return

⚠️ *Akkus sind riskanter - nur mit kleinem Einsatz!*
"""
        
        return msg
    
    def _send_message(self, bot_token: str, chat_id: str, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️  Telegram error: {e}")
            return False


def main():
    """Main execution"""
    print("\n" + "🚀"*30)
    print("   ULTIMATE MATCH PREDICTOR")
    print("🚀"*30 + "\n")
    
    # Initialize with €1000 bankroll
    predictor = UltimateMatchPredictor(bankroll=1000.0)
    
    # Analyze next 3 days of matches
    analyses = predictor.analyze_all_matches(days_ahead=3)
    
    # Send to Telegram
    if analyses:
        predictor.send_to_telegram(analyses)
    else:
        print("⚠️  No betting recommendations available")


if __name__ == "__main__":
    main()

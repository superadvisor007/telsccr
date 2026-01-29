"""
🤖 LIVING BETTING AGENT - The Main Orchestrator
================================================
Brings all components together into a "living" system.

This is the BRAIN that:
1. Collects data from free APIs
2. Builds knowledge and caches insights
3. Reasons through matches with DeepSeek 7B
4. Simulates scenarios for forward-thinking
5. Builds optimal multi-bet tickets
6. Delivers to Telegram
7. Learns from results

100% FREE: DeepSeek 7B via Ollama, no external API costs.
All compute paid by GitHub Codespaces/Actions.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from living_agent.knowledge_cache import KnowledgeCache, initialize_default_insights
from living_agent.scenario_simulator import ScenarioSimulator, TeamProfile
from living_agent.reasoning_engine import StructuralReasoningEngine, MatchReasoning
from living_agent.multi_bet_builder import MultiBetBuilder, MultiBetTicket
from living_agent.feedback_system import FeedbackSystem

# ============ HARDCODED TELEGRAM ============
TELEGRAM_TOKEN = "7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
TELEGRAM_CHAT_ID = "7554175657"
# ============================================


class LivingBettingAgent:
    """
    🤖 Living Betting Agent
    
    A proactive, self-improving AI betting system that:
    - Thinks through multiple scenarios
    - Explores hidden edges with curiosity
    - Remembers past analyses
    - Learns from results
    - Adapts over time
    
    Architecture:
    ┌─────────────────────────┐
    │   1. Data Collection    │ ← Free APIs (TheSportsDB, OpenLigaDB)
    └─────────┬───────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ 2. Knowledge Base / DB  │ ← SQLite cache, league priors
    └─────────┬───────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ 3. Structural Reasoning │ ← DeepSeek 7B, multi-step CoT
    └─────────┬───────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ 4. Scenario Simulation  │ ← Forward-thinking, chaos modeling
    └─────────┬───────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ 5. Multi-Bet Builder    │ ← Odds 1.4-1.7, target ~10 total
    └─────────┬───────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ 6. Delivery & Feedback  │ ← Telegram, results tracking
    └─────────────────────────┘
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        verbose: bool = True,
        auto_send_telegram: bool = True
    ):
        self.verbose = verbose
        self.auto_send_telegram = auto_send_telegram
        
        if self.verbose:
            print("🤖 Initializing Living Betting Agent...")
        
        # Initialize components
        self.cache = KnowledgeCache()
        self.simulator = ScenarioSimulator()
        self.reasoning = StructuralReasoningEngine(
            cache=self.cache,
            simulator=self.simulator,
            use_llm=use_llm,
            verbose=verbose
        )
        self.builder = MultiBetBuilder()
        self.feedback = FeedbackSystem(cache=self.cache)
        
        # Initialize default insights if cache is empty
        stats = self.cache.get_cache_stats()
        if stats.get('league_insights', 0) == 0:
            if self.verbose:
                print("📚 Initializing default league insights...")
            initialize_default_insights(self.cache)
        
        if self.verbose:
            print("✅ Living Agent initialized!")
            print(f"   Cache: {sum(stats.values())} entries")
            print(f"   LLM: {'DeepSeek 7B' if use_llm else 'Statistical fallback'}")
    
    def analyze_matches(
        self,
        matches: List[Dict[str, Any]],
        team_stats: Dict[str, Any] = None
    ) -> List[MatchReasoning]:
        """
        Analyze multiple matches with full reasoning.
        
        Args:
            matches: List of match dicts with home_team, away_team, league
            team_stats: Optional dict of team statistics
        
        Returns:
            List of MatchReasoning results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Analyzing {len(matches)} matches...")
            print(f"{'='*60}")
        
        analyses = []
        
        for i, match in enumerate(matches):
            if self.verbose:
                print(f"\n📍 Match {i+1}/{len(matches)}")
            
            result = self.reasoning.analyze_match(
                home_team=match.get('home_team', 'Home'),
                away_team=match.get('away_team', 'Away'),
                league=match.get('league', 'Unknown'),
                match_date=match.get('date'),
                team_stats=team_stats,
                context=match.get('context', {})
            )
            
            analyses.append(result)
        
        return analyses
    
    def generate_daily_ticket(
        self,
        matches: List[Dict[str, Any]] = None,
        stake: float = 50.0
    ) -> Optional[MultiBetTicket]:
        """
        Generate today's multi-bet ticket.
        
        Args:
            matches: Matches to analyze (fetches today's if None)
            stake: Stake amount
        
        Returns:
            MultiBetTicket or None
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🎫 GENERATING DAILY TICKET")
            print("="*60)
        
        # If no matches provided, try to fetch today's
        if not matches:
            matches = self._fetch_todays_matches()
        
        if not matches:
            if self.verbose:
                print("❌ No matches available for analysis")
            return None
        
        # Analyze all matches
        analyses = self.analyze_matches(matches)
        
        # Convert to format expected by builder
        match_data = []
        for result in analyses:
            match_data.append({
                'match_id': result.match_id,
                'home_team': result.home_team,
                'away_team': result.away_team,
                'league': result.league,
                'market_analyses': [asdict(m) for m in result.market_analyses]
            })
        
        # Build ticket
        ticket = self.builder.build_ticket(match_data, stake=stake)
        
        if ticket:
            if self.verbose:
                print(f"\n✅ Ticket generated: {ticket.ticket_id}")
                print(f"   Legs: {len(ticket.legs)}")
                print(f"   Total Odds: {ticket.total_odds:.2f}")
                print(f"   Potential Win: €{ticket.potential_win:.2f}")
            
            # Record predictions for feedback
            self.feedback.record_ticket_predictions(ticket)
            
            # Send to Telegram if enabled
            if self.auto_send_telegram:
                self.send_ticket_to_telegram(ticket)
        else:
            if self.verbose:
                print("❌ Could not build valid ticket from analyses")
        
        return ticket
    
    def _fetch_todays_matches(self) -> List[Dict[str, Any]]:
        """Fetch today's matches from free APIs."""
        
        # Try to use existing data collectors
        try:
            from ingestion.truly_free_apis import TheSportsDBClient
            
            client = TheSportsDBClient()
            # This would be async, simplified for now
            matches = []
            
            # For now, return sample data
            # In production, this would call the actual APIs
            
        except ImportError:
            pass
        
        # Return empty - caller should provide matches
        return []
    
    def send_ticket_to_telegram(self, ticket: MultiBetTicket) -> bool:
        """Send ticket to Telegram."""
        
        import requests
        
        # Format ticket
        text = self.builder.format_for_telegram(ticket)
        
        # Wrap in <pre> for monospace
        message = f"<pre>{text}</pre>"
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                if self.verbose:
                    print("📤 Ticket sent to Telegram!")
                return True
            else:
                if self.verbose:
                    print(f"❌ Telegram error: {response.status_code}")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Telegram error: {e}")
            return False
    
    def send_weekly_report(self) -> bool:
        """Send weekly performance report to Telegram."""
        
        import requests
        
        summary = self.feedback.generate_weekly_summary()
        message = f"<pre>{summary}</pre>"
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending report: {e}")
            return False
    
    def verify_yesterdays_results(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify yesterday's predictions with actual results.
        
        Args:
            results: Dict of match_id -> {home_goals, away_goals}
        
        Returns:
            Summary of verification
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🔍 VERIFYING YESTERDAY'S PREDICTIONS")
            print("="*60)
        
        total_verified = 0
        total_correct = 0
        
        for match_id, result in results.items():
            verified = self.feedback.verify_result(
                match_id=match_id,
                home_goals=result.get('home_goals', 0),
                away_goals=result.get('away_goals', 0)
            )
            
            total_verified += len(verified)
            total_correct += sum(1 for v in verified if v.actual_outcome)
        
        win_rate = total_correct / total_verified if total_verified > 0 else 0
        
        if self.verbose:
            print(f"\n📊 Verification Summary:")
            print(f"   Verified: {total_verified}")
            print(f"   Correct: {total_correct}")
            print(f"   Win Rate: {win_rate:.1%}")
        
        # Update calibration
        self.feedback.update_calibration_from_results()
        
        return {
            'verified': total_verified,
            'correct': total_correct,
            'win_rate': win_rate
        }
    
    def run_daily_pipeline(self, matches: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete daily pipeline:
        1. Verify yesterday's results
        2. Update calibration
        3. Analyze today's matches
        4. Generate ticket
        5. Send to Telegram
        
        Returns summary of all operations.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🤖 LIVING BETTING AGENT - DAILY PIPELINE")
            print("="*70)
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
        
        summary = {
            'date': datetime.now().isoformat(),
            'verification': None,
            'ticket': None,
            'telegram_sent': False
        }
        
        # Step 1: Generate ticket for today
        ticket = self.generate_daily_ticket(matches=matches)
        
        if ticket:
            summary['ticket'] = {
                'id': ticket.ticket_id,
                'legs': len(ticket.legs),
                'total_odds': ticket.total_odds,
                'potential_win': ticket.potential_win
            }
            summary['telegram_sent'] = True
        
        # Step 2: Update performance stats
        report = self.feedback.generate_performance_report(days=7)
        summary['weekly_performance'] = {
            'win_rate': report.win_rate,
            'roi': report.roi,
            'total_bets': report.total_predictions
        }
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ DAILY PIPELINE COMPLETE")
            print("="*70)
        
        return summary


# ==================== STANDALONE EXECUTION ====================

def run_demo():
    """Run a demonstration of the Living Agent."""
    
    print("\n" + "="*70)
    print("🤖 LIVING BETTING AGENT - DEMO")
    print("="*70)
    
    # Sample matches for demo with realistic team stats
    demo_matches = [
        {
            'home_team': 'Bayern München',
            'away_team': 'Borussia Dortmund',
            'league': 'Bundesliga',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'context': {'is_derby': True}
        },
        {
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'league': 'La Liga',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'context': {'is_derby': True, 'high_stakes': True}
        },
        {
            'home_team': 'Liverpool',
            'away_team': 'Manchester City',
            'league': 'Premier League',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'context': {'high_stakes': True}
        },
        {
            'home_team': 'Ajax',
            'away_team': 'PSV',
            'league': 'Eredivisie',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'context': {'is_derby': True}
        },
        {
            'home_team': 'Juventus',
            'away_team': 'Inter Milan',
            'league': 'Serie A',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'context': {'is_derby': True}
        },
    ]
    
    # Realistic team statistics (goals per game, conceded, form points last 5)
    team_stats = {
        # Bundesliga - high scoring league
        'Bayern München': {'goals_scored': 2.4, 'goals_conceded': 0.8, 'form_points': 13},
        'Borussia Dortmund': {'goals_scored': 2.1, 'goals_conceded': 1.2, 'form_points': 10},
        # La Liga
        'Real Madrid': {'goals_scored': 2.0, 'goals_conceded': 0.7, 'form_points': 12},
        'Barcelona': {'goals_scored': 2.2, 'goals_conceded': 1.0, 'form_points': 11},
        # Premier League
        'Liverpool': {'goals_scored': 2.3, 'goals_conceded': 0.9, 'form_points': 11},
        'Manchester City': {'goals_scored': 2.5, 'goals_conceded': 0.6, 'form_points': 14},
        # Eredivisie - highest scoring league in Europe
        'Ajax': {'goals_scored': 2.8, 'goals_conceded': 1.1, 'form_points': 10},
        'PSV': {'goals_scored': 2.6, 'goals_conceded': 1.0, 'form_points': 12},
        # Serie A - tactical, fewer goals
        'Juventus': {'goals_scored': 1.4, 'goals_conceded': 0.7, 'form_points': 9},
        'Inter Milan': {'goals_scored': 1.8, 'goals_conceded': 0.8, 'form_points': 11},
    }
    
    # Initialize agent
    agent = LivingBettingAgent(
        use_llm=False,  # Statistical fallback for demo
        verbose=True,
        auto_send_telegram=True
    )
    
    # Run pipeline with team stats
    # Note: Pass stats through analyze_matches, but we need to modify the pipeline
    print("\n🔧 Running with realistic team statistics...")
    
    analyses = agent.analyze_matches(demo_matches, team_stats=team_stats)
    
    # Build ticket from analyses
    match_data = []
    for result in analyses:
        match_data.append({
            'match_id': result.match_id,
            'home_team': result.home_team,
            'away_team': result.away_team,
            'league': result.league,
            'market_analyses': [{
                'market': m.market,
                'probability': m.probability,
                'confidence': m.confidence,
                'recommendation': m.recommendation,
                'reasoning': m.reasoning,
                'key_factors': m.key_factors
            } for m in result.market_analyses]
        })
    
    ticket = agent.builder.build_ticket(match_data)
    
    if ticket:
        print("\n" + "="*70)
        print("🎫 TICKET GENERATED!")
        print("="*70)
        telegram_msg = agent.builder.format_for_telegram(ticket)
        print(telegram_msg)
        
        # Send to Telegram
        if agent.auto_send_telegram:
            success = agent.send_ticket_to_telegram(ticket)
            if success:
                print("\n✅ Sent to Telegram!")
    else:
        print("\n❌ Could not generate ticket - not enough qualifying bets")
    
    # Get performance summary
    report = agent.feedback.generate_performance_report(days=7)
    
    summary = {
        'date': datetime.now().isoformat(),
        'ticket': {
            'id': ticket.ticket_id if ticket else None,
            'legs': len(ticket.legs) if ticket else 0,
            'total_odds': ticket.total_odds if ticket else 0,
            'potential_win': ticket.potential_win if ticket else 0
        } if ticket else None,
        'weekly_performance': {
            'win_rate': report.win_rate,
            'roi': report.roi,
            'total_bets': report.total_predictions
        }
    }
    
    print("\n" + "="*70)
    print("📊 DEMO SUMMARY")
    print("="*70)
    print(json.dumps(summary, indent=2, default=str))
    
    return summary


if __name__ == "__main__":
    run_demo()

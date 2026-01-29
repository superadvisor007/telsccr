"""
🎯 Unified Betting Pipeline
===========================
Central orchestrator connecting all system components.

Data Flow:
1. Data Collection (StatsBomb, Free APIs)
2. Feature Engineering (SPADL, Structural Features)
3. Foundation Models (DeepSeek 7B reasoning)
4. Betting Logic (Multi-bet builder)
5. Delivery (Telegram)
6. Feedback (Result verification, self-improvement)

This is the main entry point for production use.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedBettingPipeline:
    """
    🎯 Unified Betting Pipeline
    
    Orchestrates the complete betting workflow:
    
    ┌─────────────────┐
    │ 1. Data Layer   │ → StatsBomb, Free APIs, Historical
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 2. Features     │ → SPADL, Structural, Tactical
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 3. Reasoning    │ → DeepSeek 7B, Scenario Simulation
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 4. Betting      │ → Value Detection, Multi-bet Building
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 5. Delivery     │ → Telegram, Reports
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ 6. Feedback     │ → Results, Calibration, Learning
    └─────────────────┘
    
    Example:
        pipeline = UnifiedBettingPipeline()
        
        # Run daily prediction workflow
        ticket = pipeline.run_daily_workflow()
        
        # Verify yesterday's results
        pipeline.verify_results()
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        verbose: bool = True
    ):
        self.config = config or self._default_config()
        self.verbose = verbose
        
        # Initialize components lazily
        self._data_sources = None
        self._feature_engine = None
        self._reasoning_engine = None
        self._bet_builder = None
        self._feedback = None
        
        if self.verbose:
            logger.info("🎯 Unified Betting Pipeline initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration."""
        return {
            # Data sources
            'use_statsbomb': True,
            'use_free_apis': True,
            'football_data_api_key': os.getenv('FOOTBALL_DATA_API_KEY', ''),
            
            # Feature engineering
            'use_spadl': True,
            'feature_cache_ttl': 24,  # hours
            
            # Reasoning
            'llm_backend': 'ollama',
            'llm_model': 'deepseek-llm:7b-chat',
            'use_scenarios': True,
            
            # Betting
            'min_leg_odds': 1.25,
            'max_leg_odds': 2.00,
            'target_total_odds': 6.0,
            'min_confidence': 0.50,
            'default_stake': 50.0,
            
            # Telegram
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            
            # Paths
            'data_dir': str(PROJECT_ROOT / 'data'),
            'models_dir': str(PROJECT_ROOT / 'models'),
        }
    
    @property
    def data_sources(self):
        """Lazy-load data sources."""
        if self._data_sources is None:
            from data_sources import StatsBombClient, FreeFootballAPIs
            
            self._data_sources = {
                'statsbomb': StatsBombClient() if self.config['use_statsbomb'] else None,
                'free_apis': FreeFootballAPIs(
                    self.config['football_data_api_key']
                ) if self.config['use_free_apis'] else None,
            }
        return self._data_sources
    
    @property
    def feature_engine(self):
        """Lazy-load feature engine."""
        if self._feature_engine is None:
            from feature_engineering import SPADLConverter, StructuralFeatureEngine
            
            self._feature_engine = {
                'spadl': SPADLConverter(),
                'structural': StructuralFeatureEngine(),
            }
        return self._feature_engine
    
    @property
    def reasoning_engine(self):
        """Lazy-load reasoning engine."""
        if self._reasoning_engine is None:
            from foundation import DeepSeekEngine, DeepSeekConfig
            
            config = DeepSeekConfig(
                backend=self.config['llm_backend'],
                model_name=self.config['llm_model'],
            )
            self._reasoning_engine = DeepSeekEngine(config)
        return self._reasoning_engine
    
    @property
    def bet_builder(self):
        """Lazy-load bet builder."""
        if self._bet_builder is None:
            from living_agent.multi_bet_builder import MultiBetBuilder
            
            self._bet_builder = MultiBetBuilder(
                min_leg_odds=self.config['min_leg_odds'],
                max_leg_odds=self.config['max_leg_odds'],
                target_total_odds=self.config['target_total_odds'],
                default_stake=self.config['default_stake'],
            )
        return self._bet_builder
    
    def collect_match_data(
        self,
        date: str = None,
        leagues: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect match data from available sources.
        
        Args:
            date: Target date (default: today)
            leagues: List of leagues to include
        
        Returns:
            List of match dictionaries
        """
        date = date or datetime.now().strftime('%Y-%m-%d')
        leagues = leagues or ['premier_league', 'bundesliga', 'la_liga', 'serie_a', 'eredivisie']
        
        matches = []
        
        # Try free APIs first
        if self.data_sources['free_apis']:
            for league in leagues:
                try:
                    league_matches = self.data_sources['free_apis'].get_upcoming_matches(
                        league=league,
                        days=1
                    )
                    matches.extend(league_matches)
                except Exception as e:
                    logger.warning(f"Failed to fetch {league}: {e}")
        
        # Deduplicate
        seen = set()
        unique_matches = []
        for match in matches:
            key = f"{match['home_team']}_{match['away_team']}_{match['date']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        logger.info(f"📊 Collected {len(unique_matches)} matches for {date}")
        return unique_matches
    
    def compute_features(
        self,
        match: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute features for a match.
        
        Uses structural features from historical data.
        """
        home_team = match['home_team']
        away_team = match['away_team']
        league = match.get('league', 'Unknown')
        
        # Get historical features
        struct_engine = self.feature_engine['structural']
        
        # For now, use simple features (would be expanded with real data)
        home_features = struct_engine.compute_from_history(
            pd.DataFrame(),  # Would have historical data
            home_team
        )
        away_features = struct_engine.compute_from_history(
            pd.DataFrame(),
            away_team
        )
        
        # Compute match features
        match_features = struct_engine.compute_match_features(
            home_features,
            away_features,
            {
                'match_id': f"{home_team}_vs_{away_team}",
                'league': league,
                'is_derby': match.get('context', {}).get('is_derby', False),
            }
        )
        
        return match_features.to_dict()
    
    def analyze_match(
        self,
        match: Dict[str, Any],
        features: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run deep reasoning on a match.
        
        Uses DeepSeek LLM for structural analysis.
        """
        home_team = match['home_team']
        away_team = match['away_team']
        league = match.get('league', 'Unknown')
        
        # Run LLM analysis
        analysis = self.reasoning_engine.analyze_match(
            home_team=home_team,
            away_team=away_team,
            league=league,
            features=features
        )
        
        return {
            'match_id': f"{home_team}_vs_{away_team}",
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'probabilities': analysis.get('probabilities', {}),
            'confidence': analysis.get('confidence', 0.5),
            'recommendations': analysis.get('recommendations', []),
            'reasoning': analysis.get('raw_response', ''),
        }
    
    def build_ticket(
        self,
        analyses: List[Dict[str, Any]],
        stake: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build multi-bet ticket from analyses.
        
        Selects best value bets and constructs accumulator.
        """
        stake = stake or self.config['default_stake']
        
        # Convert to builder format
        match_data = []
        for analysis in analyses:
            market_analyses = []
            
            probs = analysis.get('probabilities', {})
            conf = analysis.get('confidence', 0.5)
            
            for market, prob in probs.items():
                # Determine recommendation
                if prob > 0.6 and conf > 0.5:
                    rec = 'BET'
                elif prob < 0.4:
                    rec = 'AVOID'
                else:
                    rec = 'SKIP'
                
                market_analyses.append({
                    'market': market,
                    'probability': prob,
                    'confidence': conf,
                    'recommendation': rec,
                    'reasoning': f"Probability: {prob:.0%}",
                    'key_factors': analysis.get('recommendations', [])[:3]
                })
            
            match_data.append({
                'match_id': analysis['match_id'],
                'home_team': analysis['home_team'],
                'away_team': analysis['away_team'],
                'league': analysis['league'],
                'market_analyses': market_analyses
            })
        
        # Build ticket
        ticket = self.bet_builder.build_ticket(match_data, stake)
        
        if ticket:
            logger.info(f"🎫 Ticket built: {len(ticket.legs)} legs @ {ticket.total_odds:.2f}")
            return {
                'ticket_id': ticket.ticket_id,
                'legs': [
                    {
                        'home_team': leg.home_team,
                        'away_team': leg.away_team,
                        'league': leg.league,
                        'market': leg.market_display,
                        'odds': leg.odds,
                        'confidence': leg.confidence,
                    }
                    for leg in ticket.legs
                ],
                'total_odds': ticket.total_odds,
                'stake': ticket.stake,
                'potential_win': ticket.potential_win,
                'confidence': ticket.overall_confidence,
            }
        
        return None
    
    def send_to_telegram(
        self,
        ticket: Dict[str, Any]
    ) -> bool:
        """Send ticket to Telegram."""
        import requests
        
        token = self.config['telegram_token']
        chat_id = self.config['telegram_chat_id']
        
        if not token or not chat_id:
            logger.warning("Telegram not configured")
            return False
        
        # Format message
        lines = [
            "═══════════════════════════════",
            "       🎫 MULTI-BET TICKET 🎫",
            "═══════════════════════════════",
            "",
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"🎟️ {ticket['ticket_id']}",
            "🤖 Powered by DeepSeek 7B",
            "",
            "─────────────────────────────────",
        ]
        
        for i, leg in enumerate(ticket['legs'], 1):
            lines.extend([
                "",
                f"Leg {i}:",
                f"  {leg['home_team']} vs {leg['away_team']}",
                f"  📍 {leg['league']}",
                f"  ⚽ {leg['market']}",
                f"  💰 Odds: {leg['odds']:.2f}",
            ])
        
        lines.extend([
            "",
            "─────────────────────────────────",
            "",
            "📋 SUMMARY",
            f"  Total Legs:    {len(ticket['legs'])}",
            f"  Total Odds:    {ticket['total_odds']:.2f}",
            f"  Stake:         €{ticket['stake']:.2f}",
            f"  Potential Win: €{ticket['potential_win']:.2f}",
            "",
            "═══════════════════════════════",
            "  ⚠️ Gamble Responsibly",
            "═══════════════════════════════",
        ])
        
        message = f"<pre>{chr(10).join(lines)}</pre>"
        
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("📤 Ticket sent to Telegram")
                return True
            else:
                logger.error(f"Telegram error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def run_daily_workflow(
        self,
        date: str = None,
        leagues: List[str] = None,
        send_telegram: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Run complete daily prediction workflow.
        
        1. Collect matches
        2. Compute features
        3. Analyze with LLM
        4. Build ticket
        5. Send to Telegram
        
        Returns ticket if generated.
        """
        logger.info("🚀 Starting daily workflow...")
        
        # 1. Collect matches
        matches = self.collect_match_data(date, leagues)
        
        if not matches:
            logger.warning("No matches found for analysis")
            return None
        
        # 2 & 3. Analyze each match
        analyses = []
        for match in matches[:10]:  # Limit to 10 for efficiency
            try:
                features = self.compute_features(match)
                analysis = self.analyze_match(match, features)
                analyses.append(analysis)
                
                if self.verbose:
                    logger.info(f"  ✓ {match['home_team']} vs {match['away_team']}")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {match['home_team']} vs {match['away_team']}: {e}")
        
        # 4. Build ticket
        ticket = self.build_ticket(analyses)
        
        if not ticket:
            logger.info("No value bets found today")
            return None
        
        # 5. Send to Telegram
        if send_telegram:
            self.send_to_telegram(ticket)
        
        return ticket
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'data_sources': {
                    'statsbomb': self.config['use_statsbomb'],
                    'free_apis': self.config['use_free_apis'],
                },
                'llm': {
                    'backend': self.config['llm_backend'],
                    'model': self.config['llm_model'],
                    'available': self.reasoning_engine._client is not None,
                },
                'betting': {
                    'min_odds': self.config['min_leg_odds'],
                    'max_odds': self.config['max_leg_odds'],
                    'target_total': self.config['target_total_odds'],
                },
                'telegram': {
                    'configured': bool(self.config['telegram_token']),
                },
            },
            'llm_stats': self.reasoning_engine.get_stats(),
        }


# Need pandas for feature computation
try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not available - some features disabled")


def main():
    """Run the daily betting pipeline."""
    print("\n" + "="*70)
    print("🎯 UNIFIED BETTING PIPELINE")
    print("="*70)
    
    pipeline = UnifiedBettingPipeline(verbose=True)
    
    # Print system status
    status = pipeline.get_system_status()
    print("\n📊 System Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Run daily workflow
    print("\n" + "="*70)
    print("🚀 Running Daily Workflow...")
    print("="*70)
    
    ticket = pipeline.run_daily_workflow()
    
    if ticket:
        print("\n" + "="*70)
        print("🎫 TICKET GENERATED")
        print("="*70)
        print(json.dumps(ticket, indent=2))
    else:
        print("\n❌ No ticket generated")
    
    return ticket


if __name__ == "__main__":
    main()

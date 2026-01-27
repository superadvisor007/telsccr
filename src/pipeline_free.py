"""Zero-cost pipeline using Ollama + Free APIs."""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

from loguru import logger

from src.betting.engine import BettingEngine
from src.core.config import settings
from src.core.database import Match, Prediction, Tip, get_db, init_db
from src.core.logging import setup_logging
from src.features.feature_engineer import FeatureEngineer
from src.ingestion.truly_free_apis import TrulyFreeQuotaManager
from src.llm.ollama_client import OllamaLLM
from src.models.predictor import PredictionModel


class ZeroCostPipeline:
    """
    Zero-cost betting pipeline using only free components:
    - Ollama (local LLM, no API costs)
    - API-Football + iSports (free tiers)
    - SQLite (local database)
    - ChromaDB (local vector DB)
    """
    
    def __init__(self):
        setup_logging()
        init_db()
        
        # FREE: Local LLM via Ollama
        self.llm = OllamaLLM(model="llama3.2:3b")
        
        # FREE: Truly free API manager (NO PAID APIS!)
        # TheSportsDB + OpenLigaDB work WITHOUT any API keys!
        # Football-Data.org is optional (only email signup)
        self.quota_manager = TrulyFreeQuotaManager(
            football_data_key=settings.api.football_data_api_key,  # Optional
        )
        
        # Existing free components
        self.feature_engineer = FeatureEngineer()
        self.prediction_model = PredictionModel()
        self.betting_engine = BettingEngine(
            initial_bankroll=settings.betting.bankroll_initial,
            target_quote=settings.betting.target_quote,
            min_probability=settings.betting.min_probability,
            max_stake_percentage=settings.betting.max_stake_percentage,
            stop_loss_percentage=settings.betting.stop_loss_percentage,
        )
        
        # Ensure Ollama model is available
        if not self.llm.ensure_model_exists():
            logger.warning("Ollama model not available, will use fallback")
        
        logger.info("Zero-Cost Pipeline initialized (100% FREE)")
    
    async def run_daily_pipeline(self) -> List[Dict[str, Any]]:
        """
        Run the complete zero-cost daily pipeline.
        
        Returns:
            List of betting tips
        """
        logger.info("=" * 80)
        logger.info("STARTING ZERO-COST DAILY PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Fetch matches from free APIs
        matches = await self._fetch_matches()
        logger.info(f"Step 1: Fetched {len(matches)} matches")
        
        if not matches:
            logger.warning("No matches found, ending pipeline")
            return []
        
        # Step 2: Enrich with odds and stats (free APIs)
        enriched_matches = await self._enrich_matches(matches)
        logger.info(f"Step 2: Enriched {len(enriched_matches)} matches")
        
        # Step 3: Engineer features
        featured_matches = self._engineer_features(enriched_matches)
        logger.info(f"Step 3: Engineered features for {len(featured_matches)} matches")
        
        # Step 4: LLM analysis (FREE - Ollama)
        analyzed_matches = await self._llm_analysis(featured_matches)
        logger.info(f"Step 4: LLM analyzed {len(analyzed_matches)} matches")
        
        # Step 5: Statistical predictions (XGBoost - free)
        predictions = self._statistical_predictions(analyzed_matches)
        logger.info(f"Step 5: Generated {len(predictions)} predictions")
        
        # Step 6: Find value bets
        value_bets = self.betting_engine.find_value_bets(predictions)
        logger.info(f"Step 6: Found {len(value_bets)} value bets")
        
        # Step 7: Build tips (accumulators)
        tips = self._build_tips(value_bets)
        logger.info(f"Step 7: Built {len(tips)} final tips")
        
        # Step 8: Save to database (SQLite - free)
        self._save_to_database(predictions, tips)
        
        # Log API quota usage
        quota_stats = self.quota_manager.get_usage_stats()
        logger.info(f"API Quota Usage: {quota_stats}")
        
        logger.info("=" * 80)
        logger.info(f"ZERO-COST PIPELINE COMPLETE - {len(tips)} TIPS GENERATED")
        logger.info("=" * 80)
        
        return tips
    
    async def _fetch_matches(self) -> List[Dict]:
        """Fetch upcoming matches from free APIs."""
        all_matches = []
        
        # Major league IDs for API-Football
        league_ids = {
            'Premier League': 39,
            'Bundesliga': 78,
            'La Liga': 140,
            'Serie A': 135,
            'Ligue 1': 61,
        }
        
        for league_name, league_id in league_ids.items():
            try:
                fixtures = await self.quota_manager.get_fixtures(
                    league_id=league_id,
                    date=None,  # Next 3 days
                )
                
                for fixture in fixtures:
                    all_matches.append({
                        'external_id': fixture['fixture']['id'],
                        'home_team': fixture['teams']['home']['name'],
                        'away_team': fixture['teams']['away']['name'],
                        'league': league_name,
                        'date': datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00')),
                        'api_data': fixture,
                    })
                
                logger.debug(f"Fetched {len(fixtures)} fixtures from {league_name}")
            
            except Exception as e:
                logger.error(f"Failed to fetch {league_name} fixtures: {e}")
        
        return all_matches
    
    async def _enrich_matches(self, matches: List[Dict]) -> List[Dict]:
        """Enrich matches with odds and stats from free APIs."""
        enriched = []
        
        for match in matches:
            try:
                # Fetch odds (free APIs with quota pooling)
                odds_data = await self.quota_manager.get_odds(
                    fixture_id=match['external_id']
                )
                
                if odds_data:
                    match['over_1_5_odds'] = odds_data.get('over_1_5_odds', 1.5)
                    match['btts_odds'] = odds_data.get('btts_odds', 1.8)
                else:
                    # Fallback odds if API quota exhausted
                    match['over_1_5_odds'] = 1.35
                    match['btts_odds'] = 1.75
                
                # Placeholder stats (in production, add more free API calls for team stats)
                match['home_goals_per_game'] = 1.8
                match['away_goals_per_game'] = 1.5
                match['home_goals_conceded_per_game'] = 1.2
                match['away_goals_conceded_per_game'] = 1.4
                match['home_clean_sheets'] = 30
                match['away_clean_sheets'] = 25
                match['home_form_ppg'] = 1.8
                match['away_form_ppg'] = 1.5
                match['home_btts_rate'] = 60
                match['away_btts_rate'] = 55
                match['h2h_avg_goals'] = 2.8
                match['h2h_over_1_5_rate'] = 75
                
                enriched.append(match)
            
            except Exception as e:
                logger.error(f"Failed to enrich match {match['home_team']} vs {match['away_team']}: {e}")
        
        return enriched
    
    def _engineer_features(self, matches: List[Dict]) -> List[Dict]:
        """Engineer ML features from match data."""
        featured = []
        
        for match in matches:
            features = self.feature_engineer.engineer_features(match)
            match['features'] = features
            featured.append(match)
        
        return featured
    
    async def _llm_analysis(self, matches: List[Dict]) -> List[Dict]:
        """Analyze matches using free local LLM (Ollama)."""
        analyzed = []
        
        for match in matches:
            try:
                # FREE LLM analysis (Ollama, runs locally)
                llm_result = await self.llm.analyze_match(
                    match_data=match,
                    market='over_1_5'
                )
                
                match['llm_over_1_5'] = llm_result
                
                # Also analyze BTTS
                llm_btts = await self.llm.analyze_match(
                    match_data=match,
                    market='btts'
                )
                
                match['llm_btts'] = llm_btts
                
                analyzed.append(match)
            
            except Exception as e:
                logger.error(f"LLM analysis failed for {match['home_team']} vs {match['away_team']}: {e}")
        
        return analyzed
    
    def _statistical_predictions(self, matches: List[Dict]) -> List[Dict]:
        """Generate XGBoost predictions (free, local inference)."""
        predictions = []
        
        for match in matches:
            features = match.get('features')
            
            if features is None:
                continue
            
            # XGBoost predictions (free, local)
            xgb_over_1_5 = self.prediction_model.predict_over_1_5(features)
            xgb_btts = self.prediction_model.predict_btts(features)
            
            # Ensemble: Blend LLM + XGBoost
            llm_over_prob = match.get('llm_over_1_5', {}).get('probability', 0.5)
            xgb_over_prob = xgb_over_1_5.get('probability', 0.5)
            
            final_over_prob = 0.6 * llm_over_prob + 0.4 * xgb_over_prob
            
            llm_btts_prob = match.get('llm_btts', {}).get('probability', 0.5)
            xgb_btts_prob = xgb_btts.get('probability', 0.5)
            
            final_btts_prob = 0.6 * llm_btts_prob + 0.4 * xgb_btts_prob
            
            predictions.append({
                'match': f"{match['home_team']} vs {match['away_team']}",
                'league': match['league'],
                'date': match['date'],
                'over_1_5_probability': final_over_prob,
                'over_1_5_odds': match['over_1_5_odds'],
                'btts_probability': final_btts_prob,
                'btts_odds': match['btts_odds'],
                'llm_reasoning': match.get('llm_over_1_5', {}).get('reasoning', ''),
                'key_factors': match.get('llm_over_1_5', {}).get('key_factors', []),
            })
        
        return predictions
    
    def _build_tips(self, value_bets: List[Dict]) -> List[Dict]:
        """Build final tips (singles and accumulators)."""
        if not value_bets:
            return []
        
        tips = []
        
        # Singles (top 5)
        for bet in value_bets[:5]:
            tips.append({
                'type': 'single',
                'matches': [bet['match']],
                'market': bet['market'],
                'odds': bet['odds'],
                'probability': bet['probability'],
                'stake_pct': 2.0,
                'reasoning': bet.get('reasoning', ''),
                'key_factors': bet.get('key_factors', []),
            })
        
        # Doubles (2-leg accumulators targeting ~1.40)
        if len(value_bets) >= 2:
            for i in range(min(3, len(value_bets) - 1)):
                for j in range(i + 1, min(i + 3, len(value_bets))):
                    bet1 = value_bets[i]
                    bet2 = value_bets[j]
                    
                    combined_odds = bet1['odds'] * bet2['odds']
                    
                    if 1.35 <= combined_odds <= 1.50:
                        tips.append({
                            'type': 'accumulator',
                            'matches': [bet1['match'], bet2['match']],
                            'market': f"{bet1['market']} + {bet2['market']}",
                            'odds': combined_odds,
                            'probability': bet1['probability'] * bet2['probability'],
                            'stake_pct': 2.0,
                            'reasoning': f"{bet1['match']}: {bet1.get('reasoning', 'N/A')} | {bet2['match']}: {bet2.get('reasoning', 'N/A')}",
                        })
        
        return tips[:10]  # Top 10 tips
    
    def _save_to_database(self, predictions: List[Dict], tips: List[Dict]):
        """Save to SQLite database (free, local)."""
        logger.info("Saving to SQLite database...")
        # TODO: Implement database persistence
        logger.debug(f"Would save {len(predictions)} predictions and {len(tips)} tips")


if __name__ == "__main__":
    pipeline = ZeroCostPipeline()
    tips = asyncio.run(pipeline.run_daily_pipeline())
    
    print("\n" + "=" * 80)
    print("ZERO-COST DAILY TIPS")
    print("=" * 80)
    
    for i, tip in enumerate(tips, 1):
        print(f"\nTip #{i}: {tip['type'].upper()}")
        print(f"  Matches: {', '.join(tip['matches'])}")
        print(f"  Market: {tip['market']}")
        print(f"  Odds: {tip['odds']:.2f}")
        print(f"  Probability: {tip['probability']:.1%}")
        print(f"  Stake: {tip['stake_pct']:.1f}%")
        if tip.get('key_factors'):
            print(f"  Key Factors: {', '.join(tip['key_factors'][:2])}")

"""Main pipeline orchestrator."""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

from loguru import logger

from src.betting.engine import BettingEngine
from src.core.config import settings
from src.core.database import Match, Prediction, Tip, get_db, init_db
from src.core.logging import setup_logging
from src.features.feature_engineer import FeatureEngineer
from src.ingestion.odds_client import OddsAPIClient
from src.ingestion.stats_client import FootballDataClient
from src.ingestion.weather_client import WeatherAPIClient
from src.llm.analyzer import LLMAnalyzer
from src.models.predictor import PredictionModel
from src.rl.agent import RLStakingAgent
from src.rag.retriever import BettingMemoryRAG
from src.meta.learner import MetaLearner, ContextualPerformanceTracker
from src.monitoring.mlflow_tracker import MLflowTracker


class Pipeline:
    """Main data and prediction pipeline."""
    
    def __init__(self):
        setup_logging()
        init_db()
        
        # Initialize components
        self.odds_client = OddsAPIClient()
        self.weather_client = WeatherAPIClient()
        self.stats_client = FootballDataClient()
        self.feature_engineer = FeatureEngineer()
        self.llm_analyzer = LLMAnalyzer()
        self.prediction_model = PredictionModel()
        self.betting_engine = BettingEngine(
            initial_bankroll=settings.betting.bankroll_initial,
            target_quote=settings.betting.target_quote,
            min_probability=settings.betting.min_probability,
            max_stake_percentage=settings.betting.max_stake_percentage,
            stop_loss_percentage=settings.betting.stop_loss_percentage,
        )
        
        # Advanced ML components
        self.rl_agent = RLStakingAgent()
        self.rag_system = BettingMemoryRAG(get_db())
        self.meta_learner = MetaLearner()
        self.performance_tracker = ContextualPerformanceTracker(get_db())
        self.mlflow_tracker = MLflowTracker()
        
        # Load trained models
        try:
            self.rl_agent.load()
            self.meta_learner.load()
            logger.info("Advanced models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load advanced models: {e}")
        
        logger.info("Pipeline initialized with advanced ML components")
    
    async def run_daily_pipeline(self) -> List[Dict[str, Any]]:
        """
        Run the complete daily pipeline.
        
        Returns:
            List of betting tips
        """
        logger.info("=" * 50)
        logger.info("STARTING DAILY PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Step 1: Fetch upcoming matches
            logger.info("Step 1: Fetching upcoming matches...")
            matches = await self._fetch_matches()
            logger.info(f"Found {len(matches)} matches")
            
            if not matches:
                logger.warning("No matches found")
                return []
            
            # Step 2: Enrich with stats and weather
            logger.info("Step 2: Enriching match data...")
            enriched_matches = await self._enrich_matches(matches)
            logger.info(f"Enriched {len(enriched_matches)} matches")
            
            # Step 3: Engineer features
            logger.info("Step 3: Engineering features...")
            featured_matches = self._engineer_features(enriched_matches)
            
            # Step 4: LLM analysis
            logger.info("Step 4: Running LLM analysis...")
            analyzed_matches = await self._llm_analysis(featured_matches)
            
            # Step 5: Statistical model predictions
            logger.info("Step 5: Running statistical models...")
            predictions = self._statistical_predictions(analyzed_matches)
            
            # Step 6: Ensemble and value detection
            logger.info("Step 6: Finding value bets...")
            value_bets = self.betting_engine.find_value_bets(predictions)
            
            if not value_bets:
                logger.warning("No value bets found")
                return []
            
            # Step 7: Build accumulators
            logger.info("Step 7: Building accumulators...")
            tips = self._build_tips(value_bets)
            
            # Step 8: Save to database
            logger.info("Step 8: Saving to database...")
            self._save_to_database(predictions, tips)
            
            logger.info("=" * 50)
            logger.info(f"PIPELINE COMPLETED - Generated {len(tips)} tips")
            logger.info("=" * 50)
            
            return tips
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return []
        finally:
            await self.odds_client.close()
            await self.weather_client.close()
            await self.stats_client.close()
    
    async def _fetch_matches(self) -> List[Dict[str, Any]]:
        """Fetch upcoming matches with odds."""
        # Get target leagues
        leagues = [
            "soccer_germany_bundesliga",
            "soccer_netherlands_eredivisie",
            "soccer_epl",
            "soccer_spain_la_liga",
        ]
        
        matches = await self.odds_client.get_upcoming_matches(leagues, hours_ahead=72)
        return matches
    
    async def _enrich_matches(self, matches: List[Dict]) -> List[Dict]:
        """Enrich matches with stats and weather."""
        enriched = []
        
        for match in matches:
            try:
                # Parse odds
                odds = self.odds_client.parse_odds(match)
                
                # Skip if no Over 1.5 odds
                if not odds.get("over_1_5"):
                    continue
                
                # Get weather (simplified - extract city from team name)
                city = match["home_team"].split()[-1]  # Last word often is city
                match_time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
                weather = await self.weather_client.get_forecast(city, match_time)
                
                if weather:
                    weather_impact = self.weather_client.assess_weather_impact(weather)
                else:
                    weather_impact = None
                
                # For MVP, we'll use simplified stats
                # In production, you'd fetch actual team IDs and stats
                home_stats = {"goals_per_game": 1.5, "matches_played": 5, "over_1_5_percentage": 70}
                away_stats = {"goals_per_game": 1.3, "matches_played": 5, "over_1_5_percentage": 65}
                h2h_stats = {"matches": 3, "avg_goals": 2.5, "btts_rate": 60}
                
                enriched.append({
                    "match_data": {
                        "external_id": match["id"],
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "league": "Unknown",  # Would extract from sport_key
                        "match_date": match_time,
                        "odds": odds,
                    },
                    "home_stats": home_stats,
                    "away_stats": away_stats,
                    "h2h_stats": h2h_stats,
                    "weather": weather_impact,
                })
                
            except Exception as e:
                logger.error(f"Failed to enrich match: {e}")
                continue
        
        return enriched
    
    def _engineer_features(self, matches: List[Dict]) -> List[Dict]:
        """Engineer features for all matches."""
        for match in matches:
            features = self.feature_engineer.engineer_features(
                match_data=match["match_data"],
                home_stats=match["home_stats"],
                away_stats=match["away_stats"],
                h2h_stats=match["h2h_stats"],
                weather=match.get("weather"),
            )
            match["features"] = features
        
        return matches
    
    async def _llm_analysis(self, matches: List[Dict]) -> List[Dict]:
        """Run LLM analysis on all matches."""
        for match in matches:
            analysis = self.llm_analyzer.analyze_match(
                match_data=match["match_data"],
                features=match["features"],
                home_stats=match["home_stats"],
                away_stats=match["away_stats"],
                h2h_stats=match["h2h_stats"],
                weather=match.get("weather"),
            )
            match["llm_analysis"] = analysis
        
        return matches
    
    def _statistical_predictions(self, matches: List[Dict]) -> List[Dict]:
        """Generate statistical model predictions and ensemble."""
        predictions = []
        
        for match in matches:
            # Create feature vector
            feature_vector = self.feature_engineer.create_feature_vector(match["features"])
            
            # LLM probabilities
            llm = match["llm_analysis"]
            
            # Ensemble prediction
            ensemble = self.prediction_model.ensemble_predict(
                X=feature_vector,
                llm_over_1_5_prob=llm["over_1_5_probability"],
                llm_btts_prob=llm["btts_probability"],
                llm_confidence=llm["confidence_score"],
            )
            
            # Combine all info
            prediction = {
                "match_id": match["match_data"]["external_id"],
                "home_team": match["match_data"]["home_team"],
                "away_team": match["match_data"]["away_team"],
                "league": match["match_data"]["league"],
                "match_date": match["match_data"]["match_date"],
                "over_1_5_odds": match["match_data"]["odds"].get("over_1_5"),
                "btts_odds": match["match_data"]["odds"].get("btts_yes"),
                "over_1_5_probability": ensemble["over_1_5_probability"],
                "btts_probability": ensemble["btts_probability"],
                "confidence_score": llm["confidence_score"],
                "key_factors": llm["key_factors"],
                "reasoning": llm["reasoning"],
                "recommendation": llm["recommendation"],
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def _build_tips(self, value_bets: List[Dict]) -> List[Dict]:
        """Build betting tips from value bets."""
        tips = []
        
        # Try to build a double (2-leg accumulator)
        accumulator = self.betting_engine.build_accumulator(value_bets, num_selections=2)
        
        if accumulator:
            stake = self.betting_engine.calculate_stake(accumulator)
            bet = self.betting_engine.place_bet(accumulator, stake)
            tips.append(bet)
        
        # Try a treble if enough value bets
        if len(value_bets) >= 3:
            accumulator = self.betting_engine.build_accumulator(value_bets, num_selections=3)
            if accumulator:
                stake = self.betting_engine.calculate_stake(accumulator)
                bet = self.betting_engine.place_bet(accumulator, stake)
                tips.append(bet)
        
        return tips
    
    def _save_to_database(self, predictions: List[Dict], tips: List[Dict]) -> None:
        """Save predictions and tips to database."""
        # TODO: Implement database saving
        logger.info("Database saving not yet implemented (MVP)")
        pass


async def main():
    """Main entry point."""
    pipeline = Pipeline()
    tips = await pipeline.run_daily_pipeline()
    
    if tips:
        logger.info("✅ Pipeline completed successfully")
        for tip in tips:
            logger.info(f"Tip: {tip['accumulator']['total_odds']:.2f} odds, €{tip['stake']:.2f} stake")
    else:
        logger.warning("⚠️ No tips generated")


if __name__ == "__main__":
    asyncio.run(main())

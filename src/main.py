"""Main application orchestrator for Swiss Soccer Tips Bot."""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict

from dotenv import load_dotenv

try:
    from .database import Database, SubscriptionTier, PredictionResult
    from .api_football import APIFootballClient
    from .prediction_engine import PredictionEngine
    from .payment_handler import StripePaymentHandler
    from .bot import TelegramBot
except ImportError:
    from database import Database, SubscriptionTier, PredictionResult
    from api_football import APIFootballClient
    from prediction_engine import PredictionEngine
    from payment_handler import StripePaymentHandler
    from bot import TelegramBot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SoccerTipsBot:
    """Main orchestrator for the soccer tips bot."""

    def __init__(self):
        """Initialize the bot application."""
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.db = Database(self.config["database_path"])
        self.api_football = APIFootballClient(
            self.config["api_football_key"],
            self.config["api_football_base_url"]
        )
        self.prediction_engine = PredictionEngine(
            self.config["groq_api_key"],
            self.config["groq_model"]
        )
        self.payment_handler = StripePaymentHandler(
            self.config["stripe_api_key"],
            self.config["stripe_webhook_secret"],
            self.config["stripe_price_basic"],
            self.config["stripe_price_premium"]
        )
        self.telegram_bot = TelegramBot(
            self.config["telegram_bot_token"],
            self.db,
            self.payment_handler,
            self.config["telegram_channel_id"],
            self.config["bot_admin_ids"]
        )

    def _load_config(self) -> Dict:
        """Load configuration from environment variables."""
        return {
            # Telegram
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "telegram_channel_id": os.getenv("TELEGRAM_CHANNEL_ID"),
            
            # Database
            "database_path": os.getenv("DATABASE_PATH", "./data/bot.db"),
            
            # Stripe
            "stripe_api_key": os.getenv("STRIPE_API_KEY"),
            "stripe_webhook_secret": os.getenv("STRIPE_WEBHOOK_SECRET"),
            "stripe_price_basic": os.getenv("STRIPE_PRICE_BASIC"),
            "stripe_price_premium": os.getenv("STRIPE_PRICE_PREMIUM"),
            
            # API Football
            "api_football_key": os.getenv("API_FOOTBALL_KEY"),
            "api_football_base_url": os.getenv(
                "API_FOOTBALL_BASE_URL",
                "https://v3.football.api-sports.io"
            ),
            
            # Groq
            "groq_api_key": os.getenv("GROQ_API_KEY"),
            "groq_model": os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            
            # Swiss League
            "swiss_league_id": int(os.getenv("SWISS_LEAGUE_ID", "207")),
            "swiss_season": int(os.getenv("SWISS_SEASON", "2024")),
            
            # Bot settings
            "bot_admin_ids": [
                int(id.strip())
                for id in os.getenv("BOT_ADMIN_IDS", "").split(",")
                if id.strip()
            ],
            "free_tier_predictions": int(os.getenv("FREE_TIER_PREDICTIONS_PER_DAY", "1")),
            "basic_tier_predictions": int(os.getenv("BASIC_TIER_PREDICTIONS_PER_DAY", "5")),
            "premium_tier_predictions": int(os.getenv("PREMIUM_TIER_PREDICTIONS_PER_DAY", "10")),
        }

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Swiss Soccer Tips Bot...")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config["database_path"]), exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Connect to database
        await self.db.connect()
        
        logger.info("Bot initialized successfully")

    async def shutdown(self):
        """Shutdown all components."""
        logger.info("Shutting down bot...")
        await self.telegram_bot.stop()
        await self.db.close()
        logger.info("Bot shutdown complete")

    async def fetch_and_generate_predictions(self):
        """Fetch matches and generate predictions (daily cron job)."""
        logger.info("Starting daily prediction generation...")
        
        try:
            # Fetch upcoming matches for the next 7 days
            matches = await self.api_football.get_upcoming_matches(
                self.config["swiss_league_id"],
                self.config["swiss_season"],
                days_ahead=7
            )
            
            logger.info(f"Found {len(matches)} upcoming matches")
            
            # Generate predictions for each match
            for fixture in matches:
                await self._generate_match_prediction(fixture)
            
            logger.info("Prediction generation completed")
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")

    async def _generate_match_prediction(self, fixture: Dict):
        """Generate prediction for a single match."""
        try:
            # Parse fixture data
            match_data = self.api_football.parse_fixture(fixture)
            
            # Parse match date properly
            from dateutil.parser import parse as parse_date
            try:
                match_date = parse_date(match_data["match_date"])
            except Exception as e:
                logger.error(f"Failed to parse match date: {e}")
                return
            
            # Check if prediction already exists
            existing_predictions = await self.db.get_predictions_for_date(
                match_date,
                SubscriptionTier.PREMIUM
            )
            
            if any(p["match_id"] == match_data["match_id"] for p in existing_predictions):
                logger.info(f"Prediction already exists for match {match_data['match_id']}")
                return
            
            # Fetch additional data for better predictions
            team_stats = None
            h2h_data = None
            
            try:
                h2h_data = await self.api_football.get_h2h_matches(
                    match_data["home_team_id"],
                    match_data["away_team_id"],
                    last=5
                )
            except Exception as e:
                logger.warning(f"Failed to fetch H2H data: {e}")
            
            # Get learning stats
            learning_stats = await self.db.get_learning_stats()
            
            # Generate AI prediction
            prediction = await self.prediction_engine.generate_prediction(
                match_data,
                team_stats,
                h2h_data,
                learning_stats
            )
            
            # Determine tier based on confidence
            if prediction["confidence"] >= 0.8:
                tier_required = "premium"
            elif prediction["confidence"] >= 0.65:
                tier_required = "basic"
            else:
                tier_required = "free"
            
            # Store prediction in database
            prediction_id = await self.db.create_prediction(
                match_data["match_id"],
                match_data["league_id"],
                match_data["home_team"],
                match_data["away_team"],
                match_date,
                prediction["prediction"],
                prediction["confidence"],
                prediction["reasoning"],
                tier_required
            )
            
            # Store learning data
            await self.db.store_learning_data(
                prediction_id,
                match_data["match_id"],
                str({"team_stats": team_stats, "h2h": h2h_data}),
                prediction["prediction"],
                confidence=prediction["confidence"]
            )
            
            logger.info(f"Generated prediction {prediction_id} for match {match_data['match_id']}")
            
        except Exception as e:
            logger.error(f"Failed to generate prediction for fixture: {e}")

    async def post_daily_predictions(self):
        """Post daily predictions to Telegram channel."""
        logger.info("Posting daily predictions...")
        
        try:
            today = datetime.now()
            
            # Get predictions for each tier
            for tier in [SubscriptionTier.FREE, SubscriptionTier.BASIC, SubscriptionTier.PREMIUM]:
                predictions = await self.db.get_predictions_for_date(today, tier)
                
                for pred in predictions:
                    # Only post predictions that haven't been posted
                    if pred["result"] == "pending":
                        await self.telegram_bot.post_prediction_to_channel(
                            {
                                "match": {
                                    "home_team": pred["home_team"],
                                    "away_team": pred["away_team"],
                                    "match_date": pred["match_date"],
                                    "venue": "N/A"
                                },
                                "prediction": {
                                    "prediction": pred["prediction"],
                                    "confidence": pred["confidence"],
                                    "reasoning": pred["reasoning"],
                                    "key_factors": [],
                                    "suggested_bet": pred["prediction"],
                                    "value_assessment": "medium"
                                }
                            },
                            pred["tier_required"]
                        )
                        
                        # Rate limiting
                        await asyncio.sleep(2)
            
            logger.info("Daily predictions posted")
            
        except Exception as e:
            logger.error(f"Failed to post predictions: {e}")

    async def update_prediction_results(self):
        """Update prediction results with actual match outcomes."""
        logger.info("Updating prediction results...")
        
        try:
            # Get predictions from the last 7 days
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            # This would need to fetch finished matches and update predictions
            # Implementation depends on API-Football's finished matches endpoint
            
            logger.info("Prediction results updated")
            
        except Exception as e:
            logger.error(f"Failed to update results: {e}")

    async def cleanup_expired_subscriptions(self):
        """Remove expired user subscriptions."""
        logger.info("Cleaning up expired subscriptions...")
        
        try:
            count = await self.db.downgrade_expired_users()
            logger.info(f"Downgraded {count} expired subscriptions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup subscriptions: {e}")

    async def run_bot(self):
        """Run the Telegram bot."""
        logger.info("Starting Telegram bot...")
        await self.telegram_bot.start()

    async def run_daily_tasks(self):
        """Run all daily tasks (for cron job)."""
        await self.initialize()
        
        try:
            # Fetch and generate predictions
            await self.fetch_and_generate_predictions()
            
            # Post predictions to channel
            await self.post_daily_predictions()
            
            # Update results from finished matches
            await self.update_prediction_results()
            
            # Cleanup expired subscriptions
            await self.cleanup_expired_subscriptions()
            
        finally:
            await self.shutdown()


async def main():
    """Main entry point."""
    bot = SoccerTipsBot()
    await bot.initialize()
    
    try:
        # Run the bot
        await bot.run_bot()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

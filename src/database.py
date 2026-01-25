"""Database models and management for the Swiss Soccer Tips Bot."""
import aiosqlite
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    """Subscription tier levels."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"


class PredictionResult(Enum):
    """Prediction outcome status."""
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    VOID = "void"


class Database:
    """Database manager for SQLite operations."""

    def __init__(self, db_path: str):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Connect to database and create tables if needed."""
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        await self.conn.executescript("""
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                subscription_tier TEXT DEFAULT 'free',
                subscription_expires_at TIMESTAMP,
                stripe_customer_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Subscriptions table (payment history)
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tier TEXT NOT NULL,
                stripe_subscription_id TEXT UNIQUE,
                stripe_payment_intent_id TEXT,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'CHF',
                status TEXT DEFAULT 'active',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Predictions table
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                league_id INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                match_date TIMESTAMP NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                tier_required TEXT DEFAULT 'free',
                result TEXT DEFAULT 'pending',
                actual_outcome TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Learning data table (for AI improvement)
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                match_id INTEGER NOT NULL,
                features TEXT NOT NULL,
                prediction TEXT NOT NULL,
                actual_result TEXT,
                was_correct INTEGER DEFAULT 0,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_tier, subscription_expires_at);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(match_date);
            CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result);
            CREATE INDEX IF NOT EXISTS idx_learning_prediction ON learning_data(prediction_id);
        """)
        await self.conn.commit()
        logger.info("Database tables created/verified")

    # User management methods
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        async with self.conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def create_or_update_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> Dict:
        """Create or update user."""
        existing = await self.get_user(user_id)
        
        if existing:
            await self.conn.execute(
                """UPDATE users SET username = ?, first_name = ?, last_name = ?, 
                   updated_at = CURRENT_TIMESTAMP WHERE user_id = ?""",
                (username, first_name, last_name, user_id)
            )
        else:
            await self.conn.execute(
                """INSERT INTO users (user_id, username, first_name, last_name) 
                   VALUES (?, ?, ?, ?)""",
                (user_id, username, first_name, last_name)
            )
        
        await self.conn.commit()
        return await self.get_user(user_id)

    async def update_user_subscription(
        self,
        user_id: int,
        tier: SubscriptionTier,
        expires_at: datetime,
        stripe_customer_id: Optional[str] = None
    ):
        """Update user subscription tier."""
        await self.conn.execute(
            """UPDATE users SET subscription_tier = ?, subscription_expires_at = ?, 
               stripe_customer_id = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE user_id = ?""",
            (tier.value, expires_at, stripe_customer_id, user_id)
        )
        await self.conn.commit()
        logger.info(f"Updated subscription for user {user_id}: {tier.value}")

    async def get_expired_subscriptions(self) -> List[Dict]:
        """Get users with expired subscriptions."""
        async with self.conn.execute(
            """SELECT * FROM users 
               WHERE subscription_tier != 'free' 
               AND subscription_expires_at < CURRENT_TIMESTAMP"""
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def downgrade_expired_users(self):
        """Downgrade expired subscriptions to free tier."""
        expired = await self.get_expired_subscriptions()
        for user in expired:
            await self.conn.execute(
                """UPDATE users SET subscription_tier = 'free', 
                   updated_at = CURRENT_TIMESTAMP WHERE user_id = ?""",
                (user['user_id'],)
            )
        await self.conn.commit()
        logger.info(f"Downgraded {len(expired)} expired subscriptions")
        return len(expired)

    # Subscription methods
    async def create_subscription(
        self,
        user_id: int,
        tier: str,
        amount: float,
        expires_at: datetime,
        stripe_subscription_id: Optional[str] = None,
        stripe_payment_intent_id: Optional[str] = None
    ):
        """Create subscription record."""
        await self.conn.execute(
            """INSERT INTO subscriptions 
               (user_id, tier, stripe_subscription_id, stripe_payment_intent_id, 
                amount, expires_at) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, tier, stripe_subscription_id, stripe_payment_intent_id, 
             amount, expires_at)
        )
        await self.conn.commit()
        logger.info(f"Created subscription for user {user_id}: {tier}")

    # Prediction methods
    async def create_prediction(
        self,
        match_id: int,
        league_id: int,
        home_team: str,
        away_team: str,
        match_date: datetime,
        prediction: str,
        confidence: float,
        reasoning: str,
        tier_required: str = "free"
    ) -> int:
        """Create a new prediction."""
        cursor = await self.conn.execute(
            """INSERT INTO predictions 
               (match_id, league_id, home_team, away_team, match_date, 
                prediction, confidence, reasoning, tier_required) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (match_id, league_id, home_team, away_team, match_date,
             prediction, confidence, reasoning, tier_required)
        )
        await self.conn.commit()
        prediction_id = cursor.lastrowid
        logger.info(f"Created prediction {prediction_id} for match {match_id}")
        return prediction_id

    async def get_predictions_for_date(
        self,
        date: datetime,
        tier: SubscriptionTier = SubscriptionTier.FREE
    ) -> List[Dict]:
        """Get predictions for a specific date and tier."""
        tier_levels = {
            SubscriptionTier.FREE: ["free"],
            SubscriptionTier.BASIC: ["free", "basic"],
            SubscriptionTier.PREMIUM: ["free", "basic", "premium"]
        }
        tiers = tier_levels[tier]
        placeholders = ",".join("?" * len(tiers))
        
        async with self.conn.execute(
            f"""SELECT * FROM predictions 
                WHERE DATE(match_date) = DATE(?) 
                AND tier_required IN ({placeholders})
                ORDER BY match_date""",
            (date, *tiers)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_prediction_result(
        self,
        prediction_id: int,
        result: PredictionResult,
        actual_outcome: Optional[str] = None
    ):
        """Update prediction result after match completion."""
        await self.conn.execute(
            """UPDATE predictions SET result = ?, actual_outcome = ?, 
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (result.value, actual_outcome, prediction_id)
        )
        await self.conn.commit()
        logger.info(f"Updated prediction {prediction_id} result: {result.value}")

    # Learning data methods
    async def store_learning_data(
        self,
        prediction_id: int,
        match_id: int,
        features: str,
        prediction: str,
        actual_result: Optional[str] = None,
        was_correct: bool = False,
        confidence: Optional[float] = None
    ):
        """Store learning data for AI improvement."""
        await self.conn.execute(
            """INSERT INTO learning_data 
               (prediction_id, match_id, features, prediction, actual_result, 
                was_correct, confidence) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, match_id, features, prediction, actual_result,
             1 if was_correct else 0, confidence)
        )
        await self.conn.commit()

    async def get_learning_stats(self) -> Dict:
        """Get learning statistics for model improvement."""
        async with self.conn.execute(
            """SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                AVG(confidence) as avg_confidence
               FROM learning_data
               WHERE actual_result IS NOT NULL"""
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {}

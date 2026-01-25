"""Telegram bot handlers using aiogram."""
import logging
from datetime import datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

try:
    from .database import Database, SubscriptionTier
    from .payment_handler import StripePaymentHandler, PaymentTier
except ImportError:
    from database import Database, SubscriptionTier
    from payment_handler import StripePaymentHandler, PaymentTier

logger = logging.getLogger(__name__)


class SubscriptionStates(StatesGroup):
    """FSM states for subscription process."""
    choosing_tier = State()


class TelegramBot:
    """Telegram bot for Swiss soccer tips."""

    def __init__(
        self,
        token: str,
        database: Database,
        payment_handler: StripePaymentHandler,
        channel_id: str,
        admin_ids: list
    ):
        """Initialize Telegram bot.
        
        Args:
            token: Telegram bot token
            database: Database instance
            payment_handler: Payment handler instance
            channel_id: Channel ID for posting tips
            admin_ids: List of admin user IDs
        """
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.db = database
        self.payment = payment_handler
        self.channel_id = channel_id
        self.admin_ids = admin_ids
        
        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register bot command and message handlers."""
        # Command handlers
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_help, Command("help"))
        self.dp.message.register(self.cmd_subscribe, Command("subscribe"))
        self.dp.message.register(self.cmd_status, Command("status"))
        self.dp.message.register(self.cmd_cancel, Command("cancel"))
        
        # Admin commands
        self.dp.message.register(self.cmd_admin_stats, Command("stats"))
        
        # Callback handlers
        self.dp.callback_query.register(
            self.callback_subscribe_tier,
            F.data.startswith("subscribe_")
        )

    async def cmd_start(self, message: types.Message):
        """Handle /start command."""
        user = message.from_user
        
        # Create or update user in database
        await self.db.create_or_update_user(
            user.id,
            user.username,
            user.first_name,
            user.last_name
        )
        
        welcome_text = f"""
🏆 **Welcome to Swiss Soccer Tips Bot!**

Hello {user.first_name}! I provide AI-powered predictions for Swiss Super League matches.

**Available Tiers:**
🆓 **Free** - 1 prediction per day
💎 **Basic** (CHF 9.90/month) - 5 predictions per day
⭐ **Premium** (CHF 19.90/month) - 10 predictions per day + detailed analysis

Use /subscribe to upgrade your subscription.
Use /help to see all available commands.

Join our channel: {self.channel_id}
"""
        
        await message.answer(welcome_text, parse_mode="Markdown")

    async def cmd_help(self, message: types.Message):
        """Handle /help command."""
        help_text = """
📚 **Available Commands:**

/start - Start the bot and see welcome message
/help - Show this help message
/subscribe - Subscribe to paid tiers
/status - Check your subscription status
/cancel - Cancel your subscription

**How it works:**
1. Daily AI-powered predictions for Swiss Super League
2. Subscribe to get more predictions and detailed analysis
3. Predictions posted in our channel: {channel}

**Payment Methods:**
💳 Credit/Debit Card
🇨🇭 TWINT (via Stripe)

For support, contact @your_support
"""
        await message.answer(help_text.format(channel=self.channel_id), parse_mode="Markdown")

    async def cmd_subscribe(self, message: types.Message):
        """Handle /subscribe command."""
        user_data = await self.db.get_user(message.from_user.id)
        
        if not user_data:
            await message.answer("Please use /start first.")
            return
        
        # Create subscription keyboard
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="💎 Basic (CHF 9.90/month)",
                    callback_data="subscribe_basic"
                )
            ],
            [
                InlineKeyboardButton(
                    text="⭐ Premium (CHF 19.90/month)",
                    callback_data="subscribe_premium"
                )
            ]
        ])
        
        subscription_text = """
💳 **Choose Your Subscription Tier:**

**💎 Basic (CHF 9.90/month)**
- 5 predictions per day
- Access to basic analysis
- Match statistics

**⭐ Premium (CHF 19.90/month)**
- 10 predictions per day
- Detailed AI analysis
- Advanced statistics
- H2H data
- Priority support

Select a tier to continue:
"""
        
        await message.answer(
            subscription_text,
            reply_markup=keyboard,
            parse_mode="Markdown"
        )

    async def callback_subscribe_tier(
        self,
        callback: types.CallbackQuery,
        state: FSMContext
    ):
        """Handle subscription tier selection."""
        tier_name = callback.data.split("_")[1]  # "subscribe_basic" -> "basic"
        user_id = callback.from_user.id
        
        # Get or create Stripe customer
        user_data = await self.db.get_user(user_id)
        
        if not user_data.get("stripe_customer_id"):
            customer_id = await self.payment.create_customer(
                user_id,
                callback.from_user.username,
                callback.from_user.first_name
            )
            await self.db.create_or_update_user(
                user_id,
                callback.from_user.username,
                callback.from_user.first_name,
                callback.from_user.last_name
            )
        else:
            customer_id = user_data["stripe_customer_id"]
        
        # Create payment link
        tier = PaymentTier.BASIC if tier_name == "basic" else PaymentTier.PREMIUM
        
        try:
            payment_url = await self.payment.create_payment_link(tier, user_id)
            
            await callback.message.answer(
                f"""
✅ **Payment Link Created**

Click the link below to complete your {tier_name.title()} subscription:

{payment_url}

After payment, your subscription will be activated automatically.
""",
                parse_mode="Markdown"
            )
            
            await callback.answer()
            
        except Exception as e:
            logger.error(f"Failed to create payment link: {e}")
            await callback.message.answer(
                "❌ Failed to create payment link. Please try again later."
            )
            await callback.answer()

    async def cmd_status(self, message: types.Message):
        """Handle /status command."""
        user_data = await self.db.get_user(message.from_user.id)
        
        if not user_data:
            await message.answer("Please use /start first.")
            return
        
        tier = user_data.get("subscription_tier", "free")
        expires_at = user_data.get("subscription_expires_at")
        
        status_text = f"""
📊 **Your Subscription Status**

**Current Tier:** {tier.title()}
"""
        
        if tier != "free" and expires_at:
            expires_dt = datetime.fromisoformat(expires_at)
            days_left = (expires_dt - datetime.now()).days
            
            status_text += f"""**Expires:** {expires_dt.strftime('%Y-%m-%d')}
**Days Left:** {days_left} days
"""
        else:
            status_text += "\nYou're on the free tier. Use /subscribe to upgrade!"
        
        await message.answer(status_text, parse_mode="Markdown")

    async def cmd_cancel(self, message: types.Message):
        """Handle /cancel command."""
        # Check if user is admin
        if message.from_user.id not in self.admin_ids:
            user_data = await self.db.get_user(message.from_user.id)
            
            if not user_data or user_data.get("subscription_tier") == "free":
                await message.answer("You don't have an active subscription.")
                return
            
            await message.answer(
                """
To cancel your subscription, please contact support or manage it via Stripe customer portal.

Note: You'll retain access until the end of your billing period.
""",
                parse_mode="Markdown"
            )

    async def cmd_admin_stats(self, message: types.Message):
        """Handle /stats command (admin only)."""
        if message.from_user.id not in self.admin_ids:
            await message.answer("❌ This command is only available to admins.")
            return
        
        # Get statistics
        learning_stats = await self.db.get_learning_stats()
        
        stats_text = f"""
📈 **Bot Statistics**

**Learning Model:**
- Total Predictions: {learning_stats.get('total_predictions', 0)}
- Correct Predictions: {learning_stats.get('correct_predictions', 0)}
- Accuracy: {learning_stats.get('correct_predictions', 0) / max(learning_stats.get('total_predictions', 1), 1) * 100:.1f}%
- Avg Confidence: {learning_stats.get('avg_confidence', 0):.2f}

Use /admin for more admin commands.
"""
        
        await message.answer(stats_text, parse_mode="Markdown")

    async def post_prediction_to_channel(
        self,
        prediction_data: dict,
        tier: str = "free"
    ):
        """Post prediction to channel.
        
        Args:
            prediction_data: Prediction information
            tier: Minimum tier required to see this prediction
        """
        match = prediction_data["match"]
        prediction = prediction_data["prediction"]
        
        # Format message based on tier
        message_text = f"""
🎯 **Match Prediction**

⚽ {match['home_team']} vs {match['away_team']}
📅 {match['match_date']}
🏟️ {match.get('venue', 'N/A')}

**Prediction:** {prediction['prediction'].replace('_', ' ').title()}
**Confidence:** {prediction['confidence']*100:.0f}%

**Analysis:**
{prediction['reasoning'][:300]}...

"""
        
        # Add more details for paid tiers
        if tier in ["basic", "premium"]:
            message_text += f"\n**Suggested Bet:** {prediction.get('suggested_bet', 'N/A')}"
            message_text += f"\n**Value Assessment:** {prediction.get('value_assessment', 'N/A').title()}"
        
        if tier == "premium" and prediction.get("key_factors"):
            message_text += f"\n\n**Key Factors:**"
            for factor in prediction["key_factors"][:3]:
                message_text += f"\n• {factor}"
        
        tier_emoji = {"free": "🆓", "basic": "💎", "premium": "⭐"}
        message_text += f"\n\n{tier_emoji.get(tier, '')} {tier.title()} Tier"
        
        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message_text,
                parse_mode="Markdown"
            )
            logger.info(f"Posted prediction to channel: {match['home_team']} vs {match['away_team']}")
        except Exception as e:
            logger.error(f"Failed to post to channel: {e}")

    async def notify_user(self, user_id: int, message: str):
        """Send notification to user.
        
        Args:
            user_id: Telegram user ID
            message: Message to send
        """
        try:
            await self.bot.send_message(chat_id=user_id, text=message)
        except Exception as e:
            logger.error(f"Failed to notify user {user_id}: {e}")

    async def start(self):
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        await self.dp.start_polling(self.bot)

    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping Telegram bot...")
        await self.bot.session.close()

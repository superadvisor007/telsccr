"""Webhook server for handling Stripe payment webhooks."""
import asyncio
import logging
from aiohttp import web
from datetime import datetime, timedelta
import os

from dotenv import load_dotenv

try:
    from .database import Database, SubscriptionTier
    from .payment_handler import StripePaymentHandler
    from .bot import TelegramBot
except ImportError:
    from database import Database, SubscriptionTier
    from payment_handler import StripePaymentHandler
    from bot import TelegramBot

load_dotenv()

logger = logging.getLogger(__name__)


class WebhookServer:
    """Webhook server for payment processing."""

    def __init__(
        self,
        db: Database,
        payment_handler: StripePaymentHandler,
        telegram_bot: TelegramBot
    ):
        """Initialize webhook server.
        
        Args:
            db: Database instance
            payment_handler: Payment handler instance
            telegram_bot: Telegram bot instance
        """
        self.db = db
        self.payment = payment_handler
        self.bot = telegram_bot
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Setup webhook routes."""
        self.app.router.add_post('/webhook/stripe', self.handle_stripe_webhook)
        self.app.router.add_get('/health', self.health_check)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_stripe_webhook(self, request: web.Request) -> web.Response:
        """Handle Stripe webhook events.
        
        Args:
            request: Webhook request
            
        Returns:
            Response
        """
        payload = await request.read()
        signature = request.headers.get('Stripe-Signature', '')
        
        try:
            # Verify and parse webhook event
            event_data = await self.payment.handle_webhook(payload, signature)
            
            # Process event based on type
            if event_data.get("event") == "subscription_created":
                await self._handle_subscription_created(event_data)
            elif event_data.get("event") == "subscription_updated":
                await self._handle_subscription_updated(event_data)
            elif event_data.get("event") == "subscription_deleted":
                await self._handle_subscription_deleted(event_data)
            elif event_data.get("event") == "payment_succeeded":
                await self._handle_payment_succeeded(event_data)
            elif event_data.get("event") == "payment_failed":
                await self._handle_payment_failed(event_data)
            
            return web.json_response({"status": "success"})
            
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=400
            )

    async def _handle_subscription_created(self, event_data: dict):
        """Handle subscription created event."""
        user_id = event_data["user_id"]
        tier = event_data["tier"]
        expires_at = event_data["expires_at"]
        subscription_id = event_data["subscription_id"]
        
        # Update user subscription in database
        tier_enum = SubscriptionTier.BASIC if tier == "basic" else SubscriptionTier.PREMIUM
        await self.db.update_user_subscription(user_id, tier_enum, expires_at)
        
        # Create subscription record
        await self.db.create_subscription(
            user_id,
            tier,
            0.0,  # Amount will be updated on payment
            expires_at,
            subscription_id
        )
        
        # Notify user
        await self.bot.notify_user(
            user_id,
            f"""
✅ **Subscription Activated!**

Your {tier.title()} subscription is now active!

You now have access to:
- Enhanced predictions
- Detailed analysis
- Priority support

Expires: {expires_at.strftime('%Y-%m-%d')}

Thank you for subscribing! 🎉
"""
        )
        
        logger.info(f"Subscription created for user {user_id}: {tier}")

    async def _handle_subscription_updated(self, event_data: dict):
        """Handle subscription updated event."""
        user_id = event_data["user_id"]
        expires_at = event_data["expires_at"]
        status = event_data["status"]
        
        # Update expiration date
        user_data = await self.db.get_user(user_id)
        if user_data:
            current_tier = user_data.get("subscription_tier", "free")
            tier_enum = SubscriptionTier[current_tier.upper()]
            await self.db.update_user_subscription(user_id, tier_enum, expires_at)
        
        logger.info(f"Subscription updated for user {user_id}: {status}")

    async def _handle_subscription_deleted(self, event_data: dict):
        """Handle subscription deleted event."""
        user_id = event_data["user_id"]
        
        # Downgrade to free tier
        await self.db.update_user_subscription(
            user_id,
            SubscriptionTier.FREE,
            datetime.now()
        )
        
        # Notify user
        await self.bot.notify_user(
            user_id,
            """
ℹ️ **Subscription Cancelled**

Your subscription has been cancelled.

You've been moved to the free tier. You can resubscribe anytime using /subscribe.

Thank you for being a subscriber!
"""
        )
        
        logger.info(f"Subscription cancelled for user {user_id}")

    async def _handle_payment_succeeded(self, event_data: dict):
        """Handle payment succeeded event."""
        user_id = event_data["user_id"]
        amount = event_data["amount"]
        currency = event_data["currency"]
        
        logger.info(f"Payment succeeded for user {user_id}: {amount} {currency}")

    async def _handle_payment_failed(self, event_data: dict):
        """Handle payment failed event."""
        user_id = event_data["user_id"]
        
        # Notify user
        await self.bot.notify_user(
            user_id,
            """
❌ **Payment Failed**

Your recent payment could not be processed.

Please update your payment method to continue your subscription.

Contact support if you need assistance.
"""
        )
        
        logger.warning(f"Payment failed for user {user_id}")

    async def start(self, host: str = '0.0.0.0', port: int = 8443):
        """Start the webhook server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Webhook server started on {host}:{port}")


async def main():
    """Main entry point for webhook server."""
    try:
        from .main import SoccerTipsBot
    except ImportError:
        from main import SoccerTipsBot
    
    # Initialize bot components
    bot = SoccerTipsBot()
    await bot.initialize()
    
    # Create webhook server
    webhook_host = os.getenv("WEBHOOK_HOST", "0.0.0.0")
    webhook_port = int(os.getenv("WEBHOOK_PORT", "8443"))
    
    server = WebhookServer(bot.db, bot.payment_handler, bot.telegram_bot)
    
    try:
        await server.start(webhook_host, webhook_port)
        
        # Keep server running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

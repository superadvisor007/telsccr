"""Stripe payment integration for subscription management."""
import stripe
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class PaymentTier(Enum):
    """Payment tier levels."""
    BASIC = "basic"
    PREMIUM = "premium"


class StripePaymentHandler:
    """Handle Stripe payments and subscriptions."""

    def __init__(
        self,
        api_key: str,
        webhook_secret: str,
        price_basic: str,
        price_premium: str
    ):
        """Initialize Stripe payment handler.
        
        Args:
            api_key: Stripe API secret key
            webhook_secret: Stripe webhook secret
            price_basic: Stripe Price ID for basic tier
            price_premium: Stripe Price ID for premium tier
        """
        stripe.api_key = api_key
        self.webhook_secret = webhook_secret
        self.prices = {
            PaymentTier.BASIC: price_basic,
            PaymentTier.PREMIUM: price_premium
        }

    async def create_customer(
        self,
        user_id: int,
        email: Optional[str] = None,
        name: Optional[str] = None
    ) -> str:
        """Create a Stripe customer.
        
        Args:
            user_id: Telegram user ID
            email: Customer email (optional)
            name: Customer name (optional)
            
        Returns:
            Stripe customer ID
        """
        try:
            customer = stripe.Customer.create(
                metadata={"telegram_user_id": str(user_id)},
                email=email,
                name=name
            )
            logger.info(f"Created Stripe customer for user {user_id}: {customer.id}")
            return customer.id
        except Exception as e:
            logger.error(f"Failed to create Stripe customer: {e}")
            raise

    async def create_checkout_session(
        self,
        customer_id: str,
        tier: PaymentTier,
        success_url: str,
        cancel_url: str
    ) -> Dict:
        """Create a Checkout Session for subscription.
        
        Args:
            customer_id: Stripe customer ID
            tier: Subscription tier
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
            
        Returns:
            Checkout session data
        """
        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[
                    {
                        "price": self.prices[tier],
                        "quantity": 1,
                    }
                ],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                locale="de",  # Swiss locale
                currency="chf",
            )
            
            logger.info(f"Created checkout session: {session.id}")
            return {
                "session_id": session.id,
                "url": session.url
            }
        except Exception as e:
            logger.error(f"Failed to create checkout session: {e}")
            raise

    async def create_payment_link(
        self,
        tier: PaymentTier,
        user_id: int
    ) -> str:
        """Create a payment link for a subscription.
        
        Args:
            tier: Subscription tier
            user_id: Telegram user ID
            
        Returns:
            Payment link URL
        """
        try:
            payment_link = stripe.PaymentLink.create(
                line_items=[
                    {
                        "price": self.prices[tier],
                        "quantity": 1,
                    }
                ],
                metadata={"telegram_user_id": str(user_id)},
            )
            
            logger.info(f"Created payment link for user {user_id}: {payment_link.url}")
            return payment_link.url
        except Exception as e:
            logger.error(f"Failed to create payment link: {e}")
            raise

    async def handle_webhook(self, payload: bytes, signature: str) -> Dict:
        """Handle Stripe webhook events.
        
        Args:
            payload: Webhook payload
            signature: Stripe signature header
            
        Returns:
            Event data
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            
            logger.info(f"Received webhook event: {event['type']}")
            
            # Handle different event types
            if event["type"] == "checkout.session.completed":
                return await self._handle_checkout_completed(event["data"]["object"])
            elif event["type"] == "customer.subscription.created":
                return await self._handle_subscription_created(event["data"]["object"])
            elif event["type"] == "customer.subscription.updated":
                return await self._handle_subscription_updated(event["data"]["object"])
            elif event["type"] == "customer.subscription.deleted":
                return await self._handle_subscription_deleted(event["data"]["object"])
            elif event["type"] == "invoice.payment_succeeded":
                return await self._handle_payment_succeeded(event["data"]["object"])
            elif event["type"] == "invoice.payment_failed":
                return await self._handle_payment_failed(event["data"]["object"])
            
            return {"status": "unhandled", "type": event["type"]}
            
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            raise

    async def _handle_checkout_completed(self, session: Dict) -> Dict:
        """Handle checkout session completed event."""
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        
        # Get customer metadata to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        return {
            "event": "checkout_completed",
            "user_id": user_id,
            "customer_id": customer_id,
            "subscription_id": subscription_id
        }

    async def _handle_subscription_created(self, subscription: Dict) -> Dict:
        """Handle subscription created event."""
        customer_id = subscription.get("customer")
        subscription_id = subscription["id"]
        
        # Get customer to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        # Determine tier from price
        price_id = subscription["items"]["data"][0]["price"]["id"]
        tier = self._get_tier_from_price(price_id)
        
        # Calculate expiration (monthly subscription)
        current_period_end = subscription["current_period_end"]
        expires_at = datetime.fromtimestamp(current_period_end)
        
        return {
            "event": "subscription_created",
            "user_id": user_id,
            "subscription_id": subscription_id,
            "tier": tier,
            "expires_at": expires_at,
            "status": subscription["status"]
        }

    async def _handle_subscription_updated(self, subscription: Dict) -> Dict:
        """Handle subscription updated event."""
        customer_id = subscription.get("customer")
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        current_period_end = subscription["current_period_end"]
        expires_at = datetime.fromtimestamp(current_period_end)
        
        return {
            "event": "subscription_updated",
            "user_id": user_id,
            "subscription_id": subscription["id"],
            "expires_at": expires_at,
            "status": subscription["status"]
        }

    async def _handle_subscription_deleted(self, subscription: Dict) -> Dict:
        """Handle subscription deleted/cancelled event."""
        customer_id = subscription.get("customer")
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        return {
            "event": "subscription_deleted",
            "user_id": user_id,
            "subscription_id": subscription["id"]
        }

    async def _handle_payment_succeeded(self, invoice: Dict) -> Dict:
        """Handle payment succeeded event."""
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        return {
            "event": "payment_succeeded",
            "user_id": user_id,
            "subscription_id": subscription_id,
            "amount": invoice["amount_paid"] / 100,  # Convert cents to CHF
            "currency": invoice["currency"]
        }

    async def _handle_payment_failed(self, invoice: Dict) -> Dict:
        """Handle payment failed event."""
        customer_id = invoice.get("customer")
        customer = stripe.Customer.retrieve(customer_id)
        user_id = int(customer.metadata.get("telegram_user_id", 0))
        
        return {
            "event": "payment_failed",
            "user_id": user_id,
            "subscription_id": invoice.get("subscription")
        }

    def _get_tier_from_price(self, price_id: str) -> str:
        """Get tier name from price ID."""
        for tier, pid in self.prices.items():
            if pid == price_id:
                return tier.value
        return "unknown"

    async def cancel_subscription(self, subscription_id: str):
        """Cancel a subscription.
        
        Args:
            subscription_id: Stripe subscription ID
        """
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            logger.info(f"Cancelled subscription: {subscription_id}")
            return subscription
        except Exception as e:
            logger.error(f"Failed to cancel subscription: {e}")
            raise

    async def get_subscription(self, subscription_id: str) -> Dict:
        """Get subscription details.
        
        Args:
            subscription_id: Stripe subscription ID
            
        Returns:
            Subscription data
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        except Exception as e:
            logger.error(f"Failed to retrieve subscription: {e}")
            raise

"""
Payment Integration for CosArt
Stripe-based subscription management and payment processing
"""

import stripe
from fastapi import HTTPException, Depends
from typing import Optional, Dict, Any
import os
from datetime import datetime
from config.settings import settings
from api.auth.dependencies import get_current_user
from api.models.user import User

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_placeholder")

# Subscription plans (price IDs from Stripe)
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price_id": None,
        "features": ["20 requests/minute", "5 generations/hour", "Basic presets"]
    },
    "pro": {
        "name": "Pro",
        "price_id": os.getenv("STRIPE_PRO_PRICE_ID", "price_pro_placeholder"),
        "features": ["100 requests/minute", "50 generations/hour", "All presets", "High resolution"]
    },
    "studio": {
        "name": "Studio",
        "price_id": os.getenv("STRIPE_STUDIO_PRICE_ID", "price_studio_placeholder"),
        "features": ["500 requests/minute", "200 generations/hour", "All presets", "4K resolution", "Custom training"]
    }
}

class PaymentService:
    """Handle Stripe payment operations"""

    @staticmethod
    def create_customer(email: str, name: Optional[str] = None) -> str:
        """Create a Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"source": "cosart"}
            )
            return customer.id
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=f"Failed to create customer: {str(e)}")

    @staticmethod
    def create_subscription(customer_id: str, price_id: str) -> Dict[str, Any]:
        """Create a subscription for a customer"""
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                metadata={"source": "cosart"}
            )
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end,
                "price_id": price_id
            }
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=f"Failed to create subscription: {str(e)}")

    @staticmethod
    def cancel_subscription(subscription_id: str) -> Dict[str, Any]:
        """Cancel a subscription"""
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "cancel_at": subscription.cancel_at
            }
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=f"Failed to cancel subscription: {str(e)}")

    @staticmethod
    def create_payment_intent(amount: int, currency: str = "usd") -> Dict[str, Any]:
        """Create a payment intent for one-time payments"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                metadata={"source": "cosart"}
            )
            return {
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id
            }
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=f"Failed to create payment intent: {str(e)}")

    @staticmethod
    def get_subscription_status(subscription_id: str) -> Dict[str, Any]:
        """Get subscription status"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return {
                "status": subscription.status,
                "current_period_end": subscription.current_period_end,
                "cancel_at": subscription.cancel_at,
                "canceled_at": subscription.canceled_at
            }
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=404, detail="Subscription not found")

def get_plan_info(plan: str) -> Dict[str, Any]:
    """Get plan information"""
    if plan not in SUBSCRIPTION_PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    return SUBSCRIPTION_PLANS[plan]

def map_stripe_price_to_tier(price_id: str) -> str:
    """Map Stripe price ID to user tier"""
    for tier, plan in SUBSCRIPTION_PLANS.items():
        if plan["price_id"] == price_id:
            return tier
    return "free"
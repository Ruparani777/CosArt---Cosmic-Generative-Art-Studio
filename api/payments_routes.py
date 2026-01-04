"""
Payment Routes for CosArt
Stripe subscription management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel, EmailStr
from api.auth.dependencies import get_current_user
from api.models.user import User
from api.repositories.user_repository import UserRepository
from api.database.session import get_db
from sqlalchemy.orm import Session
from api.payment_service import PaymentService, get_plan_info, SUBSCRIPTION_PLANS
import stripe
import stripe
import stripe
import stripe
import os

router = APIRouter(prefix="/payments", tags=["payments"])

class SubscriptionRequest(BaseModel):
    plan: str  # "pro" or "studio"
    payment_method_id: Optional[str] = None

class PaymentIntentRequest(BaseModel):
    amount: int  # Amount in cents
    currency: str = "usd"

class CustomerCreateRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None

@router.get("/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return {
        "plans": SUBSCRIPTION_PLANS,
        "current_features": {
            "free": ["20 requests/minute", "5 generations/hour", "Basic presets"],
            "pro": ["100 requests/minute", "50 generations/hour", "All presets", "High resolution"],
            "studio": ["500 requests/minute", "200 generations/hour", "All presets", "4K resolution", "Custom training"]
        }
    }

@router.post("/customers")
async def create_customer(
    request: CustomerCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a Stripe customer for the current user"""
    try:
        customer_id = PaymentService.create_customer(
            email=request.email,
            name=request.name
        )

        # Update user with Stripe customer ID
        user_repo = UserRepository(db)
        user_repo.update_stripe_customer(current_user.id, customer_id)

        return {"customer_id": customer_id, "message": "Customer created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/subscriptions")
async def create_subscription(
    request: SubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a subscription for the current user"""
    # Validate plan
    plan_info = get_plan_info(request.plan)
    if not plan_info["price_id"]:
        raise HTTPException(status_code=400, detail="Free plan doesn't require subscription")

    # Check if user already has a customer ID
    user_repo = UserRepository(db)
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="Customer not created. Call /customers first")

    try:
        subscription = PaymentService.create_subscription(
            current_user.stripe_customer_id,
            plan_info["price_id"]
        )

        # Update user tier
        user_repo.update_tier(current_user.id, request.plan)

        return {
            "subscription": subscription,
            "message": f"Successfully subscribed to {request.plan} plan"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/subscriptions/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel a subscription"""
    try:
        result = PaymentService.cancel_subscription(subscription_id)

        # Optionally downgrade user to free tier
        # This would be handled by webhooks in production

        return {
            "subscription": result,
            "message": "Subscription will be canceled at the end of the billing period"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/subscriptions/{subscription_id}")
async def get_subscription_status(subscription_id: str):
    """Get subscription status"""
    try:
        return PaymentService.get_subscription_status(subscription_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Subscription not found")

@router.post("/payment-intents")
async def create_payment_intent(request: PaymentIntentRequest):
    """Create a payment intent for one-time payments"""
    try:
        intent = PaymentService.create_payment_intent(
            amount=request.amount,
            currency=request.currency
        )
        return intent
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/billing-portal")
async def create_billing_portal_session(
    current_user: User = Depends(get_current_user)
):
    """Create a Stripe billing portal session"""
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No customer found")

    try:
        session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=os.getenv("FRONTEND_URL", "http://localhost:3000")
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Webhook endpoint for Stripe events
@router.post("/webhooks")
async def stripe_webhook(
    request: Dict[str, Any],
    stripe_signature: str = None
):
    """Handle Stripe webhooks"""
    # In production, verify webhook signature
    # For now, just log the event
    event = request

    if event["type"] == "customer.subscription.updated":
        # Handle subscription updates
        pass
    elif event["type"] == "customer.subscription.deleted":
        # Handle subscription cancellations
        pass

    return {"status": "ok"}
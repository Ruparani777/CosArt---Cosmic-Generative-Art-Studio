"""
Monitoring and Logging for CosArt
Structured logging and Prometheus metrics
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from config.settings import settings

# Configure structured logging
def setup_logging():
    """Configure structured logging for the application"""
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.DEBUG:
        # Development logging
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Production logging
        shared_processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    )

# Prometheus metrics
REQUEST_COUNT = Counter(
    'cosart_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'cosart_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

GENERATION_COUNT = Counter(
    'cosart_generations_total',
    'Total number of image generations',
    ['preset', 'resolution', 'user_tier']
)

ACTIVE_USERS = Gauge(
    'cosart_active_users',
    'Number of active users'
)

SYSTEM_HEALTH = Gauge(
    'cosart_system_health',
    'System health status (1=healthy, 0=unhealthy)'
)

class MonitoringMiddleware:
    """FastAPI middleware for request monitoring"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        # Create a custom response handler
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Record metrics
                REQUEST_COUNT.labels(
                    method=scope["method"],
                    endpoint=scope["path"],
                    status_code=status_code
                ).inc()

                duration = time.time() - start_time
                REQUEST_LATENCY.labels(
                    method=scope["method"],
                    endpoint=scope["path"]
                ).observe(duration)

            await send(message)

        await self.app(scope, receive, send_wrapper)

def record_generation_metrics(preset: str, resolution: int, user_tier: str = "free"):
    """Record metrics for image generation"""
    GENERATION_COUNT.labels(
        preset=preset,
        resolution=str(resolution),
        user_tier=user_tier
    ).inc()

def update_system_health(healthy: bool):
    """Update system health status"""
    SYSTEM_HEALTH.set(1 if healthy else 0)

def get_metrics() -> str:
    """Get Prometheus metrics"""
    return generate_latest()

# Structured logging helpers
logger = structlog.get_logger()

def log_request(request: Request, response: Optional[Response] = None, **extra):
    """Log HTTP requests with structured data"""
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None,
    }

    if response:
        log_data["status_code"] = response.status_code

    log_data.update(extra)

    if response and response.status_code >= 400:
        logger.error("HTTP request failed", **log_data)
    else:
        logger.info("HTTP request", **log_data)

def log_generation(user_id: str, preset: str, seed: int, duration: float, **extra):
    """Log image generation events"""
    logger.info("Image generation completed", {
        "user_id": user_id,
        "preset": preset,
        "seed": seed,
        "duration": duration,
        **extra
    })

def log_error(error: Exception, **extra):
    """Log errors with structured data"""
    logger.error("Application error", {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **extra
    }, exc_info=True)

# Health check functions
async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        from api.database.session import get_db
        from sqlalchemy.orm import Session

        db: Session = next(get_db())
        # Simple query to test connection
        db.execute("SELECT 1")
        return {"status": "healthy", "component": "database"}
    except Exception as e:
        return {"status": "unhealthy", "component": "database", "error": str(e)}

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity"""
    try:
        import redis
        client = redis.Redis(
            host=settings.REDIS_URL.replace("redis://", "").split(":")[0] if ":" in settings.REDIS_URL.replace("redis://", "") else "localhost",
            port=int(settings.REDIS_URL.replace("redis://", "").split(":")[1]) if ":" in settings.REDIS_URL.replace("redis://", "") else 6379,
        )
        client.ping()
        return {"status": "healthy", "component": "redis"}
    except Exception as e:
        return {"status": "unhealthy", "component": "redis", "error": str(e)}

async def comprehensive_health_check() -> Dict[str, Any]:
    """Comprehensive health check of all system components"""
    results = []

    # Check database
    db_health = await check_database_health()
    results.append(db_health)

    # Check Redis
    redis_health = await check_redis_health()
    results.append(redis_health)

    # Overall status
    overall_healthy = all(r["status"] == "healthy" for r in results)

    # Update Prometheus metric
    update_system_health(overall_healthy)

    return {
        "overall_status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": time.time(),
        "checks": results
    }
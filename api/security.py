"""CORS allow-list, in-memory rate limit, and optional admin bearer auth."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from config.settings import get_settings

logger = logging.getLogger(__name__)


def cors_origin_list() -> list[str]:
    """Parse ``CORS_ORIGINS``. Production never keeps a wildcard."""
    settings = get_settings()
    origins = [part.strip() for part in settings.cors_origins.split(",") if part.strip()]
    if settings.environment == "production":
        origins = [origin for origin in origins if origin != "*"]
    return origins or [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    ]


class RateLimiter:
    """Sliding 60-second window keyed by client IP."""

    def __init__(self, per_min: int) -> None:
        self.per_min = per_min
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Return True when ``key`` is still under the per-minute budget."""
        moment = now if now is not None else time.time()
        window_start = moment - 60.0
        with self._lock:
            recent = [hit for hit in self._hits.get(key, []) if hit > window_start]
            if len(recent) >= self.per_min:
                self._hits[key] = recent
                return False
            recent.append(moment)
            self._hits[key] = recent
            return True


_LIMITED_PREFIXES = (
    "/api/v1/generate",
    "/api/v1/review",
    "/api/v1/sandbox",
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject bursty write endpoints with HTTP 429."""

    def __init__(self, app: Callable, limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = limiter or RateLimiter(get_settings().rate_limit_per_min)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if request.method in {"POST", "PUT", "PATCH"} and any(
            path.startswith(prefix) for prefix in _LIMITED_PREFIXES
        ):
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                logger.warning("Rate limit exceeded for %s on %s", client, path)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Retry in a minute."},
                )
        return await call_next(request)


def require_admin(authorization: str | None) -> None:
    """Require ``Authorization: Bearer`` matching ``ADMIN_TOKEN``."""
    token = get_settings().admin_token.get_secret_value()
    if not token:
        raise HTTPException(
            status_code=403,
            detail="Admin token is not configured on this server.",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    provided = authorization.split(" ", 1)[1].strip()
    if provided != token:
        raise HTTPException(status_code=403, detail="Invalid admin token.")

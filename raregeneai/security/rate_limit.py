"""Rate limiting for RareGeneAI API.

Protects against:
  - Brute force attacks on /auth/token
  - DOS via unlimited analysis submissions
  - Resource exhaustion on admin endpoints

Uses in-memory sliding window counters. For multi-instance deployments,
replace with Redis-backed storage.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status
from loguru import logger


@dataclass
class _RateWindow:
    """Sliding window rate counter."""
    timestamps: list[float] = field(default_factory=list)

    def count_within(self, window_seconds: float) -> int:
        cutoff = time.monotonic() - window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps)

    def record(self) -> None:
        self.timestamps.append(time.monotonic())


class RateLimiter:
    """In-memory rate limiter with per-IP and per-user tracking.

    Usage:
        limiter = RateLimiter()

        @app.post("/auth/token")
        async def login(request: Request):
            limiter.check("login", request, max_requests=5, window_seconds=60)
            ...
    """

    def __init__(self):
        # key -> {identifier -> RateWindow}
        self._windows: dict[str, dict[str, _RateWindow]] = defaultdict(
            lambda: defaultdict(_RateWindow)
        )

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise client IP
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(
        self,
        key: str,
        request: Request,
        max_requests: int,
        window_seconds: int,
        identifier: str | None = None,
    ) -> None:
        """Check rate limit. Raises HTTP 429 if exceeded.

        Args:
            key: Rate limit bucket name (e.g., "login", "analyze")
            request: FastAPI request for IP extraction
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
            identifier: Optional override for client ID (e.g., username)
        """
        client_id = identifier or self._get_client_id(request)
        window = self._windows[key][client_id]

        current = window.count_within(window_seconds)
        if current >= max_requests:
            logger.warning(
                f"Rate limit exceeded: {key} by {client_id} "
                f"({current}/{max_requests} in {window_seconds}s)"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.",
                headers={"Retry-After": str(window_seconds)},
            )

        window.record()


# ── Default rate limits (configurable via environment) ───────────────────────

# Login: 5 attempts per minute per IP
LOGIN_MAX = int(os.environ.get("RAREGENEAI_LOGIN_RATE_MAX", "5"))
LOGIN_WINDOW = int(os.environ.get("RAREGENEAI_LOGIN_RATE_WINDOW", "60"))

# Analysis: 10 per hour per user
ANALYZE_MAX = int(os.environ.get("RAREGENEAI_ANALYZE_RATE_MAX", "10"))
ANALYZE_WINDOW = int(os.environ.get("RAREGENEAI_ANALYZE_RATE_WINDOW", "3600"))

# Registration: 3 per hour per IP
REGISTER_MAX = int(os.environ.get("RAREGENEAI_REGISTER_RATE_MAX", "3"))
REGISTER_WINDOW = int(os.environ.get("RAREGENEAI_REGISTER_RATE_WINDOW", "3600"))

# Global singleton
rate_limiter = RateLimiter()

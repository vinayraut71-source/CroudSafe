"""
Lightweight in-memory rate limiter for NLP Assistant APIs.
"""

import time
import threading
from collections import defaultdict, deque
from fastapi import Request, HTTPException


class RateLimiter:
    """Sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, max_requests: int = None, window_seconds: int = None) -> bool:
        limit = max_requests or self.max_requests
        window = window_seconds or self.window_seconds
        now = time.time()

        with self._lock:
            q = self._history[key]
            # Evict timestamps older than window
            while q and q[0] <= now - window:
                q.popleft()

            if len(q) >= limit:
                return False

            q.append(now)
            return True


default_limiter = RateLimiter(max_requests=60, window_seconds=60)


def require_rate_limit(request: Request, max_requests: int = 40, window_seconds: int = 60):
    """FastAPI dependency to enforce rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    if not default_limiter.check(key, max_requests=max_requests, window_seconds=window_seconds):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds}s allowed."
        )


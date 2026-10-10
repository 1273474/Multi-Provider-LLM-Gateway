
import os
import time
from threading import Lock

from fastapi import HTTPException


RATE_LIMIT = int(os.getenv("SENTINEL_RATE_LIMIT", "60"))
WINDOW_SECONDS = 60

# tenant_id -> (window_start, request_count)
REQUEST_COUNTS = {}

_lock = Lock()


def enforce_rate_limit(tenant_id: str) -> None:
    now = time.monotonic()

    with _lock:
        window_start, count = REQUEST_COUNTS.get(
            tenant_id, (now, 0)
        )

        # Start a new window after 60 seconds.
        if now - window_start >= WINDOW_SECONDS:
            window_start = now
            count = 0

        # Reject requests beyond the configured limit.
        if count >= RATE_LIMIT:
            REQUEST_COUNTS[tenant_id] = (
                window_start,
                count,
            )

            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded",
                    "tenant_id": tenant_id,
                    "limit": RATE_LIMIT,
                    "window_seconds": WINDOW_SECONDS,
                },
            )

        # Count this request.
        REQUEST_COUNTS[tenant_id] = (
            window_start,
            count + 1,
        )

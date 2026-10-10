
import hmac
import os

from fastapi import Header, HTTPException


def get_current_tenant(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )

    configured_keys = os.getenv("SENTINEL_API_KEYS", "")
    credentials = {}

    for item in configured_keys.split(","):
        if ":" not in item:
            continue

        key, tenant_id = item.strip().split(":", 1)
        if key and tenant_id:
            credentials[key] = tenant_id

    for key, tenant_id in credentials.items():
        if hmac.compare_digest(x_api_key, key):
            return tenant_id

    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
    )

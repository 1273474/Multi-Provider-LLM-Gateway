
import logging
import os
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from app import settings  # Loads local environment variables.
from app.auth import get_current_tenant
from app.budgets import (
    BudgetExceededError,
    release_budget,
    reserve_budget,
    settle_budget,
)
from app.observability import (
    HTTP_REQUESTS,
    HTTP_REQUEST_DURATION,
    record_llm_success,
    record_llm_failure,
)
from app.rate_limit import enforce_rate_limit
from app.reliability import generate_with_fallback
from app.schemas import ChatRequest, ChatResponse, Usage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentinel AI",
    description="A unified API gateway for LLM providers.",
    version="0.1.0",
)


# -------------------------
# HTTP Observability
# -------------------------

@app.middleware("http")
async def observe_http_requests(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    route = request.scope.get("route")
    endpoint = getattr(route, "path", "unmatched")

    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()

    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())


# -------------------------
# Health Check
# -------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------
# LLM Chat Endpoint
# -------------------------

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    request_id = str(uuid4())
    start_time = time.perf_counter()

    # 1. Enforce tenant rate limit.
    enforce_rate_limit(tenant_id)

    # 2. Include configured fallback models in budget estimation.
    fallback_models = [
        item.strip()
        for item in os.getenv(
            "SENTINEL_FALLBACK_MODELS", ""
        ).split(",")
        if item.strip()
    ]

    candidates = [request.model] + fallback_models

    # 3. Reserve estimated budget before calling a provider.
    try:
        reservation_id = reserve_budget(
            tenant_id=tenant_id,
            models=candidates,
            messages=request.messages,
            max_tokens=request.max_tokens,
        )

    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=402,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # 4. Call the provider with resilience and fallback.
    try:
        provider_name, model_name, result = (
            await generate_with_fallback(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        )

    except ValueError as exc:
        release_budget(reservation_id)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        release_budget(reservation_id)

        record_llm_failure(provider="unknown")

        logger.exception(
            "Provider request failed: request_id=%s "
            "tenant_id=%s",
            request_id,
            tenant_id,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": "The model provider request failed",
                "request_id": request_id,
            },
        ) from None

    # 5. Record actual cost using reported token usage.
    try:
        settle_budget(
            reservation_id=reservation_id,
            model_key=f"{provider_name}/{model_name}",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    except Exception:
        # Avoid silently losing track of a reservation if
        # accounting fails after a provider call succeeds.
        logger.exception(
            "Budget settlement failed: request_id=%s "
            "tenant_id=%s provider=%s model=%s",
            request_id,
            tenant_id,
            provider_name,
            model_name,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Usage accounting failed",
                "request_id": request_id,
            },
        ) from None

    # 6. Record successful LLM metrics.
    record_llm_success(
        provider=provider_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # 7. Log operational details without logging the prompt.
    logger.info(
        "request_completed request_id=%s tenant_id=%s "
        "provider=%s model=%s latency_ms=%.2f "
        "input_tokens=%d output_tokens=%d",
        request_id,
        tenant_id,
        provider_name,
        result.model,
        latency_ms,
        result.input_tokens,
        result.output_tokens,
    )

    # 8. Return the standardized response.
    return ChatResponse(
        request_id=request_id,
        tenant_id=tenant_id,
        provider=provider_name,
        model=result.model,
        content=result.content,
        usage=Usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
        finish_reason=result.finish_reason,
        latency_ms=round(latency_ms, 2),
    )

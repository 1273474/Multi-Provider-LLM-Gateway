
import time

from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "sentinel_http_requests_total",
    "Total HTTP requests handled by Sentinel AI",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "sentinel_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

LLM_REQUESTS = Counter(
    "sentinel_llm_requests_total",
    "LLM provider requests",
    ["provider", "outcome"],
)

LLM_TOKENS = Counter(
    "sentinel_llm_tokens_total",
    "LLM tokens reported by providers",
    ["provider", "direction"],
)


def record_llm_success(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    LLM_REQUESTS.labels(
        provider=provider,
        outcome="success",
    ).inc()

    LLM_TOKENS.labels(
        provider=provider,
        direction="input",
    ).inc(input_tokens)

    LLM_TOKENS.labels(
        provider=provider,
        direction="output",
    ).inc(output_tokens)


def record_llm_failure(provider: str) -> None:
    LLM_REQUESTS.labels(
        provider=provider,
        outcome="failure",
    ).inc()

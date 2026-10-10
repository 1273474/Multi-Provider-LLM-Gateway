
import asyncio
import os
import time
from dataclasses import dataclass, field

from app.providers.registry import resolve_provider


# -------------------------
# Reliability configuration
# -------------------------

TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 30


# -------------------------
# Circuit breaker
# -------------------------

class CircuitOpenError(Exception):
    """Raised when a provider's circuit breaker blocks a request."""


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None
    half_open_in_flight: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


CIRCUITS: dict[str, CircuitState] = {}


def get_circuit(provider_name: str) -> CircuitState:
    """Get or create the circuit state for a provider."""
    if provider_name not in CIRCUITS:
        CIRCUITS[provider_name] = CircuitState()

    return CIRCUITS[provider_name]


# -------------------------
# Retry policy
# -------------------------

def is_retryable(error: Exception) -> bool:
    """Determine whether a provider error should be retried."""

    if isinstance(
        error,
        (TimeoutError, asyncio.TimeoutError, ConnectionError),
    ):
        return True

    status_code = getattr(error, "status_code", None)

    if status_code == 429:
        return True

    if isinstance(status_code, int) and status_code >= 500:
        return True

    # Common connection/timeout exceptions from SDKs.
    error_name = type(error).__name__.lower()

    return "connection" in error_name or "timeout" in error_name


# -------------------------
# Provider call with retry
# and circuit breaker
# -------------------------

async def call_provider(
    provider_name: str,
    provider,
    messages,
    model: str,
    temperature: float,
    max_tokens: int,
):
    circuit = get_circuit(provider_name)
    is_trial = False

    # Check circuit state safely before calling the provider.
    async with circuit.lock:
        if circuit.opened_at is not None:
            elapsed = time.monotonic() - circuit.opened_at

            if elapsed < COOLDOWN_SECONDS:
                raise CircuitOpenError(
                    f"{provider_name} circuit is open"
                )

            # Cooldown expired. Allow one half-open trial.
            if circuit.half_open_in_flight:
                raise CircuitOpenError(
                    f"{provider_name} circuit trial is already in progress"
                )

            circuit.half_open_in_flight = True
            is_trial = True

    try:
        for attempt in range(MAX_ATTEMPTS):
            try:
                result = await asyncio.wait_for(
                    provider.generate(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )

                # Provider recovered: close the circuit.
                async with circuit.lock:
                    circuit.failures = 0
                    circuit.opened_at = None
                    circuit.half_open_in_flight = False

                return result

            except Exception as error:
                # Retry only retryable errors.
                if (
                    not is_retryable(error)
                    or attempt == MAX_ATTEMPTS - 1
                ):
                    raise

                # Exponential backoff: 0.5s, 1s, ...
                delay = 0.5 * (2 ** attempt)
                await asyncio.sleep(delay)

    except Exception:
        # Count one failed provider operation after retries
        # have been exhausted, not every individual attempt.
        async with circuit.lock:
            circuit.failures += 1

            # A failed half-open trial reopens the circuit.
            if (
                is_trial
                or circuit.failures >= FAILURE_THRESHOLD
            ):
                circuit.opened_at = time.monotonic()

            circuit.half_open_in_flight = False

        raise

    except BaseException:
        # Cancellation should release the half-open trial
        # without counting cancellation as a provider failure.
        if is_trial:
            async with circuit.lock:
                circuit.half_open_in_flight = False

        raise


# -------------------------
# Provider fallback
# -------------------------

async def generate_with_fallback(
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
):
    fallback_models = [
        item.strip()
        for item in os.getenv(
            "SENTINEL_FALLBACK_MODELS", ""
        ).split(",")
        if item.strip()
    ]

    # Validate the requested model first.
    # An invalid primary model should not silently fall back.
    resolve_provider(model)

    candidates = [model] + fallback_models
    last_error = None

    for candidate in candidates:
        # Resolve the candidate separately so invalid fallback
        # configuration doesn't stop later candidates.
        try:
            provider_name, provider, model_name = (
                resolve_provider(candidate)
            )
        except ValueError as error:
            last_error = RuntimeError(
                f"Invalid fallback model configuration: {candidate}"
            )
            last_error.__cause__ = error
            continue

        try:
            result = await call_provider(
                provider_name=provider_name,
                provider=provider,
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return provider_name, model_name, result

        except Exception as error:
            # Try the next configured model if this provider
            # fails or its circuit is open.
            last_error = error

    raise RuntimeError(
        "All configured providers failed"
    ) from last_error

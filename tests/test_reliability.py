
import asyncio

import pytest

import app.reliability as reliability
from app.providers.base import ProviderResult
from tests.fake_provider import FakeProvider, SlowFakeProvider
from app.reliability import CircuitOpenError


def successful_result():
    return ProviderResult(
        content="Test response",
        model="test-model",
        input_tokens=5,
        output_tokens=3,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_timeout_stops_slow_provider(monkeypatch):
    provider_name = "timeout-test"
    reliability.CIRCUITS.pop(provider_name, None)

    monkeypatch.setattr(
        reliability, "TIMEOUT_SECONDS", 0.01
    )
    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 1
    )

    with pytest.raises(asyncio.TimeoutError):
        await reliability.call_provider(
            provider_name=provider_name,
            provider=SlowFakeProvider(),
            messages=[],
            model="test-model",
            temperature=0.2,
            max_tokens=100,
        )


@pytest.mark.asyncio
async def test_retry_succeeds_after_temporary_failures(
    monkeypatch,
):
    provider_name = "retry-success-test"
    reliability.CIRCUITS.pop(provider_name, None)

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(
        reliability.asyncio, "sleep", no_wait
    )
    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 3
    )

    provider = FakeProvider([
        ConnectionError("Temporary failure"),
        ConnectionError("Temporary failure"),
        successful_result(),
    ])

    result = await reliability.call_provider(
        provider_name=provider_name,
        provider=provider,
        messages=[],
        model="test-model",
        temperature=0.2,
        max_tokens=100,
    )

    assert result.content == "Test response"
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_retries_stop_after_max_attempts(monkeypatch):
    provider_name = "retry-exhausted-test"
    reliability.CIRCUITS.pop(provider_name, None)

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(
        reliability.asyncio, "sleep", no_wait
    )
    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 3
    )

    provider = FakeProvider([
        ConnectionError("Failure 1"),
        ConnectionError("Failure 2"),
        ConnectionError("Failure 3"),
    ])

    with pytest.raises(ConnectionError):
        await reliability.call_provider(
            provider_name=provider_name,
            provider=provider,
            messages=[],
            model="test-model",
            temperature=0.2,
            max_tokens=100,
        )

    assert provider.calls == 3





@pytest.mark.asyncio
async def test_circuit_opens_after_repeated_failures(
    monkeypatch,
):
    provider_name = "circuit-test"
    reliability.CIRCUITS.pop(provider_name, None)

    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 1
    )
    monkeypatch.setattr(
        reliability, "FAILURE_THRESHOLD", 2
    )

    provider = FakeProvider([
        ConnectionError("Failure 1"),
        ConnectionError("Failure 2"),
        successful_result(),
    ])

    # First failure: circuit remains closed.
    with pytest.raises(ConnectionError):
        await reliability.call_provider(
            provider_name, provider, [],
            "test-model", 0.2, 100,
        )

    # Second failure: circuit opens.
    with pytest.raises(ConnectionError):
        await reliability.call_provider(
            provider_name, provider, [],
            "test-model", 0.2, 100,
        )

    # Third request: blocked without calling provider.
    with pytest.raises(CircuitOpenError):
        await reliability.call_provider(
            provider_name, provider, [],
            "test-model", 0.2, 100,
        )

    assert provider.calls == 2




@pytest.mark.asyncio
async def test_fallback_provider_succeeds(monkeypatch):
    reliability.CIRCUITS.pop("primary", None)
    reliability.CIRCUITS.pop("fallback", None)

    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 1
    )
    monkeypatch.setenv(
        "SENTINEL_FALLBACK_MODELS",
        "fallback/test-model",
    )

    primary = FakeProvider([
        ConnectionError("Primary provider unavailable"),
    ])
    fallback = FakeProvider([
        successful_result(),
    ])

    providers = {
        "primary/test-model": ("primary", primary),
        "fallback/test-model": ("fallback", fallback),
    }

    def fake_resolve_provider(model):
        if model not in providers:
            raise ValueError("Unknown provider")

        provider_name, provider = providers[model]
        return provider_name, provider, "test-model"

    monkeypatch.setattr(
        reliability, "resolve_provider",
        fake_resolve_provider,
    )

    provider_name, model_name, result = (
        await reliability.generate_with_fallback(
            model="primary/test-model",
            messages=[],
            temperature=0.2,
            max_tokens=100,
        )
    )

    assert provider_name == "fallback"
    assert model_name == "test-model"
    assert result.content == "Test response"
    assert primary.calls == 1
    assert fallback.calls == 1



@pytest.mark.asyncio
async def test_circuit_allows_request_after_cooldown(
    monkeypatch,
):
    provider_name = "circuit-recovery-test"
    reliability.CIRCUITS.pop(provider_name, None)

    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 1
    )
    monkeypatch.setattr(
        reliability, "FAILURE_THRESHOLD", 1
    )
    monkeypatch.setattr(
        reliability, "COOLDOWN_SECONDS", 0
    )

    provider = FakeProvider([
        ConnectionError("Temporary failure"),
        successful_result(),
    ])

    # First request fails and opens the circuit.
    with pytest.raises(ConnectionError):
        await reliability.call_provider(
            provider_name, provider, [],
            "test-model", 0.2, 100,
        )

    # With cooldown set to zero, the next request
    # should be allowed as a trial request.
    result = await reliability.call_provider(
        provider_name, provider, [],
        "test-model", 0.2, 100,
    )

    assert result.content == "Test response"
    assert provider.calls == 2
    assert reliability.CIRCUITS[provider_name].failures == 0
    assert reliability.CIRCUITS[provider_name].opened_at is None


@pytest.mark.asyncio
async def test_invalid_fallback_is_skipped(
    monkeypatch,
):
    reliability.CIRCUITS.pop("primary", None)
    reliability.CIRCUITS.pop("fallback", None)

    monkeypatch.setattr(
        reliability, "MAX_ATTEMPTS", 1
    )
    monkeypatch.setenv(
        "SENTINEL_FALLBACK_MODELS",
        "invalid/model,fallback/test-model",
    )

    primary = FakeProvider([
        ConnectionError("Primary unavailable"),
    ])
    fallback = FakeProvider([
        successful_result(),
    ])

    providers = {
        "primary/test-model": ("primary", primary),
        "fallback/test-model": ("fallback", fallback),
    }

    def fake_resolve_provider(model):
        if model not in providers:
            raise ValueError("Unknown provider")

        provider_name, provider = providers[model]
        return provider_name, provider, "test-model"

    monkeypatch.setattr(
        reliability, "resolve_provider",
        fake_resolve_provider,
    )

    provider_name, model_name, result = (
        await reliability.generate_with_fallback(
            model="primary/test-model",
            messages=[],
            temperature=0.2,
            max_tokens=100,
        )
    )

    assert provider_name == "fallback"
    assert model_name == "test-model"
    assert result.content == "Test response"
    assert primary.calls == 1
    assert fallback.calls == 1




@pytest.mark.asyncio
async def test_only_one_half_open_trial_is_allowed(
    monkeypatch,
):
    provider_name = "half-open-concurrency-test"

    monkeypatch.setattr(
        reliability, "COOLDOWN_SECONDS", 0
    )
    monkeypatch.setattr(
        reliability, "TIMEOUT_SECONDS", 2
    )
    monkeypatch.setattr(
        reliability, "CIRCUITS", {}
    )

    circuit = reliability.get_circuit(provider_name)
    circuit.failures = reliability.FAILURE_THRESHOLD
    circuit.opened_at = 0.0

    entered_provider = asyncio.Event()
    release_provider = asyncio.Event()

    class WaitingProvider:
        def __init__(self):
            self.calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            entered_provider.set()
            await release_provider.wait()
            return successful_result()

    provider = WaitingProvider()

    first_task = asyncio.create_task(
        reliability.call_provider(
            provider_name, provider, [],
            "test-model", 0.2, 100,
        )
    )

    try:
        await asyncio.wait_for(
            entered_provider.wait(), timeout=1
        )

        with pytest.raises(CircuitOpenError):
            await reliability.call_provider(
                provider_name, provider, [],
                "test-model", 0.2, 100,
            )

        assert provider.calls == 1

    finally:
        release_provider.set()

    result = await first_task

    assert result.content == "Test response"
    assert provider.calls == 1
    assert circuit.opened_at is None
    assert circuit.half_open_in_flight is False


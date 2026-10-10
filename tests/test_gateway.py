
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_mock_provider_returns_standard_response(monkeypatch):
    monkeypatch.setenv(
        "SENTINEL_API_KEYS",
        "test-key:team-alpha",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "test-key"},
            json={
                "model": "mock/echo",
                "messages": [
                    {"role": "user", "content": "Hello Sentinel"}
                ],
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "mock"
    assert data["tenant_id"] == "team-alpha"
    assert data["content"] == (
        "Mock response received: Hello Sentinel"
    )
    assert data["usage"]["input_tokens"] == 0
    assert data["usage"]["output_tokens"] == 0
    assert data["request_id"]
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_missing_api_key_is_rejected():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/echo",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
            },
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv(
        "SENTINEL_API_KEYS",
        "test-key:team-alpha",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "test-key"},
            json={
                "model": "unknown/model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
            },
        )

    assert response.status_code == 400



@pytest.mark.asyncio
async def test_gateway_falls_back_when_primary_provider_fails(
    monkeypatch,
):
    from app.providers.base import ProviderResult
    import app.main as main_module

    monkeypatch.setenv(
        "SENTINEL_API_KEYS",
        "test-key:team-alpha",
    )
    monkeypatch.setenv(
        "SENTINEL_FALLBACK_MODELS",
        "mock/echo",
    )

    async def fake_generate_with_fallback(
        model,
        messages,
        temperature,
        max_tokens,
    ):
        # Simulate the primary provider failing,
        # followed by a successful mock fallback.
        return (
            "mock",
            "echo",
            ProviderResult(
                content="Fallback response succeeded",
                model="echo",
                input_tokens=0,
                output_tokens=0,
                finish_reason="stop",
            ),
        )

    monkeypatch.setattr(
        main_module,
        "generate_with_fallback",
        fake_generate_with_fallback,
    )

    async with AsyncClient(
        transport=ASGITransport(app=main_module.app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "test-key"},
            json={
                "model": "groq/openai/gpt-oss-20b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Test fallback",
                    }
                ],
                "max_tokens": 50,
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "mock"
    assert data["model"] == "echo"
    assert data["content"] == "Fallback response succeeded"
    assert data["tenant_id"] == "team-alpha"





@pytest.mark.asyncio
async def test_gateway_uses_fallback_after_primary_failure(monkeypatch):
    import app.reliability as reliability_module
    from app.providers.registry import PROVIDERS

    monkeypatch.setenv("SENTINEL_API_KEYS", "test-key:team-alpha")
    monkeypatch.setenv("SENTINEL_FALLBACK_MODELS", "mock/echo")

    # Isolate circuit-breaker state for this test.
    monkeypatch.setattr(reliability_module, "CIRCUITS", {})

    # Simulate a primary Groq provider failure.
    # The actual fallback selection logic remains enabled.
    async def fail_primary(messages, model, temperature, max_tokens):
        raise RuntimeError("Simulated Groq failure")

    monkeypatch.setattr(PROVIDERS["groq"], "generate", fail_primary)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "test-key"},
            json={
                "model": "groq/openai/gpt-oss-20b",
                "messages": [
                    {"role": "user", "content": "Hello Sentinel"}
                ],
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "mock"
    assert data["model"] == "echo"
    assert data["tenant_id"] == "team-alpha"
    assert data["content"] == (
        "Mock response received: Hello Sentinel"
    )


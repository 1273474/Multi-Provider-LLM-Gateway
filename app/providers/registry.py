
from app.providers.mock import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqProvider

PROVIDERS = {
    "mock": MockProvider(),
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "groq": GroqProvider(),
}


def resolve_provider(model: str):
    if "/" not in model:
        raise ValueError(
            "Model must use provider/model format, such as mock/echo"
        )

    provider_name, model_name = model.split("/", 1)

    provider = PROVIDERS.get(provider_name)

    if provider is None:
        raise ValueError(f"Unsupported provider: {provider_name}")

    if not model_name:
        raise ValueError("Model name cannot be empty")

    return provider_name, provider, model_name

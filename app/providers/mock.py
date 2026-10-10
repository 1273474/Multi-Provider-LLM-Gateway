
from app.providers.base import ProviderResult
from app.schemas import Message


class MockProvider:
    async def generate(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        last_message = messages[-1].content

        return ProviderResult(
            content=f"Mock response received: {last_message}",
            model=model,
            input_tokens=0,
            output_tokens=0,
            finish_reason="stop",
        )

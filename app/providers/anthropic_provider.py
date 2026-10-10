
import os

from anthropic import AsyncAnthropic

from app.providers.base import ProviderResult
from app.schemas import Message


class AnthropicProvider:
    async def generate(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        client = AsyncAnthropic(api_key=api_key)

        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        chat_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role in ("user", "assistant")
        ]

        result = await client.messages.create(
            model=model,
            system="\n\n".join(system_messages),
            messages=chat_messages,
            max_tokens=max_tokens,
        )

        content = "".join(
            block.text
            for block in result.content
            if block.type == "text"
        )

        return ProviderResult(
            content=content,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            finish_reason=result.stop_reason or "unknown",
        )

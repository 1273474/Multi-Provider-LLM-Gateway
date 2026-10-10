
import os

from openai import AsyncOpenAI

from app.providers.base import ProviderResult
from app.schemas import Message


class OpenAIProvider:
    async def generate(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        client = AsyncOpenAI(api_key=api_key)

        result = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        choice = result.choices[0]
        usage = result.usage

        return ProviderResult(
            content=choice.message.content or "",
            model=result.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "unknown",
        )


from typing import Protocol

from app.schemas import Message, Usage


class ProviderResult:
    def __init__(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        finish_reason: str,
    ):
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason


class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        ...

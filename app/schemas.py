
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    model: str = Field(default="mock/echo", min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, ge=1, le=4096)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    request_id: str
    tenant_id: str
    provider: str
    model: str
    content: str
    usage: Usage
    finish_reason: str
    latency_ms: float

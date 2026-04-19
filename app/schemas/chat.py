from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolFunctionSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ChatTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)

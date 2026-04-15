from typing import Any, Literal

from pydantic import BaseModel


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
    stream: bool = False
    temperature: float | None = None

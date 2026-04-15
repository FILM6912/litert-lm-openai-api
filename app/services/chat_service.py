import asyncio
import base64
import json
import re
import uuid
from typing import Any, AsyncIterator

import httpx
import litert_lm

from app.core.engine import get_engine
from app.schemas.chat import ChatCompletionRequest, ChatMessage, ChatTool, ToolFunctionSpec


class _AllowToolsHandler(litert_lm.ToolEventHandler):
    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        return True

    def process_tool_response(
        self, tool_response: dict[str, Any]
    ) -> dict[str, Any]:
        return tool_response


def _data_url_to_blob(data_url: str) -> tuple[str, str | None]:
    m = re.match(r"^data:([^;]+);base64,(.+)$", data_url.strip(), re.DOTALL)
    if not m:
        raise ValueError("รูปแบบ data URL ไม่รองรับ")
    return m.group(2).strip(), m.group(1)


async def _image_url_to_blob(url: str) -> str:
    if url.startswith("data:"):
        b64, _ = _data_url_to_blob(url)
        return b64
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return base64.standard_b64encode(r.content).decode("ascii")


def _openai_part_to_litert(part: dict[str, Any]) -> dict[str, Any]:
    t = part.get("type")
    if t == "text":
        return {"type": "text", "text": part.get("text") or ""}
    if t == "image_url":
        inner = part.get("image_url") or {}
        url = inner.get("url") if isinstance(inner, dict) else str(inner)
        if not url:
            return {"type": "text", "text": ""}
        if url.startswith("data:"):
            b64, _ = _data_url_to_blob(url)
            return {"type": "image", "blob": b64}
        raise ValueError("image_url แบบ HTTP ต้องใช้ async path")
    if t == "input_audio":
        ia = part.get("input_audio") or {}
        data = ia.get("data") or ""
        return {"type": "audio", "blob": data}
    return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}


async def _openai_content_to_litert(
    content: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out: list[dict[str, Any]] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "image_url":
            inner = p.get("image_url") or {}
            url = inner.get("url") if isinstance(inner, dict) else str(inner)
            if url.startswith("data:"):
                out.append(_openai_part_to_litert(p))
            else:
                b64 = await _image_url_to_blob(url)
                out.append({"type": "image", "blob": b64})
        else:
            try:
                out.append(_openai_part_to_litert(p))
            except ValueError:
                out.append({"type": "text", "text": f"[unsupported part: {p.get('type')}]"})
    return out if out else [{"type": "text", "text": ""}]


def _assistant_openai_to_litert(msg: ChatMessage) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if isinstance(msg.content, str) and msg.content:
        parts.append({"type": "text", "text": msg.content})
    elif isinstance(msg.content, list):
        for p in msg.content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text") or ""})
    if msg.tool_calls:
        parts.append(
            {
                "type": "text",
                "text": "\n[tool_calls] " + json.dumps(msg.tool_calls, ensure_ascii=False),
            }
        )
    if not parts:
        parts.append({"type": "text", "text": ""})
    return {"role": "assistant", "content": parts}


def _tool_openai_to_litert(msg: ChatMessage) -> dict[str, Any]:
    body = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
    text = f"[tool result id={msg.tool_call_id}]\n{body}"
    return {"role": "user", "content": [{"type": "text", "text": text}]}


async def _openai_messages_to_litert(slice_msgs: list[ChatMessage]) -> list[dict[str, Any]]:
    litert_msgs: list[dict[str, Any]] = []
    for m in slice_msgs:
        if m.role == "system":
            c = await _openai_content_to_litert(m.content)
            litert_msgs.append({"role": "system", "content": c})
        elif m.role == "user":
            c = await _openai_content_to_litert(m.content)
            litert_msgs.append({"role": "user", "content": c})
        elif m.role == "assistant":
            litert_msgs.append(_assistant_openai_to_litert(m))
        elif m.role == "tool":
            litert_msgs.append(_tool_openai_to_litert(m))
    return litert_msgs


def _chunk_to_sse(chunk: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"


def _make_tool_callables(tools: list[ChatTool]) -> list[Any]:
    def _make_one(spec: ToolFunctionSpec):
        def _runner(**kwargs: Any) -> str:
            return json.dumps(
                {
                    "tool": spec.name,
                    "arguments": kwargs,
                    "note": "ฝั่งเซิร์ฟเวอร์ยังไม่ได้ผูกการทำงานจริง — แก้ _runner ตามต้องการ",
                },
                ensure_ascii=False,
            )

        _runner.__name__ = spec.name
        desc = spec.description or ""
        params = spec.parameters or {}
        props = params.get("properties") or {}
        req = params.get("required") or []
        import inspect
        try:
            sig = inspect.Signature(
                [
                    inspect.Parameter(
                        name=name,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        default=(
                            inspect.Parameter.empty
                            if name in req
                            else (p.get("default") if "default" in p else None)
                        ),
                    )
                    for name, p in props.items()
                ]
            )
            _runner.__signature__ = sig
        except Exception:
            pass
        _runner.__doc__ = desc
        return _runner

    return [_make_one(t.function) for t in tools]


async def stream_chat_completion(body: ChatCompletionRequest) -> AsyncIterator[bytes]:
    engine = get_engine()
    preface, last_msg = await _preface_and_last_payload(body.messages)
    tools_py: list[Any] = []
    if body.tools:
        tools_py = _make_tool_callables(body.tools)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(uuid.uuid1().time)

    conv_kw: dict[str, Any] = {"messages": preface}
    if tools_py:
        conv_kw["tools"] = tools_py
        conv_kw["tool_event_handler"] = _AllowToolsHandler()

    conversation = engine.create_conversation(**conv_kw)
    try:
        for chunk in conversation.send_message_async(last_msg):
            for item in chunk.get("content") or []:
                if item.get("type") != "text":
                    continue
                piece = item.get("text") or ""
                if not piece:
                    continue
                for ch in piece:
                    yield _chunk_to_sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": body.model,
                            "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}],
                        }
                    )
                    await asyncio.sleep(0)
    finally:
        conversation.__exit__(None, None, None)

    yield _chunk_to_sse(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


async def nonstream_chat_completion(body: ChatCompletionRequest) -> dict[str, Any]:
    engine = get_engine()
    preface, last_msg = await _preface_and_last_payload(body.messages)
    tools_py: list[Any] = []
    if body.tools:
        tools_py = _make_tool_callables(body.tools)

    conv_kw: dict[str, Any] = {"messages": preface}
    if tools_py:
        conv_kw["tools"] = tools_py
        conv_kw["tool_event_handler"] = _AllowToolsHandler()

    conversation = engine.create_conversation(**conv_kw)
    full_text = ""
    try:
        for chunk in conversation.send_message_async(last_msg):
            for item in chunk.get("content") or []:
                if item.get("type") == "text":
                    full_text += item.get("text") or ""
    finally:
        conversation.__exit__(None, None, None)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(uuid.uuid1().time)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _preface_and_last_payload(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from fastapi import HTTPException

    if not messages:
        raise HTTPException(400, detail="messages ว่างไม่ได้")
    last = messages[-1]
    if last.role == "user":
        preface = await _openai_messages_to_litert(messages[:-1])
        content = await _openai_content_to_litert(last.content)
        return preface, {"role": "user", "content": content}
    if last.role == "tool":
        preface = await _openai_messages_to_litert(messages)
        cont = {
            "role": "user",
            "content": [{"type": "text", "text": "ดำเนินการต่อจากผลลัพธ์เครื่องมือด้านบน (ตอบผู้ใช้)"}],
        }
        return preface, cont
    if last.role == "assistant":
        preface = await _openai_messages_to_litert(messages[:-1])
        cont = {"role": "user", "content": [{"type": "text", "text": "ตอบสานต่อจากข้อความผู้ช่วยล่าสุด"}]}
        return preface, cont
    raise HTTPException(400, detail="role สุดท้ายไม่รองรับ")

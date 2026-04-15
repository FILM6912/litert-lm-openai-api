"""
เซิร์ฟเวอร์แบบ OpenAI Chat Completions API บน LiteRT-LM (litert_lm)
รองรับ text, รูป (image_url / data URL), เสียง (input_audio base64), และ tools

รัน: uvicorn openai_litert_server:app --host 0.0.0.0 --port 8000

ตั้งค่า:
  LITERT_MODEL_PATH — path โมเดลเริ่มต้น (default models/gemma-4-E2B-it.litertlm ใต้ download root)
  LITERT_MODEL_DOWNLOAD_DIR — รากที่อนุญาต download/load (default cwd)
  LITERT_MODEL_SCAN_SUBDIR — โฟลเดอร์ที่สแกนหา .litertlm (default models; ตั้งเป็นสตริงว่าง = สแกนทั้งราก)
  LITERT_MODEL_SCAN_MAX_DEPTH — ความลึกสูงสุดจากจุดสแกน (default 2)
  LITERT_MODEL_CATALOG_SCAN_TIMEOUT_SEC — จำกัดเวลาสแกน (default 5 วินาที; 0 = ไม่จำกัด)
  LITERT_MODEL_CATALOG_TTL_SEC — cache รายการ catalog (default 3 วินาที)
  LITERT_ADMIN_TOKEN — ถ้าตั้ง ต้องส่ง Authorization: Bearer สำหรับ download/load/unload

API เสริม:
  GET  /ui                 หน้าเว็บทดสอบแชท + สตรีมข้อความ (HTML ฝังใน openai_litert_server.py)
  GET  /v1/model/status
  GET  /v1/models            รายการ model ตามไฟล์ .litertlm ใน models/ (id = path สัมพันธ์ download root)
  GET  /v1/model/catalog    รายการ *.litertlm ใต้โฟลเดอร์ models/ (default) หรือตาม SCAN_SUBDIR
  POST /v1/model/download  { "url", "path" }
  POST /v1/model/load     { "path", "model_id?" }
  POST /v1/model/unload

สตรีมแชท: POST /v1/chat/completions + "stream": true
  → Content-Type: text/event-stream; charset=utf-8 (SSE, บรรทัด data: ...)
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
import litert_lm
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

# --- config ---
_DEFAULT_MODEL_PATH = os.environ.get(
    "LITERT_MODEL_PATH", "models/gemma-4-E2B-it.litertlm"
)
_DEFAULT_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gemma-litert")
# โฟลเดอร์ที่อนุญาตให้ download ไฟล์ลง (realpath ต้องอยู่ใต้นี้)
_MODEL_DOWNLOAD_ROOT = os.path.realpath(
    os.environ.get("LITERT_MODEL_DOWNLOAD_DIR", os.getcwd())
)
# ถ้าตั้งค่า ต้องส่ง header Authorization: Bearer <token> สำหรับ download/load/unload
_ADMIN_TOKEN = os.environ.get("LITERT_ADMIN_TOKEN", "").strip()
# สแกนหา .litertlm: ไม่ลงทุกซับดิเรกทอรี (กัน cwd ใหญ่ / .git / cache)
_LITERT_SCAN_MAX_DEPTH = max(
    0, int(os.environ.get("LITERT_MODEL_SCAN_MAX_DEPTH", "2"))
)
_LITERT_CATALOG_SCAN_TIMEOUT = float(
    os.environ.get("LITERT_MODEL_CATALOG_SCAN_TIMEOUT_SEC", "5")
)
_LITERT_SCAN_SKIP: frozenset[str] = frozenset(
    ".git .svn .hg node_modules .cursor __pycache__ .venv .uv venv env .env "
    ".mypy_cache .pytest_cache .tox .nox dist build .idea .vscode "
    "anaconda3 miniconda3 conda pkgs site-packages .eggs "
    "htmlcov .hypothesis .ruff_cache .gradle target".split()
) | frozenset(
    x.strip()
    for x in os.environ.get("LITERT_MODEL_SCAN_EXTRA_SKIP", "").split(",")
    if x.strip()
)
# สแกน catalog เฉพาะโฟลเดอร์นี้ภายใต้ download root (default models); ถ้า env ตั้งเป็น "" = สแกนทั้งราก
if "LITERT_MODEL_SCAN_SUBDIR" in os.environ:
    _LITERT_SCAN_SUBDIR = os.environ["LITERT_MODEL_SCAN_SUBDIR"].strip()
else:
    _LITERT_SCAN_SUBDIR = "models"
# cache รายการ catalog ชั่วคราว (วินาที); ล้างหลัง load/unload/download
_LITERT_CATALOG_TTL = float(os.environ.get("LITERT_MODEL_CATALOG_TTL_SEC", "3"))
_catalog_cache_list: list[dict[str, Any]] | None = None
_catalog_cache_mono: float = 0.0

_engine: litert_lm.Engine | None = None
_engine_lock = threading.Lock()
_active_model_path: str = (
    os.path.realpath(_DEFAULT_MODEL_PATH)
    if os.path.isabs(_DEFAULT_MODEL_PATH)
    else os.path.realpath(
        os.path.join(_MODEL_DOWNLOAD_ROOT, _DEFAULT_MODEL_PATH)
    )
)
_model_id: str = _DEFAULT_MODEL_ID
# หลัง unload ต้องเรียก load ก่อนถึงจะแชทได้ (กันโหลดโมเดลซ้ำโดยไม่ตั้งใจ)
_explicit_unload: bool = False


def _require_admin(authorization: str | None) -> None:
    if not _ADMIN_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="ต้องมี Authorization: Bearer")
    if authorization[7:].strip() != _ADMIN_TOKEN:
        raise HTTPException(403, detail="token ไม่ถูกต้อง")


def _resolve_under_download_root(path: str) -> str:
    """คืน absolute path ที่อยู่ภายใต้ _MODEL_DOWNLOAD_ROOT เท่านั้น"""
    p = path.strip()
    if not p or p.endswith("/") or p.endswith("\\"):
        raise HTTPException(400, detail="path ไฟล์ไม่ถูกต้อง")
    if os.path.isabs(p):
        full = os.path.realpath(p)
    else:
        full = os.path.realpath(os.path.join(_MODEL_DOWNLOAD_ROOT, p))
    root = os.path.realpath(_MODEL_DOWNLOAD_ROOT)
    if full != root and not full.startswith(root + os.sep):
        raise HTTPException(
            403,
            detail=f"path ต้องอยู่ภายใต้ LITERT_MODEL_DOWNLOAD_DIR ({root})",
        )
    return full


def _skip_catalog_dir(name: str) -> bool:
    if name in _LITERT_SCAN_SKIP:
        return True
    low = name.lower()
    if low.endswith("_cache") or low.endswith(".cache"):
        return True
    if "xnnpack_cache" in low:
        return True
    if "litertlm" in low and "cache" in low:
        return True
    return False


def invalidate_model_catalog_cache() -> None:
    global _catalog_cache_list, _catalog_cache_mono
    _catalog_cache_list = None
    _catalog_cache_mono = 0.0


def _litert_catalog_entries_uncached() -> list[dict[str, Any]]:
    """สแกน *.litertlm แบบ BFS + จำกัดความลึก + ข้ามโฟลเดอร์หนัก (ไม่ลง .git / cache)"""
    root = os.path.realpath(_MODEL_DOWNLOAD_ROOT)
    if _LITERT_SCAN_SUBDIR:
        scan_root = os.path.realpath(os.path.join(root, _LITERT_SCAN_SUBDIR))
        if not scan_root.startswith(root + os.sep) and scan_root != root:
            scan_root = root
    else:
        scan_root = root

    ap_active = os.path.realpath(_active_model_path)
    in_mem = _engine is not None
    items: list[dict[str, Any]] = []
    if not os.path.isdir(scan_root):
        return items

    rel_base = root
    queue: deque[tuple[str, int]] = deque([(scan_root, 0)])
    # กันวนโฟลเดอร์จำนวนมหาศาล
    _max_dir_visits = 50_000
    visits = 0

    while queue:
        dirpath, depth = queue.popleft()
        visits += 1
        if visits > _max_dir_visits:
            break
        try:
            with os.scandir(dirpath) as it:
                for ent in it:
                    try:
                        if ent.is_dir(follow_symlinks=False):
                            if _skip_catalog_dir(ent.name):
                                continue
                            next_depth = depth + 1
                            if next_depth > _LITERT_SCAN_MAX_DEPTH:
                                continue
                            queue.append((ent.path, next_depth))
                        elif ent.is_file(follow_symlinks=False) and ent.name.endswith(
                            ".litertlm"
                        ):
                            abs_p = os.path.realpath(ent.path)
                            try:
                                rel = os.path.relpath(abs_p, rel_base).replace(
                                    "\\", "/"
                                )
                            except ValueError:
                                rel = abs_p.replace("\\", "/")
                            items.append(
                                {
                                    "path": rel,
                                    "label": rel,
                                    "name": ent.name,
                                    "absolute_path": abs_p,
                                    "loaded": in_mem and abs_p == ap_active,
                                }
                            )
                    except OSError:
                        continue
        except OSError:
            continue

    items.sort(key=lambda x: x["path"].lower())
    return items


def _catalog_merge_active_model(
    items: list[dict[str, Any]], root: str
) -> list[dict[str, Any]]:
    """ถ้าโมเดลที่ใช้อยู่ไม่อยู่ในรายการ (เช่น สแกนหมดเวลา) ใส่เข้าไปให้เลือกได้"""
    ap = os.path.realpath(_active_model_path)
    if not ap.endswith(".litertlm") or not os.path.isfile(ap):
        return items
    paths = {m["path"] for m in items}
    try:
        rel = os.path.relpath(ap, os.path.realpath(root)).replace("\\", "/")
    except ValueError:
        rel = ap.replace("\\", "/")
    if rel in paths:
        return items
    in_mem = _engine is not None
    items = list(items)
    items.insert(
        0,
        {
            "path": rel,
            "label": rel + " (active)",
            "name": os.path.basename(ap),
            "absolute_path": ap,
            "loaded": in_mem,
        },
    )
    return items


def _litert_catalog_scan_in_thread() -> tuple[list[dict[str, Any]], bool]:
    """คืน (รายการ, timed_out) — รันในเธรดแยกเพื่อไม่บล็อก event loop"""
    if _LITERT_CATALOG_SCAN_TIMEOUT <= 0:
        return _litert_catalog_entries_uncached(), False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_litert_catalog_entries_uncached)
        try:
            return fut.result(timeout=_LITERT_CATALOG_SCAN_TIMEOUT), False
        except concurrent.futures.TimeoutError:
            return [], True


async def _async_merged_catalog_entries() -> tuple[
    list[dict[str, Any]], bool, bool
]:
    """รายการโมเดลบนดิสก์หลัง merge active — ใช้ร่วมกับ /v1/model/catalog และ /v1/models
    คืน (entries, scan_timed_out, catalog_from_cache)
    """
    global _catalog_cache_list, _catalog_cache_mono
    now = time.monotonic()
    if (
        _LITERT_CATALOG_TTL > 0
        and _catalog_cache_list is not None
        and now - _catalog_cache_mono < _LITERT_CATALOG_TTL
    ):
        scanned = list(_catalog_cache_list)
        timed_out = False
        from_cache = True
    else:
        scanned, timed_out = await asyncio.to_thread(_litert_catalog_scan_in_thread)
        from_cache = False
        if not timed_out:
            _catalog_cache_list = list(scanned)
            _catalog_cache_mono = time.monotonic()
    entries = _catalog_merge_active_model(list(scanned), _MODEL_DOWNLOAD_ROOT)
    return entries, timed_out, from_cache


def get_engine() -> litert_lm.Engine:
    global _engine, _explicit_unload
    if _explicit_unload:
        raise HTTPException(
            503,
            detail="โมเดลถูก unload แล้ว — เรียก POST /v1/model/load ก่อนใช้แชท",
        )
    with _engine_lock:
        if _engine is None:
            if not os.path.isfile(_active_model_path):
                raise HTTPException(
                    503,
                    detail=f"ไม่พบไฟล์โมเดล: {_active_model_path}",
                )
            _engine = litert_lm.Engine(
                _active_model_path,
                backend=litert_lm.Backend.CPU,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
        return _engine


def _current_model_id() -> str:
    return _model_id


# --- OpenAI request models (subset) ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str = Field(default_factory=_current_model_id)
    messages: list[ChatMessage]
    tools: list[ChatTool] | None = None
    stream: bool = False
    temperature: float | None = None


class _AllowToolsHandler(litert_lm.ToolEventHandler):
    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        return True

    def process_tool_response(
        self, tool_response: dict[str, Any]
    ) -> dict[str, Any]:
        return tool_response


class ModelDownloadRequest(BaseModel):
    """ดาวน์โหลดไฟล์โมเดลจาก URL ลง path ภายใต้ LITERT_MODEL_DOWNLOAD_DIR"""

    url: str
    path: str


class ModelLoadRequest(BaseModel):
    """โหลดโมเดลเข้าแรม (unload โมเดลเดิมก่อนถ้ามี)"""

    path: str
    model_id: str | None = None


def _data_url_to_blob(data_url: str) -> tuple[str, str | None]:
    """คืน (raw_base64_payload, mime_or_none) จาก data:image/png;base64,..."""
    m = re.match(
        r"^data:([^;]+);base64,(.+)$", data_url.strip(), re.DOTALL
    )
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
        raise ValueError("image_url แบบ HTTP ต้องใช้ async path — ใช้ parse แบบ async")
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
                out.append(
                    {
                        "type": "text",
                        "text": f"[unsupported part: {p.get('type')}]",
                    }
                )
    return out if out else [{"type": "text", "text": ""}]


def _assistant_openai_to_litert(msg: ChatMessage) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if isinstance(msg.content, str) and msg.content:
        parts.append({"type": "text", "text": msg.content})
    elif isinstance(msg.content, list):
        for p in msg.content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(
                    {"type": "text", "text": p.get("text") or ""}
                )
    if msg.tool_calls:
        parts.append(
            {
                "type": "text",
                "text": "\n[tool_calls] "
                + json.dumps(msg.tool_calls, ensure_ascii=False),
            }
        )
    if not parts:
        parts.append({"type": "text", "text": ""})
    return {"role": "assistant", "content": parts}


def _tool_openai_to_litert(msg: ChatMessage) -> dict[str, Any]:
    body = msg.content if isinstance(msg.content, str) else json.dumps(
        msg.content, ensure_ascii=False
    )
    text = f"[tool result id={msg.tool_call_id}]\n{body}"
    return {"role": "user", "content": [{"type": "text", "text": text}]}


async def _openai_messages_to_litert(
    slice_msgs: list[ChatMessage],
) -> list[dict[str, Any]]:
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


async def _preface_and_last_payload(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """คืน (preface_messages, last_turn) สำหรับ create_conversation + send_message_async"""
    if not messages:
        raise HTTPException(400, detail="messages ว่างไม่ได้")
    last = messages[-1]
    if last.role == "user":
        preface = await _openai_messages_to_litert(messages[:-1])
        content = await _openai_content_to_litert(last.content)
        return preface, {"role": "user", "content": content}
    if last.role == "tool":
        # ประวัติรวมผล tool ล่าสุดทั้งหมดอยู่ใน preface แล้วค่อยถามต่อ
        preface = await _openai_messages_to_litert(messages)
        cont = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "ดำเนินการต่อจากผลลัพธ์เครื่องมือด้านบน (ตอบผู้ใช้)",
                }
            ],
        }
        return preface, cont
    if last.role == "assistant":
        preface = await _openai_messages_to_litert(messages[:-1])
        cont = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "ตอบสานต่อจากข้อความผู้ช่วยล่าสุด",
                }
            ],
        }
        return preface, cont
    raise HTTPException(400, detail="role สุดท้ายไม่รองรับ")


def _make_tool_callables(tools: list[ChatTool]) -> list[Any]:
    """สร้างฟังก์ชัน Python จาก OpenAI tools schema สำหรับ litert_lm."""

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
        lines = [desc or f"เรียกใช้ {spec.name}", "", "Args:"]
        for name, schema in props.items():
            st = schema.get("type", "string") if isinstance(schema, dict) else "string"
            opt = "" if name in req else " (optional)"
            lines.append(f"    {name} ({st}){opt}: {schema.get('description', '')}")
        _runner.__doc__ = "\n".join(lines)
        ann: dict[str, type] = {}
        for name in props:
            ann[name] = Any
        if ann:
            _runner.__annotations__ = {**getattr(_runner, "__annotations__", {}), **ann}
        return _runner

    return [_make_one(t.function) for t in tools]


def _chunk_to_sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_chat_completion(body: ChatCompletionRequest) -> AsyncIterator[str]:
    engine = get_engine()
    preface, last_msg = await _preface_and_last_payload(body.messages)
    tools_py: list[Any] = []
    if body.tools:
        tools_py = _make_tool_callables(body.tools)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield _chunk_to_sse(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )

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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": ch},
                                    "finish_reason": None,
                                }
                            ],
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


_UI_TEST_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LiteRT — Chat</title>
  <style>
    :root {
      --bg: #0d0d0f;
      --sidebar: #1a1a1f;
      --panel: #222228;
      --border: #2e2e36;
      --text: #ececf1;
      --muted: #8e8ea0;
      --accent: #10a37f;
      --accent-dim: #1a7f64;
      --warn: #f59e0b;
      --err: #ef4444;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Sarabun", "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    .app { display: flex; flex: 1; min-height: 0; }
    aside.sidebar {
      width: 280px; min-width: 260px; background: var(--sidebar);
      border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 0.85rem;
    }
    .brand { font-weight: 700; font-size: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); margin-bottom: 0.85rem; letter-spacing: 0.02em; }
    .brand span { color: var(--accent); }
    label.lbl { display: block; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
    select#modelSelect {
      width: 100%; padding: 0.55rem 0.6rem; border-radius: 8px; border: 1px solid var(--border);
      background: var(--panel); color: var(--text); font-size: 0.88rem; cursor: pointer;
    }
    select#modelSelect:focus { outline: none; border-color: var(--accent); }
    .model-row { margin-top: 0.65rem; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .pill {
      display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.75rem; padding: 0.25rem 0.55rem;
      border-radius: 999px; background: var(--panel); border: 1px solid var(--border);
    }
    .pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
    .pill.loaded .dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .pill.warn .dot { background: var(--warn); }
    .pill.err .dot { background: var(--err); }
    .btn-row { display: flex; gap: 0.45rem; margin-top: 0.65rem; flex-wrap: wrap; }
    button {
      cursor: pointer; border: none; border-radius: 8px; padding: 0.45rem 0.75rem; font-size: 0.82rem; font-weight: 600;
      background: var(--accent); color: #fff;
    }
    button:hover:not(:disabled) { filter: brightness(1.08); }
    button.secondary { background: var(--panel); color: var(--text); border: 1px solid var(--border); }
    button.danger { background: #3d2020; color: #fca5a5; border: 1px solid #5c2a2a; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    details.adv { margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 0.75rem; font-size: 0.85rem; }
    details.adv summary { cursor: pointer; color: var(--muted); user-select: none; }
    details.adv .inner { margin-top: 0.65rem; }
    input[type="text"], input[type="url"], textarea {
      width: 100%; padding: 0.45rem 0.55rem; border-radius: 6px; border: 1px solid var(--border);
      background: var(--bg); color: var(--text); font-size: 0.85rem;
    }
    textarea { min-height: 72px; resize: vertical; font-family: inherit; }
    main.main {
      flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0;
      padding: 0; margin: 0; width: 100%; max-width: none;
    }
    .chat-toolbar {
      flex-shrink: 0; display: flex; flex-wrap: wrap; gap: 0.65rem; align-items: flex-end;
      padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); background: var(--bg);
    }
    .chat-toolbar .tf { flex: 1; min-width: 140px; }
    .chat-toolbar .tf .lbl { margin-bottom: 0.25rem; }
    .chat-toolbar input { width: 100%; font-size: 0.82rem; padding: 0.4rem 0.5rem; }
    .chat-toolbar .tb-actions { display: flex; align-items: center; gap: 0.5rem; padding-bottom: 0.15rem; }
    .chat-toolbar a { color: var(--accent); font-size: 0.85rem; }
    .chat-messages {
      flex: 1; min-height: 0; overflow-y: auto; padding: 1rem 1.15rem;
      display: flex; flex-direction: column; gap: 0.85rem;
    }
    .msg { display: flex; width: 100%; }
    .msg.user { justify-content: flex-end; }
    .msg.assistant { justify-content: flex-start; }
    .bubble {
      max-width: min(90%, 720px); padding: 0.65rem 0.95rem; border-radius: 14px;
      line-height: 1.55; font-size: 0.9rem; white-space: pre-wrap; word-break: break-word;
    }
    .msg.user .bubble {
      background: linear-gradient(145deg, var(--accent-dim), #145a47); color: #f4fffb;
      border: 1px solid var(--accent);
    }
    .msg.assistant .bubble {
      background: var(--panel); border: 1px solid var(--border); color: var(--text);
    }
    .msg.assistant .bubble.streaming { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(16, 163, 127, 0.25); }
    .bubble.err-bubble { border-color: var(--err) !important; color: #fecaca !important; background: #2a1515 !important; }
    .chat-welcome {
      margin: auto; text-align: center; color: var(--muted); font-size: 0.88rem; max-width: 360px; line-height: 1.5; padding: 2rem 1rem;
    }
    .chat-composer {
      flex-shrink: 0; border-top: 1px solid var(--border); padding: 0.75rem 1rem 1rem;
      background: var(--sidebar);
    }
    .composer-row { display: flex; gap: 0.65rem; align-items: flex-end; max-width: 960px; margin: 0 auto; width: 100%; }
    #chatInput {
      flex: 1; min-height: 48px; max-height: 200px; resize: none; padding: 0.65rem 0.85rem;
      border-radius: 12px; font-family: inherit; font-size: 0.9rem; line-height: 1.45;
    }
    .composer-actions { display: flex; flex-direction: column; gap: 0.4rem; align-items: stretch; }
    #btnSend { min-width: 92px; }
    .composer-meta { font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem; max-width: 960px; margin-left: auto; margin-right: auto; min-height: 1.2em; }
    .row { display: flex; gap: 0.65rem; flex-wrap: wrap; align-items: flex-end; margin-bottom: 0.5rem; }
    .row > div { flex: 1; min-width: 140px; }
    .check { display: flex; align-items: center; gap: 0.4rem; padding-bottom: 0.2rem; }
    .check label, .check span { margin: 0; font-size: 0.8rem; cursor: pointer; user-select: none; }
    .err { color: var(--err); }
    #modelLoadNote { font-size: 0.72rem; color: var(--muted); margin-top: 0.35rem; line-height: 1.35; }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">Lite<span>RT</span> · LM</div>
      <label class="lbl" for="modelSelect">โมเดลจาก <code>GET /v1/models</code></label>
      <select id="modelSelect">
        <option value="">— กำลังโหลดรายการโมเดล… —</option>
      </select>
      <div id="modelLoadNote">รายการจาก <code>/v1/models</code> (ไฟล์ใน <code>models/</code>) — เลือกแล้วโหลดเข้าแรมอัตโนมัติ</div>
      <div class="model-row">
        <span class="pill" id="loadPill"><span class="dot"></span><span id="loadPillText">ไม่ทราบสถานะ</span></span>
      </div>
      <div class="btn-row">
        <button type="button" class="danger secondary" id="btnUnloadModel" title="Unload จากแรม">Unload</button>
        <button type="button" class="secondary" id="btnRefreshCatalog">รีเฟรช /v1/models</button>
      </div>
      <label class="lbl" for="systemPrompt" style="margin-top:0.85rem">System prompt</label>
      <textarea id="systemPrompt" rows="3" placeholder="You are a helpful assistant." style="min-height:68px;font-size:0.82rem"></textarea>
      <details class="adv">
        <summary>ดาวน์โหลด / Admin token</summary>
        <div class="inner">
          <label class="lbl" for="adminToken">Bearer (ถ้าเซิร์ฟเวอร์ตั้ง LITERT_ADMIN_TOKEN)</label>
          <input type="text" id="adminToken" placeholder="optional" autocomplete="off" />
          <label class="lbl" for="dlUrl" style="margin-top:0.5rem">Download URL</label>
          <input type="url" id="dlUrl" placeholder="https://..." />
          <label class="lbl" for="dlPath">path ภายใต้ download root</label>
          <input type="text" id="dlPath" placeholder="models/foo.litertlm" />
          <div class="btn-row">
            <button type="button" class="secondary" id="btnDownload">Download</button>
          </div>
          <pre id="adminOut" style="margin-top:0.5rem;font-size:0.75rem;white-space:pre-wrap;max-height:120px;overflow:auto;color:var(--muted)"></pre>
        </div>
      </details>
    </aside>
    <main class="main">
      <div class="chat-toolbar">
        <div class="tf">
          <label class="lbl" for="baseUrl">Base URL</label>
          <input type="text" id="baseUrl" placeholder="ว่าง = โดเมนนี้" />
        </div>
        <div class="tf">
          <label class="lbl" for="modelId">model id</label>
          <input type="text" id="modelId" placeholder="sync จากโมเดลที่โหลด" />
        </div>
        <div class="tb-actions">
          <button type="button" class="secondary" id="btnClearChat" title="ล้างประวัติในหน้าจอ">ล้างแชท</button>
          <a href="/docs">API docs</a>
        </div>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="chat-welcome" id="chatWelcome">ส่งข้อความด้านล่าง · ใช้ <code>/v1/chat/completions</code> · โมเดลจาก <code>GET /v1/models</code></div>
      </div>
      <div class="chat-composer">
        <div class="composer-row">
          <textarea id="chatInput" rows="2" placeholder="พิมพ์ข้อความ… (Enter ส่ง · Shift+Enter ขึ้นบรรทัด)"></textarea>
          <div class="composer-actions">
            <button type="button" id="btnSend">ส่ง</button>
            <label class="check">
              <input type="checkbox" id="useStream" checked />
              <span>สตรีม</span>
            </label>
          </div>
        </div>
        <div class="composer-meta" id="chatMeta"></div>
      </div>
    </main>
  </div>
  <script>
(function () {
  const $ = (id) => document.getElementById(id);
  let _catalogActiveRel = "";
  let _loadingModel = false;
  /** @type {{role:string,content:string}[]} */
  let chatHistory = [];

  function apiBase() {
    const u = $("baseUrl").value.trim();
    return u ? u.replace(/\/$/, "") : "";
  }

  function adminHeaders(json) {
    const h = json ? { "Content-Type": "application/json" } : {};
    const t = $("adminToken").value.trim();
    if (t) h["Authorization"] = "Bearer " + t;
    return h;
  }

  const FETCH_TIMEOUT_MS = 20000;

  async function fetchWithTimeout(url, options, ms) {
    const ctrl = new AbortController();
    const id = setTimeout(function () { ctrl.abort(); }, ms);
    try {
      return await fetch(url, Object.assign({}, options || {}, { signal: ctrl.signal }));
    } finally {
      clearTimeout(id);
    }
  }

  async function fetchModelsList() {
    const res = await fetchWithTimeout(apiBase() + "/v1/models", {}, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error("models " + res.status);
    return res.json();
  }

  async function fetchStatus() {
    const res = await fetchWithTimeout(apiBase() + "/v1/model/status", {}, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error("status " + res.status);
    return res.json();
  }

  function setPill(loaded, explicitUnload, engineInMemory) {
    const pill = $("loadPill");
    const txt = $("loadPillText");
    pill.classList.remove("loaded", "warn", "err");
    if (explicitUnload && !engineInMemory) {
      pill.classList.add("warn");
      txt.textContent = "Unloaded — เลือกโมเดลเพื่อโหลด";
      return;
    }
    if (loaded && engineInMemory) {
      pill.classList.add("loaded");
      txt.textContent = "โหลดในแรมแล้ว";
      return;
    }
    if (engineInMemory) {
      pill.classList.add("loaded");
      txt.textContent = "โหลดแล้ว";
      return;
    }
    txt.textContent = "ยังไม่โหลดในแรม (จะโหลดตอนแชทหรือเลือกโมเดล)";
  }

  function idMatchesActiveModel(id, activeAbs) {
    if (!id || !activeAbs) return false;
    const a = activeAbs.replace(/\\/g, "/");
    const i = id.replace(/\\/g, "/");
    return a === i || a.endsWith("/" + i);
  }

  async function refreshCatalog(selectPath) {
    const sel = $("modelSelect");
    try {
      const [listData, st] = await Promise.all([
        fetchModelsList(),
        fetchStatus(),
      ]);
      const rows = listData.data || [];
      _catalogActiveRel = "";
      const activeAbs = (st.active_model_path || "").replace(/\\/g, "/");
      sel.innerHTML = '<option value="">— เลือกโมเดล —</option>';
      for (let k = 0; k < rows.length; k++) {
        const row = rows[k];
        const id = row.id;
        if (!id) continue;
        const opt = document.createElement("option");
        opt.value = id;
        let label = id;
        if (idMatchesActiveModel(id, activeAbs) && st.engine_in_memory) {
          label += "  ● โหลดอยู่";
          _catalogActiveRel = id;
        }
        opt.textContent = label;
        sel.appendChild(opt);
      }
      const want = selectPath || _catalogActiveRel;
      if (want && [...sel.options].some(function (o) { return o.value === want; })) sel.value = want;
      setPill(
        st.engine_in_memory && !st.must_load_before_chat,
        st.must_load_before_chat,
        st.engine_in_memory
      );
      if (st.model_id && !$("modelId").value) $("modelId").value = st.model_id;
      if (st.model_id) $("modelId").placeholder = st.model_id;
      const note = $("modelLoadNote");
      note.textContent =
        "รายการจาก GET /v1/models — เลือกแล้วโหลดเข้าแรม (path เช่น models/xxx.litertlm)";
      note.style.color = "";
    } catch (e) {
      const msg = (e && e.name === "AbortError") ? "หมดเวลาเชื่อมต่อ — ตรวจ Base URL / เซิร์ฟเวอร์" : String(e.message || e);
      sel.innerHTML = '<option value="">— โหลดรายการไม่ได้ —</option>';
      $("loadPillText").textContent = msg;
      $("loadPill").classList.add("err");
    }
  }

  async function autoLoadSelected() {
    const path = $("modelSelect").value;
    if (!path || _loadingModel) return;
    _loadingModel = true;
    $("modelSelect").disabled = true;
    $("loadPillText").textContent = "กำลังโหลดโมเดล…";
    $("loadPill").classList.remove("loaded", "warn", "err");
    try {
      const res = await fetch(apiBase() + "/v1/model/load", {
        method: "POST",
        headers: adminHeaders(true),
        body: JSON.stringify({ path: path, model_id: null }),
      });
      const text = await res.text();
      let j;
      try { j = JSON.parse(text); } catch { j = {}; }
      if (!res.ok) throw new Error(text.slice(0, 400));
      if (j.model_id) $("modelId").value = j.model_id;
      await refreshCatalog(path);
    } catch (e) {
      $("loadPill").classList.add("err");
      $("loadPillText").textContent = "โหลดไม่สำเร็จ: " + (e.message || e);
    } finally {
      _loadingModel = false;
      $("modelSelect").disabled = false;
    }
  }

  $("modelSelect").addEventListener("change", () => {
    autoLoadSelected();
  });

  $("btnRefreshCatalog").onclick = () => refreshCatalog($("modelSelect").value);
  $("btnUnloadModel").onclick = async () => {
    $("loadPillText").textContent = "กำลัง unload…";
    try {
      const res = await fetch(apiBase() + "/v1/model/unload", {
        method: "POST",
        headers: adminHeaders(false),
      });
      await res.text();
      await refreshCatalog($("modelSelect").value);
    } catch (e) {
      $("loadPill").classList.add("err");
      $("loadPillText").textContent = String(e.message || e);
    }
  };

  $("btnDownload").onclick = async () => {
    const o = $("adminOut");
    const url = $("dlUrl").value.trim();
    const path = $("dlPath").value.trim();
    if (!url || !path) { o.textContent = "ใส่ URL และ path"; return; }
    o.textContent = "กำลังดาวน์โหลด…";
    try {
      const res = await fetch(apiBase() + "/v1/model/download", {
        method: "POST",
        headers: adminHeaders(true),
        body: JSON.stringify({ url, path }),
      });
      const text = await res.text();
      try { o.textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch { o.textContent = text; }
      await refreshCatalog($("modelSelect").value);
    } catch (e) { o.textContent = String(e); }
  };

  function buildApiMessages() {
    const sys = $("systemPrompt").value.trim();
    const msgs = [];
    if (sys) msgs.push({ role: "system", content: sys });
    for (let i = 0; i < chatHistory.length; i++) msgs.push(chatHistory[i]);
    return msgs;
  }

  function scrollChatToBottom() {
    const el = $("chatMessages");
    el.scrollTop = el.scrollHeight;
  }

  function hideWelcomeIfAny() {
    const w = $("chatWelcome");
    if (w && w.parentNode) w.remove();
  }

  function appendMessageBubble(role, text) {
    hideWelcomeIfAny();
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    wrap.appendChild(b);
    $("chatMessages").appendChild(wrap);
    scrollChatToBottom();
    return b;
  }

  function modelName() {
    const sel = $("modelSelect").value.trim();
    if (sel) return sel;
    const m = $("modelId").value.trim();
    return m || "gemma-litert";
  }

  async function parseSseStream(response, onDelta) {
    const reader = response.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    const t0 = performance.now();
    while (true) {
      const { done, value } = await reader.read();
      buf += dec.decode(value || new Uint8Array(), { stream: !done });
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const j = JSON.parse(payload);
            const c = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
            if (c) onDelta(c);
          } catch (_) {}
        }
      }
      if (done) break;
    }
    return performance.now() - t0;
  }

  $("btnClearChat").onclick = () => {
    chatHistory = [];
    const box = $("chatMessages");
    box.innerHTML = "";
    const w = document.createElement("div");
    w.className = "chat-welcome";
    w.id = "chatWelcome";
    w.innerHTML = "ส่งข้อความด้านล่าง · ใช้ <code>/v1/chat/completions</code> · โมเดลจาก <code>GET /v1/models</code>";
    box.appendChild(w);
    $("chatMeta").textContent = "";
  };

  $("chatInput").addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      $("btnSend").click();
    }
  });

  $("btnSend").onclick = async () => {
    const meta = $("chatMeta");
    meta.textContent = "";
    const userText = $("chatInput").value.trim();
    if (!userText) {
      meta.innerHTML = '<span class="err">กรุณาใส่ข้อความ</span>';
      return;
    }
    const stream = $("useStream").checked;
    chatHistory.push({ role: "user", content: userText });
    $("chatInput").value = "";
    appendMessageBubble("user", userText);
    const assistantBubble = appendMessageBubble("assistant", "");
    assistantBubble.classList.add("streaming");
    const url = apiBase() + "/v1/chat/completions";
    const messages = buildApiMessages();
    $("btnSend").disabled = true;
    let assistantText = "";
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName(), messages, stream }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(res.status + " " + t.slice(0, 500));
      }
      if (stream) {
        const ms = await parseSseStream(res, function (c) {
          assistantText += c;
          assistantBubble.textContent = assistantText;
          scrollChatToBottom();
        });
        assistantBubble.classList.remove("streaming");
        chatHistory.push({ role: "assistant", content: assistantText });
        meta.textContent = "สตรีมเสร็จ · " + Math.round(ms) + " ms";
      } else {
        const data = await res.json();
        assistantText =
          (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
        assistantBubble.textContent = assistantText;
        assistantBubble.classList.remove("streaming");
        chatHistory.push({ role: "assistant", content: assistantText });
        meta.textContent = "ไม่สตรีม";
      }
    } catch (e) {
      assistantBubble.classList.remove("streaming");
      assistantBubble.classList.add("err-bubble");
      assistantBubble.textContent = String(e.message || e);
      meta.innerHTML = '<span class="err">' + String(e.message || e) + "</span>";
    } finally {
      $("btnSend").disabled = false;
      scrollChatToBottom();
    }
  };

  refreshCatalog();
})();
  </script>
</body>
</html>
"""


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    """สร้างโฟลเดอร์ models/ ใต้ download root ถ้ายังไม่มี (เก็บไฟล์ .litertlm)"""
    try:
        os.makedirs(os.path.join(_MODEL_DOWNLOAD_ROOT, "models"), exist_ok=True)
    except OSError:
        pass
    yield


app = FastAPI(
    title="OpenAI-compatible LiteRT-LM",
    lifespan=_app_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect_ui():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui_test_page():
    """หน้าเว็บทดสอบแชท + สตรีมข้อความ + สถานะโมเดล (HTML ฝังในไฟล์นี้)"""
    return HTMLResponse(content=_UI_TEST_PAGE_HTML, media_type="text/html; charset=utf-8")


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible: แต่ละรายการคือไฟล์ .litertlm ที่เจอใน models/ (หรือ SCAN_SUBDIR)
    field `id` = path สำหรับส่งใน chat ว่า model (เช่น models/gemma-4-E2B-it.litertlm)
    """
    entries, _, _ = await _async_merged_catalog_entries()
    created = int(time.time())
    data: list[dict[str, Any]] = [
        {
            "id": e["path"].replace("\\", "/"),
            "object": "model",
            "created": created,
            "owned_by": "litert-lm",
        }
        for e in entries
    ]
    if not data:
        data = [
            {
                "id": _model_id,
                "object": "model",
                "created": created,
                "owned_by": "litert-lm",
            }
        ]
    return {"object": "list", "data": data}


@app.get("/v1/model/status")
def model_status():
    """สถานะโมเดล: โหลดในแรมหรือไม่, path, model id สำหรับ OpenAI client"""
    return {
        "engine_in_memory": _engine is not None,
        "must_load_before_chat": _explicit_unload,
        "active_model_path": _active_model_path,
        "model_id": _model_id,
        "download_root": _MODEL_DOWNLOAD_ROOT,
        "admin_auth_required": bool(_ADMIN_TOKEN),
    }


@app.get("/v1/model/catalog")
async def model_catalog():
    """รายการโมเดลบนดิสก์ — สแกนในเธรด + จำกัดเวลา กัน UI ค้าง"""
    models, timed_out, from_cache = await _async_merged_catalog_entries()
    return {
        "models": models,
        "files": [e["path"].replace("\\", "/") for e in models],
        "download_root": _MODEL_DOWNLOAD_ROOT,
        "scan_subdir": _LITERT_SCAN_SUBDIR or None,
        "scan_max_depth": _LITERT_SCAN_MAX_DEPTH,
        "scan_timeout_sec": _LITERT_CATALOG_SCAN_TIMEOUT,
        "active_model_path": _active_model_path,
        "engine_in_memory": _engine is not None,
        "must_load_before_chat": _explicit_unload,
        "model_id": _model_id,
        "scan_timed_out": timed_out,
        "catalog_from_cache": from_cache,
    }


@app.post("/v1/model/download")
async def model_download(
    body: ModelDownloadRequest,
    authorization: str | None = Header(None),
):
    _require_admin(authorization)
    dest = _resolve_under_download_root(body.path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    total = 0
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", body.url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        out.write(chunk)
                        total += len(chunk)
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "path": dest,
        "bytes_written": total,
        "source_url": body.url,
    }


@app.post("/v1/model/load")
def model_load(
    body: ModelLoadRequest,
    authorization: str | None = Header(None),
):
    _require_admin(authorization)
    global _engine, _active_model_path, _model_id, _explicit_unload
    resolved = _resolve_under_download_root(body.path)
    if not os.path.isfile(resolved):
        raise HTTPException(404, detail=f"ไม่พบไฟล์โมเดล: {resolved}")
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.__exit__(None, None, None)
            finally:
                _engine = None
        _active_model_path = resolved
        if body.model_id is not None:
            _model_id = body.model_id
        else:
            _model_id = os.path.splitext(os.path.basename(resolved))[0]
        _engine = litert_lm.Engine(
            resolved,
            backend=litert_lm.Backend.CPU,
            audio_backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.CPU,
        )
        _explicit_unload = False
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": True,
        "path": resolved,
        "model_id": _model_id,
    }


@app.post("/v1/model/unload")
def model_unload(authorization: str | None = Header(None)):
    _require_admin(authorization)
    global _engine, _explicit_unload
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.__exit__(None, None, None)
            finally:
                _engine = None
        _explicit_unload = True
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": False,
        "active_model_path": _active_model_path,
        "model_id": _model_id,
        "note": "เรียก POST /v1/model/load ก่อนใช้ /v1/chat/completions",
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    if not body.messages:
        raise HTTPException(400, detail="messages ว่างไม่ได้")
    if body.stream:
        return StreamingResponse(
            _stream_chat_completion(body),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    preface, last_msg = await _preface_and_last_payload(body.messages)
    engine = get_engine()
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
    created = int(time.time())
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

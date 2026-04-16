import asyncio
import base64
import io
import json
import logging
import re
import wave
import shutil
import subprocess
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import litert_lm
from fastapi import HTTPException

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


# โมเดล Gemma4 / mel pipeline คาดหวังเส้นเสียงยาวพอ — 16kHz mono s16 ต่อวินาที = 32000 bytes
_MIN_PCM16_MONO_16K_BYTES = 30 * 32000
# apad whole_len เป็นจำนวน sample หลัง resample เป็น 16k mono
_MIN_AUDIO_SAMPLES_16K_MONO = 30 * 16000


def _silent_wav_pcm16le_mono(duration_sec: float = 30.0) -> bytes:
    """WAV PCM s16le mono 16kHz ล้วนความเงียบ — ใช้เป็นตัวยึดเมื่อ Gemma4 ต้องการ audio ในพรอมปต์แต่เทิร์นนี้มีแค่รูป"""
    nframes = int(16000 * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * (nframes * 2))
    return buf.getvalue()


def _inject_silent_audio_for_vision_only_turn(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """ถ้ามีรูปในเทิร์นนี้แต่ไม่มีเสียง ให้แทรก WAV ความเงียบนำหน้า (แก้ INVALID_ARGUMENT less audio หลังสตริปเสียงใน preface)."""
    if not content:
        return content
    has_audio = any(isinstance(p, dict) and p.get("type") == "audio" for p in content)
    has_image = any(isinstance(p, dict) and p.get("type") == "image" for p in content)
    if not has_image or has_audio:
        return content
    blob = base64.b64encode(_silent_wav_pcm16le_mono(30.0)).decode("ascii")
    return [{"type": "audio", "blob": blob}, *content]


def _try_ffmpeg_normalize_to_wav_pcm16k(raw: bytes) -> bytes | None:
    """แปลงเสียงใดๆ เป็น WAV PCM s16le mono 16kHz และ pad ความเงียบให้ยาวอย่างน้อย ~30s (แก้ MP3/AAC ที่ pad ด้วย 0x00 ไม่ได้)"""
    if not raw or not shutil.which("ffmpeg"):
        return None
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                "-af",
                f"apad=whole_len={_MIN_AUDIO_SAMPLES_16K_MONO}",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as ex:
        logging.warning("ffmpeg normalize ไม่สำเร็จ: %s", ex)
        return None
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:500]
        logging.warning(
            "ffmpeg คืน code=%s stderr=%s",
            proc.returncode,
            err or "(ว่าง)",
        )
        return None
    return proc.stdout


def _process_audio_blob(b64_data: str) -> str:
    """แปลง/ขยายเสียงให้ยาวและรูปแบบตรงกับที่ litert คาดหวัง (ลด INVALID_ARGUMENT audio)"""
    if not b64_data:
        return b64_data
    try:
        import struct

        raw_bytes = base64.b64decode(b64_data)
        is_riff_wav = raw_bytes.startswith(b"RIFF") and len(raw_bytes) >= 44
        has_ffmpeg = shutil.which("ffmpeg") is not None

        if has_ffmpeg:
            wav_out = _try_ffmpeg_normalize_to_wav_pcm16k(raw_bytes)
            if wav_out is not None:
                logging.info(
                    "ประมวลผลเสียงด้วย ffmpeg แล้ว: %s bytes (WAV PCM 16k mono)",
                    len(wav_out),
                )
                return base64.b64encode(wav_out).decode("ascii")
            raise HTTPException(
                status_code=400,
                detail="แปลงไฟล์เสียงด้วย ffmpeg ไม่สำเร็จ — ตรวจสอบว่าไฟล์ไม่เสียหรือลองส่งเป็น WAV",
            )

        if not is_riff_wav:
            raise HTTPException(
                status_code=400,
                detail="ไม่พบ ffmpeg บนเซิร์ฟเวอร์ — รองรับเสียงอัด (MP3/AAC) ไม่ได้ ให้ส่งเป็น WAV "
                "หรือติดตั้ง ffmpeg (ใน Docker image นี้มีให้แล้วหลัง rebuild)",
            )

        MIN_BYTES = _MIN_PCM16_MONO_16K_BYTES
        if len(raw_bytes) < MIN_BYTES:
            padding_len = MIN_BYTES - len(raw_bytes)
            logging.info(
                "ข้อมูล WAV สั้น (%s bytes) กำลัง pad ความเงียบเป็น ~30s (%s bytes)",
                len(raw_bytes),
                MIN_BYTES,
            )
            try:
                new_bytes = raw_bytes + b"\x00" * padding_len
                new_file_size = len(new_bytes) - 8
                new_bytes = new_bytes[:4] + struct.pack("<I", new_file_size) + new_bytes[8:]
                data_pos = new_bytes.find(b"data", 12)
                if data_pos != -1:
                    current_data_size = struct.unpack("<I", new_bytes[data_pos + 4 : data_pos + 8])[0]
                    new_data_size = current_data_size + padding_len
                    new_bytes = (
                        new_bytes[: data_pos + 4]
                        + struct.pack("<I", new_data_size)
                        + new_bytes[data_pos + 8 :]
                    )
                raw_bytes = new_bytes
            except Exception as ex:
                logging.warning("ไม่สามารถอัปเดต WAV header ได้อย่างสมบูรณ์: %s", ex)
                raw_bytes = raw_bytes + b"\x00" * padding_len

        logging.info("ส่งข้อมูลเสียงขนาดสุดท้าย: %s bytes", len(raw_bytes))
        return base64.b64encode(raw_bytes).decode("ascii")
    except HTTPException:
        raise
    except Exception as e:
        logging.error("เกิดข้อผิดพลาดขณะประมวลผล audio blob: %s", e)
        return b64_data


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
        if isinstance(data, str) and data.startswith("data:"):
            data, _ = _data_url_to_blob(data)
        processed_data = _process_audio_blob(data)
        return {"type": "audio", "blob": processed_data}
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


def _strip_audio_from_litert_preface(preface: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """เอาเสียงออกจากข้อความใน preface — หลายเทิร์น (เสียงแล้วตามด้วยรูป) ทำให้ LiteRT รายงาน less audio than expected."""
    out: list[dict[str, Any]] = []
    note = (
        "[ผู้ใช้ส่งไฟล์เสียงในข้อความก่อนหน้า — ใช้บริบทจากข้อความผู้ช่วยถัดไปที่สรุป/ตอบจากเสียงนั้น]"
    )
    for msg in preface:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_parts: list[dict[str, Any]] = []
        had_audio = False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "audio":
                had_audio = True
                continue
            new_parts.append(part)
        if had_audio:
            if not new_parts:
                new_parts = [{"type": "text", "text": note}]
            elif new_parts[0].get("type") == "text":
                t0 = new_parts[0].get("text") or ""
                new_parts[0] = {"type": "text", "text": f"{note}\n{t0}"}
            else:
                new_parts.insert(0, {"type": "text", "text": note})
            out.append({**msg, "content": new_parts})
        else:
            out.append(msg)
    return out


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
    try:
        engine = get_engine()
        preface, last_msg = await _preface_and_last_payload(body.messages)
        tools_py: list[Any] = []
        if body.tools:
            tools_py = _make_tool_callables(body.tools)

        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

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
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logging.error("เกิดข้อผิดพลาดขณะสตรีม: %s", e)
        traceback.print_exc()
        yield _chunk_to_sse(
            {
                "id": f"err-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.model,
                "choices": [{"index": 0, "delta": {"content": f"\n[เกิดข้อผิดพลาด: {str(e)}]"}, "finish_reason": "error"}],
            }
        )
        yield b"data: [DONE]\n\n"


async def nonstream_chat_completion(body: ChatCompletionRequest) -> dict[str, Any]:
    try:
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
        created = int(time.time())
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logging.error("เกิดข้อผิดพลาดขณะส่งข้อความแบบไม่สตรีม: %s", e)
        traceback.print_exc()
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


async def _preface_and_last_payload(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not messages:
        raise HTTPException(400, detail="messages ว่างไม่ได้")
    last = messages[-1]
    if last.role == "user":
        preface = _strip_audio_from_litert_preface(
            await _openai_messages_to_litert(messages[:-1])
        )
        content = await _openai_content_to_litert(last.content)
        if isinstance(content, list):
            content = _inject_silent_audio_for_vision_only_turn(content)
        return preface, {"role": "user", "content": content}
    if last.role == "tool":
        preface = _strip_audio_from_litert_preface(
            await _openai_messages_to_litert(messages)
        )
        cont = {
            "role": "user",
            "content": [{"type": "text", "text": "ดำเนินการต่อจากผลลัพธ์เครื่องมือด้านบน (ตอบผู้ใช้)"}],
        }
        return preface, cont
    if last.role == "assistant":
        preface = _strip_audio_from_litert_preface(
            await _openai_messages_to_litert(messages[:-1])
        )
        cont = {"role": "user", "content": [{"type": "text", "text": "ตอบสานต่อจากข้อความผู้ช่วยล่าสุด"}]}
        return preface, cont
    raise HTTPException(400, detail="role สุดท้ายไม่รองรับ")

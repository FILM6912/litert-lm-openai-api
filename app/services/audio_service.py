import asyncio
import base64
import io
import json
import logging
import time
import uuid
import wave
import struct
from typing import Any

from fastapi import HTTPException

from app.schemas.audio import SpeechRequest, TranscriptionRequest
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.services.chat_service import nonstream_chat_completion


async def generate_speech(body: SpeechRequest) -> bytes:
    try:
        prompt_text = body.input[:500]
        duration_sec = min(max(len(prompt_text) * 0.05, 2.0), 30.0)

        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful assistant. The user is requesting text-to-speech output. "
                    "Simply repeat the user's text back exactly as provided. "
                    "Do not add any commentary or explanation."
                ),
            ),
            ChatMessage(role="user", content=prompt_text),
        ]

        req = ChatCompletionRequest(
            model=body.model,
            messages=msgs,
            stream=False,
            temperature=0.1,
        )
        result = await nonstream_chat_completion(req)
        text_out = ""
        choices = result.get("choices", [])
        if choices:
            text_out = choices[0].get("message", {}).get("content", prompt_text)

        wav_bytes = _generate_silent_wav(text_out, duration_sec)

        if body.response_format == "mp3":
            wav_bytes = _wav_to_mp3_fallback(wav_bytes)

        return wav_bytes
    except HTTPException:
        raise
    except Exception as e:
        logging.error("เกิดข้อผิดพลาดขณะสร้างเสียง: %s", e)
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาดในการสร้างเสียง: {str(e)}")


async def transcribe_audio(body: TranscriptionRequest) -> dict[str, Any]:
    try:
        audio_b64 = base64.b64encode(body.file).decode("ascii")

        language_instruction = ""
        if body.language:
            language_instruction = f" Respond in {body.language} language."

        prompt_hint = ""
        if body.prompt:
            prompt_hint = f"\nContext hint: {body.prompt}"

        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "You are an audio transcription assistant. The user has provided an audio file. "
                    "Listen to the audio and provide an accurate transcription of what is said. "
                    "If you cannot determine the audio content, respond with a best-effort description."
                    + language_instruction
                ),
            ),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": f"Please transcribe this audio file.{prompt_hint}"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            ),
        ]

        req = ChatCompletionRequest(
            model=body.model,
            messages=msgs,
            stream=False,
            temperature=body.temperature,
        )
        result = await nonstream_chat_completion(req)

        transcription = ""
        choices = result.get("choices", [])
        if choices:
            transcription = choices[0].get("message", {}).get("content", "")

        if body.response_format == "text":
            return transcription

        if body.response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": body.language or "en",
                "duration": len(body.file) / 32000,
                "text": transcription,
            }

        if body.response_format == "srt":
            return _text_to_srt(transcription)

        if body.response_format == "vtt":
            return _text_to_vtt(transcription)

        return {"text": transcription}
    except HTTPException:
        raise
    except Exception as e:
        logging.error("เกิดข้อผิดพลาดขณะถอดเสียง: %s", e)
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาดในการถอดเสียง: {str(e)}")


def _generate_silent_wav(text: str, duration_sec: float) -> bytes:
    nframes = int(24000 * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x80" * nframes)
    return buf.getvalue()


def _wav_to_mp3_fallback(wav_bytes: bytes) -> bytes:
    return wav_bytes


def _text_to_srt(text: str) -> str:
    lines = text.split(". ")
    srt = ""
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        start_h = i * 3
        end_h = start_h + 3
        srt += f"{i+1}\n00:00:{start_h:02d},000 --> 00:00:{end_h:02d},000\n{line.strip()}\n\n"
    return srt or "1\n00:00:00,000 --> 00:00:03,000\n(text)\n"


def _text_to_vtt(text: str) -> str:
    lines = text.split(". ")
    vtt = "WEBVTT\n\n"
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        start_h = i * 3
        end_h = start_h + 3
        vtt += f"00:00:{start_h:02d}.000 --> 00:00:{end_h:02d}.000\n{line.strip()}\n\n"
    return vtt or "WEBVTT\n\n00:00:00.000 --> 00:00:03.000\n(text)\n"

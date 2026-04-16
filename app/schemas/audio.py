from typing import Any, Literal

from pydantic import BaseModel


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


class TranscriptionRequest(BaseModel):
    model: str
    file: bytes
    filename: str = "audio.wav"
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0

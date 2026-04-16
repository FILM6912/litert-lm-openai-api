from typing import Any, Literal

from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    response_format: str = "url"
    style: str | None = None
    user: str | None = None

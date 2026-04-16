import asyncio
import base64
import io
import json
import logging
import time
import uuid
from typing import Any

from fastapi import HTTPException

from app.schemas.image import ImageGenerationRequest
from app.services.chat_service import nonstream_chat_completion
from app.schemas.chat import ChatMessage


async def generate_image(body: ImageGenerationRequest) -> dict[str, Any]:
    created = int(time.time())
    prompt = body.prompt

    try:
        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "You are an image generation assistant. When the user asks you to generate an image, "
                    "respond with a detailed textual description of what the image would look like. "
                    "Since this is a local LiteRT server without a dedicated image generation model, "
                    "provide the best possible visual description. "
                    "Format your response as a vivid, detailed description."
                ),
            ),
            ChatMessage(role="user", content=f"Generate an image: {prompt}"),
        ]

        chat_body = {
            "model": body.model,
            "messages": msgs,
            "stream": False,
            "temperature": 0.7,
        }
        from app.schemas.chat import ChatCompletionRequest

        req = ChatCompletionRequest(**chat_body)
        result = await nonstream_chat_completion(req)

        description = ""
        choices = result.get("choices", [])
        if choices:
            description = choices[0].get("message", {}).get("content", "")

        placeholder_svg = _generate_placeholder_svg(prompt, description)

        b64_png = _text_to_placeholder_image(prompt, description)

        images = []
        for _ in range(body.n):
            if body.response_format == "b64_json":
                images.append({"b64_json": b64_png, "revised_prompt": description})
            else:
                images.append(
                    {
                        "url": f"data:image/svg+xml;base64,{base64.b64encode(placeholder_svg.encode()).decode()}",
                        "revised_prompt": description,
                    }
                )

        return {
            "created": created,
            "data": images,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error("เกิดข้อผิดพลาดขณะสร้างภาพ: %s", e)
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาดในการสร้างภาพ: {str(e)}")


def _generate_placeholder_svg(prompt: str, description: str) -> str:
    import html

    safe_prompt = html.escape(prompt[:100])
    safe_desc = html.escape((description or "")[:200])

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1a2e"/>
      <stop offset="100%" style="stop-color:#16213e"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#bg)" rx="16"/>
  <rect x="20" y="20" width="472" height="472" fill="none" stroke="#10a37f" stroke-width="2" rx="12" stroke-dasharray="8,4"/>
  <text x="256" y="200" font-family="sans-serif" font-size="16" fill="#10a37f" text-anchor="middle">LiteRT Image Generation</text>
  <text x="256" y="240" font-family="sans-serif" font-size="12" fill="#8e8ea0" text-anchor="middle">{safe_prompt}</text>
  <text x="256" y="300" font-family="sans-serif" font-size="10" fill="#6b6b7b" text-anchor="middle">(placeholder - no image generation model loaded)</text>
  <text x="256" y="340" font-family="sans-serif" font-size="10" fill="#6b6b7b" text-anchor="middle">{safe_desc}</text>
</svg>'''
    return svg


def _text_to_placeholder_image(prompt: str, description: str) -> str:
    svg = _generate_placeholder_svg(prompt, description)
    return base64.b64encode(svg.encode()).decode()

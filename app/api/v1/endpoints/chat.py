from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatCompletionRequest
from app.services import stream_chat_completion, nonstream_chat_completion

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    if not body.messages:
        raise HTTPException(400, detail="messages ว่างไม่ได้")
    if body.stream:
        return StreamingResponse(
            stream_chat_completion(body),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return await nonstream_chat_completion(body)

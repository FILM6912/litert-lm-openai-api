import base64
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.schemas.audio import SpeechRequest
from app.services.audio_service import generate_speech, transcribe_audio

router = APIRouter()


@router.post("/audio/speech")
async def create_speech(body: SpeechRequest):
    if not body.input:
        raise HTTPException(400, detail="input ว่างไม่ได้")
    try:
        audio_bytes = await generate_speech(body)
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        ct = content_types.get(body.response_format, "audio/mpeg")
        return Response(content=audio_bytes, media_type=ct)
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"เกิดข้อผิดพลาดในการสร้างเสียง: {e}")
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("gemma-litert"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    if not file:
        raise HTTPException(400, detail="file ว่างไม่ได้")
    try:
        file_bytes = await file.read()
        filename = file.filename or "audio.wav"

        from app.schemas.audio import TranscriptionRequest

        req = TranscriptionRequest(
            model=model,
            file=file_bytes,
            filename=filename,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        result = await transcribe_audio(req)

        if isinstance(result, str):
            return Response(content=result, media_type="text/plain; charset=utf-8")

        if isinstance(result, dict):
            fmt = response_format
            if fmt == "srt":
                text = result if isinstance(result, str) else result.get("text", "")
                return Response(
                    content=text, media_type="text/plain; charset=utf-8"
                )
            if fmt == "vtt":
                text = result if isinstance(result, str) else result.get("text", "")
                return Response(
                    content=text, media_type="text/vtt; charset=utf-8"
                )

        return result
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"เกิดข้อผิดพลาดในการถอดเสียง: {e}")
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

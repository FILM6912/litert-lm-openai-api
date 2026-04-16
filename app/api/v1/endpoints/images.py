from fastapi import APIRouter, HTTPException

from app.schemas.image import ImageGenerationRequest
from app.services.image_service import generate_image

router = APIRouter()


@router.post("/images/generations")
async def create_image(body: ImageGenerationRequest):
    if not body.prompt:
        raise HTTPException(400, detail="prompt ว่างไม่ได้")
    try:
        return await generate_image(body)
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"เกิดข้อผิดพลาดในการสร้างภาพ: {e}")
        raise HTTPException(500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

from fastapi import APIRouter

from app.api.v1.endpoints import admin, audio, chat, images, models

api_router = APIRouter()
api_router.include_router(models.router, tags=["models"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(admin.router, tags=["admin"])
api_router.include_router(images.router, tags=["images"])
api_router.include_router(audio.router, tags=["audio"])

from fastapi import APIRouter

from app.api.v1.endpoints import admin, chat, models

api_router = APIRouter()
api_router.include_router(models.router, tags=["models"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(admin.router, tags=["admin"])

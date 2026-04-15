from fastapi import HTTPException

from app.core.config import ADMIN_TOKEN


def require_admin(authorization: str | None) -> None:
    if not ADMIN_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="ต้องมี Authorization: Bearer")
    if authorization[7:].strip() != ADMIN_TOKEN:
        raise HTTPException(403, detail="token ไม่ถูกต้อง")

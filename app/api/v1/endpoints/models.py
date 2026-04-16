import time
from typing import Any

from fastapi import APIRouter

import app.core.engine as core_engine
from app.services import async_merged_catalog_entries

router = APIRouter()


@router.get("/models")
async def list_models():
    entries, _, _ = await async_merged_catalog_entries()
    created = int(time.time())
    data: list[dict[str, Any]] = [
        {
            "id": e["path"].replace("\\", "/"),
            "object": "model",
            "created": created,
            "owned_by": "litert-lm",
        }
        for e in entries
    ]
    if not data:
        data = [
            {
                "id": core_engine._model_id,
                "object": "model",
                "created": created,
                "owned_by": "litert-lm",
            }
        ]
    return {"object": "list", "data": data}

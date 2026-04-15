import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from app.core.config import (
    ADMIN_TOKEN,
    LITERT_CATALOG_SCAN_TIMEOUT,
    LITERT_SCAN_MAX_DEPTH,
    LITERT_SCAN_SUBDIR,
    MODEL_DOWNLOAD_ROOT,
)
from app.core.engine import _active_model_path, _engine, _model_id
from app.core.security import require_admin
from app.schemas.model import ModelDownloadRequest, ModelLoadRequest
from app.services import (
    async_merged_catalog_entries,
    download_model,
    invalidate_model_catalog_cache,
    load_model,
    unload_model,
)

router = APIRouter()


@router.get("/model/status")
def model_status():
    return {
        "engine_in_memory": _engine is not None,
        "must_load_before_chat": False,
        "active_model_path": _active_model_path,
        "model_id": _model_id,
        "download_root": MODEL_DOWNLOAD_ROOT,
        "admin_auth_required": bool(ADMIN_TOKEN),
    }


@router.get("/model/catalog")
async def model_catalog():
    models, timed_out, from_cache = await async_merged_catalog_entries()
    return {
        "models": models,
        "files": [e["path"].replace("\\", "/") for e in models],
        "download_root": MODEL_DOWNLOAD_ROOT,
        "scan_subdir": LITERT_SCAN_SUBDIR or None,
        "scan_max_depth": LITERT_SCAN_MAX_DEPTH,
        "scan_timeout_sec": LITERT_CATALOG_SCAN_TIMEOUT,
        "active_model_path": _active_model_path,
        "engine_in_memory": _engine is not None,
        "must_load_before_chat": False,
        "model_id": _model_id,
        "scan_timed_out": timed_out,
        "catalog_from_cache": from_cache,
    }


@router.post("/model/download")
async def model_download(
    body: ModelDownloadRequest,
    authorization: str | None = Header(None),
):
    require_admin(authorization)
    return await download_model(body.url, body.path)


@router.post("/model/load")
def model_load(
    body: ModelLoadRequest,
    authorization: str | None = Header(None),
):
    require_admin(authorization)
    try:
        return load_model(body.path, body.model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))


@router.post("/model/unload")
def model_unload(authorization: str | None = Header(None)):
    require_admin(authorization)
    return unload_model()

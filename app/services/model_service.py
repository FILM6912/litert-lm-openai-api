import os
import httpx
import litert_lm

from app.core.config import MODEL_DOWNLOAD_ROOT
from app.core.engine import (
    _engine,
    _engine_lock,
    _active_model_path,
    _model_id,
    _explicit_unload,
    resolve_path,
)
from app.services.catalog_service import invalidate_model_catalog_cache


def load_model(path: str, model_id: str | None = None) -> dict[str, bool | str]:
    global _engine, _active_model_path, _model_id, _explicit_unload
    resolved = resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {resolved}")
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.__exit__(None, None, None)
            finally:
                _engine = None
        _active_model_path = resolved
        if model_id is not None:
            _model_id = model_id
        else:
            _model_id = os.path.splitext(os.path.basename(resolved))[0]
        _engine = litert_lm.Engine(
            resolved,
            backend=litert_lm.Backend.CPU,
            audio_backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.CPU,
        )
        _explicit_unload = False
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": True,
        "path": resolved,
        "model_id": _model_id,
    }


def unload_model() -> dict[str, bool | str]:
    global _engine, _explicit_unload
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.__exit__(None, None, None)
            finally:
                _engine = None
        _explicit_unload = True
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": False,
        "active_model_path": _active_model_path,
        "model_id": _model_id,
        "note": "เรียก POST /v1/model/load ก่อนใช้ /v1/chat/completions",
    }


async def download_model(url: str, path: str) -> dict[str, bool | str | int]:
    dest = resolve_path(path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    total = 0
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        out.write(chunk)
                        total += len(chunk)
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "path": dest,
        "bytes_written": total,
        "source_url": url,
    }


def get_status() -> dict[str, bool | str]:
    return {
        "engine_in_memory": _engine is not None,
        "must_load_before_chat": _explicit_unload,
        "active_model_path": _active_model_path,
        "model_id": _model_id,
        "download_root": MODEL_DOWNLOAD_ROOT,
    }

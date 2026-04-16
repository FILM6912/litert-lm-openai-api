import os
import httpx
import litert_lm

from app.core.config import MODEL_DOWNLOAD_ROOT
import app.core.engine as core_engine
from app.services.catalog_service import invalidate_model_catalog_cache


def load_model(path: str, model_id: str | None = None) -> dict[str, bool | str]:
    resolved = core_engine.resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {resolved}")
    with core_engine._engine_lock:
        if core_engine._engine is not None:
            try:
                core_engine._engine.__exit__(None, None, None)
            finally:
                core_engine._engine = None
        core_engine._active_model_path = resolved
        if model_id is not None:
            core_engine._model_id = model_id
        else:
            core_engine._model_id = os.path.splitext(os.path.basename(resolved))[0]
        try:
            core_engine._engine = litert_lm.Engine(
                resolved,
                backend=litert_lm.Backend.CPU,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
        except Exception as e:
            import logging
            logging.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
            raise HTTPException(
                500, detail=f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"
            )
        core_engine._explicit_unload = False
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": True,
        "path": resolved,
        "model_id": core_engine._model_id,
    }


def unload_model() -> dict[str, bool | str]:
    with core_engine._engine_lock:
        if core_engine._engine is not None:
            try:
                core_engine._engine.__exit__(None, None, None)
            finally:
                core_engine._engine = None
        core_engine._explicit_unload = True
    invalidate_model_catalog_cache()
    return {
        "ok": True,
        "loaded": False,
        "active_model_path": core_engine._active_model_path,
        "model_id": core_engine._model_id,
        "note": "เรียก POST /v1/model/load ก่อนใช้ /v1/chat/completions",
    }


async def download_model(url: str, path: str) -> dict[str, bool | str | int]:
    dest = core_engine.resolve_path(path)
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
        "engine_in_memory": core_engine._engine is not None,
        "must_load_before_chat": core_engine._explicit_unload,
        "active_model_path": core_engine._active_model_path,
        "model_id": core_engine._model_id,
        "download_root": MODEL_DOWNLOAD_ROOT,
    }

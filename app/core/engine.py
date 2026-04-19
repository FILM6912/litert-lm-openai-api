import os
import threading
from typing import TYPE_CHECKING, Optional

import litert_lm
from fastapi import HTTPException

if TYPE_CHECKING:
    from litert_lm import Engine as EngineType

from app.core.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    ENGINE_MAX_NUM_TOKENS,
    LITERT_SCAN_SUBDIR,
    MODEL_DOWNLOAD_ROOT,
)

_engine: Optional["EngineType"] = None
_engine_lock = threading.Lock()

if os.path.isabs(DEFAULT_MODEL_PATH):
    _active_model_path: str = os.path.realpath(DEFAULT_MODEL_PATH)
else:
    _active_model_path = os.path.realpath(
        os.path.join(MODEL_DOWNLOAD_ROOT, DEFAULT_MODEL_PATH)
    )

_model_id: str = DEFAULT_MODEL_ID
_explicit_unload: bool = False


def get_engine() -> litert_lm.Engine:
    global _engine, _explicit_unload
    if _explicit_unload:
        raise HTTPException(
            503,
            detail="โมเดลถูก unload แล้ว — เรียก POST /v1/model/load ก่อนใช้แชท",
        )
    with _engine_lock:
        if _engine is None:
            if not os.path.isfile(_active_model_path):
                raise HTTPException(
                    503, detail=f"ไม่พบไฟล์โมเดล: {_active_model_path}"
                )
            try:
                _engine = litert_lm.Engine(
                    _active_model_path,
                    backend=litert_lm.Backend.CPU,
                    audio_backend=litert_lm.Backend.CPU,
                    vision_backend=litert_lm.Backend.CPU,
                    max_num_tokens=ENGINE_MAX_NUM_TOKENS,
                )
            except Exception as e:
                import logging
                logging.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
                raise HTTPException(
                    500, detail=f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"
                )
        return _engine


def _resolve_model_to_path(model_name: str) -> str | None:
    if not model_name:
        return None
    p = model_name.strip().replace("\\", "/")

    candidates = [
        os.path.realpath(os.path.join(MODEL_DOWNLOAD_ROOT, p)),
    ]
    if not p.endswith(".litertlm"):
        candidates.append(
            os.path.realpath(os.path.join(MODEL_DOWNLOAD_ROOT, p + ".litertlm"))
        )
    if LITERT_SCAN_SUBDIR:
        candidates.append(
            os.path.realpath(
                os.path.join(MODEL_DOWNLOAD_ROOT, LITERT_SCAN_SUBDIR, p)
            )
        )
        if not p.endswith(".litertlm"):
            candidates.append(
                os.path.realpath(
                    os.path.join(
                        MODEL_DOWNLOAD_ROOT, LITERT_SCAN_SUBDIR, p + ".litertlm"
                    )
                )
            )

    for c in candidates:
        root = os.path.realpath(MODEL_DOWNLOAD_ROOT)
        if c != root and not c.startswith(root + os.sep):
            continue
        if os.path.isfile(c):
            return c
    return None


def ensure_engine_for_model(model_name: str) -> "litert_lm.Engine":
    global _engine, _explicit_unload, _active_model_path, _model_id

    resolved = _resolve_model_to_path(model_name)

    with _engine_lock:
        if resolved and os.path.realpath(_active_model_path) == os.path.realpath(resolved) and _engine is not None:
            _explicit_unload = False
            return _engine

        target_path = resolved or _active_model_path

        if not os.path.isfile(target_path):
            raise HTTPException(
                404,
                detail=f"ไม่พบไฟล์โมเดลสำหรับ model='{model_name}' — ค้นหาที่ {target_path}",
            )

        if _engine is not None:
            import logging as _log
            _log.info("Auto-switching model: %s -> %s", _active_model_path, target_path)
            try:
                _engine.__exit__(None, None, None)
            except Exception:
                pass
            _engine = None

        try:
            _engine = litert_lm.Engine(
                target_path,
                backend=litert_lm.Backend.CPU,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
                max_num_tokens=ENGINE_MAX_NUM_TOKENS,
            )
        except Exception as e:
            import logging
            logging.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล {target_path}: {e}")
            raise HTTPException(
                500, detail=f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"
            )

        _active_model_path = target_path
        _model_id = os.path.splitext(os.path.basename(target_path))[0]
        _explicit_unload = False

        try:
            from app.services.catalog_service import invalidate_model_catalog_cache
            invalidate_model_catalog_cache()
        except Exception:
            pass

        return _engine


def resolve_path(path: str) -> str:
    p = path.strip()
    if not p or p.endswith("/") or p.endswith("\\"):
        raise HTTPException(400, detail="path ไฟล์ไม่ถูกต้อง")
    if os.path.isabs(p):
        full = os.path.realpath(p)
    else:
        full = os.path.realpath(os.path.join(MODEL_DOWNLOAD_ROOT, p))
    root = os.path.realpath(MODEL_DOWNLOAD_ROOT)
    if full != root and not full.startswith(root + os.sep):
        raise HTTPException(
            403,
            detail=f"path ต้องอยู่ภายใต้ LITERT_MODEL_DOWNLOAD_DIR ({root})",
        )
    return full

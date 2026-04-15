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
            _engine = litert_lm.Engine(
                _active_model_path,
                backend=litert_lm.Backend.CPU,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
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

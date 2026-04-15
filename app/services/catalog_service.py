import asyncio
import concurrent.futures
import os
import time
from collections import deque
from typing import Any

from app.core.config import (
    LITERT_CATALOG_TTL,
    LITERT_CATALOG_SCAN_TIMEOUT,
    LITERT_SCAN_MAX_DEPTH,
    LITERT_SCAN_SKIP,
    LITERT_SCAN_SUBDIR,
    MODEL_DOWNLOAD_ROOT,
)
from app.core.engine import _active_model_path, _engine

_catalog_cache_list: list[dict[str, Any]] | None = None
_catalog_cache_mono: float = 0.0


def _skip_catalog_dir(name: str) -> bool:
    return name in LITERT_SCAN_SKIP


def invalidate_model_catalog_cache() -> None:
    global _catalog_cache_list, _catalog_cache_mono
    _catalog_cache_list = None
    _catalog_cache_mono = 0.0


def _litert_catalog_entries_uncached() -> list[dict[str, Any]]:
    root = os.path.realpath(MODEL_DOWNLOAD_ROOT)
    if LITERT_SCAN_SUBDIR:
        scan_root = os.path.realpath(
            os.path.join(root, LITERT_SCAN_SUBDIR)
        )
        if not scan_root.startswith(root + os.sep) and scan_root != root:
            scan_root = root
    else:
        scan_root = root

    ap_active = os.path.realpath(_active_model_path)
    in_mem = _engine is not None
    items: list[dict[str, Any]] = []
    if not os.path.isdir(scan_root):
        return items

    rel_base = root
    queue: deque[tuple[str, int]] = deque([(scan_root, 0)])
    _max_dir_visits = 50_000
    visits = 0

    while queue:
        dirpath, depth = queue.popleft()
        visits += 1
        if visits > _max_dir_visits:
            break
        try:
            with os.scandir(dirpath) as it:
                for ent in it:
                    try:
                        if ent.is_dir(follow_symlinks=False):
                            if _skip_catalog_dir(ent.name):
                                continue
                            next_depth = depth + 1
                            if next_depth > LITERT_SCAN_MAX_DEPTH:
                                continue
                            queue.append((ent.path, next_depth))
                        elif ent.is_file(follow_symlinks=False) and ent.name.endswith(
                            ".litertlm"
                        ):
                            abs_p = os.path.realpath(ent.path)
                            try:
                                rel = os.path.relpath(abs_p, rel_base).replace(
                                    "\\", "/"
                                )
                            except ValueError:
                                rel = abs_p.replace("\\", "/")
                            items.append(
                                {
                                    "path": rel,
                                    "label": rel,
                                    "name": ent.name,
                                    "absolute_path": abs_p,
                                    "loaded": in_mem and abs_p == ap_active,
                                }
                            )
                    except OSError:
                        continue
        except OSError:
            continue

    items.sort(key=lambda x: x["path"].lower())
    return items


def _catalog_merge_active_model(
    items: list[dict[str, Any]], root: str
) -> list[dict[str, Any]]:
    ap = os.path.realpath(_active_model_path)
    if not ap.endswith(".litertlm") or not os.path.isfile(ap):
        return items
    paths = {m["path"] for m in items}
    try:
        rel = os.path.relpath(ap, os.path.realpath(root)).replace("\\", "/")
    except ValueError:
        rel = ap.replace("\\", "/")
    if rel in paths:
        return items
    in_mem = _engine is not None
    items = list(items)
    items.insert(
        0,
        {
            "path": rel,
            "label": rel + " (active)",
            "name": os.path.basename(ap),
            "absolute_path": ap,
            "loaded": in_mem,
        },
    )
    return items


def _litert_catalog_scan_in_thread() -> tuple[list[dict[str, Any]], bool]:
    if LITERT_CATALOG_SCAN_TIMEOUT <= 0:
        return _litert_catalog_entries_uncached(), False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_litert_catalog_entries_uncached)
        try:
            return fut.result(timeout=LITERT_CATALOG_SCAN_TIMEOUT), False
        except concurrent.futures.TimeoutError:
            return [], True


async def async_merged_catalog_entries() -> tuple[list[dict[str, Any]], bool, bool]:
    global _catalog_cache_list, _catalog_cache_mono
    now = time.monotonic()
    if (
        LITERT_CATALOG_TTL > 0
        and _catalog_cache_list is not None
        and now - _catalog_cache_mono < LITERT_CATALOG_TTL
    ):
        scanned = list(_catalog_cache_list)
        timed_out = False
        from_cache = True
    else:
        scanned, timed_out = await asyncio.to_thread(
            _litert_catalog_scan_in_thread
        )
        from_cache = False
        if not timed_out:
            _catalog_cache_list = list(scanned)
            _catalog_cache_mono = time.monotonic()
    entries = _catalog_merge_active_model(list(scanned), MODEL_DOWNLOAD_ROOT)
    return entries, timed_out, from_cache

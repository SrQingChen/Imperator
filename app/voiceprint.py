"""声纹库：跨任务说话人身份（命名/合并/启停/导入导出）。"""
from __future__ import annotations

import json
import threading
import time
import uuid

import numpy as np

from . import config

_lock = threading.RLock()


def _load() -> list[dict]:
    if not config.VOICEPRINT_JSON.exists():
        return []
    try:
        return json.loads(config.VOICEPRINT_JSON.read_text("utf-8"))
    except Exception:
        return []


def _save(items: list[dict]) -> None:
    config.VOICEPRINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.VOICEPRINT_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), "utf-8")


def list_all() -> list[dict]:
    with _lock:
        return _load()


def get(vid: str) -> dict | None:
    with _lock:
        return next((v for v in _load() if v["id"] == vid), None)


def _validate_embedding(embedding) -> np.ndarray:
    arr = np.asarray(embedding, dtype="float32").flatten()
    if arr.size < 64 or not np.all(np.isfinite(arr)):
        raise ValueError(f"embedding 应为 ≥64 维的有限浮点向量（收到 {arr.size} 维）")
    return arr


def create(name: str, embedding, note: str = "") -> dict:
    arr = _validate_embedding(embedding)
    with _lock:
        items = _load()
        v = {"id": uuid.uuid4().hex[:12], "name": name, "enabled": True,
             "embedding": arr.tolist(),
             "dim": int(arr.shape[0]),
             "note": note, "created_at": time.time()}
        items.append(v)
        _save(items)
        return v


def update(vid: str, patch: dict) -> dict | None:
    with _lock:
        items = _load()
        v = next((x for x in items if x["id"] == vid), None)
        if v is None:
            return None
        for k in ("name", "enabled", "note"):
            if k in patch:
                v[k] = patch[k]
        _save(items)
        return v


def delete(vid: str) -> bool:
    with _lock:
        items = _load()
        rest = [x for x in items if x["id"] != vid]
        if len(rest) == len(items):
            return False
        _save(rest)
        return True


def merge(dst_id: str, src_id: str) -> dict | None:
    with _lock:
        items = _load()
        dst = next((x for x in items if x["id"] == dst_id), None)
        src = next((x for x in items if x["id"] == src_id), None)
        if dst is None or src is None or dst_id == src_id:
            return None
        e1, e2 = np.asarray(dst["embedding"], "float32"), np.asarray(src["embedding"], "float32")
        if e1.shape != e2.shape:
            raise ValueError(f"维度不一致：{e1.size} vs {e2.size}，无法合并")
        cen = (e1 + e2) / 2.0
        dst["embedding"] = (cen / (np.linalg.norm(cen) + 1e-9)).tolist()
        dst["note"] = (dst.get("note", "") + f"；合并自 {src['name']}").strip("；")
        _save([x for x in items if x["id"] != src_id])
        return dst


def export_json() -> str:
    with _lock:
        return json.dumps(_load(), ensure_ascii=False, indent=2)


def import_json(text: str, replace: bool = False) -> int:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("声纹包格式错误：应为列表")
    with _lock:
        items = [] if replace else _load()
        existing = {v["id"] for v in items}
        n = 0
        for v in data:
            if not isinstance(v, dict) or "embedding" not in v or "name" not in v:
                continue
            if v.get("id") in existing:
                v["id"] = uuid.uuid4().hex[:12]
            items.append(v)
            n += 1
        _save(items)
        return n

"""事件日志：内存环形 + 文件双写 + since 游标增量查询。"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import config

_lock = threading.Lock()
_ring: deque[dict] = deque(maxlen=800)
_seq = 0


def _append(level: str, msg: str, scope: str = "") -> dict:
    global _seq
    with _lock:
        _seq += 1
        ev = {"seq": _seq, "ts": time.time(), "level": level, "msg": msg, "scope": scope}
        _ring.append(ev)
    try:
        with open(config.LOGS_DIR / "events.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev['ts']))} [{level}]"
                    f"{'(' + scope + ')' if scope else ''} {msg}\n")
    except Exception:
        pass
    return ev


def info(msg: str, scope: str = "") -> None:
    _append("info", msg, scope)


def warn(msg: str, scope: str = "") -> None:
    _append("warn", msg, scope)


def error(msg: str, scope: str = "") -> None:
    _append("error", msg, scope)


def success(msg: str, scope: str = "") -> None:
    _append("success", msg, scope)


def stage(msg: str, scope: str = "") -> None:
    _append("stage", msg, scope)


def since(seq: int, limit: int = 200) -> list[dict]:
    with _lock:
        return [e for e in _ring if e["seq"] > seq][-limit:]

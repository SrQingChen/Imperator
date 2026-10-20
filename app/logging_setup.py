"""统一日志：文件 DEBUG + 控制台 INFO + 事件流桥接 + 线程异常兜底。"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading

from . import config, events

_configured = False


class _EventBridge(logging.Handler):
    """把第三方库的 INFO+ 日志送进事件流（UI 可见），DEBUG 只落文件。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            name = record.name
            text = f"[{name}] {msg}"
            if record.levelno >= logging.ERROR:
                events.error(text, scope="lib")
            elif record.levelno >= logging.WARNING:
                events.warn(text, scope="lib")
            elif record.levelno >= logging.INFO:
                events.info(text, scope="lib")
        except Exception:
            pass


def setup() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 测试/多次 create_app 时避免重复挂载
    for h in list(root.handlers):
        if getattr(h, "_imperator", False):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")

    fh = logging.handlers.RotatingFileHandler(
        config.LOGS_DIR / "server.log", maxBytes=8 * 1024 * 1024, backupCount=3,
        encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh._imperator = True
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch._imperator = True
    root.addHandler(ch)

    bridge = _EventBridge()
    bridge.setLevel(logging.INFO)
    bridge._imperator = True
    root.addHandler(bridge)

    # 第三方库：DEBUG 落文件，INFO 进事件流；避免 numba/torch 等刷屏
    for name in ("audio_separator", "common_separator", "roformer", "clearvoice",
                 "funasr", "modelscope", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    for name in ("urllib3", "PIL", "matplotlib", "numba", "torch", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    def _thread_hook(args):
        try:
            events.error(
                f"线程 {args.thread.name} 未捕获异常：{args.exc_type.__name__}: {args.exc_value}",
                scope="server")
        except Exception:
            pass

    threading.excepthook = _thread_hook
    events.info(f"日志已就绪：{config.LOGS_DIR / 'server.log'}", scope="server")

"""英白拉多 Imperator · 薄入口：argparse + uvicorn + 打开浏览器。"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import threading
import webbrowser

faulthandler.enable()


def _ensure_utf8_console() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def main() -> None:
    _ensure_utf8_console()
    parser = argparse.ArgumentParser(description="英白拉多 Imperator · 智能音轨分离器")
    parser.add_argument("--host", default=None, help="监听地址（默认取设置，初始 127.0.0.1）")
    parser.add_argument("--port", default=None, type=int, help="监听端口（默认取设置，初始 8766）")
    parser.add_argument("--data-dir", default=None, help="数据目录（模型/输出/日志）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--log-level", default="warning", help="uvicorn 日志级别")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["IMPERATOR_DATA_DIR"] = args.data_dir

    from app import config, settings as settings_mod
    cfg = settings_mod.load()
    host = args.host or cfg.listen_host
    port = args.port or cfg.listen_port

    # clearvoice 默认下载到 ./clearvoice —— 把进程工作目录固定到数据目录
    os.chdir(config.DATA_DIR)

    import uvicorn
    from app.api import create_app
    app = create_app()

    if not args.no_browser:
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  英白拉多 Imperator · 智能音轨分离器\n"
          f"  本地地址: http://{host}:{port}\n"
          f"  数据目录: {config.DATA_DIR}\n")
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()

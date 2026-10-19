"""依赖体检与 pip 引导安装（torch CUDA 单列官方源）。"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
import threading

from . import events, network

# (import 名, pip 名, 用途, 是否核心)
DEPS: list[tuple[str, str, str, bool]] = [
    ("fastapi", "fastapi", "Web 框架", True),
    ("uvicorn", "uvicorn[standard]", "HTTP 服务器", True),
    ("multipart", "python-multipart", "文件上传", True),
    ("numpy", "numpy", "数值计算", True),
    ("scipy", "scipy", "重采样/聚类", True),
    ("soundfile", "soundfile", "WAV 读写", True),
    ("psutil", "psutil", "性能监控", True),
    ("imageio_ffmpeg", "imageio-ffmpeg", "自带 ffmpeg", False),
    ("torch", "torch", "深度学习运行时", True),
    ("silero_vad", "silero-vad", "语音活动检测", True),
    ("audio_separator", "audio-separator", "伴奏分离", True),
    ("funasr", "funasr", "声纹/说话人", True),
    ("clearvoice", "clearvoice", "重叠语音分离", True),
]

_install_lock = threading.RLock()
_installing: dict = {"status": "idle", "log": [], "error": ""}


def check() -> list[dict]:
    out = []
    for mod, pip_name, purpose, core in DEPS:
        ok = importlib.util.find_spec(mod) is not None
        ver = ""
        if ok:
            for dist in (pip_name.split("[")[0], mod):
                try:
                    ver = importlib.metadata.version(dist)
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
        out.append({"module": mod, "pip": pip_name, "purpose": purpose, "core": core,
                    "status": "ok" if ok else "missing", "version": ver})
    return out


def install_status() -> dict:
    with _install_lock:
        return dict(_installing)


def install(modules: list[str] | None = None, gpu: bool | None = None) -> dict:
    """后台线程安装缺失依赖；modules 为 import 名列表（None=全部缺失）。"""
    with _install_lock:
        if _installing["status"] == "running":
            return {"accepted": False, "reason": "已有安装任务进行中"}
        _installing.update(status="running", log=[], error="")

    from . import config as cfg
    if gpu is None:
        gpu = cfg.get_device().has_nvidia
    plan = network.get_plan()

    def _log(line: str):
        with _install_lock:
            _installing["log"] = (_installing["log"] + [line])[-200:]
        events.info(line, scope="env")

    def _work():
        try:
            missing = []
            if modules:
                wanted = set(modules)
                missing = [d for d in check() if d["module"] in wanted and d["status"] == "missing"]
            else:
                missing = [d for d in check() if d["status"] == "missing"]
            if not missing:
                _log("没有缺失的依赖。")
                _installing.update(status="done", error="")
                return
            # torch 单独处理（CUDA 轮子只在官方源）
            normal = [d for d in missing if d["module"] != "torch"]
            torch_dep = next((d for d in missing if d["module"] == "torch"), None)
            if normal:
                pkgs = [d["pip"] for d in normal]
                cmd = [sys.executable, "-m", "pip", "install", *pkgs]
                if plan.pip_index:
                    cmd += ["-i", plan.pip_index]
                _log(f"安装：{' '.join(pkgs)}")
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                _log((proc.stdout or "")[-2000:])
                if proc.returncode != 0:
                    raise RuntimeError(f"pip 安装失败：{(proc.stderr or '')[-800:]}")
            if torch_dep:
                if gpu:
                    src = "https://download.pytorch.org/whl/cu121"
                    _log(f"安装 CUDA 版 torch（官方源 {src}，体积大请耐心）…")
                else:
                    src = plan.pip_index or None
                    _log("安装 CPU 版 torch…")
                cmd = [sys.executable, "-m", "pip", "install", "torch"]
                if src:
                    cmd += ["--index-url", src]
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                _log((proc.stdout or "")[-2000:])
                if proc.returncode != 0:
                    _log(f"CUDA torch 安装失败：{(proc.stderr or '')[-500:]}；回退 CPU 版")
                    cmd = [sys.executable, "-m", "pip", "install", "torch"]
                    if plan.pip_index:
                        cmd += ["-i", plan.pip_index]
                    subprocess.run(cmd, capture_output=True, text=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _installing.update(status="done", error="")
            events.success("依赖安装完成", scope="env")
        except Exception as e:
            _installing.update(status="error", error=str(e))
            events.error(f"依赖安装失败：{e}", scope="env")

    threading.Thread(target=_work, name="pip-install", daemon=True).start()
    return {"accepted": True}

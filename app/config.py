"""路径解析、设备探测、全局目录布局（源码/打包双态）。"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

def _base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):  # PyInstaller
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_dir()
WEB_DIR = BASE_DIR / "web"
DATA_DIR = Path(os.environ.get("IMPERATOR_DATA_DIR", "")) if os.environ.get("IMPERATOR_DATA_DIR") else BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
WORKSPACE_DIR = DATA_DIR / "workspace" / "inputs"
LOGS_DIR = DATA_DIR / "logs"
CLEARVOICE_CKPT_DIR = DATA_DIR / "checkpoints" / "MossFormer2_SS_16K"  # clearvoice 的 yaml 用 CWD 相对路径 checkpoints/…
TMP_DIR = DATA_DIR / "tmp"
CONFIG_JSON = DATA_DIR / "config.json"
VOICEPRINT_JSON = DATA_DIR / "voiceprints.json"
NETWORK_STATE_JSON = DATA_DIR / "network_state.json"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, WORKSPACE_DIR, LOGS_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DeviceInfo:
    python: str = ""
    os: str = ""
    cpu: str = ""
    cpu_cores: int = 0
    ram_total_mb: int = 0
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    has_nvidia: bool = False
    ffmpeg_path: str = ""
    ffmpeg_source: str = ""
    ffprobe_path: str = ""
    ffprobe_source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "python": self.python, "os": self.os, "cpu": self.cpu, "cpu_cores": self.cpu_cores,
            "ram_total_mb": self.ram_total_mb, "gpu_name": self.gpu_name, "gpu_vram_mb": self.gpu_vram_mb,
            "has_nvidia": self.has_nvidia, "ffmpeg_path": self.ffmpeg_path, "ffmpeg_source": self.ffmpeg_source,
            "ffprobe_path": self.ffprobe_path, "ffprobe_source": self.ffprobe_source,
            "notes": self.notes,
        }


def _probe_gpu() -> tuple[str, int]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "", 0
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        name, _, vram = line.partition(",")
        return name.strip(), int(float(vram.strip() or 0))
    except Exception:
        return "", 0


def _find_ffmpeg() -> tuple[str, str]:
    p = shutil.which("ffmpeg")
    if p:
        return p, "system"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"
    except Exception:
        return "", "missing"


def _find_ffprobe(ffmpeg_path: str) -> tuple[str, str]:
    """ffprobe：系统 PATH → ffmpeg 同目录（gyan/BtbN 构建自带）→ 缺失。
    imageio-ffmpeg 只带 ffmpeg，不带 ffprobe，故最后留空由上层解析 ffmpeg -i 兜底。"""
    p = shutil.which("ffprobe")
    if p:
        return p, "system"
    if ffmpeg_path:
        base = Path(ffmpeg_path)
        for name in ("ffprobe.exe", "ffprobe"):
            cand = base.with_name(name)
            if cand.exists():
                return str(cand), "ffmpeg-dir"
    return "", "missing"


_DEVICE: DeviceInfo | None = None


def get_device() -> DeviceInfo:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import sys as _sys

    import psutil

    name, vram = _probe_gpu()
    ff, ff_src = _find_ffmpeg()
    fp, fp_src = _find_ffprobe(ff)
    vm = psutil.virtual_memory()
    info = DeviceInfo(
        python=_sys.version.split()[0],
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or "CPU",
        cpu_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_mb=int(vm.total / 1024 / 1024),
        gpu_name=name, gpu_vram_mb=vram, has_nvidia=bool(name),
        ffmpeg_path=ff, ffmpeg_source=ff_src,
        ffprobe_path=fp, ffprobe_source=fp_src,
    )
    if not name:
        info.notes.append("未检测到 NVIDIA GPU，将使用 CPU 模式（建议选择轻量 MSS 档位）")
    if name and vram < 6 * 1024:
        info.notes.append(f"显存 {vram}MB 偏低，任务结束后将自动释放模型显存；建议标准/轻量档")
    if not ff:
        info.notes.append("未找到 ffmpeg：请安装 ffmpeg 或 pip install imageio-ffmpeg")
    _DEVICE = info
    return info


def torch_device() -> str:
    """惰性探测 torch 可用设备；探测失败回退 CPU，绝不抛异常。"""
    try:
        ensure_torch()
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_torch_lock = threading.Lock()


def ensure_torch():
    """串行化 torch 首次导入：多线程并发 import torch 会触发循环导入竞态。"""
    with _torch_lock:
        import torch  # noqa: F401
        return torch

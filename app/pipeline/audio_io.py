"""ffmpeg 抽取、WAV 读写、重采样工具。数组约定：np.float32 [n, ch]。"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config


def ffmpeg_exe() -> str:
    dev = config.get_device()
    if dev.ffmpeg_path:
        return dev.ffmpeg_path
    raise RuntimeError("ffmpeg 不可用：请安装 ffmpeg 或 pip install imageio-ffmpeg")


def run_ffmpeg(args: list[str], timeout: int = 1800) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{proc.stderr[-800:]}")


def extract_master(input_path: str | Path, out_wav: str | Path,
                   sr: int = 44100, channels: int = 2) -> Path:
    """抽取主轴音频（44.1k 立体声 f32 WAV）。时长=解码后精确采样数。"""
    out_wav = Path(out_wav)
    run_ffmpeg(["-i", str(input_path), "-vn", "-ac", str(channels), "-ar", str(sr),
                "-c:a", "pcm_f32le", str(out_wav)])
    return out_wav


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, sr


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    if x.ndim == 2:
        return np.stack([resample_poly(x[:, c], up, down) for c in range(x.shape[1])], axis=1).astype("float32")
    return resample_poly(x, up, down).astype("float32")


def save_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path = Path(path)
    if x.ndim == 1:
        x = x[:, None]
    sf.write(str(path), x, sr, subtype=subtype)
    return path


def dbfs(x: np.ndarray) -> float:
    x = to_mono(x)
    e = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return 20 * math.log10(e)


def align_length(x: np.ndarray, n: int) -> tuple[np.ndarray, bool]:
    """零填充/截断到精确长度 n；返回 (数组, 是否发生了调整)。"""
    if x.shape[0] == n:
        return x, False
    if x.shape[0] > n:
        return x[:n], True
    pad = np.zeros((n - x.shape[0], *x.shape[1:]), dtype="float32")
    return np.concatenate([x, pad], axis=0), True

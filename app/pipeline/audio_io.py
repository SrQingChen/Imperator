"""ffmpeg 抽取、WAV 读写、重采样工具、媒体探测。数组约定：np.float32 [n, ch]。"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
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


# ---------------------------------------------------------------- 媒体探测

@dataclass
class MediaInfo:
    path: str = ""
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    sample_rate: int = 0
    channels: int = 0
    pix_fmt: str = ""
    rotation: int = 0
    source: str = ""      # ffprobe | ffmpeg-parse

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3), "has_video": self.has_video,
            "has_audio": self.has_audio, "width": self.width, "height": self.height,
            "fps": round(self.fps, 3), "vcodec": self.vcodec, "acodec": self.acodec,
            "sample_rate": self.sample_rate, "channels": self.channels,
            "pix_fmt": self.pix_fmt, "rotation": self.rotation, "source": self.source,
        }


def ffprobe_exe() -> str:
    dev = config.get_device()
    if dev.ffprobe_path:
        return dev.ffprobe_path
    raise RuntimeError("ffprobe 不可用：请安装完整 ffmpeg（需含 ffprobe）")


def _rational(s: str) -> float:
    if not s or s in ("N/A", "0/0", "0:1"):
        return 0.0
    for sep in ("/", ":"):
        if sep in s:
            num, _, den = s.partition(sep)
            try:
                d = float(den)
                return float(num) / d if d else 0.0
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _probe_via_ffprobe(path: str) -> MediaInfo:
    cmd = [ffprobe_exe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe 失败：{(p.stderr or '')[:300]}")
    data = json.loads(p.stdout or "{}")
    info = MediaInfo(path=path, source="ffprobe")
    for st in data.get("streams", []):
        ctype = st.get("codec_type")
        if ctype == "video" and not info.has_video:
            info.has_video = True
            info.vcodec = st.get("codec_name") or ""
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.fps = _rational(st.get("avg_frame_rate") or st.get("r_frame_rate") or "")
            info.pix_fmt = st.get("pix_fmt") or ""
            rot = 0
            try:
                rot = int(st.get("tags", {}).get("rotate", 0) or 0)
            except (TypeError, ValueError):
                rot = 0
            for sd in st.get("side_data_list", []) or []:
                if "rotation" in sd:
                    try:
                        rot = int(float(sd["rotation"]))
                    except (TypeError, ValueError):
                        pass
            info.rotation = rot
        elif ctype == "audio" and not info.has_audio:
            info.has_audio = True
            info.acodec = st.get("codec_name") or ""
            info.sample_rate = int(st.get("sample_rate") or 0)
            info.channels = int(st.get("channels") or 0)
    try:
        info.duration = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        info.duration = 0.0
    return info


def _probe_via_ffmpeg(path: str) -> MediaInfo:
    """无 ffprobe 时解析 `ffmpeg -i` 的 stderr（退出码 1 属正常）。"""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    txt = p.stderr or ""
    info = MediaInfo(path=path, source="ffmpeg-parse")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info.duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if re.search(r"Stream #\d+:\d+.*?: Video:", txt):
        info.has_video = True
        m = re.search(r"(\d{2,5})x(\d{2,5})", txt)
        if m:
            info.width, info.height = int(m.group(1)), int(m.group(2))
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
        if m:
            info.fps = float(m.group(1))
    if re.search(r"Stream #\d+:\d+.*?: Audio:", txt):
        info.has_audio = True
        m = re.search(r"(\d{4,6})\s*Hz", txt)
        if m:
            info.sample_rate = int(m.group(1))
    return info


_probe_cache: dict[str, tuple[float, MediaInfo]] = {}


def probe_media(path: str | Path) -> MediaInfo:
    """探测媒体（缓存按 mtime）。无 ffprobe 时回退 ffmpeg -i 解析。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    key = str(p.resolve())
    mtime = p.stat().st_mtime
    cached = _probe_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        info = _probe_via_ffprobe(str(p))
    except Exception:
        info = _probe_via_ffmpeg(str(p))
    _probe_cache[key] = (mtime, info)
    return info


# ---------------------------------------------------------------- 可取消的 ffmpeg

class ProcessCancelled(Exception):
    """ffmpeg 被用户取消。"""


def run_ffmpeg_progress(args: list[str], duration: float, on_progress=None,
                        cancel=None, timeout: float = 3600.0,
                        cwd: str | None = None) -> None:
    """带进度的 ffmpeg 执行：解析 -progress，可即时取消。

    on_progress(frac) 接受 0..1；cancel() 返回 True 时 terminate→kill。
    """
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-progress", "pipe:1", "-nostats", *args]
    err_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, cwd=cwd,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    last_frac = -1.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if cancel and cancel():
                _stop_process(proc)
                raise ProcessCancelled("已取消")
            if timeout and time.time() - t0 > timeout:
                _stop_process(proc)
                raise RuntimeError(f"ffmpeg 超时（>{int(timeout)}s）")
            if on_progress and duration > 0:
                sec = _parse_progress_seconds(line)
                if sec is not None:
                    frac = max(0.0, min(1.0, sec / duration))
                    if frac - last_frac >= 0.01:
                        last_frac = frac
                        on_progress(frac)
        proc.wait(timeout=30)
    except Exception:
        if proc.poll() is None:
            _stop_process(proc)
        raise
    if proc.returncode != 0:
        try:
            err_file.seek(0)
            err = err_file.read() or ""
        except Exception:
            err = ""
        raise RuntimeError(f"ffmpeg 失败：{err[-800:]}")
    if on_progress:
        on_progress(1.0)
    try:
        err_file.close()
    except Exception:
        pass


def _parse_progress_seconds(line: str) -> float | None:
    for key in ("out_time_us=", "out_time_ms="):
        if line.startswith(key):
            try:
                return float(line.split("=", 1)[1]) / 1_000_000.0
            except ValueError:
                return None
    if line.startswith("out_time="):
        v = line.split("=", 1)[1].strip()
        try:
            h, m, s = v.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, TypeError):
            return None
    return None


def _stop_process(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

"""Silero VAD 封装。"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


def _load_model(device: str):
    """torch.jit.load 在 Windows 上无法打开含非 ASCII 字符的路径——复制到
    纯 ASCII 临时目录再加载。"""
    import torch
    from silero_vad import load_silero_vad
    import silero_vad as _pkg
    src = Path(_pkg.__file__).resolve().parent / "data" / "silero_vad.jit"
    try:
        return load_silero_vad(device=device if device == "cuda" else None)
    except Exception:
        dst = Path(tempfile.gettempdir()) / "imperator_silero_vad.jit"
        if not dst.exists():
            shutil.copyfile(src, dst)
        m = torch.jit.load(str(dst), map_location="cpu")
        m.eval()
        if device == "cuda" and torch.cuda.is_available():
            m = m.to(device)
        return m


class Vad:
    def __init__(self, device: str = "cpu"):
        from .. import config as _cfg
        _cfg.ensure_torch()
        self._torch = __import__("torch")
        self.model = _load_model(device)

    def segments(self, wav16k_mono: np.ndarray) -> list[tuple[float, float]]:
        """返回 [(start_s, end_s)]；音频须为 16k 单声道 float32。"""
        from silero_vad import get_speech_timestamps
        t = self._torch.from_numpy(np.ascontiguousarray(wav16k_mono, dtype="float32"))
        ts = get_speech_timestamps(
            t, self.model, return_seconds=True,
            min_speech_duration_ms=250, min_silence_duration_ms=350,
            speech_pad_ms=120, max_speech_duration_s=30.0,
        )
        return [(float(w["start"]), float(w["end"])) for w in ts]

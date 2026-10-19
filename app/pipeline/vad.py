"""Silero VAD 封装。"""
from __future__ import annotations

import numpy as np


class Vad:
    def __init__(self, device: str = "cpu"):
        from .. import config as _cfg
        _cfg.ensure_torch()
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        if device == "cuda" and torch.cuda.is_available():
            self.model = load_silero_vad(device=device)
        else:
            self.model = load_silero_vad()

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

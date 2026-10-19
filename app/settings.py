"""用户设置：schema、默认值、持久化（data/config.json）。"""
from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from typing import Literal

from . import config

_lock = threading.RLock()


@dataclass
class Settings:
    # 说话人 / 置信度
    merge_threshold: float = 0.62          # τ：cos ≥ τ 视为同一人
    min_new_speaker_duration: float = 5.0  # 新说话人持续证据：累计秒数
    min_new_speaker_segments: int = 3      # 新说话人持续证据：段数
    smoothing_window: float = 1.2          # 归属平滑窗（秒）
    hysteresis: float = 0.05
    window_seconds: float = 1.5            # 声纹滑窗
    window_hop: float = 0.75
    # 偶发排除
    outlier_mode: Literal["off", "ratio", "auto"] = "ratio"
    outlier_ratio_percent: float = 4.0     # 占比模式：说话人总时长占比低于此值并入最相似
    auto_merge_score: float = 0.62         # 自动模式并入阈值
    auto_unknown_score: float = 0.35       # 自动模式 unknown 阈值
    # 重叠
    overlap_margin: float = 0.06
    overlap_context: float = 1.5           # 重叠处理上下文（秒）
    # 输出
    output_format: Literal["float32", "s16", "s24"] = "float32"
    output_sr: Literal[44100, 48000, "source"] = 44100
    # 模型
    mss_tier: Literal["standard", "fast", "light", "skip"] = "standard"
    release_vram: bool = True
    # 声纹库
    voiceprint_match: bool = True
    # 网络 / 服务
    mirror_mode: Literal["auto", "china", "direct"] = "auto"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8765

    def validate(self) -> list[str]:
        errs = []
        if not (0.3 <= self.merge_threshold <= 0.99):
            errs.append("merge_threshold 应在 0.3~0.99")
        if not (0.0 < self.outlier_ratio_percent <= 50):
            errs.append("outlier_ratio_percent 应在 0~50")
        if self.outlier_mode not in ("off", "ratio", "auto"):
            errs.append("outlier_mode 非法")
        if self.mss_tier not in ("standard", "fast", "light", "skip"):
            errs.append("mss_tier 非法")
        return errs

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        valid = {f.name for f in dataclasses.fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in valid})
        # 字面量类型容错
        if s.output_sr not in (44100, 48000, "source"):
            s.output_sr = 44100
        return s


def load() -> Settings:
    with _lock:
        if config.CONFIG_JSON.exists():
            try:
                return Settings.from_dict(json.loads(config.CONFIG_JSON.read_text("utf-8")))
            except Exception:
                pass
        return Settings()


def save(s: Settings) -> None:
    with _lock:
        config.CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
        config.CONFIG_JSON.write_text(
            json.dumps(s.to_dict(), ensure_ascii=False, indent=2), "utf-8")

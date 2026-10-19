"""重叠语音：三信号触发 → MossFormer2 双流分离 → 嵌入匹配 → 软比率掩码。

分支（规划 §2）：
- 双流分别命中两个不同说话人 → 重叠：软比率掩码 m_a+m_b=1
- 仅单流命中 → 误触发：整窗硬归属
- 零命中 → unknown（进 other）
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import diarize as dz

_ENVELOPE_SMOOTH_S = 0.032   # 掩码包络平滑窗
_P = 2.0                      # 软比率指数
_EPS = 1e-9
_MIN_STREAM_RMS = 0.002       # 分离流能量下限（低于视为静音流）


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


class ClearVoiceSeparator:
    """clearvoice MossFormer2_SS_16K 封装（2 说话人盲分离）。"""

    def __init__(self, model_dir: str | None = None):
        from .. import config as _cfg
        _cfg.ensure_torch()
        from clearvoice import ClearVoice
        kwargs: dict = {"task": "speech_separation", "model_names": ["MossFormer2_SS_16K"]}
        try:
            self.cv = ClearVoice(**kwargs, model_root=model_dir) if model_dir else ClearVoice(**kwargs)
        except TypeError:
            self.cv = ClearVoice(**kwargs)

    def separate(self, wav_path: str) -> list[np.ndarray]:
        """返回 2 条 16k 单声道波形（numpy）。"""
        import torch
        try:
            out = self.cv(input_path=wav_path, online_write=False)
            arr = out["output"] if isinstance(out, dict) and "output" in out else out
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if isinstance(arr, (list, tuple)):
                arr = np.asarray([np.asarray(x) for x in arr])
            arr = np.asarray(arr, dtype="float32").squeeze()
            if arr.ndim == 1:            # (T,) 单源（不应发生）
                return [arr]
            if arr.ndim == 2:            # (S, T) 或 (T, S)：S=2 很小、T 很大
                if arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]
            if arr.ndim == 3:            # (B, S, T)
                return [np.ascontiguousarray(arr[0, i]) for i in range(arr.shape[1])]
            raise RuntimeError(f"分离输出形状异常：{arr.shape}")
        except Exception:
            # 回退：online_write 写盘后读取
            import soundfile as sf
            outs = self.cv(input_path=wav_path, online_write=True)
            files = outs if isinstance(outs, list) else [outs]
            streams = []
            for f in files:
                try:
                    x, _ = sf.read(str(f), dtype="float32")
                    streams.append(x if x.ndim == 1 else x.mean(axis=1))
                except Exception:
                    pass
            return streams


@dataclass
class OverlapRegion:
    start: float
    end: float
    kind: str = "pending"                 # overlap | hard | unknown | skip
    speakers: list[int] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # speaker -> 16k 包络掩码
    sims: dict[int, float] = field(default_factory=dict)


def detect_trigger_windows(res: dz.DiarizeResult, threshold: float, margin: float) -> set[int]:
    """三触发：①歧义窗 ②切人边界窗 ③自洽性差窗。"""
    trig: set[int] = set()
    by_idx = {w.idx: w for w in res.windows}
    order = sorted(res.windows, key=lambda w: w.start)
    for i, w in enumerate(order):
        if w.top1 < threshold or (w.top1 - w.top2) < margin:
            trig.add(w.idx)
        if w.top1 < threshold - 0.05:  # 自洽性差（与所有质心都不像）
            trig.add(w.idx)
        for j in (i - 1, i + 1):
            if 0 <= j < len(order):
                o = order[j]
                if o.assigned != w.assigned:  # 切人边界
                    trig.add(w.idx)
                    trig.add(o.idx)
    # 只保留两侧确有不同说话人候选的（前后 3 窗内出现 ≥2 个 assigned 值）
    keep: set[int] = set()
    for i, w in enumerate(order):
        if w.idx not in trig:
            continue
        lo, hi = max(0, i - 3), min(len(order), i + 4)
        nearby = {order[k].assigned for k in range(lo, hi)}
        nearby.discard(w.assigned)
        if nearby or w.top1 < threshold:  # 有其他候选 或 本身无人认领
            keep.add(w.idx)
    return {i for i in keep if i in by_idx}


def group_regions(triggered: set[int], windows: list[dz.Window], max_len: float = 12.0) -> list[tuple[float, float]]:
    if not triggered:
        return []
    ws = sorted((w for w in windows if w.idx in triggered), key=lambda w: w.start)
    regions: list[list[float]] = [[ws[0].start, ws[0].end]]
    for w in ws[1:]:
        if w.start - regions[-1][1] <= 1.0:
            regions[-1][1] = max(regions[-1][1], w.end)
        else:
            regions.append([w.start, w.end])
    # 限制单区长度（超长切分为 max_len 块）
    out: list[tuple[float, float]] = []
    for s, e in regions:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len * 0.9  # 10% 重叠衔接
        out.append((s, e))
    return out


def _envelope_mask(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|a|^p/(|a|^p+|b|^p+ε)，32ms 平滑，归一互补。"""
    ea, eb = np.abs(a) ** _P, np.abs(b) ** _P
    k = max(3, int(_ENVELOPE_SMOOTH_S * 16000) | 1)
    kernel = np.ones(k, dtype="float32") / k
    ea = np.convolve(ea, kernel, mode="same")
    eb = np.convolve(eb, kernel, mode="same")
    denom = ea + eb + _EPS
    return ea / denom, eb / denom


def resolve_region(region: tuple[float, float], ctx: float,
                    vocals16k: np.ndarray, sr16k: int,
                    separator: ClearVoiceSeparator, embed_func,
                    centroids: dict[int, np.ndarray], threshold: float,
                    tmp_wav) -> OverlapRegion:
    """处理单个候选区：切出（带上下文）→ 盲分离 → 嵌入匹配 → 三分支。"""
    import soundfile as sf

    s0, e0 = region
    s = max(0.0, s0 - ctx)
    e = min(len(vocals16k) / sr16k, e0 + ctx)
    i0, i1 = int(round(s * sr16k)), int(round(e * sr16k))
    if i1 - i0 < int(0.5 * sr16k):
        return OverlapRegion(s0, e0, kind="skip")
    clip = np.ascontiguousarray(vocals16k[i0:i1], dtype="float32")
    sf.write(str(tmp_wav), clip, sr16k)
    streams = separator.separate(str(tmp_wav))
    if len(streams) < 2:
        return OverlapRegion(s0, e0, kind="unknown")
    a, b = streams[0], streams[1]
    if float(np.sqrt(np.mean(b ** 2))) < _MIN_STREAM_RMS and float(np.sqrt(np.mean(a ** 2))) >= _MIN_STREAM_RMS:
        return _hard_assign(a, i0, i1, s0, e0, sr16k, embed_func, centroids, threshold, tmp_wav)
    if float(np.sqrt(np.mean(a ** 2))) < _MIN_STREAM_RMS and float(np.sqrt(np.mean(b ** 2))) >= _MIN_STREAM_RMS:
        return _hard_assign(b, i0, i1, s0, e0, sr16k, embed_func, centroids, threshold, tmp_wav)
    ea = embed_func(a, tmp_wav, 0)
    eb = embed_func(b, tmp_wav, 1)
    if ea is None or eb is None:
        return _hard_assign(a if ea is not None else b, i0, i1, s0, e0, sr16k,
                            embed_func, centroids, threshold, tmp_wav)
    match_a = _best_speaker(ea, centroids)
    match_b = _best_speaker(eb, centroids)
    floor = threshold - 0.05
    if match_a and match_b and match_a[1] != match_b[1] and match_a[0] >= floor and match_b[0] >= floor:
        m_a, m_b = _envelope_mask(a, b)
        r = i1 - i0
        rel0 = int(round((s0 - s) * sr16k))
        rel1 = int(round((e0 - s) * sr16k))
        rel0, rel1 = max(0, min(rel0, r)), max(0, min(rel1, r))
        reg = OverlapRegion(s0, e0, kind="overlap", speakers=[match_a[1], match_b[1]],
                            sims={match_a[1]: round(match_a[0], 3), match_b[1]: round(match_b[0], 3)})
        # 掩码裁剪到无上下文核心区，上下文交还常规段处理
        reg.masks = {match_a[1]: m_a[rel0:rel1].astype("float32"),
                     match_b[1]: m_b[rel0:rel1].astype("float32")}
        return reg
    single = None
    if match_a and match_a[0] >= floor:
        single = (a, match_a)
    elif match_b and match_b[0] >= floor:
        single = (b, match_b)
    if single:
        return _hard_assign(single[0], i0, i1, s0, e0, sr16k, embed_func, centroids, threshold, tmp_wav,
                            known=single[1])
    return OverlapRegion(s0, e0, kind="unknown")


def _best_speaker(emb: np.ndarray, centroids: dict[int, np.ndarray]) -> tuple[float, int] | None:
    if emb is None or not centroids:
        return None
    sims = sorted(((_cos(emb, c), sid) for sid, c in centroids.items()), key=lambda t: -t[0])
    return sims[0] if sims else None


def _hard_assign(stream: np.ndarray, i0: int, i1: int, s0: float, e0: float, sr16k: int,
                 embed_func, centroids: dict[int, np.ndarray], threshold: float,
                 tmp_wav, known: tuple[float, int] | None = None) -> OverlapRegion:
    emb = embed_func(stream, tmp_wav, 2)
    match = known or (_best_speaker(emb, centroids) if emb is not None else None)
    if match and match[0] >= threshold - 0.05:
        sid = match[1]
        # 单流命中 = 误触发/单人段：转为普通 speech 归属（走账本淡变，无掩码）
        return OverlapRegion(s0, e0, kind="speech", speakers=[sid],
                             sims={sid: round(match[0], 3)})
    return OverlapRegion(s0, e0, kind="unknown")


def merge_regions_into_segments(res: dz.DiarizeResult, regions: list[OverlapRegion]) -> list[dict]:
    """把区域结果叠加进段结构：
    - speech 区（单流命中）→ 普通 speech 项（走淡变）
    - overlap 区 → 软掩码项（m_a+m_b=1）
    - unknown 区 → 强制进 other（无增益）
    """
    items: list[dict] = []
    covered: list[tuple[float, float, str, dict]] = []
    for reg in regions:
        if reg.kind == "skip":
            continue
        covered.append((reg.start, reg.end, reg.kind,
                        {"masks": reg.masks, "sims": reg.sims, "speakers": reg.speakers}))
    for seg in res.segments:
        # 段被区域覆盖 → 切碎为未覆盖子区间（保留原 speech 部分）
        cur = seg.start
        for rs, re_, kind, payload in covered:
            if re_ <= cur or rs >= seg.end:
                continue
            if rs > cur:
                items.append({"start": cur, "end": min(rs, seg.end), "speaker": seg.speaker,
                              "kind": "speech" if seg.speaker >= 0 else "unknown",
                              "src_seg": seg.seg_id})
            cur = max(cur, re_)
        if cur < seg.end - 1e-4:
            items.append({"start": cur, "end": seg.end, "speaker": seg.speaker,
                          "kind": "speech" if seg.speaker >= 0 else "unknown", "src_seg": seg.seg_id})
    for rs, re_, kind, payload in covered:
        if kind == "speech":
            items.append({"start": rs, "end": re_, "kind": "speech",
                          "speaker": payload["speakers"][0] if payload["speakers"] else -1})
        else:
            items.append({"start": rs, "end": re_, "kind": kind, "speaker": payload["speakers"][0]
                          if payload["speakers"] else -1, **payload})
    items.sort(key=lambda it: it["start"])
    return items

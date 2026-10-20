"""重叠语音：三信号触发 → MossFormer2 双流分离 →（可选）递归拆出第三人 → 软掩码。

分支：
- ≥2 个流各自命中不同说话人 → overlap：按能量归一化掩码（Σm≡1）
- 2 人流都命中同一人 → 视为单人（掩码=1）
- 单流命中 → 误触发，整区普通归属
- 零命中 → unknown（进 other）

三人重叠：初始 2 流中若有一路是「混合体」（与最佳质心不够纯，或与次优差距小），
对该流再做一次双流分离；若拆出的子流命中**新的**说话人则接受，最多到 max_overlap_speakers。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import diarize as dz

_ENVELOPE_SMOOTH_S = 0.032   # 掩码包络平滑窗
_P = 2.0                      # 软比率指数
_EPS = 1e-9
_MIN_STREAM_RMS = 0.002       # 分离流能量下限（低于视为静音流）
_SUB_FLOOR_DROP = 0.12         # 子流验收门槛比主阈值再放宽（分离后的第三人更"脏"）
_SUB_ANCHOR = 0.08             # 至少一路子流需达到 τ−此值 作为身份锚点


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype="float64") ** 2)) + _EPS)


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
        """返回若干条 16k 单声道波形（numpy）。"""
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
            if arr.ndim == 2:            # (S, T) 或 (T, S)：S 很小、T 很大
                if arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]
            if arr.ndim == 3:            # (B, S, T)
                return [np.ascontiguousarray(arr[0, i]) for i in range(arr.shape[1])]
            raise RuntimeError(f"分离输出形状异常：{arr.shape}")
        except Exception:
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
    kind: str = "pending"                 # overlap | speech | unknown | skip
    speakers: list[int] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # speaker -> 16k 掩码
    sims: dict[int, float] = field(default_factory=dict)


def detect_trigger_windows(res: dz.DiarizeResult, threshold: float, margin: float) -> set[int]:
    """三触发：①歧义窗 ②切人边界窗 ③自洽性差窗。"""
    trig: set[int] = set()
    by_idx = {w.idx: w for w in res.windows}
    order = sorted(res.windows, key=lambda w: w.start)
    for i, w in enumerate(order):
        if w.top1 < threshold or (w.top1 - w.top2) < margin:
            trig.add(w.idx)
        if w.top1 < threshold - 0.05:
            trig.add(w.idx)
        for j in (i - 1, i + 1):
            if 0 <= j < len(order) and order[j].assigned != w.assigned:
                trig.add(w.idx)
                trig.add(order[j].idx)
    keep: set[int] = set()
    for i, w in enumerate(order):
        if w.idx not in trig:
            continue
        lo, hi = max(0, i - 3), min(len(order), i + 4)
        nearby = {order[k].assigned for k in range(lo, hi)}
        nearby.discard(w.assigned)
        if nearby or w.top1 < threshold:
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
    out: list[tuple[float, float]] = []
    for s, e in regions:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len * 0.9
        out.append((s, e))
    return out


def _envelopes(streams: list[np.ndarray]) -> list[np.ndarray]:
    """各流 |x|^p 平滑包络；归一化到总和 1（含静音流，剩余能量自然进 other）。"""
    k = max(3, int(_ENVELOPE_SMOOTH_S * 16000) | 1)
    kernel = np.ones(k, dtype="float32") / k
    envs = []
    for s in streams:
        e = np.convolve(np.abs(np.asarray(s, dtype="float32")) ** _P, kernel, mode="same")
        envs.append(e.astype("float32"))
    n = max((len(e) for e in envs), default=0)
    if n == 0:
        return []
    total = np.zeros(n, dtype="float64")
    for e in envs:
        total[: len(e)] += e
    total += _EPS
    return [(e / total[: len(e)]).astype("float32") for e in envs]


def _best_two(emb: np.ndarray | None, centroids: dict[int, np.ndarray]) -> tuple[tuple[float, int] | None, tuple[float, int] | None]:
    """返回 (最优匹配, 次优匹配)。"""
    if emb is None or not centroids:
        return None, None
    sims = sorted((_cos(emb, c), sid) for sid, c in centroids.items())[::-1]
    best = sims[0]
    second = sims[1] if len(sims) > 1 else None
    return best, second


def resolve_region(region: tuple[float, float], ctx: float,
                    vocals16k: np.ndarray, sr16k: int,
                    sep_fn, embed_fn,
                    centroids: dict[int, np.ndarray], threshold: float,
                    max_speakers: int = 3) -> OverlapRegion:
    """处理单个候选区：切出（带上下文）→ 分离 →（必要时）递归拆第三人 → 归属。"""
    s0, e0 = region
    s = max(0.0, s0 - ctx)
    e = min(len(vocals16k) / sr16k, e0 + ctx)
    i0, i1 = int(round(s * sr16k)), int(round(e * sr16k))
    if i1 - i0 < int(0.5 * sr16k):
        return OverlapRegion(s0, e0, kind="skip")
    clip = np.ascontiguousarray(vocals16k[i0:i1], dtype="float32")

    streams = [np.asarray(x, dtype="float32") for x in sep_fn(clip)]
    streams = [x for x in streams if _rms(x) >= _MIN_STREAM_RMS]
    if len(streams) < 2:
        # 只有一条有效流：整区按该流归属（或 unknown）
        if streams:
            return _single_stream_region(streams[0], s0, e0, s, sr16k, embed_fn, centroids, threshold)
        return OverlapRegion(s0, e0, kind="unknown")

    floor = threshold - 0.05
    floor_sub = threshold - _SUB_FLOOR_DROP
    # 每条流：{a: 音频, m: (sim, sid)|None, m2: 次优匹配, relaxed: 是否来自二次拆分}
    items = []
    for x in streams:
        m, m2 = _best_two(embed_fn(x, 0), centroids)
        items.append({"a": x, "m": m, "m2": m2, "relaxed": False})
    tag = 1

    # 递归拆第三人（最多到 max_speakers）
    for _ in range(max(0, max_speakers - len(items))):
        distinct = {it["m"][1] for it in items if it["m"] and it["m"][0] >= floor}
        idx = _pick_mixed_stream(items, threshold, floor)
        if idx is None:
            break
        sub = [np.asarray(x, dtype="float32") for x in sep_fn(items[idx]["a"])]
        sub = [x for x in sub if _rms(x) >= _MIN_STREAM_RMS]
        if len(sub) < 2:
            break
        sub_items = []
        for x in sub:
            m, m2 = _best_two(embed_fn(x, tag), centroids)
            tag += 1
            sub_items.append({"a": x, "m": m, "m2": m2, "relaxed": True})
        ok = [it for it in sub_items if it["m"] and it["m"][0] >= floor_sub]
        sub_distinct = {it["m"][1] for it in ok}
        anchored = any(it["m"][0] >= threshold - _SUB_ANCHOR for it in sub_items if it["m"])
        if len(sub_distinct) < 2 or not anchored or not (sub_distinct - distinct):
            break  # 没有新说话人 / 拆分不干净 → 放弃
        items = items[:idx] + sub_items + items[idx + 1:]
        if len(items) >= max_speakers:
            break

    # 汇总到说话人
    def _conf(it) -> bool:
        if not it["m"]:
            return False
        return it["m"][0] >= (floor_sub if it["relaxed"] else floor)

    confident = [(it["a"], it["m"]) for it in items if _conf(it)]
    distinct_ids = sorted({m[1] for _, m in confident})
    if not distinct_ids:
        return OverlapRegion(s0, e0, kind="unknown")
    if len(distinct_ids) == 1:
        sid = distinct_ids[0]
        return OverlapRegion(s0, e0, kind="speech", speakers=[sid], sims={sid: round(confident[0][1][0], 3)})

    envs = _envelopes([it["a"] for it in items])
    per_spk: dict[int, np.ndarray] = {}
    for env, it in zip(envs, items):
        if not _conf(it):
            continue  # 未归属流的能量留白，由补集掩码进 other
        sid = it["m"][1]
        cur = per_spk.get(sid)
        per_spk[sid] = env if cur is None else cur + env

    # 裁到无上下文核心区（上下文交还常规段处理，避免重复计入）
    r = i1 - i0
    rel0 = max(0, min(int(round((s0 - s) * sr16k)), r))
    rel1 = max(0, min(int(round((e0 - s) * sr16k)), r))
    masks = {sid: np.clip(arr[rel0:rel1], 0.0, 1.0).astype("float32") for sid, arr in per_spk.items()}
    sims = {sid: round(max((it["m"][0] for it in items
                            if it["m"] and it["m"][1] == sid and _conf(it)), default=0.0), 3)
            for sid in masks}
    return OverlapRegion(s0, e0, kind="overlap", speakers=sorted(masks), masks=masks, sims=sims)


def _pick_mixed_stream(items: list[dict], threshold: float, floor: float) -> int | None:
    """挑选疑似「混合体」的流。两条触发条件（避免对干净双人重叠做无谓的二次分离）：
    A) 最佳相似度低于 τ（身份不够纯）；
    B) 与次优身份明显歧义（margin < 0.03 且次优 ≥ floor）——混合体在两个人之间摇摆。"""
    best_idx, best_score = None, -1.0
    for i, it in enumerate(items):
        if _rms(it["a"]) < _MIN_STREAM_RMS * 3:
            continue
        m, m2 = it["m"], it["m2"]
        if m is None:
            score = 1.0
        else:
            sim = m[0]
            score = max(0.0, threshold - sim)          # 条件 A
            if m2 is not None and m2[0] >= floor and (sim - m2[0]) < 0.03:
                score = max(score, 0.03 - (sim - m2[0]) + 0.01)  # 条件 B
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx if best_score > 1e-6 else None


def _single_stream_region(stream: np.ndarray, s0: float, e0: float, s: float, sr16k: int,
                          embed_fn, centroids, threshold) -> OverlapRegion:
    m, _m2 = _best_two(embed_fn(stream, 9), centroids)
    if m and m[0] >= threshold - 0.05:
        return OverlapRegion(s0, e0, kind="speech", speakers=[m[1]], sims={m[1]: round(m[0], 3)})
    return OverlapRegion(s0, e0, kind="unknown")


def merge_regions_into_segments(res: dz.DiarizeResult, regions: list[OverlapRegion]) -> list[dict]:
    """区域结果叠加进段结构：
    - speech 区 → 普通 speech 项（走淡变）
    - overlap 区 → 软掩码项（Σm ≤ 1，剩余进 other）
    - unknown 区 → 强制进 other
    """
    items: list[dict] = []
    covered: list[tuple[float, float, str, dict]] = []
    for reg in regions:
        if reg.kind == "skip":
            continue
        covered.append((reg.start, reg.end, reg.kind,
                        {"masks": reg.masks, "sims": reg.sims, "speakers": reg.speakers}))
    for seg in res.segments:
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

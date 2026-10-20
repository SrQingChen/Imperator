"""偶发排除：聚类级（占比/时长小簇并入最相似）+ 段级（自动模式综合评分）。

设计原则（规划 §4）：
- 三种模式：off / ratio（占比）/ auto（评分函数）
- 默认保守：宁可 unknown，不强行合并；每个决定带可解释分项分数进审计
- 新建说话人必须满足持续证据（时长+段数），杜绝「咳嗽变第四说话人」
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import diarize as dz

_D_REF = 3.0      # 时长因子参考值（秒）：短于此越可能是一次性偶发
_SIGMA_E = 6.0    # 能量常态性尺度（dB）
_WEIGHTS = {"sim": 0.35, "margin": 0.20, "dur": 0.15, "ctx": 0.10, "energy": 0.10, "nspeech": 0.10}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def auto_score(seg: dz.Segment, best_spk: int, threshold: float,
               cluster_centroids: dict[int, np.ndarray],
               speaker_energy_db: dict[int, float],
               segments: list[dz.Segment]) -> tuple[float, dict]:
    """A(j→k*) = 0.35σ((s1−τ)/0.08)+0.20σ(margin/0.06)+0.15 短暂度
               +0.10 上下文 +0.10 能量常态性 +0.10 语音性。返回 (总分, 分项)。"""
    s1 = seg.top1
    margin = max(0.0, seg.top1 - seg.top2)
    f_sim = _sigmoid((s1 - threshold) / 0.08)
    f_margin = _sigmoid(margin / 0.06)
    f_dur = 1.0 - min(seg.dur / _D_REF, 1.0)
    # 上下文：前后 10s 内属于 best_spk 的段时长占比
    ctx_total = ctx_hit = 0.0
    for o in segments:
        if o is seg or o.kind != "speech":
            continue
        if o.start < seg.end + 10.0 and o.end > seg.start - 10.0:
            ctx_total += o.dur
            if o.speaker == best_spk:
                ctx_hit += o.dur
    f_ctx = ctx_hit / ctx_total if ctx_total > 0 else 0.5
    # 能量常态性：与该说话人平均能量（dB）偏离程度
    ref_e = speaker_energy_db.get(best_spk, seg.energy_db)
    f_energy = math.exp(-abs(seg.energy_db - ref_e) / _SIGMA_E)
    # 语音性：VAD 已过滤，能量极低视作疑似非语音
    f_nspeech = 0.2 if seg.energy_db < -45.0 else 0.95
    breakdown = {"sim": round(f_sim, 3), "margin": round(f_margin, 3), "dur": round(f_dur, 3),
                 "ctx": round(f_ctx, 3), "energy": round(f_energy, 3), "nspeech": round(f_nspeech, 3),
                 "raw_sim": round(s1, 3), "raw_margin": round(margin, 3)}
    score = (_WEIGHTS["sim"] * f_sim + _WEIGHTS["margin"] * f_margin + _WEIGHTS["dur"] * f_dur
             + _WEIGHTS["ctx"] * f_ctx + _WEIGHTS["energy"] * f_energy + _WEIGHTS["nspeech"] * f_nspeech)
    return round(score, 4), breakdown


def cluster_level(res: dz.DiarizeResult, cfg, audit: list[dict]) -> None:
    """占比/时长过小的聚类 → 候选偶发簇：并入最相似聚类（cos ≥ τ−0.05），否则保留并标记。"""
    if cfg.outlier_mode == "off" or len(res.clusters) <= 1:
        return
    total_dur = sum(c.duration for c in res.clusters) + 1e-9
    floor = cfg.merge_threshold - 0.05
    max_abs = float(getattr(cfg, "outlier_max_seconds", 8.0))
    # 候选偶发簇：占比低于阈值「且」绝对时长短（长视频里合法的少话者不会被 4% 规则吞掉），
    # 或时长低于新说话人证据下限
    small = [c for c in res.clusters
             if (c.duration / total_dur * 100 < cfg.outlier_ratio_percent and c.duration <= max_abs)
             or c.duration < cfg.min_new_speaker_duration]
    for c in sorted(small, key=lambda c: c.duration):
        others = [o for o in res.clusters if o.speaker_id != c.speaker_id and o.duration > c.duration]
        if not others:
            continue
        sims = sorted(((_cos(c.centroid, o.centroid), o) for o in others), key=lambda t: -t[0])
        best_sim, target = sims[0]
        share = c.duration / total_dur * 100
        if best_sim >= floor:
            audit.append({"ts": time.time(), "action": "cluster_merge",
                          "detail": f"聚类 {c.label}（时长 {c.duration:.1f}s，占比 {share:.1f}%）"
                                    f"并入 {target.label}（cos={best_sim:.3f} ≥ {floor:.3f}）",
                          "sim": round(best_sim, 3), "share_pct": round(share, 2)})
            for w in res.windows:
                if w.assigned == c.speaker_id:
                    w.assigned = target.speaker_id
            for s in res.segments:
                if s.speaker == c.speaker_id:
                    s.speaker = target.speaker_id
            target.duration += c.duration
            c.duration = 0.0
            c.flags.append("已并入 " + target.name)
        else:
            c.flags.append(f"可能偶发（占比 {share:.1f}%，最近 cos={best_sim:.3f} < {floor:.3f}，保留）")
            audit.append({"ts": time.time(), "action": "cluster_keep_flagged",
                          "detail": f"聚类 {c.label} 疑似偶发但相似度不足（cos={best_sim:.3f}），保留独立",
                          "sim": round(best_sim, 3), "share_pct": round(share, 2)})
    _reindex(res)


def segment_level(res: dz.DiarizeResult, cfg, audit: list[dict]) -> None:
    """自动模式：歧义段/超短段的去留由评分函数决定。"""
    if cfg.outlier_mode != "auto":
        return
    centroids = {c.speaker_id: c.centroid for c in res.clusters}
    # 每说话人平均能量（dB）
    e_map: dict[int, list[float]] = {}
    for s in res.segments:
        if s.speaker >= 0 and s.kind == "speech":
            e_map.setdefault(s.speaker, []).append(s.energy_db)
    energy_db = {k: float(np.mean(v)) for k, v in e_map.items()}
    snapshot = list(res.segments)
    for seg in snapshot:
        if seg.kind != "speech":
            continue
        ambiguous = seg.top1 < cfg.merge_threshold or (seg.top1 - seg.top2) < 0.05
        very_short = seg.dur < 0.8
        if not (ambiguous or very_short):
            continue
        best = seg.speaker
        if best < 0 or best not in centroids:
            continue
        score, breakdown = auto_score(seg, best, cfg.merge_threshold, centroids, energy_db, snapshot)
        seg.audit = {"auto_score": score, **breakdown}
        if score >= cfg.auto_merge_score:
            seg.audit["decision"] = "merge"
            audit.append({"ts": time.time(), "action": "segment_merge",
                          "seg": [round(seg.start, 2), round(seg.end, 2)],
                          "detail": f"段 [{seg.start:.1f}s,{seg.end:.1f}s] 评分 {score:.2f} ≥ {cfg.auto_merge_score}"
                                    f" → 保留归属 {best}", "score": score, "breakdown": breakdown})
        elif score <= cfg.auto_unknown_score:
            seg.audit["decision"] = "unknown"
            audit.append({"ts": time.time(), "action": "segment_unknown",
                          "seg": [round(seg.start, 2), round(seg.end, 2)],
                          "detail": f"段 [{seg.start:.1f}s,{seg.end:.1f}s] 评分 {score:.2f} ≤ {cfg.auto_unknown_score}"
                                    f" → 归入未归属（other）", "score": score, "breakdown": breakdown})
            seg.speaker = -1
            seg.kind = "unknown"
        else:
            seg.audit["decision"] = "review"
            audit.append({"ts": time.time(), "action": "segment_review",
                          "seg": [round(seg.start, 2), round(seg.end, 2)],
                          "detail": f"段 [{seg.start:.1f}s,{seg.end:.1f}s] 评分 {score:.2f} 处于复核区间，"
                                    f"保留归属但已标记", "score": score, "breakdown": breakdown})


def guard_new_speakers(res: dz.DiarizeResult, cfg, audit: list[dict]) -> None:
    """持续证据原则：不足证据的小簇不作为独立说话人输出（并入最近邻或 unknown）。"""
    if cfg.outlier_mode == "off":
        return
    for c in res.clusters:
        if c.duration <= 0:
            continue
        segs = [s for s in res.segments if s.speaker == c.speaker_id]
        if c.duration >= cfg.min_new_speaker_duration and len(segs) >= cfg.min_new_speaker_segments:
            continue
        others = [o for o in res.clusters if o.speaker_id != c.speaker_id and o.duration > 0]
        if not others:
            continue
        sims = sorted(((_cos(c.centroid, o.centroid), o) for o in others), key=lambda t: -t[0])
        best_sim, target = sims[0]
        if best_sim >= cfg.merge_threshold - 0.05:
            for s in segs:
                s.speaker = target.speaker_id
            audit.append({"ts": time.time(), "action": "evidence_guard_merge",
                          "detail": f"聚类 {c.label} 证据不足（{c.duration:.1f}s/{len(segs)}段 < "
                                    f"{cfg.min_new_speaker_duration}s/{cfg.min_new_speaker_segments}段）"
                                    f"并入 {target.label}（cos={best_sim:.3f}）"})
            c.duration = 0.0
            c.flags.append("证据不足已并入")
        else:
            c.flags.append(f"证据不足但无近邻（cos={best_sim:.3f}）：保留，请人工复核")
            audit.append({"ts": time.time(), "action": "evidence_guard_flag",
                          "detail": f"聚类 {c.label} 证据不足且无足够近邻，保留并标记复核"})
    _reindex(res)


def _reindex(res: dz.DiarizeResult) -> None:
    ids = sorted({c.speaker_id for c in res.clusters if c.duration > 0})
    remap = {old: new for new, old in enumerate(ids)}
    for c in res.clusters:
        c.speaker_id = remap.get(c.speaker_id, c.speaker_id) if c.duration > 0 else -1
        if c.duration > 0:
            c.name = f"说话人 {c.speaker_id + 1}" if not any(f.startswith("声纹库") for f in c.flags) else c.name
    for s in res.segments:
        if s.speaker >= 0:
            s.speaker = remap.get(s.speaker, s.speaker)


def run_outlier_stage(res: dz.DiarizeResult, cfg) -> list[dict]:
    audit: list[dict] = []
    cluster_level(res, cfg, audit)
    segment_level(res, cfg, audit)
    guard_new_speakers(res, cfg, audit)
    # 重新计算每说话人总时长
    dur: dict[int, float] = {}
    for s in res.segments:
        if s.speaker >= 0:
            dur[s.speaker] = dur.get(s.speaker, 0.0) + s.dur
    for c in res.clusters:
        if c.speaker_id >= 0:
            c.duration = dur.get(c.speaker_id, 0.0)
    _reindex(res)
    return audit

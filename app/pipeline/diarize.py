"""声纹滑窗 + 可配置阈值凝聚聚类 + 段归属 + 声纹库命名。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


@dataclass
class Window:
    idx: int
    start: float
    end: float
    emb: np.ndarray | None = None
    label: int = -1           # 聚类标签
    assigned: int = -1        # 最终说话人 id（0 起）
    top1: float = 0.0
    top2: float = 0.0
    best1: int = -1
    best2: int = -1
    energy_db: float = -120.0


@dataclass
class Cluster:
    label: int
    centroid: np.ndarray
    members: list[int] = field(default_factory=list)
    speaker_id: int = -1
    name: str = ""
    duration: float = 0.0
    flags: list[str] = field(default_factory=list)


@dataclass
class Segment:
    start: float
    end: float
    speaker: int                 # -1 = unknown（进 other）
    kind: str = "speech"         # speech | overlap | unknown
    top1: float = 0.0
    top2: float = 0.0
    margin: float = 0.0
    dur: float = 0.0
    energy_db: float = -120.0
    audit: dict = field(default_factory=dict)
    seg_id: int = 0


@dataclass
class DiarizeResult:
    windows: list[Window]
    clusters: list[Cluster]
    segments: list[Segment]
    audit: list[dict] = field(default_factory=list)
    vad_segments: list[tuple[float, float]] = field(default_factory=list)

    def next_seg_id(self) -> int:
        return (max((s.seg_id for s in self.segments), default=0)) + 1


def sliding_windows(vad_segments: list[tuple[float, float]], total_dur: float,
                    win: float = 1.5, hop: float = 0.75, min_len: float = 0.4) -> list[Window]:
    if win <= 0 or hop <= 0:
        raise ValueError(f"滑窗参数非法：win={win}, hop={hop}（必须 > 0）")
    out: list[Window] = []
    idx = 0
    for s, e in vad_segments:
        s, e = max(0.0, s), min(total_dur, e)
        if e - s < min_len:  # 短语音段整窗保留（偶发声音来源之一）
            out.append(Window(idx=idx, start=s, end=e))
            idx += 1
            continue
        t = s
        while t < e - 1e-6:
            w_end = min(t + win, e)
            if w_end - t >= min_len or (w_end >= e and e - t > 0.2):
                out.append(Window(idx=idx, start=t, end=w_end))
                idx += 1
            if w_end >= e:
                break
            t += hop
    return out


def average_linkage_cluster(embs: np.ndarray, threshold: float) -> np.ndarray:
    """凝聚聚类（平均链接，距离=1-cos），cos ≥ threshold 合并。返回标签数组。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    if len(embs) == 0:
        return np.zeros(0, dtype=int)
    if len(embs) == 1:
        return np.zeros(1, dtype=int)
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    dist = pdist(X, metric="cosine")  # 1 - cos
    Z = linkage(dist, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return labels.astype(int)


def smooth_labels(windows: list[Window], hop: float, smoothing_s: float) -> None:
    """时间平滑：滑窗多数票，防标签抖动。"""
    k = max(1, int(round(smoothing_s / max(hop, 0.1))) | 1)
    if len(windows) < 3 or k <= 1:
        return
    labels = [w.label for w in windows]
    smoothed = labels[:]
    half = k // 2
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        smoothed[i] = int(vals[np.argmax(counts)])
    for w, lb in zip(windows, smoothed):
        w.label = lb


def build_clusters(windows: list[Window], embs: np.ndarray) -> list[Cluster]:
    labels = sorted({w.label for w in windows})
    clusters: list[Cluster] = []
    for lb in labels:
        mem = [w.idx for w in windows if w.label == lb]
        vecs = embs[mem]
        cen = vecs.mean(axis=0)
        cen = cen / (np.linalg.norm(cen) + 1e-9)
        dur = sum(windows[i].end - windows[i].start for i in mem)
        clusters.append(Cluster(label=lb, centroid=cen, members=mem, duration=dur))
    clusters.sort(key=lambda c: -c.duration)  # 说话人 id 按时长降序分配
    for sid, c in enumerate(clusters):
        c.speaker_id = sid
        c.name = f"说话人 {sid + 1}"
    for w in windows:
        for c in clusters:
            if c.label == w.label:
                w.assigned = c.speaker_id
                break
    return clusters


def score_against(w: Window, clusters: list[Cluster]) -> None:
    sims = [(_cos(w.emb, c.centroid), c.speaker_id) for c in clusters if w.emb is not None]
    sims.sort(reverse=True)
    w.top1, w.best1 = (sims[0] if sims else (0.0, -1))
    w.top2, w.best2 = (sims[1] if len(sims) > 1 else (0.0, -1))


def segmentize(windows: list[Window], vad_segments: list[tuple[float, float]],
               clusters: list[Cluster], min_seg: float = 0.25) -> list[Segment]:
    """窗口归属 → 连续同说话人段（段边界=窗口运行边界中点），短段并入前段。"""
    segments: list[Segment] = []
    for s, e in vad_segments:
        ws = [w for w in windows if w.start >= s - 1e-6 and w.end <= e + 1e-6]
        if not ws:
            continue
        runs: list[list] = []  # [start, end, speaker, last_window]
        for w in ws:
            if runs and runs[-1][2] == w.assigned:
                runs[-1][1] = w.end
                runs[-1][3] = w
            else:
                runs.append([w.start, w.end, w.assigned, w])
        # 运行边界中点细化
        refined: list[dict] = []
        for i, (rs, re_, spk, wlast) in enumerate(runs):
            bs = rs if i == 0 else (runs[i - 1][1] + rs) / 2
            be = re_ if i == len(runs) - 1 else (re_ + runs[i + 1][0]) / 2
            refined.append({"start": bs, "end": be, "spk": spk, "w": wlast})
        for r in refined:
            if r["end"] - r["start"] <= 0.05:
                continue
            w: Window = r["w"]
            segments.append(Segment(
                start=r["start"], end=r["end"], speaker=r["spk"],
                top1=w.top1, top2=w.top2, margin=w.top1 - w.top2,
                dur=r["end"] - r["start"], energy_db=w.energy_db))
    segments.sort(key=lambda x: x.start)
    # 短段并入前段：仅同说话人且紧邻（跨说话人吸收会抹掉快速轮换）
    merged: list[Segment] = []
    for seg in segments:
        if (merged and seg.dur < min_seg and seg.start - merged[-1].end < 0.3
                and merged[-1].speaker == seg.speaker):
            prev = merged[-1]
            prev.end, prev.dur = seg.end, seg.end - prev.start
        else:
            merged.append(seg)
    for i, seg in enumerate(merged):
        seg.seg_id = i
    return merged


def apply_voiceprints(res: "DiarizeResult", library: list[dict], threshold: float,
                      audit: list[dict]) -> None:
    """声纹库命名 + 同库条目聚类合并（同一真实说话人）。
    簇 id 重排时必须同步重排 segments/windows（否则账本 KeyError / 归属旋转）。"""
    entries = [v for v in library if v.get("enabled", True) and v.get("embedding")]
    if not entries:
        return
    clusters = res.clusters
    # 1) 每簇找最佳库匹配（只考虑存活簇，死簇不参与合并/命名）
    matched: dict[int, tuple[dict, float]] = {}   # id(cluster) -> (entry, sim)
    for c in clusters:
        if c.speaker_id < 0 or c.duration <= 0:
            continue
        best, bs = None, -1.0
        for v in entries:
            sim = _cos(c.centroid, np.asarray(v["embedding"], dtype="float32"))
            if sim > bs:
                bs, best = sim, v
        if best is not None and bs >= threshold:
            matched[id(c)] = (best, bs)
    if not matched:
        return
    # 2) 命中同一条目的多个簇 → 合并为同一 speaker_id（保留最大簇）
    by_entry: dict[str, list[Cluster]] = {}
    for c in clusters:
        if id(c) in matched:
            by_entry.setdefault(matched[id(c)][0]["id"], []).append(c)
    for vid, hits in by_entry.items():
        if len(hits) >= 2:
            keep = max(hits, key=lambda c: c.duration)
            for c in hits:
                if c is not keep:
                    c.speaker_id = keep.speaker_id
            audit.append({"ts": time.time(), "action": "voiceprint_merge",
                          "detail": f"聚类 {[c.label for c in hits if c is not keep]} 并入 {keep.label}"
                                    f"（同命中声纹库「{matched[id(keep)][0]['name']}」）"})
    # 3) 命名
    for c in clusters:
        if id(c) in matched:
            v, sim = matched[id(c)]
            c.name = v["name"]
            audit.append({"ts": time.time(), "action": "voiceprint_name",
                          "detail": f"聚类 {c.label} 命中声纹库「{v['name']}」(sim={sim:.3f}) → 重命名"})
    # 4) id 连续化：只重排存活簇（speaker_id≥0），死簇固定 -1；segments/windows 同步
    live = [c for c in clusters if c.speaker_id >= 0]
    ids = sorted({c.speaker_id for c in live})
    remap = {old: new for new, old in enumerate(ids)}
    for c in clusters:
        c.speaker_id = remap.get(c.speaker_id, -1) if c.speaker_id >= 0 else -1
    for w in res.windows:
        if w.assigned in remap:
            w.assigned = remap[w.assigned]
    for s in res.segments:
        if s.speaker in remap:
            s.speaker = remap[s.speaker]


def diarize(embs: np.ndarray, windows: list[Window], vad_segments: list[tuple[float, float]],
            threshold: float, smoothing_s: float, hop: float) -> DiarizeResult:
    labels = average_linkage_cluster(embs, threshold)
    for i, (w, lb) in enumerate(zip(windows, labels)):
        w.label = int(lb)
        w.emb = embs[i]  # 必须挂回窗口：置信度/outlier 评分都依赖它
    smooth_labels(windows, hop, smoothing_s)
    clusters = build_clusters(windows, embs)
    for w in windows:
        score_against(w, clusters)
    segments = segmentize(windows, vad_segments, clusters)
    return DiarizeResult(windows=windows, clusters=clusters, segments=segments,
                         vad_segments=list(vad_segments))


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def refine_clusters(res: DiarizeResult, embs: np.ndarray, threshold: float,
                    max_speakers: int = 0) -> DiarizeResult:
    """多说话人场景的二次精修：
    1) 质心再指派：与自身质心相似度 < τ 的窗改派到 ≥τ 的最近质心；
    2) 质心合并：两个簇质心 cos ≥ τ 视为同一人被拆开 → 合并（解决 10+ 人时的过度分裂）；
    3) 可选硬上限：超过 max_speakers 时把最小簇并入最近邻。
    重建 segments，保持 id 连续。"""
    n = len(res.windows)
    if n == 0:
        return res
    X = np.stack([_norm(np.asarray(embs[i], "float32")) for i in range(n)])
    labels = np.array([w.assigned for w in res.windows], dtype=int)

    def centroids_of(lbl: np.ndarray) -> dict[int, np.ndarray]:
        ids = sorted(set(lbl.tolist()) - {-1})
        out = {}
        for i in ids:
            m = lbl == i
            if m.any():
                out[i] = _norm(X[m].mean(axis=0))
        return out

    # 1) 质心再指派
    for _ in range(2):
        cents = centroids_of(labels)
        if not cents:
            break
        ids = sorted(cents)
        M = np.stack([cents[i] for i in ids])
        sims = X @ M.T
        changed = False
        for k in range(n):
            if labels[k] < 0:
                continue
            cur = ids.index(labels[k])
            cur_sim = float(sims[k, cur])
            best = int(np.argmax(sims[k]))
            if cur_sim < threshold and float(sims[k, best]) >= threshold and ids[best] != labels[k]:
                labels[k] = ids[best]
                changed = True
        if not changed:
            break

    # 2) 质心合并（迭代到无 ≥τ 的簇对）
    while True:
        cents = centroids_of(labels)
        ids = sorted(cents)
        if len(ids) < 2:
            break
        best = (-1.0, None, None)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                s = float(np.dot(cents[ids[a]], cents[ids[b]]))
                if s > best[0]:
                    best = (s, ids[a], ids[b])
        if best[0] < threshold:
            break
        labels[labels == best[2]] = best[1]

    # 3) 硬上限：最小簇并入最近簇
    if max_speakers and max_speakers > 0:
        while len(set(labels.tolist()) - {-1}) > max_speakers:
            cents = centroids_of(labels)
            ids = sorted(cents)
            counts = {i: int((labels == i).sum()) for i in ids}
            small = min(ids, key=lambda i: counts[i])
            others = [i for i in ids if i != small]
            nearest = max(others, key=lambda o: float(np.dot(cents[small], cents[o])))
            labels[labels == small] = nearest

    return _rebuild(res, labels, embs)


def _rebuild(res: DiarizeResult, labels: np.ndarray, embs: np.ndarray) -> DiarizeResult:
    ids = sorted(set(labels.tolist()) - {-1})
    remap = {old: new for new, old in enumerate(ids)}
    windows = res.windows
    clusters: list[Cluster] = []
    for old in ids:
        mem = [i for i in range(len(labels)) if labels[i] == old]
        vecs = np.stack([embs[i] for i in mem])
        cen = _norm(vecs.mean(axis=0))
        dur = sum(windows[i].end - windows[i].start for i in mem)
        c = Cluster(label=remap[old] + 1, centroid=cen, members=mem,
                    speaker_id=remap[old], duration=dur)
        c.name = f"说话人 {remap[old] + 1}"
        clusters.append(c)
    by_id = {c.speaker_id: c for c in clusters}
    for i, w in enumerate(windows):
        old = int(labels[i])
        w.assigned = remap.get(old, -1)
        w.label = by_id[w.assigned].label if w.assigned >= 0 else -1
    for w in windows:
        score_against(w, clusters)
    segments = segmentize(windows, getattr(res, "vad_segments", []), clusters)
    return DiarizeResult(windows=windows, clusters=clusters, segments=segments,
                         audit=res.audit, vad_segments=getattr(res, "vad_segments", []))

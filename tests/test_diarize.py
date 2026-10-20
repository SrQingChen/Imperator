"""声纹库应用回归（第二轮验收 P1）与设置校验测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import diarize as dz  # noqa: E402
from app.pipeline import ledger  # noqa: E402
from app.settings import Settings  # noqa: E402


def _res_with_dead_cluster():
    """3 活簇 + 1 死簇（speaker_id=-1，模拟偶发排除后），段/窗挂在各簇上。"""
    rng = np.random.default_rng(3)
    def emb(seed):
        v = rng.normal(0, 1, 8); return v / np.linalg.norm(v)
    windows, clusters, segments = [], [], []
    wi = 0
    for cid in range(3):
        c = dz.Cluster(label=cid + 1, centroid=emb(cid), speaker_id=cid,
                       duration=10.0 - cid)
        for k in range(4):
            w = dz.Window(idx=wi, start=wi * 0.75, end=wi * 0.75 + 1.5,
                          label=cid + 1, assigned=cid, emb=None)
            windows.append(w); wi += 1
        c.members = [x.idx for x in windows if x.assigned == cid]
        clusters.append(c)
        s = dz.Segment(start=cid * 10, end=cid * 10 + 8, speaker=cid, dur=8)
        s.seg_id = cid
        segments.append(s)
    dead = dz.Cluster(label=9, centroid=emb(9), speaker_id=-1, duration=0.0)
    clusters.append(dead)
    return dz.DiarizeResult(windows=windows, clusters=clusters, segments=segments)


def test_voiceprint_nonempty_library_no_keyerror():
    """库非空 + 命中：id 重排必须同步 segments/windows，账本不得 KeyError。"""
    res = _res_with_dead_cluster()
    lib_entry = {"id": "abc", "name": "库主人", "enabled": True,
                 "embedding": res.clusters[0].centroid.tolist()}
    audit = []
    dz.apply_voiceprints(res, [lib_entry], 0.5, audit)
    live_ids = sorted({c.speaker_id for c in res.clusters if c.speaker_id >= 0})
    assert live_ids == list(range(len(live_ids))), "id 应连续"
    assert all(c.speaker_id == -1 for c in res.clusters if c.duration <= 0)
    seg_ids = {s.speaker for s in res.segments}
    assert seg_ids <= set(live_ids), f"段残留越界 id：{seg_ids} ⊄ {live_ids}"
    win_ids = {w.assigned for w in res.windows}
    assert win_ids <= set(live_ids)
    assert res.clusters[0].name == "库主人"
    # 账本构建（P1 崩溃点）
    items = [{"start": s.start, "end": s.end, "speaker": s.speaker, "kind": "speech"}
             for s in res.segments if s.speaker >= 0]
    gains = ledger.build_ctrl_gains(items, live_ids, 30.0)
    assert set(gains.keys()) == set(live_ids)


def test_voiceprint_same_entry_merges_clusters():
    """两个簇命中同一条目 → 合并为同一说话人且 id 一致。"""
    res = _res_with_dead_cluster()
    e = res.clusters[0].centroid
    lib = [{"id": "x", "name": "同一人", "enabled": True, "embedding": e.tolist()}]
    res.clusters[1].centroid = e * 0.99 + 0.01  # 近似 → 也命中
    audit = []
    dz.apply_voiceprints(res, lib, 0.5, audit)
    live = [c for c in res.clusters if c.speaker_id >= 0]
    assert any(a["action"] == "voiceprint_merge" for a in audit)
    merged = {c.speaker_id for c in live}
    assert len(merged) < len(res.clusters) - 1  # 簇数减少
    seg_ids = {s.speaker for s in res.segments}
    assert seg_ids <= merged | {-1}


def test_settings_validation_hard_bounds():
    ok = Settings().validate()
    assert ok == []
    bad_cases = [
        dict(window_hop=0), dict(window_hop=-0.5), dict(window_seconds=0.1),
        dict(window_hop=2.0, window_seconds=1.0),   # hop > win
        dict(overlap_context=-1), dict(listen_port=99999), dict(listen_port=0),
        dict(output_format="xyz"), dict(output_sr=22050), dict(mss_precision="int8"), dict(mss_overlap=1), dict(mss_overlap=9),
        dict(auto_merge_score=0.3, auto_unknown_score=0.5),  # unknown > merge
        dict(min_new_speaker_duration=-1),
        dict(listen_host="  "),
        dict(max_overlap_speakers=1), dict(max_speakers=-1),
        dict(outlier_max_seconds=-2),
    ]
    for case in bad_cases:
        errs = Settings(**{**Settings().to_dict(), **case}).validate()
        assert errs, f"应拒绝非法配置：{case}"


# ---------------------------------------------------------------- 聚类精修

def _embs_split_same_speaker():
    """同一人被拆成两个相似度约 0.8 的子簇（τ=0.9 时会分开）+ 一个完全不同的 C。"""
    rng = np.random.default_rng(11)
    base_a = rng.normal(0, 1, 16)
    base_a /= np.linalg.norm(base_a)
    orth = rng.normal(0, 1, 16)
    orth -= base_a * np.dot(orth, base_a)
    orth /= np.linalg.norm(orth)
    base_a2 = 0.8 * base_a + 0.6 * orth        # cos(base_a, base_a2)=0.8
    base_c = rng.normal(0, 1, 16)
    base_c /= np.linalg.norm(base_c)
    rows = []
    for _ in range(5):
        v = base_a + 0.01 * rng.normal(0, 1, 16)
        rows.append(v / np.linalg.norm(v))
    for _ in range(4):
        v = base_a2 + 0.01 * rng.normal(0, 1, 16)
        rows.append(v / np.linalg.norm(v))
    for _ in range(3):
        v = base_c + 0.01 * rng.normal(0, 1, 16)
        rows.append(v / np.linalg.norm(v))
    return np.stack(rows).astype("float32")


def test_refine_merges_split_same_speaker_and_sets_emb():
    embs = _embs_split_same_speaker()
    windows = [dz.Window(idx=i, start=i * 0.75, end=i * 0.75 + 1.5) for i in range(len(embs))]
    # τ=0.9 故意把同一人拆开
    res = dz.diarize(embs, windows, [(0.0, 30.0)], 0.9, 0.0, 0.75)
    assert len([c for c in res.clusters if c.speaker_id >= 0]) >= 3
    refined = dz.refine_clusters(res, embs, 0.62, max_speakers=0)
    live = [c for c in refined.clusters if c.speaker_id >= 0]
    assert len(live) == 2, f"同一人的两簇应合并为 2 人，实际 {len(live)}"
    ids = sorted(c.speaker_id for c in live)
    assert ids == list(range(len(ids)))
    # 置信度必须真实计算（历史 bug：w.emb 未赋值导致 top1 恒为 0）
    assert all(w.emb is not None for w in refined.windows)
    assert any(w.top1 > 0.5 for w in refined.windows)
    assert {s.speaker for s in refined.segments} <= set(ids)


def test_refine_max_speakers_cap():
    embs = _embs_split_same_speaker()
    windows = [dz.Window(idx=i, start=i * 0.75, end=i * 0.75 + 1.5) for i in range(len(embs))]
    res = dz.diarize(embs, windows, [(0.0, 30.0)], 0.9, 0.0, 0.75)
    refined = dz.refine_clusters(res, embs, 0.99, max_speakers=1)  # 极高阈值 → 只留上限
    live = [c for c in refined.clusters if c.speaker_id >= 0]
    assert len(live) == 1
    assert {s.speaker for s in refined.segments} == {0}


def test_many_speakers_12_kept_and_ids_continuous():
    """十几位说话人（不同时说话）：精修不得把它们合并，id 连续，账本能构建。"""
    rng = np.random.default_rng(21)
    K, PER = 12, 4
    dirs = []
    for _ in range(K):
        v = rng.normal(0, 1, 32)
        dirs.append(v / np.linalg.norm(v))
    rows = []
    for d in dirs:
        for _ in range(PER):
            v = d + 0.015 * rng.normal(0, 1, 32)
            rows.append(v / np.linalg.norm(v))
    embs = np.stack(rows).astype("float32")
    windows = [dz.Window(idx=i, start=i * 1.0, end=i * 1.0 + 1.5) for i in range(len(embs))]
    res = dz.diarize(embs, windows, [(0.0, len(embs) * 1.0 + 2)], 0.55, 0.0, 1.0)
    res = dz.refine_clusters(res, embs, 0.55, max_speakers=0)
    live = [c for c in res.clusters if c.speaker_id >= 0]
    assert len(live) == K, f"12 位说话人应全部保留，实际 {len(live)}"
    assert sorted(c.speaker_id for c in live) == list(range(K))
    items = [{"start": s.start, "end": s.end, "speaker": s.speaker, "kind": "speech"}
             for s in res.segments if s.speaker >= 0]
    gains = ledger.build_ctrl_gains(items, list(range(K)), len(embs) * 1.0 + 2)
    assert len(gains) == K

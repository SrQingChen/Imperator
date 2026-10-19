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
        dict(output_format="xyz"), dict(output_sr=22050),
        dict(auto_merge_score=0.3, auto_unknown_score=0.5),  # unknown > merge
        dict(min_new_speaker_duration=-1),
        dict(listen_host="  "),
    ]
    for case in bad_cases:
        errs = Settings(**{**Settings().to_dict(), **case}).validate()
        assert errs, f"应拒绝非法配置：{case}"

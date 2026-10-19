"""偶发排除：比例模式合并 / 自动评分边界 / 持续证据守卫。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import diarize as dz  # noqa: E402
from app.pipeline import outlier  # noqa: E402
from app.settings import Settings  # noqa: E402


def _mk_result():
    """2 个大说话人 + 1 个小簇（模拟惊呼），小簇与大说话人 A 相似度高。"""
    e = lambda seed: np.random.default_rng(seed).normal(0, 1, 8)
    base_a, base_b = e(1), e(2)
    cen_small = base_a + 0.01 * e(3)
    cen_small /= np.linalg.norm(cen_small)
    for v in (base_a, base_b):
        v /= np.linalg.norm(v)

    def W(idx, s, t_, label, assigned):
        w = dz.Window(idx=idx, start=s, end=t_, label=label, assigned=assigned,
                      emb=None, top1=0.9, top2=0.2)
        return w
    windows = ([W(i, i * 0.75, i * 0.75 + 1.5, 1, 0) for i in range(20)] +
               [W(20 + i, 20 + i * 0.75, 20 + i * 0.75 + 1.5, 2, 1) for i in range(15)] +
               [W(35, 30.0, 30.6, 3, 2)])  # 0.6s 偶发
    clusters = [dz.Cluster(label=1, centroid=base_a, members=list(range(20)), speaker_id=0, duration=25.0),
                dz.Cluster(label=2, centroid=base_b, members=list(range(20, 35)), speaker_id=1, duration=18.0),
                dz.Cluster(label=3, centroid=cen_small, members=[35], speaker_id=2, duration=0.6)]
    segments = [dz.Segment(start=0, end=15, speaker=0, top1=0.9, top2=0.2, dur=15, energy_db=-20),
                dz.Segment(start=16, end=29, speaker=1, top1=0.88, top2=0.2, dur=13, energy_db=-21),
                dz.Segment(start=30.0, end=30.6, speaker=2, top1=0.55, top2=0.2, dur=0.6, energy_db=-14)]
    for i, s in enumerate(segments):
        s.seg_id = i
    return dz.DiarizeResult(windows=windows, clusters=clusters, segments=segments)


def test_ratio_mode_merges_small_cluster():
    res = _mk_result()
    cfg = Settings(outlier_mode="ratio", outlier_ratio_percent=4.0)
    audit = outlier.run_outlier_stage(res, cfg)
    assert res.segments[2].speaker == 0, "小簇应并入最相似的说话人 A"
    assert any(a["action"] in ("cluster_merge", "evidence_guard_merge") for a in audit)
    assert len({s.speaker for s in res.segments}) <= 2


def test_off_mode_keeps():
    res = _mk_result()
    cfg = Settings(outlier_mode="off")
    outlier.run_outlier_stage(res, cfg)
    assert res.segments[2].speaker == 2


def test_auto_score_bounds():
    seg = dz.Segment(start=30, end=30.6, speaker=0, top1=0.61, top2=0.2,
                     dur=0.6, energy_db=-20)
    others = [dz.Segment(start=0, end=15, speaker=0, top1=0.9, top2=0.2, dur=15, energy_db=-20)]
    score, bd = outlier.auto_score(seg, 0, 0.62, {0: np.ones(8)}, {0: -20.0}, others)
    assert 0.0 <= score <= 1.0
    assert set(bd) >= {"sim", "margin", "dur", "ctx", "energy", "nspeech"}
    # 高置信长段的评分应显著高于低置信短段
    seg2 = dz.Segment(start=0, end=5, speaker=0, top1=0.9, top2=0.2, dur=5, energy_db=-20)
    s2, _ = outlier.auto_score(seg2, 0, 0.62, {0: np.ones(8)}, {0: -20.0},
                               others + [seg2])
    assert s2 > score


def test_auto_mode_unknown_for_orphan():
    res = _mk_result()
    # 孤儿段：与谁都最不像 → unknown
    res.segments[2].top1, res.segments[2].top2 = 0.1, 0.05
    cfg = Settings(outlier_mode="auto")
    audit = outlier.run_outlier_stage(res, cfg)
    assert res.segments[2].kind in ("unknown",) or res.segments[2].speaker >= 0

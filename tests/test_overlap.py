"""三人重叠递归分离与软掩码归一化测试（假分离器，确定性）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import overlap  # noqa: E402

SR = 16000
DUR = 3.0
N = int(SR * DUR)
T = np.arange(N) / SR
TONES = [np.sin(2 * np.pi * f * T).astype("float32") for f in (440.0, 880.0, 1320.0)]
CENTROIDS = {0: np.array([1.0, 0, 0], "float32"),
             1: np.array([0, 1.0, 0], "float32"),
             2: np.array([0, 0, 1.0], "float32")}


def _presence(x: np.ndarray) -> np.ndarray:
    v = np.array([abs(float(np.dot(x, t))) / (np.linalg.norm(x) * np.linalg.norm(t) + 1e-9)
                  for t in TONES], dtype="float32")
    return v


def embed_fn(x: np.ndarray, tag: int = 0) -> np.ndarray:
    v = np.clip(_presence(x), 0, None)
    return v / (np.linalg.norm(v) + 1e-9)


def sep_fn(x: np.ndarray) -> list[np.ndarray]:
    pres = _presence(x) > 0.15
    if pres.sum() >= 3:
        return [TONES[0].copy(), (TONES[1] + TONES[2]).astype("float32")]
    if pres[1] and pres[2]:
        return [TONES[1].copy(), TONES[2].copy()]
    return [x.copy(), np.zeros_like(x)]


def test_three_speaker_recursive_split():
    mix = sum(t * 0.3 for t in TONES).astype("float32")
    reg = overlap.resolve_region((0.0, DUR), 0.0, mix, SR, sep_fn, embed_fn,
                                 CENTROIDS, threshold=0.6, max_speakers=3)
    assert reg.kind == "overlap"
    assert len(reg.speakers) == 3, f"应拆出 3 人，实际 {reg.speakers}"
    assert set(reg.masks) == {0, 1, 2}
    total = np.sum([m for m in reg.masks.values()], axis=0)
    assert np.allclose(total, 1.0, atol=0.05), f"掩码和应≈1，实际 [{total.min():.3f},{total.max():.3f}]"
    for m in reg.masks.values():
        assert m.min() >= 0.0 and m.max() <= 1.0 + 1e-6


def test_two_speaker_mode_stops_at_two():
    """2 人模式：第三人的声音会归到最相似轨（掩码和=1，不丢音频），但不会被独立拆出。"""
    mix = sum(t * 0.3 for t in TONES).astype("float32")
    reg = overlap.resolve_region((0.0, DUR), 0.0, mix, SR, sep_fn, embed_fn,
                                 CENTROIDS, threshold=0.6, max_speakers=2)
    assert reg.kind == "overlap"
    assert len(reg.speakers) == 2
    total = np.sum([m for m in reg.masks.values()], axis=0)
    assert np.all(total <= 1.0 + 1e-5)
    assert np.allclose(total, 1.0, atol=0.05)  # 无丢弃（未拆出的声音并入最相似轨）


def test_single_speaker_region_is_speech():
    reg = overlap.resolve_region((0.0, DUR), 0.0, TONES[0].copy(), SR, sep_fn, embed_fn,
                                 CENTROIDS, threshold=0.6, max_speakers=3)
    assert reg.kind == "speech"
    assert reg.speakers == [0] and not reg.masks


def test_envelopes_sum_to_one():
    envs = overlap._envelopes([TONES[0], TONES[1], TONES[2]])
    total = np.sum(envs, axis=0)
    assert np.allclose(total, 1.0, atol=1e-4)

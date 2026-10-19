"""合成测试媒体：两人「语音」+ BGM，含重叠段与偶发（短促异响）。

用正弦调幅模拟不同音色的「说话」（基频与共振峰不同即可被声纹区分的
合成替代；用于端到端流程与账本断言，不用于质量评测）。
用法：python scripts/make_test_media.py --out test_media --dur 40
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _speech_like(f0: float, dur: float, sr: int = 44100, seed: int = 0,
                 formants=((700, 1200), (1100, 2600))) -> np.ndarray:
    """共振峰叠加 + 音节包络的类语音信号。"""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    # 音节包络：4Hz 左右的通断
    syl = 4.0
    env = ((np.sin(2 * math.pi * syl * t + seed) > -0.2)).astype("float32")
    k = int(0.02 * sr)
    env = np.convolve(env, np.ones(k, "float32") / k, mode="same")
    env *= (0.7 + 0.3 * np.sin(2 * math.pi * 1.3 * t + seed))  # 音节内起伏
    sig = np.zeros(n, "float32")
    for f, amp in [(f0, 1.0), (f0 * 2, 0.4), (f0 * 3, 0.2)]:
        jitter = 1.0 + 0.02 * np.sin(2 * math.pi * 5.5 * t + seed * 3)
        sig += amp * np.sin(2 * math.pi * f * jitter * t + rng.uniform(0, 6.28))
    for (f1, f2), amp in zip(formants, (0.5, 0.25)):
        noise = rng.normal(0, 1, n).astype("float32")
        # 简易共振峰：带通噪声叠加
        from scipy.signal import butter, lfilter
        b, a = butter(2, [f1 / (sr / 2), f2 / (sr / 2)], btype="band")
        sig += amp * lfilter(b, a, noise)
    return (sig * env).astype("float32")


def _bgm(dur: float, sr: int = 44100, seed: int = 7) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    music = np.zeros_like(t)
    for f in (110.0, 164.8, 220.0, 329.6):  # Am 和弦
        music += np.sin(2 * math.pi * f * t + f) * 0.18
    beat = ((t * 2) % 1 < 0.1).astype("float32") * 0.25
    music = music * 0.5 + beat * np.sin(2 * math.pi * 60 * t)
    return music.astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_media")
    ap.add_argument("--dur", type=float, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr = 44100
    dur = args.dur
    n = int(dur * sr)
    rng = np.random.default_rng(42)

    a = np.zeros(n, "float32")   # 说话人 A：低音色 f0=110
    b = np.zeros(n, "float32")   # 说话人 B：高音色 f0=220
    plan_a = [(1, 5), (8, 13), (20, 26), (31, 33.5), (36, dur)]
    plan_b = [(6, 10), (16, 19), (24, 29), (33, 36)]
    overlap = (24, 26)           # A 与 B 重叠
    for s, e in plan_a:
        a[int(s * sr):int(e * sr)] = _speech_like(110, e - s, sr, seed=1)
    for s, e in plan_b:
        b[int(s * sr):int(e * sr)] = _speech_like(220, e - s, sr, seed=2)
    # 偶发：A 的短促惊呼（0.6s，30.0~30.6，音色偏离）
    a[int(30.0 * sr):int(30.6 * sr)] = _speech_like(150, 0.6, sr, seed=9) * 1.6
    bgm = _bgm(dur, sr) * 0.4
    mix = a + b + bgm + rng.normal(0, 0.003, n).astype("float32")
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    import soundfile as sf
    sf.write(str(out / "mix_stereo.wav"), np.stack([mix, mix * 0.98], axis=1), sr, subtype="FLOAT")
    sf.write(str(out / "truth_a.wav"), a, sr, subtype="FLOAT")
    sf.write(str(out / "truth_b.wav"), b, sr, subtype="FLOAT")
    sf.write(str(out / "truth_bgm.wav"), bgm, sr, subtype="FLOAT")
    meta = {"sr": sr, "dur": dur, "plan_a": plan_a, "plan_b": plan_b,
            "overlap": overlap, "outlier": [30.0, 30.6]}
    import json
    (out / "meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"已生成测试媒体：{out / 'mix_stereo.wav'}（{dur}s，A/B/重叠/偶发 已知真值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

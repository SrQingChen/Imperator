"""合成测试媒体：真人语音 A/B + BGM，含重叠段与偶发（第三者短促插入）。

默认使用本机 IndexTTS voice_bank 的真实语音（参数可覆盖；缺失时回退合成音）。
用法：python scripts/make_test_media.py --out test_media --dur 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import audio_io  # noqa: E402

DEFAULT_A = r"E:\AI\indextts2.5\index-tts-2.5.0\voice_bank\audio\爱弥斯.wav"
DEFAULT_B = r"E:\AI\indextts2.5\index-tts-2.5.0\voice_bank\audio\男漂泊者.wav"
DEFAULT_C = r"E:\AI\indextts2.5\index-tts-2.5.0\voice_bank\audio\达妮娅.wav"
SR = 44100


def _load_speech(path: str, sr: int = SR) -> np.ndarray:
    x, x_sr = audio_io.load_audio(path)
    return audio_io.resample(audio_io.to_mono(x), x_sr, sr)


def _synth_speech(f0: float, dur: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """回退用：共振峰叠加 + 音节包络的类语音信号。"""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = (np.sin(2 * math.pi * 4.0 * t + seed) > -0.2).astype("float32")
    k = int(0.02 * sr)
    env = np.convolve(env, np.ones(k, "float32") / k, mode="same")
    sig = np.zeros(n, "float32")
    for f, amp in [(f0, 1.0), (f0 * 2, 0.4), (f0 * 3, 0.2)]:
        jit = 1.0 + 0.02 * np.sin(2 * math.pi * 5.5 * t + seed * 3)
        sig += amp * np.sin(2 * math.pi * f * jit * t + rng.uniform(0, 6.28))
    from scipy.signal import butter, lfilter
    for (f1, f2), amp in [((700, 1200), 0.4), ((1100, 2600), 0.2)]:
        b, a = butter(2, [f1 / (sr / 2), f2 / (sr / 2)], btype="band")
        sig += amp * lfilter(b, a, rng.normal(0, 1, n).astype("float32"))
    return sig * env


def _bgm(dur: float, sr: int = SR, seed: int = 7) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    music = np.zeros_like(t)
    for f in (110.0, 164.8, 220.0, 329.6):
        music += np.sin(2 * math.pi * f * t + f) * 0.18
    beat = ((t * 2) % 1 < 0.1).astype("float32") * 0.25
    music = music * 0.5 + beat * np.sin(2 * math.pi * 60 * t)
    return music.astype("float32")


def _place(track: np.ndarray, t0: float, clip: np.ndarray, sr: int = SR,
           gain: float = 1.0, fade: float = 0.005) -> None:
    i0 = int(round(t0 * sr))
    i1 = min(len(track), i0 + len(clip))
    if i1 <= i0:
        return
    seg = clip[: i1 - i0].copy()
    n = min(int(fade * sr), len(seg) // 4)
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype="float32")
        seg[:n] *= ramp
        seg[-n:] *= ramp[::-1]
    track[i0:i1] += seg * gain


def _cut(src: np.ndarray, offset: float, dur: float, sr: int = SR, gain: float = 0.9) -> np.ndarray:
    i0 = int(offset * sr)
    need = int(dur * sr)
    out = np.zeros(need, "float32")
    i1 = min(len(src), i0 + need)
    if i1 > i0:
        out[: i1 - i0] = src[i0:i1]
    peak = float(np.max(np.abs(out))) + 1e-9
    return (out / peak * gain).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_media")
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--speech-a", default=DEFAULT_A)
    ap.add_argument("--speech-b", default=DEFAULT_B)
    ap.add_argument("--speech-c", default=DEFAULT_C)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr, dur = SR, args.dur
    n = int(dur * sr)

    a_src = _load_speech(args.speech_a) if Path(args.speech_a).exists() else _synth_speech(110, 15)
    b_src = _load_speech(args.speech_b) if Path(args.speech_b).exists() else _synth_speech(220, 15)
    c_src = _load_speech(args.speech_c) if Path(args.speech_c).exists() else _synth_speech(160, 5)
    la, lb, lc = len(a_src) / sr, len(b_src) / sr, len(c_src) / sr

    A = np.zeros(n, "float32")
    B = np.zeros(n, "float32")
    C = np.zeros(n, "float32")
    # 时间线：A、B 独白若干轮 + 一段重叠 + 一次偶发（第三者 0.7s）
    plan_a = [(2.0, 8.0), (20.0, 27.0), (26.0, 33.0), (44.0, 52.0)]
    plan_b = [(12.0, 19.0), (26.0, 33.0), (38.0, 43.0), (55.0, 59.0)]
    overlap = (26.0, 33.0)
    outlier = (35.0, 35.7)
    cursor_a = cursor_b = 0.0
    for s, e in plan_a:
        d = e - s
        _place(A, s, _cut(a_src, cursor_a % max(la - d - 0.2, 0.5), d, gain=0.85), sr)
        cursor_a += d
    for s, e in plan_b:
        d = e - s
        _place(B, s, _cut(b_src, cursor_b % max(lb - d - 0.2, 0.5), d, gain=0.85), sr)
        cursor_b += d
    _place(C, outlier[0], _cut(c_src, 0.5, outlier[1] - outlier[0], gain=1.1), sr)

    bgm = _bgm(dur, sr) * 0.35
    mix = A + B + C + bgm
    peak = float(np.max(np.abs(mix)))
    if peak > 0.95:
        mix *= 0.95 / peak

    sf.write(str(out / "mix_stereo.wav"), np.stack([mix, mix * 0.98], axis=1), sr, subtype="FLOAT")
    sf.write(str(out / "truth_a.wav"), A, sr, subtype="FLOAT")
    sf.write(str(out / "truth_b.wav"), B, sr, subtype="FLOAT")
    sf.write(str(out / "truth_bgm.wav"), bgm, sr, subtype="FLOAT")
    meta = {"sr": sr, "dur": dur, "plan_a": plan_a, "plan_b": plan_b,
            "overlap": overlap, "outlier": list(outlier),
            "sources": {"a": args.speech_a, "b": args.speech_b, "c": args.speech_c}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    print(f"已生成测试媒体：{out / 'mix_stereo.wav'}（{dur}s；A/B 独白+重叠 {overlap}s+偶发 {outlier}s，真值已知）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

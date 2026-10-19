"""阈值校准脚本：在合成标注集上扫描 τ / 偶发参数，输出 reliability 报告。

用法：
  python scripts/calibrate.py --media test_media --sweep 0.5 0.55 0.6 0.65 0.7 0.75
（需要已生成的 make_test_media 素材与已就绪的声纹模型；仅跑声纹/聚类层，不跑 MSS）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.pipeline import audio_io, diarize as dz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default="test_media")
    ap.add_argument("--sweep", nargs="*", type=float,
                    default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    args = ap.parse_args()
    media = Path(args.media)
    meta = json.loads((media / "meta.json").read_text("utf-8"))
    x, sr = audio_io.load_audio(media / "mix_stereo.wav")
    mono16 = audio_io.resample(audio_io.to_mono(x), sr, 16000)

    from app.pipeline.engine import Embedder
    import soundfile as sf
    emb_model = Embedder(device=config.torch_device())
    truth_segs = ([(s, e, "A") for s, e in meta["plan_a"]] +
                  [(s, e, "B") for s, e in meta["plan_b"]])

    rows = []
    for tau in args.sweep:
        windows = dz.sliding_windows(truth_segs_truth(truth_segs, mono16), len(mono16) / 16000)
        embs = []
        ok_windows = []
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            for w in windows:
                i0, i1 = int(w.start * 16000), int(w.end * 16000)
                clip = np.ascontiguousarray(mono16[i0:i1], "float32")
                p = Path(td) / "w.wav"
                sf.write(str(p), clip, 16000)
                e = emb_model.embed_path(p)
                if e is not None:
                    embs.append(e)
                    ok_windows.append(w)
        embs = np.stack(embs)
        res = dz.diarize(embs, ok_windows, [(s, e) for s, e, _ in truth_segs], tau, 1.2, 0.75)
        # 用真值标签评测：窗口级多数映射后计算纯度/覆盖
        purity, coverage, n_spk = evaluate(res, truth_segs)
        rows.append({"tau": tau, "speakers": n_spk, "purity": round(purity, 3),
                     "coverage": round(coverage, 3)})
        print(f"τ={tau:.2f} → 说话人数={n_spk} 纯度={purity:.3f} 覆盖={coverage:.3f}")
    out = media / "calibration.json"
    out.write_text(json.dumps(rows, indent=2), "utf-8")
    print(f"已写入 {out}")
    return 0


def truth_segs_truth(truth_segs, mono16):
    """真值时间线直接作为 VAD 输入（校准只考察聚类层）。"""
    return [(s, e) for s, e, _ in truth_segs]


def evaluate(res: dz.DiarizeResult, truth_segs):
    """窗口级评测：把每个聚类映射到使其纯度最高的真值说话人。"""
    total = correct = 0
    tmap: dict[int, str] = {}
    for c in res.clusters:
        votes: dict[str, float] = {}
        for w in (res.windows[i] for i in c.members):
            for s, e, who in truth_segs:
                ov = max(0, min(w.end, e) - max(w.start, s))
                if ov > 0:
                    votes[who] = votes.get(who, 0) + ov
        if votes:
            tmap[c.speaker_id] = max(votes, key=votes.get)
    covered = 0
    for w in res.windows:
        for s, e, who in truth_segs:
            ov = max(0, min(w.end, e) - max(w.start, s))
            if ov > 0:
                total += ov
                covered += ov
                if tmap.get(w.assigned) == who:
                    correct += ov
    return (correct / max(total, 1e-9), covered / max(sum(e - s for s, e, _ in truth_segs), 1e-9),
            len({c.speaker_id for c in res.clusters if c.duration > 0}))


if __name__ == "__main__":
    raise SystemExit(main())

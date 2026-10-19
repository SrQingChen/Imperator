"""端到端实测：合成媒体 → 全流水线 → 契约断言。

前置：模型已就绪（scripts/fetch_models.py --all）。
用法：.venv/Scripts/python scripts/run_e2e.py [--tier standard] [--dur 40]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="standard")
    ap.add_argument("--dur", type=float, default=40)
    ap.add_argument("--skip-media", action="store_true")
    args = ap.parse_args()
    media = ROOT / "test_media"
    if args.skip_media and (media / "mix_stereo.wav").exists():
        pass
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts/make_test_media.py"),
                        "--out", str(media), "--dur", str(args.dur)], check=True)

    from app.settings import Settings
    from app.pipeline.engine import Engine

    cfg = Settings(mss_tier=args.tier, release_vram=False)
    engine = Engine()
    job_dir = ROOT / "data" / "outputs" / "e2e_manual"
    import shutil
    shutil.rmtree(job_dir, ignore_errors=True)

    t0 = time.time()
    result = engine.run(media / "mix_stereo.wav", job_dir, cfg,
                        lambda s, f, n: print(f"  [{s}] {f*100:5.1f}% {n}", flush=True)
                        if f in (0.0, 1.0) or int(f * 100) % 25 == 0 else None,
                        lambda: False)
    print(f"\n流水线耗时 {time.time()-t0:.1f}s，QC：")
    print(json.dumps(result["qc"], indent=2, ensure_ascii=False, default=str))

    # ---- 契约断言
    import soundfile as sf
    ok = True
    m, sr = sf.read(str(job_dir / "master.wav"), dtype="float32", always_2d=True)
    acc = np.zeros_like(m)
    for t in result["tracks"]:
        p = job_dir / t["file"]
        x, x_sr = sf.read(str(p), dtype="float32", always_2d=True)
        assert x.shape[0] == m.shape[0], f"{t['file']} 时长不一致 {x.shape[0]} != {m.shape[0]}"
        assert x_sr == sf.info(str(job_dir / "master.wav")).samplerate
        acc += x
    err = float(np.max(np.abs(acc - m)))
    print(f"Σ 恒等断言：max|Σtracks − master| = {err:.3e}")
    ok &= err < 1e-3

    # 说话人轨在真值静音区的静音校验：B 独占区间 = plan_b 减去 plan_a/重叠/偶发
    meta = json.loads((media / "meta.json").read_text("utf-8"))
    busy = list(meta["plan_a"]) + [tuple(meta["overlap"]), tuple(meta["outlier"])]

    def b_only_intervals():
        out = []
        for s, e in meta["plan_b"]:
            cur = [(s, e)]
            for bs, be in busy:
                nxt = []
                for cs, ce in cur:
                    if be <= cs or bs >= ce:
                        nxt.append((cs, ce))
                        continue
                    if cs < bs:
                        nxt.append((cs, min(ce, bs)))
                    if ce > be:
                        nxt.append((max(cs, be), ce))
                cur = nxt
            out += [(cs, ce) for cs, ce in cur if ce - cs >= 2.0]
        return out
    a_track = next((job_dir / t["file"] for t in result["tracks"]
                    if t["key"].startswith("speaker")), None)
    if a_track:
        spk_files = sorted(job_dir.glob("speaker_*.wav"))
        # 找出 A 对应哪条轨（按与真值的内积）
        ta, _ = sf.read(str(media / "truth_a.wav"), dtype="float32")
        best, best_e = None, -1
        for p in spk_files:
            x, _ = sf.read(str(p), dtype="float32", always_2d=True)
            mono = x.mean(axis=1)
            n = min(len(mono), len(ta))
            e = float(np.dot(mono[:n], ta[:n]))
            if e > best_e:
                best_e, best = e, p
        print(f"A 对应轨道：{best.name if best else '未检出'}（内积 {best_e:.2f}）")
        if best:
            x, _ = sf.read(str(best), dtype="float32", always_2d=True)
            mono = x.mean(axis=1)
            for s, e in b_only_intervals():
                i0, i1 = int(s * sr), int(e * sr)
                peak = float(np.max(np.abs(mono[i0:i1]))) if i1 <= len(mono) else None
                print(f"  B 独占区间 [{s},{e}]s A 轨峰值 = {peak}")
    print("说话人：", [(s["name"], s["duration_s"], s["share_pct"]) for s in result["speakers"]])
    print("重叠段：", sum(1 for s in result["segments"] if s["kind"] == "overlap"))
    print("审计条目：", len(result["audit"]))
    print("\nE2E", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

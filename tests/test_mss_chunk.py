"""分块 MSS：重叠淡入淡出、真实进度、取消；性能规划策略。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import mss  # noqa: E402

SR = 44100


def _make_input(path: Path, dur: float = 10.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (0.4 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    sf.write(str(path), np.stack([x, x], axis=1), SR, subtype="FLOAT")
    return x


def _fake_separator(tmp_path: Path):
    obj = mss.MssSeparator.__new__(mss.MssSeparator)
    obj._output_dir = tmp_path / "mss_out"
    obj._output_dir.mkdir(parents=True, exist_ok=True)
    obj.model_name = "fake"

    def fake_separate(wav_path):
        x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        stem = Path(wav_path).stem
        vp = obj._output_dir / f"{stem}_(Vocals).wav"
        ip = obj._output_dir / f"{stem}_(Instrumental).wav"
        sf.write(str(vp), x, sr, subtype="FLOAT")
        sf.write(str(ip), np.zeros_like(x), sr, subtype="FLOAT")
        return vp, ip

    obj.separate = fake_separate
    return obj


def test_chunked_mss_crossfade_and_progress(tmp_path):
    inp = tmp_path / "in.wav"
    expected = _make_input(inp, 10.0)
    sep = _fake_separator(tmp_path)
    voc = tmp_path / "vocals.wav"
    inst = tmp_path / "inst.wav"
    fracs, processed = [], []

    def prog(f, p, rtf):
        fracs.append(f)
        processed.append(p)

    meta = sep.separate_chunked(inp, voc, inst, chunk_seconds=4.0,
                                overlap_seconds=1.0, progress=prog)
    v, sr = sf.read(str(voc), dtype="float32", always_2d=True)
    i, _ = sf.read(str(inst), dtype="float32", always_2d=True)
    assert sr == SR
    assert v.shape[0] == len(expected), (v.shape[0], len(expected))
    assert i.shape[0] == len(expected)
    assert np.max(np.abs(i)) == 0.0
    # 假分离器把原音作为 vocals：重叠淡入淡出后应还原原信号
    err = float(np.max(np.abs(v[:, 0] - expected)))
    assert err < 1e-4, err
    assert fracs and fracs[-1] == pytest.approx(1.0)
    assert processed[-1] == pytest.approx(10.0, abs=0.01)
    assert meta["chunks"] >= 3
    assert meta["rtf"] > 0


def test_chunked_mss_cancel_cleans_outputs(tmp_path):
    inp = tmp_path / "in.wav"
    _make_input(inp, 10.0)
    sep = _fake_separator(tmp_path)
    voc = tmp_path / "vocals.wav"
    inst = tmp_path / "inst.wav"
    calls = {"n": 0}

    def prog(f, p, rtf):
        calls["n"] += 1

    def cancel():
        return calls["n"] >= 1

    with pytest.raises(mss.MssCancelled):
        sep.separate_chunked(inp, voc, inst, chunk_seconds=4.0,
                             overlap_seconds=1.0, progress=prog, cancel=cancel)
    assert not voc.exists() and not inst.exists()
    assert not (tmp_path / "mss_out" / "_mss_chunks").exists() or \
        not any((tmp_path / "mss_out" / "_mss_chunks").iterdir())


def test_plan_for_duration_long_audio(monkeypatch):
    monkeypatch.setattr(mss.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=6 * 2**30))
    monkeypatch.setattr(mss.config, "get_device",
                        lambda: SimpleNamespace(has_nvidia=True, gpu_vram_mb=8188,
                                                gpu_name="RTX 4060"))
    monkeypatch.setattr(mss, "load_benchmark", lambda: {})
    p = mss.plan_for_duration(6109.0)
    assert p["chunk_seconds"] > 0
    assert 0 < p["overlap_seconds"] < p["chunk_seconds"]
    assert p["chunks"] >= 2
    assert "分块" in p["reason"]
    short = mss.plan_for_duration(120.0)
    assert short["chunk_seconds"] == 0
    assert "整段" in short["reason"]


def test_plan_for_duration_low_memory(monkeypatch):
    monkeypatch.setattr(mss.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1.4 * 2**30))
    monkeypatch.setattr(mss.config, "get_device",
                        lambda: SimpleNamespace(has_nvidia=True, gpu_vram_mb=4096,
                                                gpu_name="GTX"))
    monkeypatch.setattr(mss, "load_benchmark", lambda: {})
    p = mss.plan_for_duration(3600.0)
    assert p["chunk_seconds"] <= 60
    assert p["chunks"] > 1


def test_stage_weights_for_long_video():
    from app.jobs import _weights_for_duration
    w = _weights_for_duration(6109.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["mss"] > 0.5
    short = _weights_for_duration(120.0)
    assert abs(sum(short.values()) - 1.0) < 1e-9
    assert short["mss"] < 0.4


def test_precision_kwargs():
    assert mss.precision_kwargs("auto", "cuda") == {"use_native_fp16": True}
    assert mss.precision_kwargs("fp16", "cuda") == {"use_native_fp16": True}
    assert mss.precision_kwargs("autocast", "cuda") == {"use_autocast": True}
    assert mss.precision_kwargs("fp32", "cuda") == {}
    assert mss.precision_kwargs("auto", "cpu") == {}
    assert mss.precision_kwargs("autocast", "cpu") == {}

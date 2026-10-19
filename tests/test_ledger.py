"""账本 Σ 恒等 / 时长一致 / 静音契约测试。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import audio_io, ledger  # noqa: E402

SR = 44100


@pytest.fixture()
def trio(tmp_path):
    rng = np.random.default_rng(0)
    n = SR * 6
    t = np.arange(n) / SR
    master = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * rng.normal(0, 1, n)).astype("float32")
    vocals = (0.25 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    inst = master - vocals  # 保证 inst = master - vocals（残差恒 0 的友好设定）
    m = tmp_path / "master.wav"
    sf.write(m, np.stack([master, master], 1), SR, subtype="FLOAT")
    v = tmp_path / "vocals.wav"
    sf.write(v, np.stack([vocals, vocals], 1), SR, subtype="FLOAT")
    i = tmp_path / "inst.wav"
    sf.write(i, np.stack([inst, inst], 1), SR, subtype="FLOAT")
    return tmp_path, m, v, i, n


def test_sum_identity_speech_only(trio):
    tmp, m, v, i, n = trio
    items = [{"start": 0.5, "end": 2.0, "speaker": 0, "kind": "speech"},
             {"start": 2.0, "end": 4.0, "speaker": 1, "kind": "speech"}]
    gains = ledger.build_ctrl_gains(items, [0, 1], n / SR)
    led = ledger.write_tracks(m, v, i, tmp / "out", gains, {0: "A", 1: "B"})
    assert led.n_samples == n
    ver = ledger.verify_sum([tmp / "out" / t["file"] for t in led.tracks], m, SR)
    assert ver["pass"], ver
    assert ver["verify_sum_max_abs_err"] < 1e-5
    # 静音契约：0~0.5s 无人说话 → speaker 轨精确静音
    a, _ = audio_io.load_audio(tmp / "out" / "speaker_01.wav")
    assert np.max(np.abs(a[: int(0.4 * SR)])) == 0.0


def test_sum_identity_with_overlap_masks(trio):
    tmp, m, v, i, n = trio
    r = int(1.0 * 16000)
    mask_a = np.linspace(0.9, 0.1, r, dtype="float32")
    mask_b = 1.0 - mask_a
    items = [{"start": 0.0, "end": 1.0, "speaker": 0, "kind": "speech"},
             {"start": 1.0, "end": 2.0, "kind": "overlap", "speaker": 0,
              "speakers": [0, 1], "masks": {0: mask_a, 1: mask_b}},
             {"start": 2.0, "end": 3.0, "speaker": 1, "kind": "speech"}]
    gains = ledger.build_ctrl_gains(items, [0, 1], n / SR)
    led = ledger.write_tracks(m, v, i, tmp / "out", gains, {0: "A", 1: "B"})
    ver = ledger.verify_sum([tmp / "out" / t["file"] for t in led.tracks], m, SR)
    assert ver["pass"], ver


def test_resample_48k_keeps_identity(trio):
    tmp, m, v, i, n = trio
    items = [{"start": 0.2, "end": 1.8, "speaker": 0, "kind": "speech"}]
    gains = ledger.build_ctrl_gains(items, [0], n / SR)
    led = ledger.write_tracks(m, v, i, tmp / "out48", gains, {0: "A"}, out_sr=48000)
    assert led.sr_out == 48000
    ver = ledger.verify_sum([tmp / "out48" / t["file"] for t in led.tracks],
                            m, 48000)
    # 48k 轨道与 44.1k master 的比对需重采样 master 后再比；verify_sum 内部按文件 sr 读，
    # 此处改为显式：Σ(48k 轨) vs resample(master)
    acc = None
    import soundfile as sf2
    for wf in [tmp / "out48" / t["file"] for t in led.tracks]:
        x, _ = audio_io.load_audio(wf)
        acc = x if acc is None else acc + x
    mst, _ = audio_io.load_audio(m)
    mst48 = audio_io.resample(mst, SR, 48000)
    k = min(acc.shape[0], mst48.shape[0])
    err = float(np.max(np.abs(acc[:k] - mst48[:k])))
    assert err < 2e-3, err  # 重采样边缘容差

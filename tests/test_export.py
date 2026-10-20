"""导出功能测试：片段检测、纯音频导出、视频剪切拼接、任务队列。"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import audio_io, export_media  # noqa: E402

SR = 44100
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _tone_bursts(path: Path, dur: float = 10.0, freq: float = 880.0) -> None:
    """1-2s、2.1-3s（间隔 0.1 应合并）、4-4.2s（过短应丢弃）、6-8s 的响声。"""
    n = int(dur * SR)
    x = np.zeros(n, dtype="float32")
    t = np.arange(n) / SR
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")
    for s, e in [(1.0, 2.0), (2.1, 3.0), (4.0, 4.2), (6.0, 8.0)]:
        i0, i1 = int(s * SR), int(e * SR)
        x[i0:i1] = tone[i0:i1]
    sf.write(str(path), np.stack([x, x], axis=1), SR, subtype="FLOAT")


@pytest.fixture()
def burst_wav(tmp_path):
    p = tmp_path / "speaker_01.wav"
    _tone_bursts(p)
    return p


# ---------------------------------------------------------------- 片段检测

def test_plan_intervals_merge_pad_and_min_clip(burst_wav):
    res = export_media.plan_intervals(burst_wav, {
        "threshold_db": -45, "rel_db": -30, "merge_gap": 0.35,
        "min_clip": 0.6, "pad_before": 0.15, "pad_after": 0.25,
    })
    iv = res["intervals"]
    assert len(iv) == 2, iv
    # 1-3s 合并后 0.85-3.25；6-8s → 5.85-8.25
    assert abs(iv[0][0] - 0.85) < 0.08 and abs(iv[0][1] - 3.25) < 0.08
    assert abs(iv[1][0] - 5.85) < 0.08 and abs(iv[1][1] - 8.25) < 0.08
    assert res["total"] == pytest.approx(sum(e - s for s, e in iv), abs=0.02)
    assert 0 < res["coverage"] < 1


def test_plan_intervals_silent_track(tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(str(p), np.zeros((SR * 2, 2), dtype="float32"), SR, subtype="FLOAT")
    res = export_media.plan_intervals(p)
    assert res["intervals"] == []


def test_validate_intervals_merges_and_clamps():
    iv = export_media.validate_intervals([[0.5, 2.0], [1.5, 3.0], [9.5, 99]], 10.0)
    assert iv == [[0.5, 3.0], [9.5, 10.0]]
    with pytest.raises(ValueError):
        export_media.validate_intervals([[5, 4]], 10)
    with pytest.raises(ValueError):
        export_media.validate_intervals([[11, 12]], 10)
    with pytest.raises(ValueError):
        export_media.validate_intervals([], 10)


# ---------------------------------------------------------------- 纯音频

def test_export_audio_clips_wav(burst_wav, tmp_path):
    iv = [[0.85, 3.25], [5.85, 8.25]]
    out = tmp_path / "clips.wav"
    meta = export_media.export_audio(burst_wav, iv, out, fmt="wav", scope="clips")
    assert out.exists()
    info = sf.info(str(out))
    assert abs(info.frames / SR - sum(e - s for s, e in iv)) < 0.05
    x, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert float(np.sqrt((x ** 2).mean())) > 0.1
    assert meta["clips"] == 2


def test_export_audio_full_wav(burst_wav, tmp_path):
    out = tmp_path / "full.wav"
    meta = export_media.export_audio(burst_wav, [], out, fmt="wav", scope="full")
    assert meta["scope"] == "full"
    assert abs(meta["duration"] - 10.0) < 0.05
    assert out.stat().st_size == burst_wav.stat().st_size


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_export_audio_mp3(burst_wav, tmp_path):
    out = tmp_path / "clips.mp3"
    meta = export_media.export_audio(burst_wav, [[1.0, 2.0]], out, fmt="mp3", scope="clips")
    assert out.exists() and out.stat().st_size > 1000
    assert abs(meta["duration"] - 1.0) < 0.3


# ---------------------------------------------------------------- 视频

def _make_video(path: Path, dur: float = 8.0, freq: float = 440.0) -> None:
    subprocess.run([
        audio_io.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=44100:duration={dur}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)


def _extract_audio(media: Path, wav: Path) -> np.ndarray:
    subprocess.run([audio_io.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   check=True, capture_output=True)
    x, _ = sf.read(str(wav), dtype="float32")
    return x


def _peak_freq(x: np.ndarray, sr: int = 16000, lo: float = 0.2, hi: float = 0.8) -> float:
    seg = x[int(len(x) * lo):int(len(x) * hi)]
    if len(seg) < 1024:
        seg = x
    win = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1 / sr)
    return float(freqs[int(np.argmax(spec))])


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_export_video_clean_audio(burst_wav, tmp_path):
    src = tmp_path / "src.mp4"
    _make_video(src, dur=8.0, freq=440.0)
    out = tmp_path / "out_clean.mp4"
    iv = [[1.0, 2.5], [4.0, 6.0]]
    meta = export_media.export_video(src, burst_wav, iv, out, tmp_path / "work",
                                     audio_mode="clean")
    assert out.exists()
    info = audio_io.probe_media(str(out))
    assert info.has_video and info.has_audio
    assert abs(info.duration - sum(e - s for s, e in iv)) < 1.0
    x = _extract_audio(out, tmp_path / "a_clean.wav")
    assert float(np.sqrt((x ** 2).mean())) > 0.05
    peak = _peak_freq(x)
    assert 700 < peak < 1100, f"干净音轨应为 880Hz 左右，实际 {peak}"


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_export_video_original_audio(burst_wav, tmp_path):
    src = tmp_path / "src.mp4"
    _make_video(src, dur=8.0, freq=440.0)
    out = tmp_path / "out_orig.mp4"
    iv = [[1.0, 2.5], [4.0, 6.0]]
    export_media.export_video(src, burst_wav, iv, out, tmp_path / "work",
                              audio_mode="original")
    x = _extract_audio(out, tmp_path / "a_orig.wav")
    peak = _peak_freq(x)
    assert 350 < peak < 520, f"原声轨应为 440Hz 左右，实际 {peak}"


# ---------------------------------------------------------------- 队列

class _FakeJobs:
    def __init__(self):
        self._lock = __import__("threading").RLock()

    def has_active(self):
        return False

    def lock_for(self, jid):
        return self._lock


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_export_runner_audio_task(tmp_path, burst_wav):
    from app.exports import ExportRunner
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    shutil.copyfile(burst_wav, job_dir / "speaker_01.wav")
    (job_dir / "timeline.json").write_text(json.dumps({
        "duration_s": 10.0,
        "tracks": [{"key": "speaker_01", "label": "说话人 1",
                    "file": "speaker_01.wav", "kind": "speaker"}],
        "speakers": [], "segments": [],
    }), encoding="utf-8")
    job = SimpleNamespace(id="job1", name="测试任务", job_dir=str(job_dir),
                          input_path=str(burst_wav))
    runner = ExportRunner(_FakeJobs())
    task = runner.create(job, {
        "track": "speaker_01", "output_kind": "audio", "audio_scope": "clips",
        "audio_format": "wav", "intervals": [[1.0, 2.0]],
        "threshold_db": -45, "rel_db": -30, "merge_gap": 0.35,
        "min_clip": 0.6, "pad_before": 0.15, "pad_after": 0.25,
    })
    deadline = time.time() + 30
    while time.time() < deadline:
        t = runner.get(task.id)
        if t.status in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    t = runner.get(task.id)
    assert t.status == "done", t.error
    out = job_dir / "exports" / task.id / "export.wav"
    assert out.exists() and out.stat().st_size > 1000
    assert t.result["url"].startswith(f"/outputs/job1/exports/{task.id}/")
    assert abs(t.result["duration"] - 1.0) < 0.2


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_export_runner_rejects_residual(tmp_path):
    from app.exports import ExportRunner
    job_dir = tmp_path / "job2"
    job_dir.mkdir()
    (job_dir / "timeline.json").write_text(json.dumps({
        "duration_s": 10.0,
        "tracks": [{"key": "residual", "label": "残差", "file": "residual.wav", "kind": "residual"}],
    }), encoding="utf-8")
    job = SimpleNamespace(id="job2", name="t", job_dir=str(job_dir),
                          input_path=str(job_dir / "x.mp4"))
    runner = ExportRunner(_FakeJobs())
    with pytest.raises(ValueError):
        runner.create(job, {"track": "residual", "output_kind": "audio"})


def test_plan_audio_input_rejects_video_but_allows_audio(tmp_path, burst_wav):
    """纯音频输入：视频导出明确报错，音频导出正常。"""
    from app.exports import plan_for_job
    job_dir = tmp_path / "job3"
    job_dir.mkdir()
    shutil.copyfile(burst_wav, job_dir / "speaker_01.wav")
    (job_dir / "timeline.json").write_text(json.dumps({
        "duration_s": 10.0,
        "tracks": [{"key": "speaker_01", "label": "说话人 1",
                    "file": "speaker_01.wav", "kind": "speaker"}],
    }), encoding="utf-8")
    audio_src = tmp_path / "only_audio.wav"
    shutil.copyfile(burst_wav, audio_src)
    with pytest.raises(ValueError, match="没有视频流"):
        plan_for_job(job_dir, audio_src, {"track": "speaker_01", "output_kind": "video"})
    plan = plan_for_job(job_dir, audio_src, {"track": "speaker_01", "output_kind": "audio"})
    assert len(plan["intervals"]) == 2
    assert plan["source"] is None

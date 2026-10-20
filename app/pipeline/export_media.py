"""导出媒体：有声音片段检测、纯音频导出、按片段剪切视频并拼接。

设计要点（与分离流水线解耦，只读任务目录）：
- 片段检测直接分析已渲染的音轨 WAV（能量门限），而不是 timeline 段——
  人工改派、重叠掩码后的真实内容都能反映出来。
- 视频剪切必须重编码才能帧精确；所有片段使用同一套编码参数，最后 concat 流拼接。
- 纯音频导出用 soundfile 流式切片，片段接缝 5ms 淡入淡出防爆音。
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio_io

DEFAULTS: dict = {
    "threshold_db": -45.0,   # 绝对能量门限（dBFS）
    "rel_db": -30.0,         # 相对该轨 95 分位能量的门限（dB）
    "merge_gap": 0.35,       # 小于此间隔的相邻片段合并（秒）
    "min_clip": 0.6,         # 短于此的片段丢弃（秒）
    "pad_before": 0.15,      # 片段前留白（秒）
    "pad_after": 0.25,       # 片段后留白（秒）
}
MAX_CLIPS = 500
AUDIO_FORMATS = {"wav": "wav", "mp3": "mp3", "m4a": "m4a", "flac": "flac"}


class ExportCancelled(Exception):
    pass


# ---------------------------------------------------------------- 片段检测

def _frame_rms_db(track_wav: str | Path, frame: int = 882, hop: int = 441) -> tuple[np.ndarray, int, float]:
    """流式计算逐帧 RMS（dBFS）。返回 (帧能量数组, 采样率, 时长)。"""
    vals: list[float] = []
    with sf.SoundFile(str(track_wav)) as f:
        sr = f.samplerate
        ch = f.channels or 1
        total = f.frames
        block = max(frame, int(sr * 1.0))
        buf = np.zeros(0, dtype="float32")
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if data.shape[0] == 0:
                break
            mono = data.mean(axis=1)
            buf = np.concatenate([buf, mono])
            if len(buf) < frame:
                continue
            n_frames = 1 + (len(buf) - frame) // hop
            idx = np.arange(n_frames)[:, None] * hop + np.arange(frame)[None, :]
            fr = buf[idx]
            rms = np.sqrt(np.mean(fr.astype("float64") ** 2, axis=1) + 1e-12)
            vals.extend((20 * np.log10(rms)).tolist())
            # 保留最后一帧的尾部重叠
            keep_from = n_frames * hop
            buf = buf[keep_from:]
        duration = total / sr if sr else 0.0
    return np.asarray(vals, dtype="float32"), sr, duration


def plan_intervals(track_wav: str | Path, params: dict | None = None,
                   duration: float | None = None) -> dict:
    """检测音轨中“有声音”的片段，返回合并/留白后的区间与统计。"""
    p = {**DEFAULTS, **(params or {})}
    frame, hop = 882, 441  # 20ms / 10ms @44.1k
    rms_db, sr, dur = _frame_rms_db(track_wav, frame, hop)
    dur = float(duration or dur)
    if len(rms_db) == 0:
        return {"intervals": [], "total": 0.0, "coverage": 0.0, "duration": dur,
                "threshold_db": p["threshold_db"], "p95_db": None, "frames": 0}
    p95 = float(np.percentile(rms_db, 95))
    thr = max(float(p["threshold_db"]), p95 + float(p["rel_db"]))
    active = rms_db > thr
    segs: list[list[float]] = []
    in_seg = False
    start_i = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg, start_i = True, i
        elif not a and in_seg:
            segs.append([start_i * hop / sr, (i * hop + frame) / sr])
            in_seg = False
    if in_seg:
        segs.append([start_i * hop / sr, (len(active) * hop + frame) / sr])

    # 合并小间隔
    merged: list[list[float]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= float(p["merge_gap"]):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    # 最短片段过滤（先过滤再留白）
    merged = [[s, e] for s, e in merged if e - s >= float(p["min_clip"])]
    # 前后留白 + clamp
    padded: list[list[float]] = []
    for s, e in merged:
        ns = max(0.0, s - float(p["pad_before"]))
        ne = min(dur, e + float(p["pad_after"]))
        if ne - ns <= 0.05:
            continue
        if padded and ns <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], ne)
        else:
            padded.append([ns, ne])
    total = sum(e - s for s, e in padded)
    return {
        "intervals": [[round(s, 3), round(e, 3)] for s, e in padded],
        "total": round(total, 3),
        "coverage": round(total / dur, 4) if dur > 0 else 0.0,
        "duration": round(dur, 3),
        "threshold_db": round(thr, 1),
        "p95_db": round(p95, 1),
        "frames": int(len(rms_db)),
    }


def validate_intervals(intervals, duration: float) -> list[list[float]]:
    """校验用户确认后的区间：排序、合并重叠、clamp、数量上限。"""
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("片段列表为空")
    if len(intervals) > MAX_CLIPS:
        raise ValueError(f"片段过多（{len(intervals)} > {MAX_CLIPS}），请增大静音合并间隔")
    out: list[list[float]] = []
    for it in intervals:
        if not isinstance(it, (list, tuple)) or len(it) != 2:
            raise ValueError("片段格式应为 [start, end]")
        s, e = float(it[0]), float(it[1])
        if not (math.isfinite(s) and math.isfinite(e)):
            raise ValueError("片段含非法数值")
        if s < 0 or e <= s or s > duration + 0.01:
            raise ValueError(f"片段越界：[{s:.2f}, {e:.2f}]，视频时长 {duration:.2f}s")
        out.append([s, min(e, duration)])
    out.sort(key=lambda x: x[0])
    merged: list[list[float]] = []
    for s, e in out:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if not merged:
        raise ValueError("没有可导出的片段")
    return [[round(s, 3), round(e, 3)] for s, e in merged]


# ---------------------------------------------------------------- 音频导出

def _audio_codec_args(fmt: str) -> list[str]:
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", "192k"]
    if fmt == "m4a":
        return ["-c:a", "aac", "-b:a", "192k"]
    if fmt == "flac":
        return ["-c:a", "flac"]
    return ["-c:a", "pcm_s16le"]


def export_audio(track_wav: str | Path, intervals: list[list[float]], out_path: str | Path,
                 fmt: str = "wav", scope: str = "clips", progress=None,
                 cancel=None) -> dict:
    """导出纯音频：scope=clips 拼接有声音片段；scope=full 完整音轨。

    返回 {duration, clips, format, scope}。取消抛 ExportCancelled。
    """
    track_wav = Path(track_wav)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt if fmt in AUDIO_FORMATS else "wav"
    if fmt == "wav" and scope == "full":
        if cancel and cancel():
            raise ExportCancelled("已取消")
        shutil.copyfile(track_wav, out_path)
        info = sf.info(str(out_path))
        if progress:
            progress(1.0)
        return {"duration": round(info.frames / info.samplerate, 3), "clips": 0,
                "format": fmt, "scope": scope}

    expected = 0.0
    tmp_wav = out_path.with_name(out_path.stem + ".tmp.wav")
    if scope == "full":
        # 完整音轨先转成临时 WAV（或直接作为源转码）
        src_wav = track_wav
    else:
        _concat_clips(track_wav, intervals, tmp_wav, progress, cancel)
        src_wav = tmp_wav
        expected = sum(e - s for s, e in intervals)

    try:
        if fmt == "wav":
            if src_wav != out_path:
                shutil.move(str(src_wav), str(out_path))
        else:
            dur = expected or audio_io.probe_media(str(src_wav)).duration
            audio_io.run_ffmpeg_progress(
                ["-i", str(src_wav), "-vn", *_audio_codec_args(fmt), str(out_path)],
                dur, on_progress=(lambda f: progress(0.9 + 0.1 * f) if progress else None),
                cancel=cancel)
        if tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
        if progress:
            progress(1.0)
        info = audio_io.probe_media(str(out_path))
        if expected > 0 and abs(info.duration - expected) > max(0.15, expected * 0.01):
            raise RuntimeError(f"音频时长校验失败：{info.duration:.2f}s ≠ 片段和 {expected:.2f}s")
        return {"duration": round(info.duration, 3),
                "clips": 0 if scope == "full" else len(intervals),
                "format": fmt, "scope": scope}
    except Exception:
        tmp_wav.unlink(missing_ok=True)
        raise


def _concat_clips(track_wav: Path, intervals: list[list[float]], out_wav: Path,
                  progress=None, cancel=None) -> None:
    """把片段从音轨 WAV 中切出并顺序写入（接缝 5ms 淡入淡出）。"""
    with sf.SoundFile(str(track_wav)) as src:
        sr, ch = src.samplerate, src.channels
        total_n = src.frames
        fade = max(2, int(0.005 * sr))
        with sf.SoundFile(str(out_wav), mode="w", samplerate=sr, channels=ch,
                          subtype="FLOAT") as dst:
            for i, (s, e) in enumerate(intervals):
                if cancel and cancel():
                    raise ExportCancelled("已取消")
                i0 = max(0, min(int(round(s * sr)), total_n))
                i1 = max(i0, min(int(round(e * sr)), total_n))
                n = i1 - i0
                if n <= 0:
                    continue
                src.seek(i0)
                remaining = n
                first = True
                while remaining > 0:
                    chunk = src.read(min(int(sr), remaining), dtype="float32", always_2d=True)
                    if chunk.shape[0] == 0:
                        break
                    if first:
                        k = min(fade, chunk.shape[0])
                        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k, dtype="float32") / k)
                        chunk[:k] *= ramp[:, None]
                        first = False
                    if chunk.shape[0] < remaining:
                        dst.write(chunk)
                        remaining -= chunk.shape[0]
                        continue
                    # 最后一块：淡出
                    k = min(fade, chunk.shape[0])
                    ramp = 0.5 + 0.5 * np.cos(np.pi * np.arange(k, dtype="float32") / k)
                    chunk[-k:] *= ramp[:, None]
                    dst.write(chunk)
                    remaining = 0
                if progress:
                    progress((i + 1) / max(len(intervals), 1) * 0.9)
    if progress:
        progress(0.9)


# ---------------------------------------------------------------- 视频导出

_NVENC: dict = {"checked": False, "ok": False}


def pick_video_encoder() -> tuple[list[str], str]:
    """自动探测 NVENC（真实试编 0.1s），失败回退 libx264。"""
    if not _NVENC["checked"]:
        _NVENC["checked"] = True
        try:
            p = subprocess.run(
                [audio_io.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "color=black:s=256x144:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _NVENC["ok"] = p.returncode == 0
        except Exception:
            _NVENC["ok"] = False
    if _NVENC["ok"]:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"], "h264_nvenc"
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"], "libx264"


def export_video(source_path: str | Path, track_wav: str | Path, intervals: list[list[float]],
                 out_path: str | Path, work_dir: str | Path, audio_mode: str = "clean",
                 progress=None, cancel=None) -> dict:
    """按区间剪出视频片段，替换/保留音频，拼接为成片。"""
    source_path = Path(source_path)
    out_path = Path(out_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(f"原始视频不存在：{source_path}")
    info = audio_io.probe_media(str(source_path))
    if not info.has_video:
        raise ValueError("该输入没有视频流，无法导出视频（可改用“仅音频”导出）")
    fps = info.fps if info.fps and info.fps > 1 else 30.0
    enc_args, enc_name = pick_video_encoder()
    expected = sum(e - s for s, e in intervals)
    n = len(intervals)
    seg_files: list[Path] = []

    for i, (s, e) in enumerate(intervals):
        if cancel and cancel():
            raise ExportCancelled("已取消")
        dur = max(0.05, e - s)
        seg = work_dir / f"seg_{i:04d}.mp4"
        args = ["-ss", f"{s:.3f}", "-i", str(source_path)]
        maps = ["-map", "0:v:0"]
        if audio_mode == "clean":
            args += ["-ss", f"{s:.3f}", "-i", str(track_wav)]
            maps += ["-map", "1:a:0"]
        else:
            if info.has_audio:
                maps += ["-map", "0:a:0"]
            else:
                args += ["-f", "lavfi", "-t", f"{dur:.3f}",
                         "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
                maps += ["-map", "1:a:0"]
        args += ["-t", f"{dur:.3f}", *maps,
                 "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-r", f"{fps:.3f}", "-fps_mode", "cfr",
                 *enc_args, "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "-shortest", "-movflags", "+faststart", str(seg)]
        audio_io.run_ffmpeg_progress(
            args, dur,
            on_progress=(lambda f, i=i: progress(0.05 + 0.75 * (i + f) / n) if progress else None),
            cancel=cancel)
        seg_files.append(seg)

    if not seg_files:
        raise ValueError("没有可导出的片段")
    if progress:
        progress(0.82)
    list_file = work_dir / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg in seg_files:
            f.write("file '" + str(seg).replace("'", "'\\''") + "'\n")
    audio_io.run_ffmpeg_progress(
        ["-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", "-movflags", "+faststart", str(out_path)],
        expected,
        on_progress=(lambda f: progress(0.82 + 0.14 * f) if progress else None),
        cancel=cancel)
    if progress:
        progress(0.97)
    out_info = audio_io.probe_media(str(out_path))
    if not (out_info.has_video and out_info.has_audio):
        raise RuntimeError("成片校验失败：缺少视频或音频流")
    if abs(out_info.duration - expected) > max(1.0, expected * 0.02):
        raise RuntimeError(f"成片时长校验失败：{out_info.duration:.2f}s ≠ 片段和 {expected:.2f}s")
    if progress:
        progress(1.0)
    return {"duration": round(out_info.duration, 3), "clips": n,
            "encoder": enc_name, "audio_mode": audio_mode,
            "width": out_info.width, "height": out_info.height, "fps": round(out_info.fps, 3)}

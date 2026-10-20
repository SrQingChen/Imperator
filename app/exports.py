"""导出任务：按音轨切出视频 / 导出纯音频（独立单 worker 队列）。"""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from . import config, events
from .pipeline import audio_io, export_media

STAGE_NAMES = {"prepare": "准备", "plan": "分析片段", "render": "渲染",
               "finalize": "拼接 / 转码", "verify": "校验"}


@dataclass
class ExportTask:
    id: str
    job_id: str
    job_name: str
    job_dir: str
    source_path: str
    track_key: str
    track_label: str
    output_kind: str          # video | audio
    audio_mode: str = "clean"
    audio_scope: str = "clips"
    audio_format: str = "wav"
    params: dict = field(default_factory=dict)
    intervals: list | None = None
    status: str = "queued"    # queued/running/done/error/cancelled
    stage: str = ""
    progress: float = 0.0
    note: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    _cancel: bool = False

    @property
    def export_dir(self) -> Path:
        return Path(self.job_dir) / "exports" / self.id

    def to_dict(self, full: bool = False) -> dict:
        d = {
            "id": self.id, "job_id": self.job_id, "job_name": self.job_name,
            "track_key": self.track_key, "track_label": self.track_label,
            "output_kind": self.output_kind, "audio_mode": self.audio_mode,
            "audio_scope": self.audio_scope, "audio_format": self.audio_format,
            "params": self.params, "status": self.status, "stage": self.stage,
            "stage_name": STAGE_NAMES.get(self.stage, self.stage),
            "progress": self.progress, "note": self.note, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if full and self.result is not None:
            d["result"] = self.result
        return d


def _num(payload: dict, key: str, lo: float, hi: float) -> float:
    raw = payload.get(key, export_media.DEFAULTS[key])
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 应为数字")
    if not math.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(f"{key} 应在 {lo}~{hi} 之间")
    return v


def parse_params(payload: dict) -> dict:
    return {
        "threshold_db": _num(payload, "threshold_db", -90.0, 0.0),
        "rel_db": _num(payload, "rel_db", -60.0, 0.0),
        "merge_gap": _num(payload, "merge_gap", 0.0, 10.0),
        "min_clip": _num(payload, "min_clip", 0.1, 60.0),
        "pad_before": _num(payload, "pad_before", 0.0, 10.0),
        "pad_after": _num(payload, "pad_after", 0.0, 10.0),
    }


def _read_timeline(job_dir: Path) -> dict:
    p = job_dir / "timeline.json"
    if not p.exists():
        raise FileNotFoundError("任务没有 timeline.json（可能未完成或已被删除）")
    return json.loads(p.read_text("utf-8"))


def _find_track(tl: dict, key: str) -> dict:
    for t in tl.get("tracks", []):
        if t.get("key") == key:
            if t.get("kind") == "residual":
                raise ValueError("残差轨是诊断用轨道，不能导出")
            return t
    raise ValueError(f"音轨不存在：{key}")


def _track_duration(track_wav: Path, fallback: float) -> float:
    try:
        with sf.SoundFile(str(track_wav)) as f:
            return f.frames / f.samplerate if f.samplerate else fallback
    except Exception:
        return fallback


def plan_for_job(job_dir: Path, source_path: Path, payload: dict) -> dict:
    """分析某条音轨的可导出片段（不创建任务）。"""
    tl = _read_timeline(job_dir)
    key = str(payload.get("track", ""))
    if not key:
        raise ValueError("缺少 track")
    track = _find_track(tl, key)
    track_wav = job_dir / track["file"]
    if not track_wav.exists():
        raise FileNotFoundError(f"音轨文件缺失：{track_wav.name}")
    params = parse_params(payload)
    out_kind = str(payload.get("output_kind", "video"))
    cap = _track_duration(track_wav, float(tl.get("duration_s") or 0))
    src_info = None
    if out_kind == "video":
        if not source_path.exists():
            raise FileNotFoundError(f"原始视频不存在：{source_path}")
        src_info = audio_io.probe_media(str(source_path))
        if not src_info.has_video:
            raise ValueError("该输入没有视频流，无法导出视频（可改用“仅音频”导出）")
        if src_info.duration > 0:
            cap = min(cap, src_info.duration)
    res = export_media.plan_intervals(track_wav, params, duration=cap)
    # bgm 等长音轨：覆盖率高时直接整段
    if track.get("kind") == "bgm" and res["coverage"] > 0.9:
        res["intervals"] = [[0.0, round(cap, 3)]]
        res["total"] = round(cap, 3)
        res["coverage"] = 1.0
    res.update({
        "track": key, "track_label": track.get("label", key),
        "track_duration": round(cap, 3), "source": src_info.to_dict() if src_info else None,
        "output_kind": out_kind,
    })
    return res


class ExportRunner:
    def __init__(self, job_runner):
        self.jobs = job_runner
        self._tasks: dict[str, ExportTask] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="export-runner", daemon=True)
        self._thread.start()

    # ------------------------------------------------ API
    def create(self, job, payload: dict) -> ExportTask:
        job_dir = Path(job.job_dir)
        with self.jobs.lock_for(job.id):
            tl = _read_timeline(job_dir)  # 与人工修正的 timeline 重写互斥
        key = str(payload.get("track", ""))
        if not key:
            raise ValueError("缺少 track")
        track = _find_track(tl, key)
        track_wav = job_dir / track["file"]
        if not track_wav.exists():
            raise FileNotFoundError(f"音轨文件缺失：{track_wav.name}")
        kind = str(payload.get("output_kind", "video"))
        if kind not in ("video", "audio"):
            raise ValueError("output_kind 应为 video 或 audio")
        mode = str(payload.get("audio_mode", "clean"))
        if mode not in ("clean", "original"):
            raise ValueError("audio_mode 应为 clean 或 original")
        scope = str(payload.get("audio_scope", "clips"))
        if scope not in ("clips", "full"):
            raise ValueError("audio_scope 应为 clips 或 full")
        fmt = str(payload.get("audio_format", "wav")).lower()
        if fmt not in export_media.AUDIO_FORMATS:
            raise ValueError("audio_format 应为 wav / mp3 / m4a / flac")
        params = parse_params(payload)
        source = Path(job.input_path)
        if kind == "video":
            if not source.exists():
                raise FileNotFoundError(f"原始视频不存在：{source}")
            info = audio_io.probe_media(str(source))
            if not info.has_video:
                raise ValueError("该输入没有视频流，无法导出视频（可改用“仅音频”导出）")
        intervals = payload.get("intervals")
        if not (kind == "audio" and scope == "full"):
            if not intervals:
                raise ValueError("缺少片段列表（请先分析并确认片段）")
            cap = _track_duration(track_wav, float(tl.get("duration_s") or 0))
            if kind == "video":
                si = audio_io.probe_media(str(source))
                cap = min(cap, si.duration) if si.duration > 0 else cap
            intervals = export_media.validate_intervals(intervals, cap)
        else:
            intervals = None
        task = ExportTask(
            id=time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
            job_id=job.id, job_name=job.name, job_dir=str(job_dir),
            source_path=str(source), track_key=key, track_label=track.get("label", key),
            output_kind=kind, audio_mode=mode, audio_scope=scope, audio_format=fmt,
            params=params, intervals=intervals,
        )
        with self._lock:
            self._tasks[task.id] = task
            self._order.append(task.id)
        self._wake.set()
        events.info(f"导出任务已排队：{task.track_label} · {kind}（{task.id}）", scope="export")
        return task

    def get(self, eid: str) -> ExportTask | None:
        with self._lock:
            return self._tasks.get(eid)

    def list(self, job_id: str | None = None) -> list[ExportTask]:
        with self._lock:
            items = [self._tasks[i] for i in self._order]
        if job_id:
            items = [t for t in items if t.job_id == job_id]
        return items[::-1]

    def cancel(self, eid: str) -> bool:
        with self._lock:
            t = self._tasks.get(eid)
            if t and t.status in ("queued", "running"):
                t._cancel = True
                t.note = "取消请求已发出（当前 ffmpeg 片段会被终止）"
                events.warn(f"导出任务 {eid} 请求取消", scope="export")
                return True
        return False

    def delete(self, eid: str) -> bool:
        with self._lock:
            t = self._tasks.get(eid)
            if not t or t.status == "running":
                return False
            del self._tasks[eid]
            self._order.remove(eid)
        shutil.rmtree(t.export_dir, ignore_errors=True)
        return True

    def has_active_for(self, job_id: str) -> bool:
        with self._lock:
            return any(t.job_id == job_id and t.status in ("queued", "running")
                       for t in self._tasks.values())

    def delete_for_job(self, job_id: str) -> None:
        with self._lock:
            ids = [t.id for t in self._tasks.values()
                   if t.job_id == job_id and t.status not in ("running",)]
            for eid in ids:
                t = self._tasks.pop(eid, None)
                if t:
                    try:
                        self._order.remove(eid)
                    except ValueError:
                        pass
        for eid in ids:
            shutil.rmtree(Path(self._tasks_dir_hint(job_id)) / eid, ignore_errors=True)

    @staticmethod
    def _tasks_dir_hint(job_id: str) -> Path:
        return config.OUTPUTS_DIR / job_id / "exports"

    # ------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            task = self._next_queued()
            if task is None:
                continue
            while self.jobs.has_active() and not task._cancel:
                task.note = "等待分离任务结束（避免抢占 CPU）"
                time.sleep(2.0)
            if task._cancel:
                self._finish_cancelled(task)
                continue
            self._run(task)

    def _next_queued(self) -> ExportTask | None:
        with self._lock:
            for eid in self._order:
                t = self._tasks[eid]
                if t.status == "queued":
                    return t
        return None

    def _set_progress(self, task: ExportTask, p: float, note: str = "") -> None:
        task.progress = max(0.0, min(1.0, p))
        if task.progress < 0.05:
            task.stage = "prepare"
        elif task.progress < 0.08:
            task.stage = "plan"
        elif task.progress < 0.95:
            task.stage = "render"
        elif task.progress < 0.99:
            task.stage = "finalize"
        else:
            task.stage = "verify"
        if note:
            task.note = note

    def _run(self, task: ExportTask) -> None:
        task.status = "running"
        task.started_at = time.time()
        self._set_progress(task, 0.01, "准备导出")
        events.stage(f"开始导出：{task.job_name} · {task.track_label} · "
                     f"{'视频' if task.output_kind == 'video' else '音频'}", scope=task.id)
        work = task.export_dir / "_work"
        out_ext = "mp4" if task.output_kind == "video" else task.audio_format
        out_path = task.export_dir / f"export.{out_ext}"
        try:
            task.export_dir.mkdir(parents=True, exist_ok=True)
            job_dir = Path(task.job_dir)
            if not job_dir.exists():
                raise FileNotFoundError("任务目录已被删除")
            with self.jobs.lock_for(task.job_id):
                tl = _read_timeline(job_dir)
            track = _find_track(tl, task.track_key)
            # 在任务锁内复制音轨快照，避免与人工修正重渲染并发读写
            src_wav = job_dir / track["file"]
            if not src_wav.exists():
                raise FileNotFoundError(f"音轨文件缺失：{src_wav.name}")
            work.mkdir(parents=True, exist_ok=True)
            snap = work / "track.wav"
            with self.jobs.lock_for(task.job_id):
                shutil.copyfile(src_wav, snap)
            self._set_progress(task, 0.04, "分析有声音片段")
            cap = _track_duration(snap, float(tl.get("duration_s") or 0))
            intervals = task.intervals
            source_info = None
            if task.output_kind == "video":
                if not Path(task.source_path).exists():
                    raise FileNotFoundError(f"原始视频不存在：{task.source_path}")
                source_info = audio_io.probe_media(task.source_path)
                if not source_info.has_video:
                    raise ValueError("该输入没有视频流，无法导出视频")
                if source_info.duration > 0:
                    cap = min(cap, source_info.duration)
                if task.audio_mode == "clean" and not snap.exists():
                    raise FileNotFoundError("分离音轨不存在，无法用干净音频导出")
            full_audio = task.output_kind == "audio" and task.audio_scope == "full"
            if full_audio:
                intervals = []
            elif intervals is None:
                res = export_media.plan_intervals(snap, task.params, duration=cap)
                intervals = res["intervals"]
                if not intervals:
                    raise ValueError("该音轨没有检测到有声音的片段（可降低能量阈值后重试）")
                task.intervals = intervals
            else:
                intervals = export_media.validate_intervals(intervals, cap)
                task.intervals = intervals
            self._set_progress(task, 0.08, f"{len(intervals)} 个片段，开始渲染")
            if task.output_kind == "video":
                meta = export_media.export_video(
                    task.source_path, snap, intervals, out_path, work,
                    audio_mode=task.audio_mode,
                    progress=lambda f: self._set_progress(task, 0.08 + 0.87 * f,
                                                          f"渲染片段 {int(f * 100)}%"),
                    cancel=lambda: task._cancel)
            else:
                meta = export_media.export_audio(
                    snap, intervals or [], out_path, fmt=task.audio_format,
                    scope=task.audio_scope,
                    progress=lambda f: self._set_progress(task, 0.08 + 0.87 * f,
                                                          f"写出音频 {int(f * 100)}%"),
                    cancel=lambda: task._cancel)
            self._set_progress(task, 0.97, "校验成片")
            size_mb = round(out_path.stat().st_size / 1048576, 2)
            url = f"/outputs/{task.job_id}/exports/{task.id}/{out_path.name}"
            task.result = {
                "file": out_path.name, "url": url, "size_mb": size_mb,
                "duration": meta.get("duration"), "clips": meta.get("clips"),
                "output_kind": task.output_kind, "audio_mode": task.audio_mode,
                "audio_scope": task.audio_scope, "audio_format": task.audio_format,
                "track_key": task.track_key, "track_label": task.track_label,
                "intervals": intervals,
                "media": {k: v for k, v in meta.items() if k not in ("intervals",)},
            }
            (task.export_dir / "meta.json").write_text(
                json.dumps({**task.to_dict(), "result": task.result}, ensure_ascii=False, indent=2),
                "utf-8")
            shutil.rmtree(work, ignore_errors=True)
            task.status = "done"
            task.progress = 1.0
            task.stage = "verify"
            task.note = f"完成：{meta.get('clips')} 段 · {meta.get('duration')}s"
            events.success(f"导出完成：{out_path.name}（{size_mb} MB）", scope=task.id)
        except (export_media.ExportCancelled, audio_io.ProcessCancelled):
            self._finish_cancelled(task)
        except Exception as e:
            task.status = "error"
            task.error = f"{type(e).__name__}: {e}"
            task.note = "失败"
            events.error(f"导出失败：{task.error}", scope=task.id)
            shutil.rmtree(work, ignore_errors=True)
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            task.finished_at = time.time()

    def _finish_cancelled(self, task: ExportTask) -> None:
        task.status = "cancelled"
        task.note = "已取消"
        task.finished_at = time.time()
        shutil.rmtree(task.export_dir / "_work", ignore_errors=True)
        try:
            for p in task.export_dir.glob("export.*"):
                p.unlink(missing_ok=True)
        except OSError:
            pass
        events.warn(f"导出已取消：{task.id}", scope=task.id)

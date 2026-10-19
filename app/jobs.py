"""任务系统：串行队列、阶段进度、取消、RTF、修正重渲染。"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config, events, settings as settings_mod
from .pipeline.engine import Cancelled, Engine

STAGES: list[tuple[str, str, float]] = [
    ("prepare", "准备", 2),
    ("extract", "抽取音频", 4),
    ("mss", "伴奏分离", 30),
    ("vad", "人声检测", 3),
    ("embed", "声纹分析", 15),
    ("cluster", "说话人聚类", 7),
    ("overlap", "重叠分离", 27),
    ("ledger", "轨道重建", 8),
    ("report", "质检与报告", 4),
]
_STAGE_KEYS = [s[0] for s in STAGES]


@dataclass
class Job:
    id: str
    name: str
    input_path: str
    settings: dict
    created_at: float = field(default_factory=time.time)
    status: str = "queued"            # queued/running/done/error/cancelled
    stage: str = ""
    stage_frac: float = 0.0
    note: str = ""
    error: str = ""
    progress: float = 0.0
    eta_s: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    stage_times: dict = field(default_factory=dict)
    result: dict | None = None
    _cancel: bool = False
    _stage_t0: float = 0.0

    @property
    def job_dir(self) -> Path:
        return config.OUTPUTS_DIR / self.id

    def to_dict(self, full: bool = False) -> dict:
        d = {"id": self.id, "name": self.name, "input_path": self.input_path,
             "status": self.status, "stage": self.stage, "stage_frac": self.stage_frac,
             "note": self.note, "error": self.error, "progress": self.progress,
             "eta_s": self.eta_s, "created_at": self.created_at,
             "started_at": self.started_at, "finished_at": self.finished_at,
             "stage_times": self.stage_times, "stages": [s[0] for s in STAGES],
             "stage_names": {s[0]: s[1] for s in STAGES}}
        if full and self.result:
            d["result"] = self.result
        return d


class JobRunner:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._corr_locks: dict[str, threading.RLock] = {}
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="job-runner", daemon=True)
        self._thread.start()

    # ------------------------------------------------ API 侧
    def create(self, input_path: str, name: str | None, overrides: dict | None) -> Job:
        cfg = settings_mod.load()
        if overrides:
            merged = {**cfg.to_dict(), **overrides}
            cfg = settings_mod.Settings.from_dict(merged)
        errs = cfg.validate()
        if errs:
            raise ValueError("参数非法：" + "; ".join(errs))
        job = Job(id=time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
                  name=name or Path(input_path).name, input_path=str(input_path),
                  settings=cfg.to_dict())
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._wake.set()
        events.info(f"任务已排队：{job.name}（{job.id}）", scope="jobs")
        return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order]

    def cancel(self, jid: str) -> bool:
        with self._lock:
            job = self._jobs.get(jid)
            if job and job.status in ("queued", "running"):
                job._cancel = True
                job.note = "取消请求已发出（将在最近的检查点生效）"
                events.warn(f"任务 {jid} 请求取消", scope="jobs")
                return True
        return False

    def delete(self, jid: str) -> bool:
        with self._lock:
            job = self._jobs.get(jid)
            if job and job.status not in ("running",):
                if job.status == "queued":
                    job._cancel = True  # 关闭与 _next_queued 的竞态窗口
                del self._jobs[jid]
                self._order.remove(jid)
        if job and job.status != "running":
            import shutil
            shutil.rmtree(job.job_dir, ignore_errors=True)
            # 清理本工具上传的输入副本（用户自选路径不动）
            try:
                inp = Path(job.input_path)
                if inp.exists() and config.WORKSPACE_DIR in inp.parents:
                    inp.unlink()
            except OSError:
                pass
            return True
        return False

    # ------------------------------------------------ 工作线程
    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            job = self._next_queued()
            if job is None:
                continue
            self._run_job(job)

    def _next_queued(self) -> Job | None:
        with self._lock:
            for jid in self._order:
                j = self._jobs[jid]
                if j.status == "queued":
                    return j
        return None

    def _progress_of(self, job: Job) -> float:
        weights = {k: w for k, _, w in STAGES}
        done = sum(weights[k] for k, t in job.stage_times.items() if t.get("done"))
        cur = weights.get(job.stage, 0) * job.stage_frac
        return min(0.999, (done + cur) / sum(weights.values()))

    def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        job._stage_t0 = time.time()
        cfg = settings_mod.Settings.from_dict(job.settings)
        events.stage(f"任务开始：{job.name}", scope=job.id)

        def progress(stage: str, frac: float, note: str = ""):
            if job.stage != stage:
                if job.stage and job.stage in job.stage_times:
                    job.stage_times[job.stage]["done"] = True
                    job.stage_times[job.stage]["secs"] = round(
                        job.stage_times[job.stage].get("secs", 0) + time.time() - job._stage_t0, 1)
                job.stage = stage
                job._stage_t0 = time.time()
                job.stage_times.setdefault(stage, {"secs": 0.0, "done": False})
            job.stage_frac = max(0.0, min(1.0, frac))
            if note:
                job.note = note
            job.progress = self._progress_of(job)
            elapsed = time.time() - job.started_at
            if job.progress > 0.02:
                job.eta_s = elapsed / job.progress - elapsed

        try:
            if not Path(job.input_path).exists():
                raise FileNotFoundError(f"输入文件不存在：{job.input_path}")
            job.stage = "prepare"
            job.stage_times["prepare"] = {"secs": 0.0, "done": True}
            result = self.engine.run(Path(job.input_path), job.job_dir, cfg,
                                     progress, lambda: job._cancel)
            job.result = result
            job.status = "done"
            job.progress = 1.0
            job.stage = "report"
            job.note = f"完成：{len(result['speakers'])} 个说话人，RTF {result['qc'].get('rtf')}"
            events.success(f"任务完成：{job.name} → {job.job_dir}", scope=job.id)
        except Cancelled:
            job.status = "cancelled"
            job.note = "已取消"
            events.warn(f"任务已取消：{job.name}", scope=job.id)
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.note = "失败"
            events.error(f"任务失败 {job.name}：{job.error}", scope=job.id)
            events.info(traceback.format_exc()[-1500:], scope=job.id)
        finally:
            job.finished_at = time.time()
            if job.stage and job.stage in job.stage_times:
                job.stage_times[job.stage]["secs"] = round(
                    job.stage_times[job.stage].get("secs", 0) + time.time() - job._stage_t0, 1)

    # ------------------------------------------------ 修正重渲染
    def apply_corrections(self, jid: str, moves: list[dict]) -> dict:
        job = self.get(jid)
        if job is None or job.status != "done" or not job.result:
            raise ValueError("任务不存在或未完成")
        tl_path = job.job_dir / "timeline.json"
        if not tl_path.exists():
            raise FileNotFoundError("timeline.json 缺失")
        with self._lock:
            lock = self._corr_locks.setdefault(jid, threading.RLock())
        with lock:  # 同一任务的修正重渲染互斥（并发写 WAV 会损坏轨道）
            def progress(stage, frac, note=""):
                pass
            tl = self.engine.rerender(job.job_dir, moves,
                                      settings_mod.Settings.from_dict(job.settings), progress)
            job.result.update({
                "tracks": tl["tracks"], "speakers": tl["speakers"],
                "segments": tl["segments"], "audit": tl["audit"], "qc": tl["qc"]})
        events.success(f"人工修正已应用（{len(moves)} 处改派，秒级重渲染）", scope=jid)
        return job.result

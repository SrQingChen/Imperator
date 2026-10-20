"""FastAPI 应用工厂：/api/* 路由 + 静态前端 + 输出目录挂载（Range 支持）。"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import (config, env_manager, events, exports, jobs, models_manager, monitor, network,
               settings as settings_mod, voiceprint)

_OK_FIELDS = {"tier", "merge_threshold", "outlier_mode", "outlier_ratio_percent",
              "output_format", "output_sr"}


def create_app() -> FastAPI:
    app = FastAPI(title="英白拉多 Imperator", version="1.0.0", docs_url=None, redoc_url=None)
    perf = monitor.PerfMonitor()
    engine = None
    runner: jobs.JobRunner | None = None
    exports_runner: exports.ExportRunner | None = None

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal engine, runner, exports_runner
        from .logging_setup import setup as setup_logging
        setup_logging()
        os.chdir(config.DATA_DIR)  # clearvoice 以 CWD 相对路径管理 checkpoints
        # 启动时串行完成 torch/CUDA 初始化，避免 PerfMonitor 与任务线程并发 import
        try:
            t_torch = time.time()
            torch = config.ensure_torch()
            cuda = bool(torch.cuda.is_available())
            events.info(f"PyTorch 初始化完成（{time.time()-t_torch:.2f}s，CUDA={cuda}"
                        + (f"，{torch.cuda.get_device_name(0)}" if cuda else "") + "）",
                        scope="perf")
        except Exception as e:
            events.error(f"PyTorch/CUDA 初始化失败：{type(e).__name__}: {e}", scope="perf")
        perf.start()
        from .pipeline.engine import Engine
        engine = Engine()
        runner = jobs.JobRunner(engine)
        exports_runner = exports.ExportRunner(runner)
        events.success(f"英白拉多 Imperator 已启动（数据目录 {config.DATA_DIR}）", scope="server")
        threading.Thread(target=network.get_plan, daemon=True).start()
        dev = config.get_device()
        events.info(f"设备：{dev.cpu} · {dev.cpu_cores} 线程 · RAM {dev.ram_total_mb//1024}GB · "
                    f"GPU {dev.gpu_name or '无'}"
                    + (f"（{dev.gpu_vram_mb//1024}GB）" if dev.gpu_name else ""), scope="server")
        for note in dev.notes:
            events.warn(note, scope="server")

        def _warm_models() -> None:
            """后台预加载 VAD/CAM++：funasr 首次 import 约 25s，放在启动期做，
            避免用户任务到声纹阶段时干等；任务若提前到达会在同一把引擎锁上等待。"""
            try:
                t0 = time.time()
                engine.ensure_vad()
                engine.ensure_embedder()
                events.info(f"VAD/CAM++ 预加载完成（{time.time()-t0:.1f}s）", scope="perf")
            except Exception as e:
                events.warn(f"VAD/CAM++ 预加载失败（任务时会重试）：{type(e).__name__}: {e}",
                            scope="perf")

        threading.Thread(target=_warm_models, name="model-warmup", daemon=True).start()

    # ------------------------------------------------ 系统
    @app.get("/api/health")
    def health():
        return {"ok": True, "ts": time.time()}

    @app.get("/api/system")
    def system():
        dev = config.get_device().to_dict()
        dev["torch_device"] = config.torch_device()
        dev["base_dir"] = str(config.BASE_DIR)
        dev["data_dir"] = str(config.DATA_DIR)
        return dev

    @app.get("/api/events")
    def get_events(since: int = 0, limit: int = 200):
        return {"events": events.since(since, limit)}

    @app.get("/api/monitor")
    def get_monitor():
        return perf.snapshot()

    # ------------------------------------------------ 设置
    @app.get("/api/settings")
    def get_settings():
        return settings_mod.load().to_dict()

    @app.post("/api/settings")
    def post_settings(body: dict):
        s = settings_mod.Settings.from_dict(body)
        errs = s.validate()
        if errs:
            raise HTTPException(400, "; ".join(errs))
        settings_mod.save(s)
        events.info("设置已保存", scope="settings")
        return {"ok": True, "settings": s.to_dict()}

    # ------------------------------------------------ 任务
    @app.post("/api/jobs")
    async def create_job(file: UploadFile | None = File(None),
                         path: str | None = Form(None),
                         tier: str | None = Form(None),
                         merge_threshold: float | None = Form(None),
                         outlier_mode: str | None = Form(None),
                         outlier_ratio_percent: float | None = Form(None),
                         output_format: str | None = Form(None),
                         output_sr: str | None = Form(None),
                         resume_from: str | None = Form(None)):
        overrides: dict = {}
        if tier:
            overrides["mss_tier"] = tier
        if merge_threshold is not None:
            overrides["merge_threshold"] = merge_threshold
        if outlier_mode:
            overrides["outlier_mode"] = outlier_mode
        if outlier_ratio_percent is not None:
            overrides["outlier_ratio_percent"] = outlier_ratio_percent
        if output_format:
            overrides["output_format"] = output_format
        if output_sr:
            try:
                overrides["output_sr"] = int(output_sr)
            except ValueError:
                overrides["output_sr"] = output_sr
        input_path: str | None = path
        if file is not None and file.filename:
            dest = config.WORKSPACE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            input_path = str(dest)
            events.info(f"已上传输入：{dest.name}", scope="jobs")
        if not input_path:
            raise HTTPException(400, "请上传文件或提供本地路径")
        p = Path(input_path)
        if not p.is_absolute():
            raise HTTPException(400, "请提供绝对路径")
        if not p.exists():
            raise HTTPException(400, f"文件不存在：{input_path}")
        resume_dir = None
        if resume_from:
            rd = Path(resume_from).resolve()
            try:
                rd.relative_to(config.OUTPUTS_DIR.resolve())
            except ValueError:
                raise HTTPException(400, "resume_from 必须位于任务输出目录内")
            for name in ("master.wav", "vocals_keep.wav", "inst_keep.wav"):
                if not (rd / name).exists():
                    raise HTTPException(400, f"续跑缺少中间文件：{name}")
            resume_dir = str(rd)
        try:
            job = runner.create(str(p), p.name, overrides, resume_from=resume_dir)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "job": job.to_dict()}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.to_dict() for j in reversed(runner.list_jobs())]}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        job = runner.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job.to_dict(full=True)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        if not runner.cancel(jid):
            raise HTTPException(400, "无法取消（不存在或不在运行）")
        return {"ok": True}

    @app.delete("/api/jobs/{jid}")
    def delete_job(jid: str):
        job = runner.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        if exports_runner.has_active_for(jid):
            raise HTTPException(409, "该任务还有导出在进行，请先取消或等待完成")
        exports_runner.delete_for_job(jid)
        if not runner.delete(jid):
            raise HTTPException(400, "无法删除（运行中或不存在）")
        return {"ok": True}

    @app.post("/api/jobs/{jid}/corrections")
    def corrections(jid: str, body: dict):
        moves = body.get("moves")
        if not isinstance(moves, list):
            raise HTTPException(400, "moves 应为列表")
        try:
            result = runner.apply_corrections(jid, moves)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "result": result}

    @app.get("/api/jobs/{jid}/timeline")
    def timeline(jid: str):
        p = config.OUTPUTS_DIR / jid / "timeline.json"
        if not p.exists():
            raise HTTPException(404, "无时间线")
        return JSONResponse(content=json.loads(p.read_text("utf-8")))

    @app.get("/api/jobs/{jid}/report")
    def report_html(jid: str):
        p = config.OUTPUTS_DIR / jid / "report.html"
        if not p.exists():
            raise HTTPException(404, "无报告")
        return FileResponse(p, media_type="text/html")

    # ------------------------------------------------ 视频/音频导出
    def _done_job(jid: str):
        job = runner.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        if job.status != "done":
            raise HTTPException(400, "任务尚未完成，无法导出")
        return job

    @app.post("/api/jobs/{jid}/exports/plan")
    def export_plan(jid: str, body: dict):
        job = _done_job(jid)
        try:
            with runner.lock_for(jid):
                plan = exports.plan_for_job(Path(job.job_dir), Path(job.input_path), body)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "plan": plan}

    @app.post("/api/jobs/{jid}/exports")
    def create_export(jid: str, body: dict):
        job = _done_job(jid)
        try:
            task = exports_runner.create(job, body)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "export": task.to_dict()}

    @app.get("/api/jobs/{jid}/exports")
    def list_exports(jid: str):
        if runner.get(jid) is None:
            raise HTTPException(404, "任务不存在")
        return {"exports": [t.to_dict() for t in exports_runner.list(jid)]}

    @app.get("/api/exports/{eid}")
    def get_export(eid: str):
        t = exports_runner.get(eid)
        if t is None:
            raise HTTPException(404, "导出任务不存在")
        return t.to_dict(full=True)

    @app.post("/api/exports/{eid}/cancel")
    def cancel_export(eid: str):
        if not exports_runner.cancel(eid):
            raise HTTPException(400, "无法取消（不存在或不在运行）")
        return {"ok": True}

    @app.delete("/api/exports/{eid}")
    def delete_export(eid: str):
        if not exports_runner.delete(eid):
            raise HTTPException(400, "无法删除（不存在或正在运行）")
        return {"ok": True}

    @app.get("/api/jobs/{jid}/source")
    def job_source(jid: str):
        job = runner.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        p = Path(job.input_path)
        if not p.exists():
            raise HTTPException(404, "原始文件不存在（可能已被移动或删除）")
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mt)

    # ------------------------------------------------ 环境 / 模型 / 网络
    @app.get("/api/env")
    def env_status():
        return {"deps": env_manager.check(), "install": env_manager.install_status()}

    @app.post("/api/env/install")
    def env_install(body: dict):
        modules = body.get("modules")
        res = env_manager.install(modules if isinstance(modules, list) else None,
                                  body.get("gpu"))
        if not res.get("accepted"):
            raise HTTPException(409, res.get("reason", "未受理"))
        return res

    @app.get("/api/models")
    def models_status():
        return {"models": models_manager.status()}

    @app.post("/api/models/download")
    def models_download(body: dict):
        key = body.get("key")
        if not key:
            raise HTTPException(400, "缺少 key")
        if not models_manager.start_download(key):
            raise HTTPException(409, "该模型正在下载")
        return {"ok": True}

    @app.get("/api/network")
    def network_status():
        plan = network.get_plan()
        return plan.to_dict()

    @app.post("/api/network/probe")
    def network_probe():
        plan = network.probe(force=True)
        return plan.to_dict()

    # ------------------------------------------------ 声纹库
    @app.get("/api/voiceprints")
    def vp_list():
        return {"voiceprints": [{k: v[k] for k in v if k != "embedding"}
                                for v in voiceprint.list_all()]}

    @app.post("/api/voiceprints")
    def vp_create(body: dict):
        name = body.get("name", "").strip()
        emb = body.get("embedding")
        if not name or not emb:
            raise HTTPException(400, "需要 name 与 embedding")
        v = voiceprint.create(name, emb, body.get("note", ""))
        events.success(f"声纹已入库：{name}", scope="voiceprint")
        return {"ok": True, "voiceprint": {k: v[k] for k in v if k != "embedding"}}

    @app.patch("/api/voiceprints/{vid}")
    def vp_update(vid: str, body: dict):
        v = voiceprint.update(vid, body)
        if v is None:
            raise HTTPException(404, "声纹不存在")
        return {"ok": True}

    @app.delete("/api/voiceprints/{vid}")
    def vp_delete(vid: str):
        if not voiceprint.delete(vid):
            raise HTTPException(404, "声纹不存在")
        return {"ok": True}

    @app.post("/api/voiceprints/merge")
    def vp_merge(body: dict):
        try:
            v = voiceprint.merge(body.get("dst", ""), body.get("src", ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if v is None:
            raise HTTPException(400, "合并失败（id 无效或相同）")
        return {"ok": True}

    @app.get("/api/voiceprints/export")
    def vp_export():
        from fastapi.responses import Response
        return Response(content=voiceprint.export_json(), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=voiceprints.json"})

    @app.post("/api/voiceprints/import")
    async def vp_import(file: UploadFile | None = File(None), text: str | None = Form(None),
                        replace: bool = Form(False)):
        content = text or ""
        if file is not None and file.filename:
            content = (await file.read()).decode("utf-8", "ignore")
        if not content:
            raise HTTPException(400, "缺少声纹包内容")
        try:
            n = voiceprint.import_json(content, replace)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "imported": n}

    # 从任务结果提取说话人 centroid 入库
    @app.post("/api/jobs/{jid}/voiceprint")
    def vp_from_job(jid: str, body: dict):
        job = runner.get(jid)
        if job is None or not job.result:
            raise HTTPException(404, "任务不存在或无结果")
        sid = body.get("speaker")
        name = body.get("name", "").strip()
        sp = next((s for s in job.result["speakers"] if s["id"] == sid), None)
        if sp is None or not name:
            raise HTTPException(400, "说话人或名称无效")
        v = voiceprint.create(name, sp["centroid"], note=f"来自任务 {jid}")
        return {"ok": True, "voiceprint": {k: v[k] for k in v if k != "embedding"}}

    # ------------------------------------------------ 静态
    app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")
    if config.WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
    else:
        @app.get("/")
        def fallback_index():
            return {"ok": True, "hint": "web/ 前端目录缺失，API 仍可用：/api/health"}
    return app

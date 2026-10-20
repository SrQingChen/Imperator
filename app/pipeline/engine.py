"""SeparationEngine：模型懒加载/互斥/释放 + 全流程编排 + 修正重渲染。"""
from __future__ import annotations

import gc
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config, events, voiceprint
from ..settings import Settings
from . import audio_io, diarize as dz, ledger, mss, outlier, overlap, report


def free_vram() -> dict:
    before = 0
    try:
        import torch
        if torch.cuda.is_available():
            before = int(torch.cuda.memory_reserved() / 1048576)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after = int(torch.cuda.memory_reserved() / 1048576)
            return {"before_mb": before, "after_mb": after, "freed_mb": max(0, before - after)}
    except Exception:
        pass
    gc.collect()
    return {"before_mb": before, "after_mb": before, "freed_mb": 0}


class Cancelled(Exception):
    pass


class Embedder:
    """funasr cam++ 声纹（192 维，L2 归一化）。"""

    def __init__(self, device: str):
        config.ensure_torch()
        from funasr import AutoModel
        self.model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common",
                               device=device, disable_update=True, hub="ms")

    @staticmethod
    def _to_numpy(v) -> np.ndarray | None:
        try:
            import torch
            if isinstance(v, torch.Tensor):
                v = v.detach().float().cpu().numpy()
        except Exception:
            pass
        try:
            return np.asarray(v, dtype="float32").flatten()
        except Exception:
            return None

    def embed_path(self, wav_path: str) -> np.ndarray | None:
        try:
            res = self.model.generate(input=str(wav_path))
        except Exception as e:
            events.warn(f"声纹提取失败 {wav_path}: {e}", scope="embed")
            return None
        first = res[0] if isinstance(res, list) and res else res
        if isinstance(first, dict):
            for key in ("spk_embedding", "spk", "embedding"):
                if key in first:
                    v = self._to_numpy(first[key])
                    if v is not None and v.size >= 64:
                        return v / (np.linalg.norm(v) + 1e-9)
            for v in first.values():  # 兜底：找第一个尺寸合适的浮点数组
                arr = self._to_numpy(v)
                if arr is not None and arr.size >= 64:
                    return arr / (np.linalg.norm(arr) + 1e-9)
        return None


class Engine:
    def __init__(self):
        self._lock = threading.RLock()
        self._mss: mss.MssSeparator | None = None
        self._mss_tier: str | None = None
        self._vad = None
        self._embedder: Embedder | None = None
        self._sep: overlap.ClearVoiceSeparator | None = None

    # ------------------------------------------------ 模型生命周期
    def ensure_mss(self, tier: str) -> mss.MssSeparator:
        with self._lock:
            if self._mss is not None and self._mss_tier == tier:
                return self._mss
            if self._mss is not None:
                self.release_mss()
            from .. import models_manager
            name = mss.tier_model_filename(tier)
            models_manager.ensure_audio_separator_model(name)
            events.info(f"加载 MSS 模型：{name}（档位 {tier}）", scope="engine")
            self._mss = mss.MssSeparator(tier=tier, output_dir=config.TMP_DIR / "mss_out")
            self._mss_tier = tier
            return self._mss

    def release_mss(self) -> None:
        with self._lock:
            self._mss = None
            self._mss_tier = None
            free_vram()
            events.info("MSS 模型已卸载（显存已释放）", scope="engine")

    def ensure_vad(self):
        with self._lock:
            if self._vad is None:
                events.info("加载 Silero VAD…", scope="engine")
                from .vad import Vad
                self._vad = Vad(device="cpu")
            return self._vad

    def ensure_embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                events.info("加载 CAM++ 声纹模型（首次将自动下载）…", scope="engine")
                self._embedder = Embedder(device=config.torch_device())
            return self._embedder

    def ensure_separator(self) -> overlap.ClearVoiceSeparator:
        with self._lock:
            if self._sep is None:
                from .. import models_manager
                models_manager.ensure_clearvoice()
                events.info("加载 MossFormer2_SS_16K 重叠分离模型…", scope="engine")
                self._sep = overlap.ClearVoiceSeparator()
            return self._sep

    def release_separator(self) -> None:
        with self._lock:
            self._sep = None
            free_vram()
            events.info("重叠分离模型已卸载（显存已释放）", scope="engine")

    def release_all(self) -> None:
        with self._lock:
            self._mss = None
            self._mss_tier = None
            self._sep = None
            free_vram()

    # ------------------------------------------------ 全流程
    def run(self, input_path: Path, job_dir: Path, cfg: Settings,
            progress, cancel) -> dict:
        t_start = time.time()
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp = job_dir / "_tmp"
        tmp.mkdir(exist_ok=True)
        os.chdir(config.DATA_DIR)  # clearvoice 以 CWD 相对路径找 checkpoints

        def _chk():
            if cancel():
                raise Cancelled()

        # ---- extract
        _chk()
        progress("extract", 0.05, "ffmpeg 抽取 44.1k 主轴音频")
        master = audio_io.extract_master(input_path, job_dir / "master.wav")
        mst_info = sf.info(str(master))
        duration = mst_info.frames / mst_info.samplerate
        progress("extract", 1.0, f"时长 {duration:.1f}s")

        # ---- mss
        _chk()
        vocals_keep = job_dir / "vocals_keep.wav"
        inst_keep = job_dir / "inst_keep.wav"
        if cfg.mss_tier == "skip":
            events.info("MSS 档位=跳过（视为纯人声输入）：vocals=master，bgm=静音", scope="pipeline")
            shutil.copyfile(master, vocals_keep)
            x, sr = audio_io.load_audio(master)
            audio_io.save_wav(inst_keep, np.zeros_like(x), sr)
            del x
            progress("mss", 1.0, "跳过（纯人声模式）")
        else:
            progress("mss", 0.02, "加载伴奏分离模型")
            sep = self.ensure_mss(cfg.mss_tier)
            progress("mss", 0.15, "分离伴奏与人声（大模型推理，长音频较久）")
            vocals_path, inst_path = sep.separate(master)
            shutil.move(str(vocals_path), vocals_keep)
            shutil.move(str(inst_path), inst_keep)
            if cfg.release_vram and config.get_device().has_nvidia:
                self.release_mss()
            progress("mss", 1.0, "完成")

        # ---- 分析支线：vocals → 16k 单声道（VAD/声纹/重叠一律用去伴奏后的人声）
        _chk()
        progress("vad", 0.1, "生成 16k 分析副本")
        voc_x, voc_sr = audio_io.load_audio(vocals_keep)
        voc16 = audio_io.resample(audio_io.to_mono(voc_x), voc_sr, 16000)
        del voc_x
        progress("vad", 0.4, "Silero VAD 检测语音段")
        vad = self.ensure_vad()
        vad_segs = vad.segments(voc16)
        progress("vad", 1.0, f"{len(vad_segs)} 个语音段")

        # ---- embed
        _chk()
        windows = dz.sliding_windows(vad_segs, duration, cfg.window_seconds,
                                     cfg.window_hop, min_len=0.4)
        emb = self.ensure_embedder()
        embs: list[np.ndarray | None] = [None] * len(windows)
        win_tmp = tmp / "win.wav"
        for i, w in enumerate(windows):
            _chk()
            i0, i1 = int(round(w.start * 16000)), int(round(w.end * 16000))
            clip = np.ascontiguousarray(voc16[i0:i1], dtype="float32")
            w.energy_db = audio_io.dbfs(clip)
            if w.energy_db > -50.0:  # 静音窗不浪费推理
                sf.write(str(win_tmp), clip, 16000)
                embs[i] = emb.embed_path(win_tmp)
            if i % 5 == 0 or i == len(windows) - 1:
                progress("embed", (i + 1) / max(len(windows), 1), f"{i + 1}/{len(windows)} 窗")
        valid = [(i, e) for i, e in enumerate(embs) if e is not None]
        if len(valid) < 2:
            raise RuntimeError("有效语音窗不足（未检测到足够人声，请检查输入或更换伴奏分离档位）")
        windows_kept = [windows[i] for i, _ in valid]
        emb_matrix = np.stack([e for _, e in valid])
        for new_idx, w in enumerate(windows_kept):
            w.idx = new_idx
        progress("embed", 1.0, f"{len(windows_kept)} 个有效窗")

        # ---- cluster + outlier + voiceprints
        _chk()
        progress("cluster", 0.2, f"凝聚聚类（τ={cfg.merge_threshold:.2f}）")
        res = dz.diarize(emb_matrix, windows_kept, vad_segs,
                         cfg.merge_threshold, cfg.smoothing_window, cfg.window_hop)
        progress("cluster", 0.4, f"{len(res.clusters)} 个候选说话人，精修中")
        res = dz.refine_clusters(res, emb_matrix, cfg.merge_threshold, cfg.max_speakers)
        progress("cluster", 0.5, f"精修后 {len([c for c in res.clusters if c.speaker_id >= 0])} 个候选说话人")
        audit = outlier.run_outlier_stage(res, cfg)
        if cfg.voiceprint_match:
            progress("cluster", 0.75, "声纹库匹配")
            dz.apply_voiceprints(res, voiceprint.list_all(), cfg.merge_threshold, audit)
        spk_ids = sorted({c.speaker_id for c in res.clusters if c.duration > 0 and c.speaker_id >= 0})
        centroids = {c.speaker_id: c.centroid for c in res.clusters if c.duration > 0 and c.speaker_id >= 0}
        label_map = {c.speaker_id: c.name for c in res.clusters if c.duration > 0 and c.speaker_id >= 0}
        progress("cluster", 1.0, f"最终 {len(spk_ids)} 个说话人")

        # ---- overlap
        _chk()
        triggered = overlap.detect_trigger_windows(res, cfg.merge_threshold, cfg.overlap_margin)
        regions: list[overlap.OverlapRegion] = []
        if triggered:
            progress("overlap", 0.05, f"{len(triggered)} 个触发窗，加载重叠分离模型")
            cv = self.ensure_separator()
            _cnt = [0]

            def sep_fn(audio: np.ndarray):
                p = tmp / f"ov_in_{_cnt[0]}.wav"
                _cnt[0] += 1
                sf.write(str(p), np.ascontiguousarray(audio, "float32"), 16000)
                return cv.separate(str(p))

            def embed_func(audio: np.ndarray, tag: int):
                p = tmp / f"ov_emb_{_cnt[0]}_{tag}.wav"
                _cnt[0] += 1
                sf.write(str(p), np.ascontiguousarray(audio, "float32"), 16000)
                return emb.embed_path(p)

            region_list = overlap.group_regions(triggered, res.windows)
            for ri, reg in enumerate(region_list):
                _chk()
                r = overlap.resolve_region(reg, cfg.overlap_context,
                                           voc16, 16000, sep_fn, embed_func, centroids,
                                           cfg.merge_threshold, cfg.max_overlap_speakers)
                regions.append(r)
                note = r.kind
                if r.kind == "overlap":
                    note += f"（{len(r.speakers)} 人）"
                progress("overlap", (ri + 1) / len(region_list),
                         f"区 {ri + 1}/{len(region_list)}：{note}")
            if cfg.release_vram and config.get_device().has_nvidia:
                self.release_separator()
        progress("overlap", 1.0, f"真重叠区 {sum(1 for r in regions if r.kind == 'overlap')} 个"
                                f"（其中多人区 {sum(1 for r in regions if len(r.speakers) >= 3)} 个，"
                                f"单人确认 {sum(1 for r in regions if r.kind == 'speech')}，"
                                f"未归属 {sum(1 for r in regions if r.kind == 'unknown')}）")

        # ---- ledger
        _chk()
        progress("ledger", 0.1, "构建增益账本")
        items = overlap.merge_regions_into_segments(res, regions)
        # 丢弃最终时长≈0 的说话人（其内容已被改派/未归属，避免输出空轨道）
        est_dur: dict[int, float] = {sid: 0.0 for sid in spk_ids}
        for it in items:
            masks = it.get("masks") or {}
            if it["kind"] == "speech" and it.get("speaker") in est_dur:
                est_dur[it["speaker"]] += it["end"] - it["start"]
            elif masks:
                share = (it["end"] - it["start"]) / max(len(masks), 1)
                for sid in masks:
                    if sid in est_dur:
                        est_dur[sid] += share
        spk_ids = [sid for sid in spk_ids if est_dur.get(sid, 0.0) >= 0.1]
        if not spk_ids:
            raise RuntimeError("未发现任何说话人（全部人声未归属，请降低合并阈值 τ 或检查素材）")
        label_map = {k: v for k, v in label_map.items() if k in spk_ids}
        gains = ledger.build_ctrl_gains(items, spk_ids, duration)
        progress("ledger", 0.3, "流式写轨（Σ 恒等校验中）")

        def on_ledger_prog(f: float):
            progress("ledger", 0.3 + 0.6 * f, f"重建 {f * 100:.0f}%")

        led = ledger.write_tracks(master, vocals_keep, inst_keep, job_dir,
                                  gains, label_map, cfg.output_format, cfg.output_sr,
                                  on_progress=on_ledger_prog)
        track_files = [job_dir / t["file"] for t in led.tracks]
        verify = ledger.verify_sum(track_files, master, led.sr_out)
        led.qc.update(verify)
        progress("ledger", 1.0, "完成")

        # ---- 报告与时间线（含掩码持久化，供修正重渲染）
        _chk()
        progress("report", 0.3, "汇总时间线与审计")
        mask_store: dict[str, np.ndarray] = {}
        segs_out: list[dict] = []
        for it in items:
            entry = {"id": len(segs_out), "start": round(it["start"], 3), "end": round(it["end"], 3),
                     "speaker": it.get("speaker", -1), "kind": it["kind"],
                     "label": label_map.get(it.get("speaker", -1), "未归属")
                     if it.get("speaker", -1) in label_map else "未归属",
                     "top1": 0.0, "margin": 0.0}
            src = it.get("src_seg")
            if src is not None:
                base = next((s for s in res.segments if s.seg_id == src), None)
                if base:
                    entry.update(top1=round(base.top1, 3), margin=round(base.margin, 3),
                                 energy_db=round(base.energy_db, 1), audit=base.audit)
            if it["kind"] in ("overlap", "hard"):
                spks = it.get("speakers", [])
                entry["speakers"] = spks
                entry["sims"] = it.get("sims", {})
                entry["mask_keys"] = {}
                for sid, arr in it.get("masks", {}).items():
                    key = f"seg{entry['id']}_{sid}"
                    mask_store[key] = arr
                    entry["mask_keys"][str(sid)] = key
                entry["speaker"] = spks[0] if spks else -1
                entry["label"] = (" + ".join(label_map.get(s, "?") for s in spks)
                                  if it["kind"] == "overlap" else label_map.get(entry["speaker"], "未归属"))
            segs_out.append(entry)
        if mask_store:
            np.savez_compressed(job_dir / "masks.npz", **mask_store)
        spk_durs = {sid: 0.0 for sid in spk_ids}
        for it in items:
            masks = it.get("masks") or {}
            if it["kind"] == "speech" and it.get("speaker") in spk_durs:
                spk_durs[it["speaker"]] += it["end"] - it["start"]
            elif masks:
                share = (it["end"] - it["start"]) / max(len(masks), 1)
                for sid in masks:
                    if sid in spk_durs:
                        spk_durs[sid] += share
        total_sp = sum(spk_durs.values()) + 1e-9
        speakers_out = [{
            "id": sid, "name": label_map.get(sid, f"说话人 {sid + 1}"),
            "duration_s": round(spk_durs[sid], 2),
            "share_pct": round(spk_durs[sid] / total_sp * 100, 2),
            "flags": [f for c in res.clusters if c.speaker_id == sid for f in c.flags],
            "centroid": centroids[sid].tolist(),
        } for sid in spk_ids]
        led.qc["elapsed_s"] = round(time.time() - t_start, 1)
        led.qc["rtf"] = round(led.qc["elapsed_s"] / max(duration, 1e-9), 3)
        tl = report.build_timeline(
            {"name": input_path.name, "master_path": str(master),
             "vocals_path": str(vocals_keep), "inst_path": str(inst_keep)},
            led.tracks, speakers_out, segs_out, audit, led.qc, led.duration_s, led.sr_out)
        tl["label_map"] = {str(k): v for k, v in label_map.items()}
        report.save_timeline(tl, job_dir)
        (job_dir / "report.html").write_text(report.render_html(tl), "utf-8")

        for p_ in tmp.glob("*.wav"):
            try:
                p_.unlink()
            except Exception:
                pass
        try:
            tmp.rmdir()
        except Exception:
            pass
        progress("report", 1.0, f"全部完成（{led.qc['elapsed_s']}s，RTF {led.qc['rtf']}）")
        return {
            "tracks": led.tracks, "speakers": speakers_out, "segments": segs_out,
            "audit": audit, "qc": led.qc, "duration_s": led.duration_s,
            "sr_out": led.sr_out, "n_samples": led.n_samples,
            "label_map": tl["label_map"],
        }

    # ------------------------------------------------ 修正回路（不重跑模型，秒级）
    def rerender(self, job_dir: Path, moves: list[dict], cfg: Settings,
                 progress=None) -> dict:
        tl = json.loads((job_dir / "timeline.json").read_text("utf-8"))
        masks_npz = job_dir / "masks.npz"
        masks = np.load(masks_npz) if masks_npz.exists() else {}
        by_id = {s["id"]: s for s in tl["segments"]}
        new_audit = []
        for mv in moves:
            sid, to = mv.get("segment"), mv.get("speaker", -1)
            seg = by_id.get(sid)
            if seg is None:
                continue
            old = seg.get("speaker", -1)
            unknown = (to == "unknown" or to == -1 or to is None)
            seg["speaker"] = -1 if unknown else int(to)
            seg["label"] = "未归属" if unknown else tl["label_map"].get(str(to), f"说话人 {to + 1}")
            if seg["kind"] in ("overlap", "hard"):
                if unknown:
                    seg["kind"] = "unknown"
                    seg.pop("mask_keys", None)
                else:
                    seg["kind"] = "hard"
                    seg.pop("mask_keys", None)  # 转为整区硬归属
                    seg["hard_forced"] = seg["speaker"]
            else:
                seg["kind"] = "unknown" if unknown else "speech"
            new_audit.append({"ts": time.time(), "action": "manual_reassign",
                              "detail": f"段 #{sid} [{seg['start']:.1f}s,{seg['end']:.1f}s] "
                                        f"{old} → {seg['speaker']}（人工改派）"})
        # 重建账本输入
        items: list[dict] = []
        for seg in tl["segments"]:
            if seg["kind"] == "unknown":
                continue
            if seg["kind"] in ("overlap", "hard"):
                if seg.get("hard_forced") is not None:
                    n16 = int(round((seg["end"] - seg["start"]) * 16000))
                    items.append({"start": seg["start"], "end": seg["end"], "kind": "hard",
                                  "speaker": seg["hard_forced"],
                                  "masks": {seg["hard_forced"]: np.ones(n16, "float32")}})
                else:
                    mk = seg.get("mask_keys") or {}
                    masks_i = {int(sid): np.asarray(masks[key], "float32") for sid, key in mk.items()
                               if key in masks}
                    items.append({"start": seg["start"], "end": seg["end"], "kind": seg["kind"],
                                  "speaker": seg.get("speaker", -1),
                                  "speakers": seg.get("speakers", []), "masks": masks_i})
            else:
                items.append({"start": seg["start"], "end": seg["end"], "kind": "speech",
                              "speaker": seg["speaker"]})
        spk_ids = sorted({it["speaker"] for it in items
                          if it["kind"] == "speech" and isinstance(it["speaker"], int)
                          and it["speaker"] >= 0} |
                         {sid for it in items if it["kind"] in ("overlap", "hard")
                          for sid in (it.get("masks") or {})})
        label_map = {int(k): v for k, v in tl["label_map"].items()}
        gains = ledger.build_ctrl_gains(items, spk_ids, tl["duration_s"])
        led = ledger.write_tracks(Path(tl["job"]["master_path"]), Path(tl["job"]["vocals_path"]),
                                  Path(tl["job"]["inst_path"]), job_dir, gains, label_map,
                                  cfg.output_format, cfg.output_sr,
                                  on_progress=(lambda f: progress("ledger", f, "重渲染")) if progress else None)
        verify = ledger.verify_sum([job_dir / t["file"] for t in led.tracks],
                                   Path(tl["job"]["master_path"]), led.sr_out)
        led.qc.update(verify)
        # 清理不再存在的说话人轨道（改派后轨数可能减少，遗留旧文件会误导下载）
        keep = {t["file"] for t in led.tracks}
        for stale in job_dir.glob("speaker_*.wav"):
            if stale.name not in keep:
                try:
                    stale.unlink()
                except OSError:
                    pass
        tl["qc"] = led.qc
        tl["tracks"] = led.tracks
        tl["audit"] = tl.get("audit", []) + new_audit
        durs: dict[int, float] = {}
        for it in items:
            per = it.get("masks") or {}
            if it["kind"] == "speech" and isinstance(it["speaker"], int) and it["speaker"] >= 0:
                durs[it["speaker"]] = durs.get(it["speaker"], 0.0) + (it["end"] - it["start"]) * 0.5
            for sid in per:
                durs[sid] = durs.get(sid, 0.0) + (it["end"] - it["start"]) * 0.5
        tot = sum(durs.values()) + 1e-9
        for sp in tl["speakers"]:
            sp["duration_s"] = round(durs.get(sp["id"], 0), 2)
            sp["share_pct"] = round(durs.get(sp["id"], 0) / tot * 100, 2)
        report.save_timeline(tl, job_dir)
        (job_dir / "report.html").write_text(report.render_html(tl), "utf-8")
        try:
            masks.close()
        except Exception:
            pass
        return tl

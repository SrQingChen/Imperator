"""MSS（伴奏/人声）分离：audio-separator 封装 + 模型清单解析（预下载用）。

0.16.x 的 audio-separator 在运行时从 TRvlvr/application_data 拉取 download_checks.json，
模型文件统一来自 https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models。
本适配层用同一份清单 + 自研镜像下载器完成预下载（RoFormer 需 ckpt+yaml 双文件）。
"""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from .. import config, events, network

TIER_MODELS: dict[str, list[str]] = {
    # 依次尝试，取清单中第一个存在的
    "standard": ["model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"],
    "fast": ["model_bs_roformer_ep_317_sdr_12.9755.ckpt",
             "model_bs_roformer_ep_368_sdr_12.9628.ckpt"],
    "light": ["UVR-MDX-NET-Inst_HQ_3.onnx", "UVR-MDX-NET-Inst_HQ_2.onnx",
              "UVR-MDX-NET-Inst_HQ.onnx"],
}
_CHECKS_URL = ("https://raw.githubusercontent.com/TRvlvr/application_data/main/"
               "filelists/download_checks.json")
_REPO_PREFIX = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
_APPDATA_PREFIX = "https://raw.githubusercontent.com/TRvlvr/application_data/main"
_YAML_PREFIX = f"{_APPDATA_PREFIX}/mdx_model_data/mdx_c_configs"


def model_dir() -> Path:
    d = config.MODELS_DIR / "audio-separator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checks_path() -> Path:
    return model_dir() / "download_checks.json"


def _load_checks() -> dict:
    p = _checks_path()
    if not p.exists():
        try:
            network.download(_CHECKS_URL, p, log_prefix="模型清单")
        except Exception as e:
            events.warn(f"模型清单不可用（离线或不稳定网络）：{e}", scope="models")
            return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return {}


_MANIFEST_CACHE: dict = {"mtime": None, "data": []}


def iter_manifest() -> list[dict]:
    """[{name, files:[{name,url}]}]——覆盖 VR/MDX/Demucs/MDXC/Roformer 全部公开模型。"""
    p = _checks_path()
    mtime = p.stat().st_mtime if p.exists() else None
    if _MANIFEST_CACHE["mtime"] == mtime and _MANIFEST_CACHE["data"]:
        return _MANIFEST_CACHE["data"]
    d = _load_checks()
    out: dict[str, dict] = {}

    def _add(fname: str, url: str | None = None, companions: list[tuple[str, str]] = []):
        if not fname or fname in out:
            return
        files = [{"name": fname, "url": url or f"{_REPO_PREFIX}/{fname}"}]
        for cname, curl in companions:
            files.append({"name": cname, "url": curl})
        out[fname] = {"name": fname, "files": files}

    for key in ("vr_download_list", "mdx_download_list", "demucs_download_list"):
        for display, val in (d.get(key) or {}).items():
            if isinstance(val, str):
                _add(val)
            elif isinstance(val, dict):
                main = next(iter(val))
                rest = [(k, v) for k, v in val.items() if k != main]
                _add(main, val.get(main) if str(val.get(main)).startswith("http") else None, rest)
    for key in ("mdx23_download_list", "mdx23c_download_list", "roformer_download_list"):
        for display, val in (d.get(key) or {}).items():
            if isinstance(val, dict):
                ckpt = next(iter(val))
                yamls = [(y, f"{_YAML_PREFIX}/{y}") for y in val.values() if y != ckpt]
                _add(ckpt, companions=yamls)
    result = list(out.values())
    _MANIFEST_CACHE.update(mtime=mtime, data=result)
    return result


def manifest_for(name: str) -> dict | None:
    for e in iter_manifest():
        if e["name"] == name:
            return e
    return None


def installed(name: str) -> bool:
    meta = manifest_for(name)
    if meta:
        return all((model_dir() / f["name"]).exists() for f in meta["files"])
    return (model_dir() / name).exists()


def ensure_model(name: str, progress_cb=None) -> None:
    """按清单下载模型全部文件（镜像路由 + 续传）。清单缺失时交还库自行下载。"""
    meta = manifest_for(name)
    if not meta:
        events.warn(f"模型 {name} 不在清单：将由 audio-separator 自行下载（不走镜像）",
                    scope="models")
        return
    for f in meta["files"]:
        if (model_dir() / f["name"]).exists():
            continue
        network.download(f["url"], model_dir() / f["name"], progress_cb=progress_cb,
                         log_prefix=name)


def tier_model_filename(tier: str) -> str:
    """解析档位对应的实际模型文件名（已装优先，其次清单命中）。"""
    cands = TIER_MODELS.get(tier) or TIER_MODELS["standard"]
    for c in cands:
        if (model_dir() / c).exists():
            return c
    manifest = {e["name"] for e in iter_manifest()}
    for c in cands:
        if c in manifest:
            return c
    return cands[0]


# ---------------------------------------------------------------- 性能规划

class MssCancelled(Exception):
    """分块分离被用户取消。"""


_BENCH_PATH = lambda: config.DATA_DIR / "mss_benchmark.json"  # noqa: E731


def load_benchmark() -> dict:
    try:
        return json.loads(_BENCH_PATH().read_text("utf-8"))
    except Exception:
        return {}


def save_benchmark(rtf: float, model: str) -> None:
    try:
        data = load_benchmark()
        data[model] = {"rtf": round(rtf, 4), "ts": time.time(),
                       "gpu": config.get_device().gpu_name}
        _BENCH_PATH().write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass


def plan_for_duration(duration_s: float, model_name: str | None = None) -> dict:
    """按可用内存/显存为长音频选择分块策略。

    分块不仅让内存有界，也让进度条按“已处理音频时长”真实推进；
    块间用 3s 重叠 + 互补淡入淡出，避免接缝爆音/断层。
    """
    dev = config.get_device()
    avail_gb = psutil.virtual_memory().available / 2**30
    vram_gb = dev.gpu_vram_mb / 1024 if dev.has_nvidia else 0.0
    model_reserve = 2.8 if dev.has_nvidia else 1.2
    usable = avail_gb - model_reserve
    if duration_s <= 240:
        chunk = 0
    elif usable >= 6 and vram_gb >= 7:
        chunk = 180.0
    elif usable >= 3:
        chunk = 120.0
    elif usable >= 1.5:
        chunk = 90.0
    elif usable >= 0.6:
        chunk = 60.0
    else:
        chunk = 30.0
    overlap = min(3.0, max(0.5, chunk * 0.08)) if chunk else 0.0
    chunks = 0 if not chunk else max(1, int(math.ceil(duration_s / max(chunk - overlap, 1.0))))
    bench = load_benchmark().get(model_name or tier_model_filename("standard"), {})
    est_rtf = float(bench.get("rtf") or 0) or None
    est_s = round(duration_s * est_rtf, 1) if est_rtf else None
    if chunk:
        reason = (f"长音频 {duration_s/60:.1f} 分钟：可用内存 {avail_gb:.1f}GB，"
                  f"分块 {chunk:.0f}s（重叠 {overlap:.1f}s，共 {chunks} 块）")
    else:
        reason = f"时长 {duration_s/60:.1f} 分钟，整段处理"
    return {
        "chunk_seconds": chunk, "overlap_seconds": overlap, "chunks": chunks,
        "available_ram_gb": round(avail_gb, 2), "gpu": dev.gpu_name or "CPU",
        "vram_gb": round(vram_gb, 1), "reason": reason,
        "estimated_rtf": est_rtf, "estimated_seconds": est_s,
        "model": model_name or tier_model_filename("standard"),
    }


def _read_exact(path: str | Path, frames: int, channels: int) -> np.ndarray:
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[0] < frames:
        x = np.concatenate([x, np.zeros((frames - x.shape[0], x.shape[1]), "float32")], axis=0)
    elif x.shape[0] > frames:
        x = x[:frames]
    if x.shape[1] != channels:
        if x.shape[1] == 1:
            x = np.repeat(x, channels, axis=1)
        elif channels == 1:
            x = x.mean(axis=1, keepdims=True)
        else:
            x = x[:, :channels] if x.shape[1] > channels else np.pad(
                x, ((0, 0), (0, channels - x.shape[1])))
    return x.astype("float32", copy=False)


def precision_kwargs(precision: str, device: str) -> dict:
    """RoFormer 精度策略：CUDA 上用原生 FP16（实测快 45%，波形差 -73dB）。

    - auto：CUDA 自动 FP16；CPU 保持 FP32
    - fp16：强制原生 FP16（仅 CUDA 有效）
    - autocast：FP32 权重 + autocast（速度略慢于原生 FP16，显存略高）
    - fp32：完全 FP32（最保守）
    """
    if device != "cuda":
        return {}
    if precision in ("auto", "fp16"):
        return {"use_native_fp16": True}
    if precision == "autocast":
        return {"use_autocast": True}
    return {}


def _gpu_snapshot() -> str:
    import subprocess
    smi = shutil.which("nvidia-smi")
    if not smi:
        return ""
    try:
        p = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=4,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        parts = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
        return f"，GPU {parts[0]}%/{parts[1]}MB"
    except Exception:
        return ""


def _heartbeat_start(label: str, interval: float = 5.0) -> threading.Event:
    """模型加载/长步骤心跳：每 5 秒报告耗时、RSS、GPU，卡住时一眼能看出来。"""
    stop = threading.Event()

    def run():
        t0 = time.time()
        while not stop.wait(interval):
            try:
                rss = psutil.Process().memory_info().rss / 2**20
                events.info(f"{label}… 已用 {time.time()-t0:.0f}s，RSS {rss:.0f}MB"
                            f"{_gpu_snapshot()}", scope="engine")
            except Exception:
                pass

    threading.Thread(target=run, name="mss-heartbeat", daemon=True).start()
    return stop


class MssSeparator:
    """薄封装：构造即加载模型，separate() 输出 vocals/instrumental 路径。"""

    def __init__(self, tier: str = "standard", output_dir: str | Path | None = None,
                 precision: str = "auto", overlap: int = 2):
        from .. import config as _cfg
        t0 = time.time()
        _cfg.ensure_torch()
        from audio_separator.separator import Separator
        self.tier = tier
        self.precision = precision
        self.overlap = int(overlap)
        self.model_name = tier_model_filename(tier)
        self._output_dir = Path(output_dir) if output_dir else None
        ckpt = model_dir() / self.model_name
        yaml_path = ckpt.with_suffix(".yaml")
        size_mb = ckpt.stat().st_size / 1048576 if ckpt.exists() else 0.0
        device = _cfg.torch_device()
        self._precision_kwargs = precision_kwargs(precision, device)
        events.info(f"MSS 模型文件：{self.model_name}（{size_mb:.1f}MB），"
                    f"配置 {yaml_path.name} {'存在' if yaml_path.exists() else '缺失（可能触发下载）'}，"
                    f"设备 {device}，精度 {precision} → {self._precision_kwargs or 'FP32'}，"
                    f"重叠窗口 {self.overlap}", scope="engine")
        kwargs: dict = {
            "model_file_dir": str(model_dir()),
            "output_format": "WAV",
            "sample_rate": 44100,
            "mdxc_params": {"segment_size": 256, "override_model_segment_size": False,
                            "batch_size": None, "overlap": self.overlap, "pitch_shift": 0},
        }
        if output_dir:
            kwargs["output_dir"] = str(output_dir)
        events.info("初始化 Separator…", scope="engine")
        try:
            self.sep = Separator(**kwargs, **self._precision_kwargs)
        except TypeError as e:  # 老版本参数名差异 / 不支持精度参数
            events.warn(f"Separator 参数降级（{e}），改用 FP32", scope="mss")
            kw = dict(kwargs)
            kw.pop("sample_rate", None)
            self.sep = Separator(**kw)
        t1 = time.time()
        events.info(f"Separator 初始化完成（{t1-t0:.2f}s），开始加载权重…", scope="engine")
        stop = _heartbeat_start("MSS 模型加载中")
        try:
            self.sep.load_model(self.model_name)
        except Exception as e:
            events.error(f"MSS 模型加载失败：{type(e).__name__}: {e}", scope="engine")
            raise
        finally:
            stop.set()
        extra = ""
        try:
            import torch
            if torch.cuda.is_available():
                extra = f"，显存已分配 {torch.cuda.memory_allocated()/1048576:.0f}MB"
        except Exception:
            pass
        events.success(f"MSS 权重加载完成（{time.time()-t1:.2f}s，总计 {time.time()-t0:.2f}s{extra}）",
                       scope="engine")

    def separate(self, wav_path: str | Path) -> tuple[Path, Path]:
        out_base = self._output_dir or Path(wav_path).resolve().parent
        out_base.mkdir(parents=True, exist_ok=True)
        for stale in out_base.glob("*(Vocals)*.wav"):
            try:
                stale.unlink()
            except OSError:
                pass
        for stale in out_base.glob("*(Instrumental)*.wav"):
            try:
                stale.unlink()
            except OSError:
                pass
        outs = self.sep.separate(str(wav_path))
        # 0.47 只返回文件名（相对 output_dir）；0.16 返回完整路径——统一解析
        resolved = []
        for o in outs:
            p = Path(o)
            resolved.append(p if p.is_absolute() else out_base / p.name)
        vocals = inst = None
        for p in resolved:
            low = p.name.lower()
            if "(vocals)" in low:
                vocals = p
            elif "(instrumental)" in low or "(no vocals)" in low or "accompaniment" in low:
                inst = p
        if vocals is None or inst is None:
            found = ", ".join(p.name for p in resolved)
            raise RuntimeError(f"未能从 MSS 输出中识别 vocals/instrumental：{found}")
        return vocals, inst


    # ------------------------------------------------------------ 分块长音频
    def separate_chunked(self, input_wav: str | Path, out_vocals: str | Path,
                         out_inst: str | Path, chunk_seconds: float,
                         overlap_seconds: float, progress=None,
                         cancel=None) -> dict:
        """分块分离长音频：块间重叠 + 互补淡入淡出，内存有界、进度真实。

        progress(frac, processed_s, measured_rtf) 按“已处理音频时长”回调。
        取消抛 MssCancelled。
        """
        info = sf.info(str(input_wav))
        sr, ch, total = info.samplerate, info.channels, info.frames
        duration = total / sr if sr else 0.0
        out_vocals, out_inst = Path(out_vocals), Path(out_inst)
        out_vocals.parent.mkdir(parents=True, exist_ok=True)
        work = out_vocals.parent / "_mss_chunks"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        chunk_seconds = max(5.0, float(chunk_seconds or duration or 30.0))
        overlap = max(0.0, min(float(overlap_seconds or 0.0), chunk_seconds * 0.25))
        step = max(1.0, chunk_seconds - overlap)
        n_chunks = max(1, int(math.ceil(duration / step))) if duration > 0 else 0
        t0 = time.time()
        src = sf.SoundFile(str(input_wav))
        vo = sf.SoundFile(str(out_vocals), mode="w", samplerate=sr, channels=ch, subtype="FLOAT")
        io_ = sf.SoundFile(str(out_inst), mode="w", samplerate=sr, channels=ch, subtype="FLOAT")
        prev_v = prev_i = None
        written = 0
        try:
            for ci in range(n_chunks):
                if cancel and cancel():
                    raise MssCancelled("已取消")
                s = ci * step
                if s >= duration - 1e-6:
                    break
                e = min(s + chunk_seconds, duration)
                is_last = e >= duration - 1e-6
                n = int(round((e - s) * sr))
                src.seek(int(round(s * sr)))
                x = src.read(n, dtype="float32", always_2d=True)
                if x.shape[0] < n:
                    x = np.pad(x, ((0, n - x.shape[0]), (0, 0)))
                chunk_wav = work / f"chunk_{ci:05d}.wav"
                sf.write(str(chunk_wav), x, sr, subtype="FLOAT")
                vp, ip = self.separate(chunk_wav)
                v = _read_exact(vp, n, ch)
                inst = _read_exact(ip, n, ch)
                core = n if is_last else min(int(round(step * sr)), n)
                head = 0
                if prev_v is not None and prev_i is not None:
                    head = min(len(prev_v), n)
                    if head > 0:
                        w = np.linspace(0.0, 1.0, head, dtype="float32")[:, None]
                        vo.write(prev_v[:head] * (1.0 - w) + v[:head] * w)
                        io_.write(prev_i[:head] * (1.0 - w) + inst[:head] * w)
                        written += head
                    prev_v = prev_i = None
                if is_last:
                    if n > head:
                        vo.write(v[head:])
                        io_.write(inst[head:])
                        written += n - head
                else:
                    if core > head:
                        vo.write(v[head:core])
                        io_.write(inst[head:core])
                        written += core - head
                    prev_v = v[core:].copy() if core < n else None
                    prev_i = inst[core:].copy() if core < n else None
                # 清理当前块产物
                for p in (vp, ip, chunk_wav):
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
                elapsed = max(time.time() - t0, 1e-6)
                rtf = elapsed / max(e, 1e-6)
                if progress:
                    progress(min(1.0, e / max(duration, 1e-6)), e, rtf)
                if cancel and cancel():
                    raise MssCancelled("已取消")
            # 数值安全：输出不足时补零（正常不应发生）
            if written < total:
                pad = np.zeros((total - written, ch), dtype="float32")
                vo.write(pad)
                io_.write(pad)
                written = total
        except Exception:
            vo.close()
            io_.close()
            src.close()
            shutil.rmtree(work, ignore_errors=True)
            for p in (out_vocals, out_inst):
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        vo.close()
        io_.close()
        src.close()
        shutil.rmtree(work, ignore_errors=True)
        elapsed = max(time.time() - t0, 1e-6)
        return {"duration": round(duration, 3), "chunks": n_chunks,
                "chunk_seconds": chunk_seconds, "overlap_seconds": overlap,
                "elapsed_s": round(elapsed, 1), "rtf": round(elapsed / max(duration, 1e-6), 3)}

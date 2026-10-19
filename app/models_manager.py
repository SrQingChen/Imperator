"""模型注册表与预下载（运行时与 fetch_models.py 共用同一代码路径）。"""
from __future__ import annotations

import os
import threading

from . import config, events, network
from .pipeline import mss

_state_lock = threading.RLock()
_downloading: dict[str, dict] = {}   # key -> {status, progress, done_mb, total_mb, error}


def _registry() -> list[dict]:
    """模型清单：key/用途/检查方式。"""
    items = []
    for tier, cands in mss.TIER_MODELS.items():
        for name in cands:
            if (mss.model_dir() / name).exists() or mss.manifest_for(name):
                items.append({
                    "key": f"mss:{name}", "name": name, "kind": "mss",
                    "tier": tier, "installed": mss.installed(name),
                    "source": "audio-separator 官方仓库（镜像加速）",
                })
    items.append({"key": "sep:MossFormer2_SS_16K", "name": "MossFormer2_SS_16K",
                  "kind": "overlap", "tier": "", "installed": clearvoice_installed(),
                  "source": "HuggingFace / ModelScope"})
    items.append({"key": "emb:campplus", "name": "CAM++ 声纹模型", "kind": "embed",
                  "tier": "", "installed": funasr_cache_installed(), "source": "ModelScope"})
    items.append({"key": "vad:silero", "name": "Silero VAD（随 pip 包内置）", "kind": "vad",
                  "tier": "", "installed": _module_ok("silero_vad"), "source": "pip"})
    return items


def _module_ok(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def clearvoice_installed() -> bool:
    ckpt = config.CLEARVOICE_CKPT_DIR
    if not ckpt.exists():
        return False
    return any(ckpt.rglob("*.pt")) or any(ckpt.rglob("*.bin")) or any(ckpt.rglob("*.safetensors"))


def funasr_cache_installed() -> bool:
    try:
        from pathlib import Path as _P
        home = _P.home() / ".cache" / "modelscope" / "hub" / "models" / "iic"
        if not home.exists():
            return False
        for d in home.iterdir():
            if "campplus" in d.name.lower() and any(d.rglob("*.pt")):
                return True
        return False
    except Exception:
        return False


def status() -> list[dict]:
    with _state_lock:
        items = _registry()
        for it in items:
            st = _downloading.get(it["key"])
            if st:
                it["download"] = dict(st)
        return items


def _set_dl(key: str, **kw) -> None:
    with _state_lock:
        _downloading.setdefault(key, {"status": "running", "progress": 0.0,
                                      "done_mb": 0.0, "total_mb": 0.0, "error": ""})
        _downloading[key].update(kw)


def ensure_audio_separator_model(name: str) -> None:
    """MSS 模型缺失时阻塞式确保（镜像路由与续传）；任务线程调用。
    若 UI 已在下载同一模型则等待其完成。"""
    if mss.installed(name):
        return
    key = f"mss:{name}"
    with _state_lock:
        cur = _downloading.get(key, {})
    if cur.get("status") == "running":
        import time as _t
        while True:
            with _state_lock:
                cur = dict(_downloading.get(key, {}))
            if cur.get("status") != "running":
                if cur.get("status") == "done" and mss.installed(name):
                    return
                raise RuntimeError(f"模型 {name} 下载未成功：{cur.get('error') or cur.get('status')}")
            _t.sleep(1.0)
    _set_dl(key, status="running", error="")
    total_files = len((mss.manifest_for(name) or {}).get("files", []))

    def cb(done: int, tot: int):
        _set_dl(key, done_mb=round(done / 1048576, 1), total_mb=round(tot / 1048576, 1),
                progress=(done / tot) if tot else 0.0)

    try:
        if total_files:
            mss.ensure_model(name, progress_cb=cb)
        else:
            events.warn(f"模型 {name} 不在清单：将由 audio-separator 自行下载（不走镜像）",
                        scope="models")
            return
        _set_dl(key, status="done", progress=1.0)
    except Exception as e:
        _set_dl(key, status="error", error=str(e))
        raise


def start_download(key: str) -> bool:
    """UI 触发的后台下载。返回是否受理。"""
    with _state_lock:
        cur = _downloading.get(key, {})
        if cur.get("status") == "running":
            return False
        _downloading[key] = {"status": "running", "progress": 0.0, "done_mb": 0.0,
                             "total_mb": 0.0, "error": ""}

    def _work():
        try:
            if key.startswith("mss:"):
                name = key.split(":", 1)[1]

                def cb(done: int, tot: int):
                    _set_dl(key, done_mb=round(done / 1048576, 1),
                            total_mb=round(tot / 1048576, 1),
                            progress=(done / tot) if tot else 0.0)

                mss.ensure_model(name, progress_cb=cb)
            elif key == "sep:MossFormer2_SS_16K":
                _download_clearvoice(key)
            elif key == "emb:campplus":
                _download_funasr(key)
            else:
                raise RuntimeError(f"未知模型 {key}")
            _set_dl(key, status="done", progress=1.0)
            events.success(f"模型就绪：{key}", scope="models")
        except Exception as e:
            _set_dl(key, status="error", error=str(e))
            events.error(f"模型下载失败 {key}：{e}", scope="models")

    threading.Thread(target=_work, name=f"dl-{key}", daemon=True).start()
    return True


def _dir_size_mb(p) -> float:
    import os
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1048576


def ensure_clearvoice() -> None:
    """确保 MossFormer2_SS_16K 权重就位：当前端点 → hf-mirror → 官方 的回退链。
    引擎加载前调用（snapshot_download 幂等，已就位时瞬时返回）。"""
    if clearvoice_installed():
        return
    from huggingface_hub import snapshot_download

    attempts: list[str | None] = []
    cur = os.environ.get("HF_ENDPOINT")
    if cur:
        attempts.append(cur)
    if "https://hf-mirror.com" not in attempts:
        attempts.append("https://hf-mirror.com")
    if None not in attempts:
        attempts.append(None)
    last: Exception | None = None
    for ep in attempts:
        if ep:
            os.environ["HF_ENDPOINT"] = ep
        else:
            os.environ.pop("HF_ENDPOINT", None)
        try:
            events.info(f"下载 MossFormer2_SS_16K（{ep or '官方源'}）", scope="models")
            snapshot_download(repo_id="alibabasglab/MossFormer2_SS_16K",
                              local_dir=str(config.CLEARVOICE_CKPT_DIR))
            if clearvoice_installed():
                return
        except Exception as e:
            last = e
            events.warn(f"源 {ep or '官方'} 下载失败：{e}", scope="models")
    raise RuntimeError(f"MossFormer2_SS_16K 全部下载源失败：{last}")


def _download_clearvoice(key: str) -> None:
    """UI 触发的后台下载：线程内跑 ensure_clearvoice，目录体积轮询显进度。"""
    done_flag = threading.Event()
    errs: list[Exception] = []

    def _load():
        try:
            ensure_clearvoice()
        except Exception as e:
            errs.append(e)
        finally:
            done_flag.set()

    threading.Thread(target=_load, daemon=True).start()
    ckpt = config.CLEARVOICE_CKPT_DIR
    last = -1.0
    while not done_flag.wait(1.5):
        sz = _dir_size_mb(ckpt) if ckpt.exists() else 0.0
        if sz > last:
            _set_dl(key, done_mb=round(sz, 1), total_mb=0, progress=0.0)
            last = sz
    if errs:
        raise errs[0]
    if not clearvoice_installed():
        raise RuntimeError("下载结束但权重缺失，请重试或手动放置")
    _set_dl(key, done_mb=round(_dir_size_mb(ckpt), 1), status="done", progress=1.0)


def _download_funasr(key: str) -> None:
    """funasr cam++：ModelScope 官方通道；缓存目录体积轮询。"""
    done_flag = threading.Event()
    errs: list[Exception] = []

    def _load():
        try:
            from . import config as _cfg
            _cfg.ensure_torch()
            from funasr import AutoModel
            AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common",
                      device="cpu", disable_update=True, hub="ms")
        except Exception as e:
            errs.append(e)
        finally:
            done_flag.set()

    threading.Thread(target=_load, daemon=True).start()
    events.info("开始下载 CAM++ 声纹模型（ModelScope 通道）", scope="models")
    from pathlib import Path as _P
    cache = _P.home() / ".cache" / "modelscope" / "hub" / "models" / "iic"
    last = -1.0
    while not done_flag.wait(1.5):
        sz = _dir_size_mb(cache) if cache.exists() else 0.0
        if sz > last:
            _set_dl(key, done_mb=round(sz, 1), total_mb=0, progress=0.0)
            last = sz
    if errs:
        raise errs[0]
    _set_dl(key, status="done", progress=1.0)

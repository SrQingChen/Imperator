"""MSS（伴奏/人声）分离：audio-separator 封装 + 模型清单解析（预下载用）。

0.16.x 的 audio-separator 在运行时从 TRvlvr/application_data 拉取 download_checks.json，
模型文件统一来自 https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models。
本适配层用同一份清单 + 自研镜像下载器完成预下载（RoFormer 需 ckpt+yaml 双文件）。
"""
from __future__ import annotations

import json
from pathlib import Path

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
        network.download(_CHECKS_URL, p, log_prefix="模型清单")
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return {}


def iter_manifest() -> list[dict]:
    """[{name, files:[{name,url}]}]——覆盖 VR/MDX/Demucs/MDXC/Roformer 全部公开模型。"""
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
    return list(out.values())


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


class MssSeparator:
    """薄封装：构造即加载模型，separate() 输出 vocals/instrumental 路径。"""

    def __init__(self, tier: str = "standard", output_dir: str | Path | None = None):
        from .. import config as _cfg
        _cfg.ensure_torch()
        from audio_separator.separator import Separator
        self.tier = tier
        self.model_name = tier_model_filename(tier)
        self._output_dir = Path(output_dir) if output_dir else None
        kwargs: dict = {
            "model_file_dir": str(model_dir()),
            "output_format": "WAV",
            "sample_rate": 44100,
        }
        if output_dir:
            kwargs["output_dir"] = str(output_dir)
        try:
            self.sep = Separator(**kwargs)
        except TypeError as e:  # 老版本参数名差异
            events.warn(f"Separator 构造参数降级：{e}", scope="mss")
            kwargs.pop("sample_rate", None)
            self.sep = Separator(**kwargs)
        self.sep.load_model(self.model_name)

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

"""audio-separator 清单解析 golden 测试（本地 fixture，不依赖网络）。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import mss  # noqa: E402

CHECKS = {
    "vr_download_list": {"VR: 1_HP-UVR": "1_HP-UVR.pth"},
    "mdx_download_list": {"MDX: Inst HQ 3": "UVR-MDX-NET-Inst_HQ_3.onnx"},
    "demucs_download_list": {},
    "mdx23_download_list": {},
    "mdx23c_download_list": {},
    "roformer_download_list": {
        "Roformer Model: Mel-Roformer-Viperx-1143": {
            "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt":
                "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
        },
        "Roformer Model: BS-Roformer-Viperx-1297": {
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt":
                "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        },
    },
}


@pytest.fixture()
def manifest_env(tmp_path, monkeypatch):
    p = tmp_path / "download_checks.json"
    p.write_text(json.dumps(CHECKS), encoding="utf-8")
    monkeypatch.setattr(mss, "_checks_path", lambda: p)
    monkeypatch.setattr(mss, "model_dir", lambda: tmp_path)
    return tmp_path


def test_manifest_parses_ckpt_and_yaml(manifest_env):
    entries = {e["name"]: e for e in mss.iter_manifest()}
    std = mss.TIER_MODELS["standard"][0]
    assert std in entries
    files = entries[std]["files"]
    assert len(files) == 2
    names = {f["name"] for f in files}
    assert std in names and std.replace(".ckpt", ".yaml") in names
    for f in files:
        assert f["url"].startswith("https://")
    # ckpt 走 GitHub Releases，yaml 走 application_data 仓库
    assert "model_repo/releases" in next(f["url"] for f in files if f["name"].endswith(".ckpt"))
    assert "application_data" in next(f["url"] for f in files if f["name"].endswith(".yaml"))


def test_manifest_parses_single_file_models(manifest_env):
    entries = {e["name"]: e for e in mss.iter_manifest()}
    assert "UVR-MDX-NET-Inst_HQ_3.onnx" in entries
    assert len(entries["UVR-MDX-NET-Inst_HQ_3.onnx"]["files"]) == 1
    assert "1_HP-UVR.pth" in entries


def test_tier_resolution_falls_back_to_candidates(manifest_env):
    assert mss.tier_model_filename("standard") == mss.TIER_MODELS["standard"][0]
    assert mss.tier_model_filename("fast") == mss.TIER_MODELS["fast"][0]
    # 轻量档清单中不存在 → 仍然返回候选名（由下载层报可读错误）
    assert mss.tier_model_filename("light").endswith(".onnx")


def test_installed_requires_all_files(manifest_env):
    std = mss.TIER_MODELS["standard"][0]
    (manifest_env / std).write_bytes(b"x")
    assert not mss.installed(std), "只有 ckpt 没有 yaml 时不应算已安装"
    (manifest_env / std.replace(".ckpt", ".yaml")).write_bytes(b"x")
    assert mss.installed(std)

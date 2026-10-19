"""audio-separator manifest 解析 golden 测试（未安装库时跳过）。"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAS_LIB = importlib.util.find_spec("audio_separator") is not None
pytestmark = pytest.mark.skipif(not HAS_LIB, reason="audio-separator 未安装")


def test_manifest_contains_tier_models():
    from app.pipeline import mss
    manifest = {e["name"] for e in mss.iter_manifest()}
    assert mss.TIER_MODELS["standard"][0] in manifest, \
        f"标准档模型不在 manifest（布局可能已变更，需适配）：{sorted(manifest)[:10]}…"
    for e in mss.iter_manifest():
        for f in e["files"]:
            assert f["url"].startswith("http")


def test_tier_resolution():
    from app.pipeline import mss
    name = mss.tier_model_filename("standard")
    assert name.endswith(".ckpt") or name.endswith(".onnx") or name.endswith(".th")

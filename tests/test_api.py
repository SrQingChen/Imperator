"""API 冒烟测试（不跑真实模型）。"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERATOR_DATA_DIR", str(tmp_path))
    # 复制真实模型清单，避免 /api/models 在临时数据目录里尝试联网下载
    real_checks = Path(__file__).resolve().parent.parent / "data" / "models" / "audio-separator" / "download_checks.json"
    if real_checks.exists():
        dst = tmp_path / "models" / "audio-separator" / "download_checks.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_checks, dst)
    # 保存原始 app 模块，测试后恢复，避免污染后续测试的数据目录
    saved = {k: v for k, v in sys.modules.items() if k.startswith("app")}
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    from fastapi.testclient import TestClient
    from app.api import create_app
    app = create_app()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        for mod in list(sys.modules):
            if mod.startswith("app"):
                del sys.modules[mod]
        sys.modules.update(saved)


def test_health_and_system(client):
    assert client.get("/api/health").json()["ok"]
    sysd = client.get("/api/system").json()
    assert "cpu" in sysd


def test_settings_roundtrip(client):
    s = client.get("/api/settings").json()
    assert s["merge_threshold"] == pytest.approx(0.62)
    s["merge_threshold"] = 0.7
    r = client.post("/api/settings", json=s)
    assert r.status_code == 200
    assert client.get("/api/settings").json()["merge_threshold"] == pytest.approx(0.7)
    # 非法值
    bad = dict(s, merge_threshold=5)
    assert client.post("/api/settings", json=bad).status_code == 400


def test_create_job_requires_input(client):
    r = client.post("/api/jobs", data={})
    assert r.status_code == 400


def test_create_job_with_missing_file(client):
    r = client.post("/api/jobs", data={"path": "Z:/not/exist.mp4"})
    assert r.status_code == 400


def test_events_and_monitor(client):
    assert "events" in client.get("/api/events").json()
    mon = client.get("/api/monitor").json()
    assert "hist_cpu" in mon


def test_env_and_models(client):
    assert isinstance(client.get("/api/env").json()["deps"], list)
    models = client.get("/api/models").json()["models"]
    assert any(m["kind"] == "vad" for m in models)


def test_voiceprint_crud(client):
    r = client.post("/api/voiceprints", json={"name": "测试者", "embedding": [0.1] * 192})
    assert r.status_code == 200
    vid = r.json()["voiceprint"]["id"]
    items = client.get("/api/voiceprints").json()["voiceprints"]
    assert len(items) == 1 and "embedding" not in items[0]
    assert client.patch(f"/api/voiceprints/{vid}", json={"name": "改名"}).status_code == 200
    assert client.delete(f"/api/voiceprints/{vid}").status_code == 200
    assert client.get("/api/voiceprints").json()["voiceprints"] == []


def test_export_endpoints_validation(client):
    assert client.get("/api/exports/nope").status_code == 404
    assert client.post("/api/jobs/nope/exports/plan", json={"track": "speaker_01"}).status_code == 404
    assert client.post("/api/jobs/nope/exports", json={"track": "speaker_01"}).status_code == 404
    assert client.get("/api/jobs/nope/source").status_code == 404
    assert client.get("/api/jobs/nope/exports").status_code == 404
    assert client.post("/api/exports/nope/cancel").status_code == 400
    assert client.delete("/api/exports/nope").status_code == 400


def test_export_page_served(client):
    r = client.get("/export.html")
    assert r.status_code == 200
    assert "导出片段" in r.text
    assert client.get("/export.js").status_code == 200

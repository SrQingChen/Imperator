"""网络探测、镜像方案、断点续传下载器（按域名路由镜像策略）。"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from . import config, events

_PROBE_TARGETS = {
    "google": "https://www.google.com/generate_204",
    "github": "https://github.com",
    "baidu": "https://www.baidu.com",
    "gitee": "https://gitee.com",
}
_GH_MIRRORS = ["https://ghproxy.net/", "https://gh-proxy.com/"]
_HF_MIRROR = "https://hf-mirror.com"
_UA = "Mozilla/5.0 (Imperator/1.0; +https://github.com/SrQingChen/Imperator)"


@dataclass
class MirrorPlan:
    region: str = "unknown"           # china / direct / unknown
    reason: str = ""
    pip_index: str | None = None
    hf_endpoint: str | None = None
    gh_prefixes: list[str] = field(default_factory=list)
    probes: dict = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "region": self.region, "reason": self.reason, "pip_index": self.pip_index,
            "hf_endpoint": self.hf_endpoint, "gh_prefixes": self.gh_prefixes,
            "probes": self.probes, "_ts": self.ts,
        }


_plan: MirrorPlan | None = None
_plan_lock = threading.Lock()


def _head(url: str, timeout: float = 2.0) -> tuple[bool, float]:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA}, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True, time.time() - t0
    except Exception:
        try:  # 某些镜像不支持 HEAD，退回 GET 小范围
            req = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=timeout):
                return True, time.time() - t0
        except Exception:
            return False, timeout


def _ipapi_country() -> str:
    try:
        with urllib.request.urlopen("http://ip-api.com/json/?fields=countryCode", timeout=3) as r:
            return json.loads(r.read().decode()).get("countryCode", "")
    except Exception:
        return ""


def probe(force: bool = False, quiet: bool = False) -> MirrorPlan:
    """并发探测网络环境，产出镜像方案；结果缓存 1 小时（可强制重测）。"""
    global _plan
    with _plan_lock:
        if not force and _plan is not None and time.time() - _plan.ts < 3600:
            return _plan
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut = {k: ex.submit(_head, u) for k, u in _PROBE_TARGETS.items()}
            probes = {k: {"ok": ok, "ms": int(ms * 1000)} for k, (ok, ms) in
                      ((k, f.result()) for k, f in fut.items())}
        g_ok = probes["google"]["ok"] or probes["github"]["ok"]
        cn_ok = probes["baidu"]["ok"] or probes["gitee"]["ok"]
        if g_ok and cn_ok:
            region, reason = "direct", "境外站点可达，使用官方源"
        elif cn_ok and not g_ok:
            region, reason = "china", "境外站点不可达而国内站点可达，启用镜像源"
        else:
            country = _ipapi_country()
            if country == "CN":
                region, reason = "china", f"连通性探测失败，出口 IP 属地 {country}，启用镜像源"
            elif country:
                region, reason = "direct", f"连通性探测失败，出口 IP 属地 {country}，使用官方源"
            else:
                region, reason = "china", "无法判定网络环境，保守启用镜像源（镜像全球可用）"
        plan = MirrorPlan(
            region=region, reason=reason, probes=probes, ts=time.time(),
            pip_index="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" if region == "china" else None,
            hf_endpoint=_HF_MIRROR if region == "china" else None,
            gh_prefixes=_GH_MIRRORS if region == "china" else [],
        )
        if region == "china" and plan.hf_endpoint:
            os.environ["HF_ENDPOINT"] = plan.hf_endpoint  # huggingface_hub 全链路生效
        _plan = plan
        try:
            config.NETWORK_STATE_JSON.write_text(
                json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass
        if not quiet:
            events.info(f"网络探测：{reason}", scope="network")
        return plan


def get_plan() -> MirrorPlan:
    global _plan
    if _plan is None:
        cached = None
        try:
            if config.NETWORK_STATE_JSON.exists():
                cached = json.loads(config.NETWORK_STATE_JSON.read_text("utf-8"))
        except Exception:
            cached = None
        if cached and time.time() - cached.get("_ts", 0) < 3600 and cached.get("region") in ("china", "direct"):
            p = MirrorPlan(**{k: cached.get(k) for k in
                              ("region", "reason", "pip_index", "hf_endpoint", "gh_prefixes", "probes")},
                           ts=cached.get("_ts", time.time()))
            if p.region == "china" and p.hf_endpoint:
                os.environ["HF_ENDPOINT"] = p.hf_endpoint
            _plan = p
        else:
            _plan = probe(quiet=False)
    return _plan


# ---------------------------------------------------------------- 下载器

def _route_urls(url: str, plan: MirrorPlan) -> list[str]:
    host = urlparse(url).netloc
    if "github.com" in host or "githubusercontent.com" in host:
        return [pre + url for pre in plan.gh_prefixes] + [url]
    if "huggingface.co" in host and plan.hf_endpoint:
        return [url.replace("https://huggingface.co", plan.hf_endpoint, 1), url]
    return [url]


def _remote_size(url: str, timeout: float = 15.0) -> int:
    """Range 探测远端体积（兼容不支持 HEAD 的镜像）；失败返回 -1。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                total = cr.rsplit("/", 1)[-1]
                if total.isdigit():
                    return int(total)
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > 1:
                return int(cl)
    except Exception:
        pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA}, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except Exception:
        pass
    return -1


class DownloadError(Exception):
    pass


def download(url: str, dest: Path, expected_size: int | None = None,
             progress_cb=None, log_prefix: str = "") -> Path:
    """断点续传下载：.part + Range + 体积校验 + 域名路由镜像 + text/html 错误页防写。"""
    import shutil as _sh

    plan = get_plan()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    last_err: Exception | None = None
    for attempt_url in _route_urls(url, plan):
        try:
            remote = _remote_size(attempt_url)
            total = expected_size if expected_size and expected_size > 0 else (remote if remote > 0 else 0)
            part = dest.with_suffix(dest.suffix + ".part")
            have = part.stat().st_size if part.exists() else 0
            if total and have >= total:
                part.unlink()
                have = 0
            headers = {"User-Agent": _UA}
            if have > 0:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(attempt_url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "text/html" in ctype and have == 0:
                    raise DownloadError(f"镜像返回了 HTML 错误页：{attempt_url}")
                mode = "ab" if have > 0 and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                done = have
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb:
                            progress_cb(done, total)
            size = part.stat().st_size
            if total and size != total:
                raise DownloadError(f"体积校验失败：{size} != {total}（{attempt_url}）")
            if expected_size and size != expected_size:
                raise DownloadError(f"体积与清单不符：{size} != {expected_size}")
            _sh.move(str(part), str(dest))
            if log_prefix:
                events.success(f"{log_prefix} 下载完成 {dest.name}（{size/1048576:.1f} MB）", scope="models")
            return dest
        except Exception as e:  # 换下一个镜像重试
            last_err = e
            if not quiet_log():
                events.warn(f"下载失败（{attempt_url}）：{e}，尝试下一个源", scope="models")
    raise DownloadError(f"全部下载源失败：{url}，最后错误：{last_err}")


def quiet_log() -> bool:
    return os.environ.get("IMPERATOR_QUIET_DOWNLOAD_LOG") == "1"

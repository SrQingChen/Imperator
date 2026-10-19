"""QC 报告与时间线导出：timeline.json / report.html（全部 HTML 转义防注入）。"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


_HTML_HEAD = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>英白拉多 · 分离报告</title>
<style>
 body{background:#14161c;color:#d7dae0;font-family:"Segoe UI",system-ui,sans-serif;margin:0;padding:32px;}
 h1{color:#e8c97a;font-weight:600;} h2{color:#e8c97a;border-bottom:1px solid #2a2e38;padding-bottom:6px;}
 table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px;}
 th,td{border:1px solid #2a2e38;padding:6px 10px;text-align:left;}
 th{background:#1b1e26;color:#e8c97a;}
 .ok{color:#7fd18a}.bad{color:#e07a6a}.warn{color:#e8c97a}
 .muted{color:#8a8f9a} code{background:#1b1e26;padding:1px 5px;border-radius:3px;}
</style></head><body>"""


def build_timeline(job_meta: dict, ledger_tracks: list[dict], speakers: list[dict],
                   segments: list[dict], audit: list[dict], qc: dict,
                   duration_s: float, sr_out: int) -> dict:
    return {
        "generated_at": time.time(),
        "job": job_meta,
        "duration_s": round(duration_s, 3),
        "sr_out": sr_out,
        "tracks": ledger_tracks,
        "speakers": speakers,
        "segments": segments,
        "audit": audit,
        "qc": qc,
    }


def save_timeline(tl: dict, out_dir: Path) -> Path:
    p = out_dir / "timeline.json"
    p.write_text(json.dumps(tl, ensure_ascii=False, indent=2, default=float), "utf-8")
    return p


def render_html(tl: dict) -> str:
    qc = tl.get("qc", {})
    rows_spk = "".join(
        f"<tr><td>{_e(s['name'])}</td><td>{s['duration_s']:.1f}s</td><td>{s['share_pct']:.1f}%</td>"
        f"<td>{_e('；'.join(s.get('flags', [])) or '—')}</td></tr>"
        for s in tl.get("speakers", []))
    rows_seg = "".join(
        f"<tr><td>{s['start']:.2f} – {s['end']:.2f}</td><td>{_e(s.get('label','—'))}</td>"
        f"<td>{_e(s.get('kind',''))}</td><td>{s.get('top1',0):.3f}</td><td>{s.get('margin',0):.3f}</td></tr>"
        for s in tl.get("segments", [])[:400])
    audit_html = "".join(
        f"<li><code>{time.strftime('%H:%M:%S', time.localtime(a.get('ts', 0)))}</code> "
        f"<b>{_e(a.get('action'))}</b>：{_e(a.get('detail'))}</li>" for a in tl.get("audit", [])) or "<li>无</li>"
    passed = bool(qc.get("verify_pass", qc.get("sum_pass")))
    sum_cls = "ok" if passed else "bad"
    tracks_rows = "".join(
        f"<tr><td>{_e(t['label'])}</td><td><code>{_e(t['file'])}</code></td></tr>" for t in tl.get("tracks", []))
    return f"""{_HTML_HEAD}
<h1>英白拉多 Imperator · 分离报告</h1>
<p class="muted">{_e(tl['job'].get('name',''))} · 时长 {tl['duration_s']:.1f}s · 采样率 {tl['sr_out']} Hz ·
生成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tl.get('generated_at', time.time())))}</p>
<h2>输出契约</h2>
<p>所有音轨（说话人 + BGM + 未归属 + 残差）逐样本相加 = 原始音频。
Σ 最大偏差：<span class="{sum_cls}">{qc.get('verify_sum_max_abs_err', qc.get('sum_max_abs_err'))}</span></p>
<h2>说话人</h2>
<table><tr><th>名称</th><th>时长</th><th>占比</th><th>标记</th></tr>{rows_spk}</table>
<h2>轨道</h2>
<table><tr><th>名称</th><th>文件</th></tr>{tracks_rows}</table>
<h2>时间线（前 400 段）</h2>
<table><tr><th>区间 (s)</th><th>说话人</th><th>类型</th><th>top1</th><th>margin</th></tr>{rows_seg}</table>
<h2>审计日志</h2>
<ul>{audit_html}</ul>
<h2>QC 明细</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>残差能量占比</td><td>{qc.get('residual_energy_ratio', 0):.4f}</td></tr>
<tr><td>负补集采样数</td><td>{qc.get('neg_complement_samples', 0)}</td></tr>
<tr><td>vocals/inst 长度补偿</td><td>{qc.get('vocals_len_adjusted')} / {qc.get('inst_len_adjusted')}</td></tr>
</table>
<p class="muted">本报告由英白拉多 Imperator 自动生成（GPL-3.0）。</p>
</body></html>"""


def save_html(html_text: str, out_dir: Path) -> Path:
    p = out_dir / "report.html"
    p.write_text(html_text, "utf-8")
    return p

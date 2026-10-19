/* 英白拉多 Imperator · 前端逻辑（原生 JS，轮询 + 游标增量） */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (s) => s == null ? "—" : (s >= 90 ? Math.round(s / 60) + "m" : Math.round(s) + "s");

const SPK_COLORS = ["#e8c97a", "#7fb3d8", "#c793d8", "#8fd1a8", "#d89393", "#9aa8e8"];
const state = {
  eventsSeq: 0, jobs: [], resultJob: null, timeline: null,
  selectedSeg: null, moves: new Map(), solo: null, settings: {},
  lastJobsJson: "", net: null, resultData: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ================= 标签页 ================= */
$$("#tabs button").forEach(b => b.onclick = () => {
  $$("#tabs button").forEach(x => x.classList.toggle("active", x === b));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "monitor") drawGauges();
});

/* ================= 日志 ================= */
function pushEvents(evs) {
  const con = $("#log-console");
  const nearBottom = con.scrollHeight - con.scrollTop - con.clientHeight < 40;
  for (const e of evs) {
    const div = document.createElement("div");
    div.className = `lv-${e.level}`;
    const t = new Date(e.ts * 1000).toTimeString().slice(0, 8);
    div.innerHTML = `<span class="t">${t}</span>${e.scope ? `[${esc(e.scope)}] ` : ""}${esc(e.msg)}`;
    con.appendChild(div);
    state.eventsSeq = Math.max(state.eventsSeq, e.seq);
  }
  while (con.children.length > 400) con.removeChild(con.firstChild);
  if (nearBottom) con.scrollTop = con.scrollHeight;
}

/* ================= 新建任务 ================= */
const drop = $("#drop");
drop.onclick = () => $("#file-input").click();
$("#file-input").onchange = (e) => { if (e.target.files[0]) $("#submit-hint").textContent = "已选择：" + e.target.files[0].name; };
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) { $("#file-input").files = e.dataTransfer.files; $("#submit-hint").textContent = "已选择：" + f.name; }
});
$("#opt-tau").oninput = (e) => $("#opt-tau-val").textContent = e.target.value;

$("#btn-submit").onclick = async () => {
  const fd = new FormData();
  const f = $("#file-input").files[0];
  const path = $("#path-input").value.trim();
  if (f) fd.append("file", f);
  else if (path) fd.append("path", path);
  else { $("#submit-hint").textContent = "请先选择文件或填写路径"; return; }
  const add = (k, v) => { if (v !== "" && v != null) fd.append(k, v); };
  add("tier", $("#opt-tier").value);
  add("outlier_mode", $("#opt-outlier").value);
  add("merge_threshold", $("#opt-tau").value !== "0.62" ? $("#opt-tau").value : "");
  try {
    $("#btn-submit").disabled = true;
    const res = await fetch("/api/jobs", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "提交失败");
    $("#submit-hint").textContent = "已提交 ✓";
    $("#path-input").value = ""; $("#file-input").value = "";
    setTimeout(() => ($("#submit-hint").textContent = ""), 3000);
    pollJobs(true);
  } catch (e) { $("#submit-hint").textContent = "提交失败：" + e.message; }
  finally { $("#btn-submit").disabled = false; }
};

/* ================= 任务列表 ================= */
const STAGE_NAMES = { prepare: "准备", extract: "抽取", mss: "伴奏分离", vad: "检测", embed: "声纹", cluster: "聚类", overlap: "重叠", ledger: "重建", report: "报告" };
const STAGE_KEYS = Object.keys(STAGE_NAMES);

function renderJobs() {
  const wrap = $("#job-list");
  const json = JSON.stringify(state.jobs.map(j => [j.id, j.status, j.stage, Math.round(j.progress * 100), j.note]));
  if (json === state.lastJobsJson) { updateRunningNumbers(); return; }
  state.lastJobsJson = json;
  wrap.innerHTML = state.jobs.map(j => {
    const curIdx = STAGE_KEYS.indexOf(j.stage);
    const chips = STAGE_KEYS.map((k, i) =>
      `<div class="st ${j.status === "done" || (curIdx > i) ? "done" : (k === j.stage && j.status === "running") ? "cur" : ""}"
        title="${STAGE_NAMES[k]}"></div>`).join("");
    const eta = (j.status === "running" && j.eta_s) ? ` · 预计剩余 ${fmt(j.eta_s)}` : "";
    const rtf = j.status === "done" && j.result ? ` · RTF ${j.result.qc.rtf ?? "—"}` : "";
    return `<div class="job" id="job-${j.id}">
      <div class="job-head">
        <div><span class="name">${esc(j.name)}</span> <span class="job-id">${j.id}</span></div>
        <span class="badge ${j.status}">${{ queued: "排队中", running: "运行中", done: "完成", error: "失败", cancelled: "已取消" }[j.status] || j.status}</span>
      </div>
      <div class="stagebar">${chips}</div>
      <div class="pbar"><div style="width:${(j.progress * 100).toFixed(1)}%"></div></div>
      <div class="meta"><span>${j.stage ? STAGE_NAMES[j.stage] || j.stage : ""}${j.note ? " · " + esc(j.note) : ""}</span>${eta}${rtf}</div>
      ${j.error ? `<div class="err">${esc(j.error)}</div>` : ""}
      <div class="actions">
        ${j.status === "running" || j.status === "queued" ? `<button onclick="cancelJob('${j.id}')">取消</button>` : ""}
        ${j.status === "done" ? `<button onclick="showResult('${j.id}')">查看结果</button>` : ""}
        ${j.status !== "running" ? `<button onclick="deleteJob('${j.id}')">删除</button>` : ""}
      </div>
    </div>`;
  }).join("") || '<p class="muted">暂无任务</p>';
}

function updateRunningNumbers() { /* 轻量更新由全量渲染承担（json 变化才重绘） */ }

window.cancelJob = async (id) => { try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (e) { alert(e.message); } };
window.deleteJob = async (id) => {
  if (!confirm("删除任务及其输出文件？")) return;
  try { await api(`/api/jobs/${id}`, { method: "DELETE" }); if (state.resultJob === id) { state.resultJob = null; $("#result-card").classList.add("hidden"); } pollJobs(true); }
  catch (e) { alert(e.message); }
};

async function pollJobs(force = false) {
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    renderJobs();
    const running = state.jobs.find(j => j.status === "running");
    $("#monitor-job").innerHTML = running
      ? `<b>${esc(running.name)}</b> · ${STAGE_NAMES[running.stage] || running.stage} · ${(running.progress * 100).toFixed(0)}%
         ${running.eta_s ? " · 剩余 " + fmt(running.eta_s) : ""}<div class="pbar" style="margin-top:8px"><div style="width:${running.progress * 100}%"></div></div>`
      : "无运行中任务";
    if (!force) {
      const done = state.jobs.find(j => j.status === "done" && j.id === state.resultJob);
      if (done && state.resultData && done.note !== state._lastDoneNote) { /* 完成态由用户手动刷新结果 */ }
    }
  } catch (e) { /* 瞬时错误忽略 */ }
}

/* ================= 结果 ================= */
window.showResult = async (id) => {
  try {
    const j = await api(`/api/jobs/${id}`);
    if (!j.result) return;
    state.resultJob = id; state.resultData = j.result; state.moves = new Map(); state.selectedSeg = null;
    $("#result-card").classList.remove("hidden");
    $("#result-name").textContent = j.name;
    $("#btn-report").href = `/api/jobs/${id}/report`;
    renderQC(j.result.qc);
    renderTracks(j.result, id);
    renderSpeakers(j.result, id);
    drawTimeline(j.result);
    $("#seg-editor").classList.add("hidden");
    $("#btn-apply-moves").disabled = true;
    $("#result-card").scrollIntoView({ behavior: "smooth" });
  } catch (e) { alert("加载结果失败：" + e.message); }
};

function renderQC(qc) {
  if (!qc) { $("#qc-strip").innerHTML = ""; return; }
  const sumErr = qc.verify_sum_max_abs_err ?? qc.sum_max_abs_err;
  const pass = qc.verify_pass ?? qc.sum_pass;
  $("#qc-strip").innerHTML = `
    <div class="qc-chip ${pass ? "pass" : "fail"}">Σ 还原偏差 <b>${sumErr == null ? "—" : sumErr.toExponential(2)}</b> ${pass ? "✓" : "✗"}</div>
    <div class="qc-chip">残差能量占比 <b>${(qc.residual_energy_ratio ?? 0).toFixed(4)}</b></div>
    <div class="qc-chip">说话人 <b>${state.resultData.speakers.length}</b></div>
    <div class="qc-chip">重叠段 <b>${state.resultData.segments.filter(s => s.kind === "overlap").length}</b></div>
    <div class="qc-chip">耗时 <b>${qc.elapsed_s ?? "—"}s</b> · RTF <b>${qc.rtf ?? "—"}</b></div>
    ${qc.neg_complement_samples ? `<div class="qc-chip">负补集采样 ${qc.neg_complement_samples}</div>` : ""}`;
}

function trackColor(key) {
  if (key.startsWith("speaker")) return SPK_COLORS[(parseInt(key.slice(7)) - 1) % SPK_COLORS.length];
  if (key === "bgm") return "var(--bgm)";
  if (key === "other") return "var(--other)";
  return "var(--res)";
}

function renderTracks(result, jid) {
  state.solo = null;
  $("#track-list").innerHTML = result.tracks.map(t => `
    <div class="track" id="tr-${t.key}">
      <span class="tname"><span class="dot" style="background:${trackColor(t.key)}"></span>${esc(t.label)}</span>
      <audio controls preload="none" src="/outputs/${jid}/${t.file}" ${t.key === "residual" || t.key === "other" ? "volume=\"0\"" : ""}></audio>
      <div class="tctl">
        <button onclick="soloTrack('${t.key}')">solo</button>
        <a class="btn" href="/outputs/${jid}/${t.file}" download>下载</a>
      </div>
    </div>`).join("");
}

window.soloTrack = (key) => {
  state.solo = state.solo === key ? null : key;
  $$("#track-list .track").forEach(el => {
    const k = el.id.slice(3);
    const audio = $("audio", el);
    audio.muted = state.solo ? k !== state.solo : false;
    el.classList.toggle("soloed", state.solo === k);
  });
};

$("#btn-sync-play").onclick = () => {
  const auds = $$("#track-list audio");
  const main = auds.find(a => !a.muted) || auds[0];
  auds.forEach(a => { a.currentTime = main.currentTime; a.play().catch(() => {}); });
};
$("#btn-sync-stop").onclick = () => $$("#track-list audio").forEach(a => a.pause());

function renderSpeakers(result, jid) {
  $("#speaker-cards").innerHTML = result.speakers.map(sp => `
    <div class="spcard">
      <span class="spname"><span class="dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${SPK_COLORS[sp.id % SPK_COLORS.length]};margin-right:6px"></span>${esc(sp.name)}</span>
      <span class="spmeta">${sp.duration_s.toFixed(1)}s · 占比 ${sp.share_pct}%${sp.flags?.length ? " · " + esc(sp.flags.join("，")) : ""}</span>
      <span class="row" style="margin:0">
        <input type="text" id="spname-${sp.id}" placeholder="命名后入库" value="${esc(sp.name)}">
        <button onclick="saveVoiceprint('${jid}', ${sp.id})">存入声纹库</button>
      </span>
    </div>`).join("");
}

window.saveVoiceprint = async (jid, sid) => {
  const name = $(`#spname-${sid}`).value.trim();
  if (!name) { alert("请先输入名称"); return; }
  try {
    const sp = state.resultData.speakers.find(s => s.id === sid);
    await api("/api/jobs/" + jid + "/voiceprint", { method: "POST", body: JSON.stringify({ speaker: sid, name }) });
    sp.name = name;
    renderSpeakers(state.resultData, jid);
  } catch (e) { alert("入库失败：" + e.message); }
};

/* ---------- 时间线 ---------- */
function drawTimeline(result) {
  const cv = $("#timeline");
  const W = cv.clientWidth || 600;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = 140 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, 140);
  const dur = Math.max(result.duration_s, 1);
  const rows = result.speakers.length + 1;
  const rh = Math.min(22, 120 / rows), top = 8;
  const x = (t) => (t / dur) * W;
  ctx.font = "10px sans-serif";
  result.speakers.forEach((sp, i) => {
    ctx.fillStyle = SPK_COLORS[i % SPK_COLORS.length];
    ctx.fillText(sp.name.slice(0, 8), 4, top + i * rh + 9);
  });
  ctx.fillStyle = "#8a8f9a";
  ctx.fillText("other", 4, top + result.speakers.length * rh + 9);
  for (const seg of result.segments) {
    if (seg.kind === "speech" && seg.speaker >= 0) {
      const i = result.speakers.findIndex(s => s.id === seg.speaker);
      if (i < 0) continue;
      ctx.fillStyle = SPK_COLORS[i % SPK_COLORS.length];
      ctx.fillRect(60 + x(seg.start), top + i * rh, Math.max(1, x(seg.end - seg.start)), rh - 3);
    } else if (seg.kind === "overlap" && seg.speakers) {
      seg.speakers.forEach(sid => {
        const i = result.speakers.findIndex(s => s.id === sid);
        if (i < 0) return;
        ctx.fillStyle = SPK_COLORS[i % SPK_COLORS.length];
        ctx.globalAlpha = 0.65;
        ctx.fillRect(60 + x(seg.start), top + i * rh, Math.max(1, x(seg.end - seg.start)), rh - 3);
        ctx.globalAlpha = 1;
      });
    } else if (seg.speaker < 0) {
      ctx.fillStyle = "#5a5f6b";
      ctx.fillRect(60 + x(seg.start), top + result.speakers.length * rh, Math.max(1, x(seg.end - seg.start)), rh - 3);
    }
  }
  // 刻度
  ctx.fillStyle = "#565d6b";
  const step = dur > 300 ? 60 : dur > 60 ? 10 : 5;
  for (let t = 0; t <= dur; t += step) {
    ctx.fillText(t < 60 ? t + "s" : Math.round(t / 60) + "m", 60 + x(t) - 6, 134);
  }
  if (state.selectedSeg != null) {
    const seg = result.segments.find(s => s.id === state.selectedSeg);
    if (seg) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      ctx.strokeRect(60 + x(seg.start), 2, Math.max(2, x(seg.end - seg.start)), 130);
    }
  }
  cv.onclick = (e) => {
    const r = cv.getBoundingClientRect();
    const t = ((e.clientX - r.left - 60) / Math.max(r.width - 60, 1)) * dur;
    const seg = result.segments.find(s => t >= s.start && t < s.end);
    if (seg) selectSeg(seg);
  };
}

function selectSeg(seg) {
  state.selectedSeg = seg.id;
  $("#seg-editor").classList.remove("hidden");
  $("#seg-info").textContent = `段 #${seg.id} [${seg.start.toFixed(2)}s – ${seg.end.toFixed(2)}s] · ${seg.label} · ${seg.kind}` +
    (seg.top1 ? ` · top1=${seg.top1} margin=${seg.margin}` : "");
  const sel = $("#seg-target");
  sel.innerHTML = state.resultData.speakers.map(s => `<option value="${s.id}">${esc(s.name)}</option>`).join("") +
    `<option value="unknown">→ 未归属（other）</option>`;
  if (seg.speaker >= 0) sel.value = seg.speaker;
  sel.onchange = () => {
    state.moves.set(seg.id, { segment: seg.id, speaker: sel.value === "unknown" ? "unknown" : parseInt(sel.value) });
    $("#btn-apply-moves").disabled = state.moves.size === 0;
    $("#btn-apply-moves").textContent = `应用人工修正（${state.moves.size} 处）`;
  };
  drawTimeline(state.resultData);
}

$("#btn-apply-moves").onclick = async () => {
  if (!state.moves.size || !state.resultJob) return;
  try {
    $("#btn-apply-moves").disabled = true;
    const data = await api(`/api/jobs/${state.resultJob}/corrections`,
      { method: "POST", body: JSON.stringify({ moves: [...state.moves.values()] }) });
    state.resultData = data.result; state.moves = new Map();
    $("#btn-apply-moves").textContent = "应用人工修正";
    renderQC(data.result.qc); renderTracks(data.result, state.resultJob);
    renderSpeakers(data.result, state.resultJob); drawTimeline(data.result);
    $("#seg-editor").classList.add("hidden");
  } catch (e) { alert("应用失败：" + e.message); $("#btn-apply-moves").disabled = false; }
};

/* ================= 设置 ================= */
async function loadSettings() {
  try {
    state.settings = await api("/api/settings");
    for (const [k, v] of Object.entries(state.settings)) {
      const el = $(`#s-${k}`);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    }
    $("#opt-tau").value = state.settings.merge_threshold;
    $("#opt-tau-val").textContent = state.settings.merge_threshold;
  } catch (e) { /* 忽略 */ }
}
$("#btn-save-settings").onclick = async () => {
  const body = {};
  for (const k of Object.keys(state.settings)) {
    const el = $(`#s-${k}`);
    if (!el) continue;
    let v = el.type === "checkbox" ? el.checked : el.value;
    if (typeof state.settings[k] === "number") v = parseFloat(v);
    if (state.settings[k] === true || state.settings[k] === false) v = el.type === "checkbox" ? el.checked : v === "true";
    body[k] = v;
  }
  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    $("#settings-msg").textContent = "已保存 ✓";
    setTimeout(() => ($("#settings-msg").textContent = ""), 2500);
    loadSettings();
  } catch (e) { $("#settings-msg").textContent = "保存失败：" + e.message; }
};

/* ================= 模型与环境 ================= */
async function loadEnv() {
  try {
    const data = await api("/api/env");
    $("#dep-table tbody").innerHTML = data.deps.map(d => `
      <tr><td>${esc(d.module)}</td><td>${esc(d.purpose)}</td>
      <td class="${d.status === "ok" ? "tag-ok" : "tag-missing"}">${d.status === "ok" ? "✓ " + esc(d.version || "") : "✗ 缺失"}</td>
      <td>${esc(d.pip)}</td></tr>`).join("");
    const missing = data.deps.filter(d => d.status === "missing").length;
    $("#btn-install-deps").textContent = missing ? `一键安装缺失依赖（${missing} 项）` : "依赖完整";
    $("#btn-install-deps").disabled = !missing;
    if (data.install.status === "running") {
      $("#install-log").classList.remove("hidden");
      $("#install-log").textContent = (data.install.log || []).join("\n").slice(-3000);
    } else if (data.install.status === "error") {
      $("#install-log").classList.remove("hidden");
      $("#install-log").textContent = "安装失败：" + data.install.error;
    }
  } catch (e) { /* 忽略 */ }
}
$("#btn-install-deps").onclick = async () => {
  try { await api("/api/env/install", { method: "POST", body: JSON.stringify({}) }); $("#install-log").classList.remove("hidden"); $("#install-log").textContent = "安装已启动…"; }
  catch (e) { alert(e.message); }
};

async function loadModels() {
  try {
    const data = await api("/api/models");
    $("#model-list").innerHTML = data.models.map(m => {
      const dl = m.download || {};
      const stat = dl.status === "running"
        ? `<div class="dl"><div class="pbar"><div style="width:${(dl.progress * 100).toFixed(0)}%"></div></div>
           <span class="mstat">${dl.done_mb} MB${dl.total_mb ? " / " + dl.total_mb + " MB" : ""}…</span></div>`
        : dl.status === "error" ? `<span class="mstat" style="color:var(--err)">失败：${esc(dl.error)}</span>` : "";
      const btn = m.installed
        ? `<span class="mstat" style="color:var(--ok)">✓ 已就绪</span>`
        : dl.status === "running" ? `<button disabled>下载中…</button>`
        : `<button class="primary" onclick="dlModel('${esc(m.key)}')">下载</button>`;
      return `<div class="model"><div><div class="mname">${esc(m.name)}</div>
        <div class="msrc">${esc(m.source)}${m.tier ? " · 档位 " + m.tier : ""}</div></div>${stat}${btn}</div>`;
    }).join("");
  } catch (e) { /* 忽略 */ }
}
window.dlModel = async (key) => {
  try { await api("/api/models/download", { method: "POST", body: JSON.stringify({ key }) }); loadModels(); }
  catch (e) { alert(e.message); }
};

async function loadNet() {
  try {
    state.net = await api("/api/network");
    const b = $("#net-badge");
    b.textContent = state.net.region === "china" ? "镜像源 · 国内" : state.net.region === "direct" ? "官方源 · 直连" : "未知网络";
    b.className = "net-badge " + (state.net.region === "china" ? "cn" : state.net.direct || state.net.region === "direct" ? "direct" : "");
    $("#net-detail").textContent = state.net.reason + " · pip：" + (state.net.pip_index || "官方源") +
      " · HF：" + (state.net.hf_endpoint || "官方") + " · GitHub：" + (state.net.gh_prefixes?.length ? "镜像加速" : "直连");
  } catch (e) { /* 忽略 */ }
}
$("#btn-net-probe").onclick = async () => {
  $("#net-detail").textContent = "探测中…";
  state.net = await api("/api/network/probe", { method: "POST" });
  loadNet();
};

/* ================= 监控 ================= */
function drawSpark(cv, hist, color) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 180, H = cv.clientHeight || 56;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#2a2e38"; ctx.beginPath(); ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1); ctx.stroke();
  if (!hist || hist.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * W;
    const y = H - 2 - Math.min(1, v / 100) * (H - 6);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}
function drawGauges() {
  api("/api/monitor").then(m => {
    $("#g-cpu").textContent = m.cpu_pct != null ? m.cpu_pct.toFixed(0) + "%" : "—";
    $("#g-ram").textContent = m.ram_pct != null ? `${m.ram_pct.toFixed(0)}% (${m.ram_used_gb}/${m.ram_total_gb}G)` : "—";
    $("#g-gpu").textContent = m.gpu_util_pct != null ? m.gpu_util_pct + "%" : "无 GPU";
    $("#g-vram").textContent = m.gpu_mem_total_mb ? `${(m.gpu_mem_used_mb / 1024).toFixed(1)}/${(m.gpu_mem_total_mb / 1024).toFixed(1)}G` : "—";
    $$(".gauge canvas").forEach(cv => {
      const key = cv.dataset.hist;
      drawSpark(cv, m[key], key.includes("gpu") || key.includes("vram") ? "#7fb3d8" : "#e8c97a");
    });
  }).catch(() => {});
}

/* ================= 声纹库 ================= */
async function loadVps() {
  try {
    const data = await api("/api/voiceprints");
    $("#vp-table tbody").innerHTML = data.voiceprints.map(v => `
      <tr><td><input type="text" value="${esc(v.name)}" onchange="vpUpdate('${v.id}', 'name', this.value)"></td>
      <td>${v.dim || "—"}</td>
      <td>${esc(v.note || "")} <span class="muted">${v.id}</span></td>
      <td><input type="checkbox" ${v.enabled ? "checked" : ""} onchange="vpUpdate('${v.id}', 'enabled', this.checked)"></td>
      <td><button onclick="vpDelete('${v.id}')">删除</button></td></tr>`).join("") || '<tr><td colspan="5" class="muted">暂无声纹</td></tr>';
  } catch (e) { /* 忽略 */ }
}
window.vpUpdate = async (id, k, v) => {
  try { await api("/api/voiceprints/" + id, { method: "PATCH", body: JSON.stringify({ [k]: v }) }); loadVps(); }
  catch (e) { alert(e.message); }
};
window.vpDelete = async (id) => {
  if (!confirm("删除该声纹？")) return;
  try { await api("/api/voiceprints/" + id, { method: "DELETE" }); loadVps(); } catch (e) { alert(e.message); }
};
$("#btn-vp-export").onclick = () => { window.open("/api/voiceprints/export"); };
$("#vp-import-file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  const res = await fetch("/api/voiceprints/import", { method: "POST", body: fd });
  const data = await res.json();
  alert(res.ok ? `已导入 ${data.imported} 条` : "导入失败：" + (data.detail || ""));
  loadVps();
};
$("#btn-vp-merge").onclick = async () => {
  const ids = $("#vp-merge-ids").value.split(",").map(s => s.trim()).filter(Boolean);
  if (ids.length !== 2) { alert("请填写：目标id,来源id"); return; }
  try { await api("/api/voiceprints/merge", { method: "POST", body: JSON.stringify({ dst: ids[0], src: ids[1] }) }); loadVps(); }
  catch (e) { alert(e.message); }
};

/* ================= 主循环 ================= */
async function tick() {
  try {
    const evs = await api(`/api/events?since=${state.eventsSeq}`);
    if (evs.events.length) pushEvents(evs.events);
  } catch (e) { /* 忽略 */ }
  pollJobs();
}
window.addEventListener("resize", () => { if (state.resultData) drawTimeline(state.resultData); });

(async function init() {
  loadSettings(); loadEnv(); loadModels(); loadNet(); loadVps();
  pollJobs(true);
  tick();
  setInterval(tick, 1000);
  setInterval(drawGauges, 2000);
  setInterval(() => { loadModels(); loadEnv(); }, 4000);
})();

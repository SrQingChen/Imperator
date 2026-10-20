/* 导出页逻辑：分析片段 → 试听/勾选 → 导出视频或音频 */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtT = (t) => {
  if (t == null || !isFinite(t)) return "—";
  const m = Math.floor(t / 60), s = t - m * 60;
  return m > 0 ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${s.toFixed(1)}s`;
};

const qs = new URLSearchParams(location.search);
const state = {
  jid: qs.get("job") || "",
  trackKey: qs.get("track") || "",
  kind: qs.get("kind") === "audio" ? "audio" : "video",
  job: null,
  plan: null,
  selected: new Set(),
  exportId: null,
  timer: null,
  planning: false,
  creating: false,
  planSeq: 0,
};
const LS_KEY = "imperator.export.params.v1";
const PARAM_IDS = ["threshold_db", "rel_db", "merge_gap", "min_clip", "pad_before", "pad_after"];
const DEFAULTS = { threshold_db: -45, rel_db: -30, merge_gap: 0.35, min_clip: 0.6,
                   pad_before: 0.15, pad_after: 0.25 };

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

function fatal(msg) {
  $("#fatal").classList.remove("hidden");
  $("#fatal-msg").textContent = "无法打开导出页：" + msg;
}

/* ---------------- 参数 ---------------- */
function loadParams() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch (e) {}
  for (const k of PARAM_IDS) {
    const el = $("#p-" + k);
    el.value = saved[k] ?? DEFAULTS[k];
    el.onchange = () => { saveParams(); schedulePlan(); };
  }
  $("#p-audio_format").value = saved.audio_format || "wav";
  $("#p-audio_format").onchange = saveParams;
}
function readParams() {
  const p = {};
  for (const k of PARAM_IDS) {
    const v = parseFloat($("#p-" + k).value);
    p[k] = isFinite(v) ? v : DEFAULTS[k];
  }
  return p;
}
function saveParams() {
  const data = readParams();
  data.audio_format = $("#p-audio_format").value;
  localStorage.setItem(LS_KEY, JSON.stringify(data));
}
function debounce(fn, ms = 500) {
  return (...a) => { clearTimeout(fn._t); fn._t = setTimeout(() => fn(...a), ms); };
}
const schedulePlan = debounce(() => plan(true), 550);

/* ---------------- 初始化 ---------------- */
async function init() {
  loadParams();
  if (!state.jid) { fatal("缺少 job 参数"); return; }
  try {
    state.job = await api("/api/jobs/" + encodeURIComponent(state.jid));
  } catch (e) { fatal(e.message); return; }
  if (!state.job.result) { fatal("任务尚未完成或结果已失效"); return; }
  $("#job-title").textContent = `${state.job.name} · ${fmtT(state.job.result.duration_s)} · ${state.job.id}`;

  const tracks = state.job.result.tracks.filter(t => t.kind !== "residual");
  if (!tracks.length) { fatal("没有可导出的音轨"); return; }
  if (!tracks.some(t => t.key === state.trackKey)) state.trackKey = tracks[0].key;
  const sel = $("#track-select");
  sel.innerHTML = tracks.map(t =>
    `<option value="${esc(t.key)}">${esc(t.label)}（${esc(t.key)}）</option>`).join("");
  sel.value = state.trackKey;
  sel.onchange = () => { state.trackKey = sel.value; updateAudio(); plan(true); };
  updateAudio();

  // 输出类型
  setKind(state.kind, false);
  $$("input[name=output_kind]").forEach(r => r.onchange = () => setKind(r.value));
  $$("input[name=audio_mode]").forEach(r => r.onchange = () => {
    cardSelect("mode", r.value === "clean" ? "clean" : "original");
  });
  $$("input[name=audio_scope]").forEach(r => r.onchange = () => {
    cardSelect("scope", r.value === "clips" ? "clips" : "full");
    renderClips();
  });
  $("#btn-plan").onclick = () => plan(true);
  $("#btn-all").onclick = () => { state.selected = new Set(state.plan.intervals.map((_, i) => i)); renderClips(); };
  $("#btn-none").onclick = () => { state.selected = new Set(); renderClips(); };
  $("#btn-generate").onclick = generate;

  // 原视频预览
  const v = $("#source-video");
  v.src = `/api/jobs/${encodeURIComponent(state.jid)}/source`;
  v.onerror = () => {
    $("#source-card").classList.add("hidden");
    $("#source-hint").textContent = "该输入没有可预览的视频流（纯音频输入仍可导出音频）。";
  };
  await plan(true);
}

function updateAudio() {
  const t = state.job.result.tracks.find(x => x.key === state.trackKey);
  if (!t) return;
  $("#track-audio").src = `/outputs/${encodeURIComponent(state.jid)}/${t.file}`;
  const sp = state.job.result.speakers.find(s => `speaker_${String(s.id + 1).padStart(2, "0")}` === t.key);
  $("#track-meta").textContent = sp
    ? `${sp.duration_s.toFixed(1)}s · 占比 ${sp.share_pct}%${sp.flags?.length ? " · " + sp.flags.join("，") : ""}`
    : "";
  $("#track-hint").textContent = t.kind === "bgm"
    ? "BGM 通常全程有声，切片会接近整段视频；建议用于“整段替换音频”场景。"
    : t.kind === "other"
      ? "未归属人声（歌声/呼吸/惊呼等）；可能有零散片段。"
      : "试听确认是目标说话人后再导出。";
}

function setKind(kind, schedule = true) {
  state.kind = kind;
  $$("input[name=output_kind]").forEach(r => r.checked = r.value === kind);
  $("#kind-video-card").classList.toggle("on", kind === "video");
  $("#kind-audio-card").classList.toggle("on", kind === "audio");
  $("#video-opts").classList.toggle("hidden", kind !== "video");
  $("#audio-opts").classList.toggle("hidden", kind !== "audio");
  if (kind === "audio") $("#source-card").classList.add("hidden");
  else $("#source-card").classList.remove("hidden");
  if (schedule) schedulePlan();
}

function cardSelect(group, value) {
  if (group === "mode") {
    $("#mode-clean-card").classList.toggle("on", value === "clean");
    $("#mode-original-card").classList.toggle("on", value === "original");
  } else {
    $("#scope-clips-card").classList.toggle("on", value === "clips");
    $("#scope-full-card").classList.toggle("on", value === "full");
  }
}

function currentMode() { return $("input[name=audio_mode]:checked")?.value || "clean"; }
function currentScope() { return $("input[name=audio_scope]:checked")?.value || "clips"; }

/* ---------------- 分析片段 ---------------- */
async function plan(force = false) {
  if (!state.job) return;
  const seq = ++state.planSeq;
  state.planning = true;
  $("#plan-hint").textContent = "分析中…";
  try {
    const body = { track: state.trackKey, output_kind: state.kind, ...readParams() };
    const data = await api(`/api/jobs/${encodeURIComponent(state.jid)}/exports/plan`,
      { method: "POST", body: JSON.stringify(body) });
    if (seq !== state.planSeq) return;  // 丢弃过期响应
    state.plan = data.plan;
    state.selected = new Set(data.plan.intervals.map((_, i) => i));
    renderPlanSummary();
    renderClips();
  } catch (e) {
    if (seq !== state.planSeq) return;
    state.plan = null;
    $("#plan-hint").textContent = "";
    $("#plan-summary").innerHTML = `<span class="err">分析失败：${esc(e.message)}</span>`;
    $("#clip-list").innerHTML = `<p class="muted" style="padding:12px">无片段</p>`;
    $("#clip-stats").textContent = "";
  } finally {
    if (seq === state.planSeq) { state.planning = false; $("#plan-hint").textContent = ""; }
  }
}

function renderPlanSummary() {
  const p = state.plan;
  if (!p) return;
  const src = p.source;
  const srcTxt = src
    ? `源视频 ${src.width}×${src.height} · ${src.fps || "?"}fps · ${fmtT(src.duration)}${src.has_audio ? "" : "（无音轨）"}`
    : "音频导出（不需要视频流）";
  $("#plan-summary").innerHTML =
    `检测到 <b>${p.intervals.length}</b> 段有声音 · 合计 <b>${fmtT(p.total)}</b> · 覆盖率 <b>${(p.coverage * 100).toFixed(1)}%</b><br>
     阈值 ${p.threshold_db} dBFS（95 分位 ${p.p95_db} dBFS）· ${esc(srcTxt)}`;
}

function renderClips() {
  const p = state.plan;
  const full = state.kind === "audio" && currentScope() === "full";
  const list = $("#clip-list");
  if (!p) { list.innerHTML = ""; return; }
  if (full) {
    $("#clip-hint").textContent = "完整音轨模式：导出与源等长的音频，无人声处为静音，无需勾选片段。";
    list.innerHTML = `<p class="muted" style="padding:12px">完整音轨模式（不裁剪片段）</p>`;
    $("#clip-stats").textContent = `· 全长 ${fmtT(p.track_duration)}`;
    updateGenerateState();
    return;
  }
  $("#clip-hint").textContent = "取消勾选误检片段；只有勾选的片段会被导出。";
  if (!p.intervals.length) {
    list.innerHTML = `<p class="muted" style="padding:12px">该音轨没有检测到有声音的片段，可降低能量阈值后重新分析。</p>`;
  } else {
    list.innerHTML = p.intervals.map(([s, e], i) => `
      <label class="clip ${state.selected.has(i) ? "" : "off"}" data-i="${i}">
        <input type="checkbox" ${state.selected.has(i) ? "checked" : ""}>
        <span>片段 ${i + 1}</span>
        <span class="time">${fmtT(s)} → ${fmtT(e)}</span>
        <span class="dur">${(e - s).toFixed(2)}s</span>
      </label>`).join("");
    $$("#clip-list .clip input").forEach(cb => cb.onchange = (ev) => {
      const row = ev.target.closest(".clip");
      const i = parseInt(row.dataset.i, 10);
      if (ev.target.checked) state.selected.add(i); else state.selected.delete(i);
      row.classList.toggle("off", !ev.target.checked);
      updateStats();
    });
    $$("#clip-list .clip").forEach(row => row.onclick = (ev) => {
      if (ev.target.tagName === "INPUT") return;
      const cb = $("input", row);
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change"));
    });
  }
  updateStats();
}

function selectedIntervals() {
  if (!state.plan) return [];
  if (state.kind === "audio" && currentScope() === "full") return [];
  return [...state.selected].sort((a, b) => a - b).map(i => state.plan.intervals[i]);
}

function updateStats() {
  const iv = selectedIntervals();
  const total = iv.reduce((a, [s, e]) => a + (e - s), 0);
  $("#clip-stats").textContent = `· 已选 ${iv.length}/${state.plan?.intervals.length ?? 0} 段 · 合计 ${fmtT(total)}`;
  updateGenerateState();
}

function updateGenerateState() {
  const ok = !!state.plan && (state.kind === "audio" && currentScope() === "full"
    ? true : selectedIntervals().length > 0);
  $("#btn-generate").disabled = !ok || !!state.exportId;
}

/* ---------------- 生成 ---------------- */
async function generate() {
  if (!state.plan || state.creating || state.exportId) return;
  state.creating = true;
  $("#btn-generate").disabled = true;
  const body = {
    track: state.trackKey,
    output_kind: state.kind,
    audio_mode: currentMode(),
    audio_scope: currentScope(),
    audio_format: $("#p-audio_format").value,
    ...readParams(),
    intervals: state.kind === "audio" && currentScope() === "full" ? null : selectedIntervals(),
  };
  $("#export-result").innerHTML = "";
  $("#export-status").innerHTML = '<p class="muted">提交导出任务…</p>';
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(state.jid)}/exports`,
      { method: "POST", body: JSON.stringify(body) });
    state.exportId = data.export.id;
    $("#btn-generate").disabled = true;
    pollExport();
  } catch (e) {
    $("#export-status").innerHTML = `<p class="err">提交失败：${esc(e.message)}</p>`;
    updateGenerateState();
  } finally {
    state.creating = false;
  }
}

async function pollExport() {
  clearTimeout(state.timer);
  if (!state.exportId) return;
  try {
    const t = await api("/api/exports/" + encodeURIComponent(state.exportId));
    renderExportStatus(t);
    if (t.status === "done") { renderExportResult(t); state.exportId = null; updateGenerateState(); return; }
    if (t.status === "error" || t.status === "cancelled") {
      state.exportId = null; updateGenerateState(); return;
    }
  } catch (e) {
    $("#export-status").innerHTML = `<p class="err">查询失败：${esc(e.message)}</p>`;
  }
  state.timer = setTimeout(pollExport, 800);
}

function renderExportStatus(t) {
  const statusMap = { queued: "排队中", running: "进行中", done: "完成", error: "失败", cancelled: "已取消" };
  const pct = Math.round((t.progress || 0) * 100);
  const canCancel = t.status === "running" || t.status === "queued";
  $("#export-status").innerHTML = `
    <div class="row" style="margin:0">
      <span class="badge ${t.status}">${statusMap[t.status] || t.status}</span>
      <span class="muted">${esc(t.stage_name || "")} · ${esc(t.note || "")}</span>
      <span style="flex:1"></span>
      ${canCancel ? `<button id="btn-cancel-export">取消导出</button>` : ""}
    </div>
    <div class="pbar"><div style="width:${pct}%"></div></div>
    <p class="hint">${pct}%${t.status === "running" ? " · ffmpeg 渲染中，取消会立即终止当前片段" : ""}</p>`;
  const btn = $("#btn-cancel-export");
  if (btn) btn.onclick = async () => {
    try { await api(`/api/exports/${encodeURIComponent(t.id)}/cancel`, { method: "POST" }); }
    catch (e) { alert(e.message); }
  };
}

function renderExportResult(t) {
  const r = t.result || {};
  const isVideo = r.output_kind === "video";
  const media = isVideo
    ? `<video class="result-media" controls src="${esc(r.url)}"></video>`
    : `<audio style="width:100%;margin-top:10px" controls src="${esc(r.url)}"></audio>`;
  const modeTxt = isVideo
    ? (r.audio_mode === "clean" ? "音频：分离音轨（干净）" : "音频：原视频")
    : (r.audio_scope === "full" ? "范围：完整音轨" : `范围：${r.clips} 个片段拼接`) +
      ` · 格式：${(r.audio_format || "").toUpperCase()}`;
  $("#export-status").innerHTML = `<p class="okline">导出完成 ✓</p>`;
  $("#export-result").innerHTML = `
    ${media}
    <div class="row" style="margin-top:10px">
      <a class="btn primary" href="${esc(r.url)}" download="${esc(r.file)}">下载 ${esc(r.file)}</a>
      <span class="muted">${fmtT(r.duration)} · ${r.size_mb} MB · ${esc(modeTxt)}</span>
    </div>
    <p class="hint">文件位于 data/outputs/${esc(t.job_id)}/exports/${esc(t.id)}/</p>`;
}

init();

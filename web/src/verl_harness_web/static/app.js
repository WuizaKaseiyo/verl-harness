/* verl harness · editorial specimen dashboard.
   Same backend API; new front shell — masthead + banner + tabs + reader. */

import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs';

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const on = (el, ev, fn) => el && el.addEventListener(ev, fn);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const S = {
  config: { live: true },
  harness: null,
  run: null,
  view: 'metrics',
  current: null,
  activeStateSkills: new Set(),
  zoom: 1,
  charts: {},                        // panel-name → Chart instance (c)
  logOffset: 0,
  lastEventTs: Date.now(),
  startTs: null,
  runOrdinal: null,
  smoothing: 0,                      // 0..0.95 EMA factor (c)
  xAxisKey: 'training/global_step',  // (c)
  mapMode: 'phase',                  // 'phase' (6 nodes, default) | 'stage' (FSM for goal) | 'all' (16)
  goal: 'train',                     // (a) derived from training_intent.md
  runId: '',                         // user-picked run from #run-select; '' = latest by mtime
  runsList: [],                      // cached list from /api/runs
  navExpanded: { overview: true, states: false, skills: false },
  submitReady: false,
  submitTasksTimer: null,
};

/* Append ?run_id=… (or &run_id=…) to a per-run endpoint when the user has
   picked a specific run from the dropdown. Generic endpoints (/api/runs,
   /api/harness, /api/config, /api/state/, /api/skill, /api/file) skip this. */
function scoped(path) {
  if (!S.runId) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}run_id=${encodeURIComponent(S.runId)}`;
}

/* ─── (a) per-goal state set used by map filter ──────────────────────────── */
const TRACK_STATES = {
  train: ['intake', 'locate_recipe', 'configure_algorithm', 'prepare_data',
          'generate_preprocess', 'configure_reward', 'select_compute',
          'provision_env', 'sanity_rollout', 'launch_training',
          'monitor_training', 'summarize', 'reflect', 'finalize'],
  resume_monitor: ['intake', 'monitor_training', 'summarize', 'finalize'],
  resume_train:   ['intake', 'launch_training', 'monitor_training', 'summarize', 'finalize'],
  generate:       ['intake', 'select_compute', 'provision_env', 'run_generate', 'finalize'],
  eval:           ['intake', 'run_eval', 'finalize'],
};

/* ─── (a2) phase view — collapses the 14-state train track into 6 user-visible phases.
   When mapMode === 'phase' the graph renders these as 5 nodes (the default).
   A phase's status aggregates from its members: any live → live; if all visited
   → visited; if it contains a terminal that's reached → terminal. Click → first
   member's spec. Goals other than `train` are too small to collapse — they just
   fall back to the goal-filtered FSM. */
const PHASES = {
  train: [
    { id: 'intent',   label: 'Intent',   members: ['intake'] },
    { id: 'setup',    label: 'Setup',    members: ['locate_recipe', 'configure_algorithm',
                                                   'prepare_data', 'generate_preprocess',
                                                   'configure_reward'] },
    { id: 'compute',  label: 'Compute',  members: ['select_compute', 'provision_env', 'sanity_rollout'] },
    { id: 'train',    label: 'Train',    members: ['launch_training', 'monitor_training'] },
    { id: 'report',   label: 'Report',   members: ['summarize', 'reflect'] },
    { id: 'done',     label: 'Done',     members: ['finalize'] },
  ],
};

/* ─── (c) namespace groups for panel grid ─────────────────────────────────── */
const NS_GROUPS = [
  { name: 'training',       match: k => k.startsWith('training/') },
  /* actor.loss now also catches `pg_clipfrac_lower` (the `_lower` suffix was missed) */
  { name: 'actor · loss',   match: k => /^actor\/(loss|pg_loss|kl_loss|entropy|grad_norm|ppo_kl|kl_coef|pg_clipfrac(_lower)?|lr)\b/.test(k) },
  { name: 'actor · perf',   match: k => k.startsWith('actor/perf/') },
  /* split critic to give 'reward signal' its own card — was 12 lines crammed in one */
  { name: 'reward signal',  match: k => /^critic\/(score|rewards|advantages|returns)\//.test(k) },
  { name: 'critic',         match: k => k.startsWith('critic/') },
  /* merge val · core + val · aux → one `validation` panel */
  { name: 'validation',     match: k => k.startsWith('val-core/') || k.startsWith('val-aux/') },
  /* response_length{_non_aborted}/* + prompt_length/* */
  { name: 'response / prompt', match: k => /^(response_length(_non_aborted)?|response|prompt_length)\//.test(k) },
  { name: 'reward',         match: k => k.startsWith('reward/') },
  /* split heavy timing namespace: a small "throughput" panel for the headline steps,
     and a bigger "timing · stages" panel for everything else. Cap render to top-N */
  { name: 'timing · core',  match: k => /^timing_s\/(gen|step|update_actor|update_weights|ref|reward|old_log_prob|adv|save_checkpoint|testing)$/.test(k) },
  { name: 'timing · per token', match: k => k.startsWith('timing_per_token_ms/') },
  { name: 'timing · agent loop', match: k => k.startsWith('timing_s/agent_loop/') },
  { name: 'perf',           match: k => k.startsWith('perf/') && !k.startsWith('actor/perf/') },
  { name: 'turns / seqlen', match: k => k.startsWith('num_turns/') || k.startsWith('global_seqlen/') },
];

/* Drop noisy / duplicate columns the csv parser sometimes emits. */
const DROP_KEYS = new Set(['step']);
const MAX_SERIES_PER_PANEL = 10;

/* line colors for series within one panel — Claude design.md palette
   (coral primary, teal/amber accents, coral-active, ink, muted, semantic).
   Stays inside the cream+coral+dark-navy trinity; no purple, no saturated blue. */
const SERIES_PALETTE = [
  '#cc785c',  /* primary (coral)        */
  '#5db8a6',  /* accent-teal            */
  '#e8a55a',  /* accent-amber           */
  '#a9583e',  /* primary-active         */
  '#5db872',  /* success                */
  '#c64545',  /* error                  */
  '#141413',  /* ink                    */
  '#6c6a64',  /* muted                  */
  /* extended HSL-distributed steps for high-series panels (>= 9 lines) — stays in
     warm/teal range, avoids saturated blue/purple per Claude design.md */
  '#9b6850',  '#74a99a',  '#bf8a4a',  '#7c4b3a',  '#4f9586',
  '#a06c45',  '#5a8a7e',  '#b78f5e',  '#8e6042',  '#6d9e8d',
];

/* ════════════════════════════ boot ════════════════════════════ */
async function boot() {
  try { S.config = await getJSON('/api/config'); }
  catch { $('#masthead-run').textContent = '— api unreachable —'; return; }

  initMermaid();

  await reloadHarness();
  await loadRunOrdinal();
  await reloadRun();
  await renderGraph();
  renderSpecNav();
  paintBanner();

  $$('.tab[data-view]').forEach(btn =>
    on(btn, 'click', () => setView(btn.dataset.view)));

  // Banner CTA jumps to the Approvals tab
  on($('#hitl-banner-cta'), 'click', () => setView('hitl'));

  on($('#edit-btn'),  'click', openEditor);
  on($('#save-btn'),  'click', saveEdit);
  on($('#cancel-btn'),'click', () => { exitEditor(); reselect(); });
  on($('#zoom-in'),  'click', () => { S.zoom = Math.min(S.zoom + 0.1, 2.0); applyZoom(); });
  on($('#zoom-out'), 'click', () => { S.zoom = Math.max(S.zoom - 0.1, 0.5); applyZoom(); });
  on($('#zoom-level'), 'click', () => { S.zoom = 1.0; applyZoom(); });

  // (c) metrics controls
  on($('#smooth-slider'), 'input', (e) => {
    S.smoothing = Number(e.target.value) / 100;
    $('#smooth-value').textContent = `${e.target.value}%`;
    if (S.view === 'metrics') refreshMetrics();
  });
  on($('#x-axis-select'), 'change', (e) => {
    S.xAxisKey = e.target.value;
    if (S.view === 'metrics') refreshMetrics();
  });

  // (a) map mode selector — 3-state: phase (5) / stage (FSM for goal) / all (15)
  on($('#map-mode'), 'change', (e) => {
    S.mapMode = e.target.value;
    if (S.view === 'map') renderGraph();
  });

  // run picker — user selects a specific run; '' falls back to latest by mtime
  on($('#run-select'), 'change', async (e) => {
    S.runId = e.target.value;        // '' means "follow latest"
    S.logOffset = 0;                 // logs differ per run; rewind
    S.startTs = null;                // recompute from new run's started_at
    // tear down any existing per-run views/charts so stale data doesn't bleed
    const stream = $('#stream'); if (stream) stream.innerHTML = '';
    Object.values(S.charts).forEach(c => { try { c.destroy(); } catch (_) {} });
    S.charts = {};
    await reloadRun();
    paintBanner();
    await refreshAll();
  });

  setView('metrics');

  if (S.config.live) connectSSE();
  setInterval(tick, 1000);
  setInterval(refreshAll, S.config.live ? 5000 : 10000);
  refreshAll();
}

/* ════════════════════════════ API helpers ════════════════════════════ */
async function getJSON(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p} → ${r.status}`);
  return r.json();
}
async function putJSON(p, body) {
  const r = await fetch(p, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || `${p} → ${r.status}`);
  return d;
}

/* ════════════════════════════ harness + run ════════════════════════════ */
async function reloadHarness() { S.harness = await getJSON('/api/harness'); }

async function loadRunOrdinal() {
  /* Count total runs (for the N° label) AND populate the run-select dropdown. */
  try {
    const data = await getJSON('/api/runs');
    const runs = data.runs || [];
    S.runsList = runs;
    S.runOrdinal = runs.length;     // latest run = total number
    populateRunSelect(runs);
  } catch { S.runOrdinal = null; }
}

function populateRunSelect(runs) {
  const sel = $('#run-select');
  if (!sel) return;
  /* Build options in mtime-DESCENDING order (latest first) — list_runs() returns ascending. */
  const opts = ['<option value="">— latest by mtime —</option>'];
  for (const r of [...runs].reverse()) {
    const id = r.id;
    const purpose = (r.meta && r.meta.purpose) ? ` — ${r.meta.purpose.substring(0, 60)}` : '';
    const status = (r.meta && r.meta.status) ? ` [${r.meta.status}]` : '';
    opts.push(`<option value="${esc(id)}">${esc(id)}${esc(status)}${esc(purpose)}</option>`);
  }
  const prev = S.runId || '';
  sel.innerHTML = opts.join('');
  sel.value = prev;
}

async function reloadRun() {
  try { S.run = await getJSON(scoped('/api/run')); }
  catch { S.run = { run_id: null, status: 'idle', entries: [], current: null }; }
  const startedRaw = S.run?.meta?.started_at;
  S.startTs = startedRaw ? new Date(startedRaw).getTime() : null;
  S.activeStateSkills = new Set();
  // (a) derive run goal — best effort, defaults to 'train'
  S.goal = S.run?.meta?.goal || 'train';
  if (S.run.current) {
    try {
      const stateData = await getJSON(`/api/state/${S.run.current}`);
      (stateData.skills || []).forEach(s => S.activeStateSkills.add(s));
    } catch (_) {}
  }
}

/* ════════════════════════════ BANNER painter ════════════════════════════ */
function paintBanner() {
  const status = (S.run?.status || 'idle').toLowerCase();
  document.body.dataset.state = status;

  const runId = S.run?.run_id || '— no run on record —';
  $('#masthead-run').textContent = runId;

  // banner number: roman-numeral-flavored
  const num = S.runOrdinal != null
    ? `№ ${String(S.runOrdinal).padStart(3, '0')}`
    : '№ —';
  $('#banner-num').textContent = num;

  // banner name = current state (if running) or status word
  const banner = $('#banner-name');
  const cur = S.run?.current;
  banner.innerHTML = cur ? `<em>${esc(cur)}</em>` : `<em>${esc(status)}</em>`;

  // tabs meta — a small italic note
  const meta = $('#tabs-meta');
  if (S.run?.meta?.purpose) {
    meta.innerHTML = `<em>${esc(S.run.meta.purpose.split('.')[0])}</em>`;
  } else if (S.harness) {
    meta.innerHTML = `<em>${S.harness.state_count} states · ${S.harness.skills?.length || 0} skills</em>`;
  }
}

function tick() {
  if (S.startTs) {
    const ms = Date.now() - S.startTs;
    $('#meta-elapsed').textContent = fmtDuration(ms);
  } else {
    $('#meta-elapsed').textContent = '—';
  }
  const age = Math.floor((Date.now() - S.lastEventTs) / 1000);
  const ageEl = $('#meta-since');
  ageEl.textContent = fmtAge(age) + ' ago';
}
function fmtDuration(ms) {
  if (!isFinite(ms) || ms < 0) return '—';
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}h ${String(m).padStart(2,'0')}m`;
  if (m) return `${m}m ${String(ss).padStart(2,'0')}s`;
  return `${ss}s`;
}
function fmtAge(s) {
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s/60)} m`;
  return `${Math.floor(s/3600)} h`;
}
function pulseSignal() {
  S.lastEventTs = Date.now();
  const dot = $('#signal-dot');
  if (!dot) return;
  dot.classList.remove('beating');
  void dot.offsetWidth;
  dot.classList.add('beating');
}

/* ════════════════════════════ view switcher ════════════════════════════ */
function setView(name) {
  S.view = name;
  document.body.dataset.view = name;
  $$('.tab[data-view]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === name);
  });
  $$('.view').forEach(v => { v.hidden = (v.dataset.view !== name); });
  if (name === 'map')     renderGraph();
  if (name === 'spec')    selectOverview();
  if (name === 'metrics') refreshMetrics();
  if (name === 'submit')  initSubmit();
  if (name === 'hitl')    refreshHitl();
  // Auto-refresh task list only while on submit view; clear otherwise.
  clearInterval(S.submitTasksTimer);
  S.submitTasksTimer = null;
  if (name === 'submit') {
    S.submitTasksTimer = setInterval(refreshSubmitTasks, 5000);
  }
}

/* ════════════════════════════ STREAM ════════════════════════════ */
async function refreshStream() {
  try {
    const data = await getJSON(scoped(`/api/logs?since=${S.logOffset}`));
    if (data.size === undefined) return;
    if (data.size < S.logOffset) S.logOffset = 0;
    if (data.content) appendStream(data.content);
    S.logOffset = data.size;
  } catch (_) {}
}
function appendStream(chunk) {
  const pre = $('#stream');
  const wasAtBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
  const lines = chunk.split('\n').map(line => {
    let cls = 'log-line-new';
    if (/error|cuda out of memory|nan|inf|traceback|failed/i.test(line)) cls += ' log-line-error';
    else if (/warn|preempt|timeout/i.test(line)) cls += ' log-line-warn';
    else if (/training\/global_step|^step:|Training Progress:/i.test(line)) cls += ' log-line-step';
    else if (/Final validation metrics:|saved|checkpoint/i.test(line)) cls += ' log-line-ok';
    return `<span class="${cls}">${esc(line)}</span>`;
  });
  pre.insertAdjacentHTML('beforeend', lines.join('\n'));
  setTimeout(() => pre.querySelectorAll('.log-line-new').forEach(e =>
    e.classList.remove('log-line-new')), 820);
  if (pre.innerHTML.length > 400_000) pre.innerHTML = pre.innerHTML.slice(-200_000);
  if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
}

/* ════════════════════════════ METRICS — panel grid ════════════════════════════ */
async function refreshMetrics() {
  try {
    const data = await getJSON(scoped('/api/progress'));
    renderPanelGrid(data);
  } catch (_) {}
  try {
    const a = await getJSON(scoped('/api/anomalies'));
    renderAnomalies(a.anomalies || []);
  } catch (_) {}
  try {
    const r = await getJSON(scoped('/api/reflect'));
    renderReflectCard(r);
  } catch (_) {}
}

/* EMA smoothing of a numeric series; alpha=0 means raw. */
function ema(ys, alpha) {
  if (!alpha || alpha <= 0) return ys;
  const out = [];
  let s = NaN;
  for (const y of ys) {
    if (typeof y !== 'number' || !isFinite(y)) { out.push(y); continue; }
    s = isFinite(s) ? (alpha * s + (1 - alpha) * y) : y;
    out.push(s);
  }
  return out;
}

function renderPanelGrid(data) {
  const grid = $('#panel-grid');
  const empty = !data || !data.rows || !data.columns?.length;
  const emptyEl = $('#metrics-empty');

  $('#metrics-meta').innerHTML = empty ? '<em>no metric rows yet</em>'
    : `<em>${data.rows} row${data.rows>1?'s':''} · ${data.columns.length} fields</em>`;

  if (empty) {
    // teardown
    for (const k of Object.keys(S.charts)) { S.charts[k].destroy(); }
    S.charts = {};
    grid.innerHTML = '<div class="empty" id="metrics-empty"><em>awaiting first metric dict from the trainer</em></div>';
    return;
  }
  if (emptyEl) emptyEl.remove();

  const cols = data.columns;
  // x-axis values + the step display in pulse banner
  let xs;
  const xKey = S.xAxisKey;
  if (xKey === '_index' || !cols.includes(xKey)) {
    xs = (data.series[cols[0]] || []).map((_, i) => i);
  } else {
    xs = data.series[xKey].map(v => typeof v === 'number' ? v : null);
  }
  // also push step into the banner cell
  const gs = data.series['training/global_step'];
  if (gs && gs.length) {
    const last = gs[gs.length - 1];
    if (last !== undefined && last !== null) $('#meta-step').textContent = String(last);
  }

  // determine groups present in this data
  const used = new Set();
  const groupsToShow = [];
  for (const grp of NS_GROUPS) {
    const keys = cols.filter(c => grp.match(c) && !used.has(c) && !DROP_KEYS.has(c));
    // only keep numeric series with at least one finite value
    const numericKeys = keys.filter(k => (data.series[k] || []).some(v => typeof v === 'number' && isFinite(v)));
    if (numericKeys.length === 0) continue;
    numericKeys.forEach(k => used.add(k));
    // Cap series per panel — sort by variance so the most informative lines render first,
    // then truncate. Spill count is surfaced in the panel header.
    let renderKeys = numericKeys;
    let spill = 0;
    if (numericKeys.length > MAX_SERIES_PER_PANEL) {
      const variance = (arr) => {
        const xs = arr.filter(v => typeof v === 'number' && isFinite(v));
        if (xs.length < 2) return 0;
        const m = xs.reduce((a,b)=>a+b,0) / xs.length;
        return xs.reduce((a,b) => a + (b-m)*(b-m), 0) / xs.length;
      };
      const scored = numericKeys.map(k => ({ k, v: variance(data.series[k] || []) }));
      scored.sort((a,b) => b.v - a.v);
      renderKeys = scored.slice(0, MAX_SERIES_PER_PANEL).map(x => x.k);
      spill = numericKeys.length - MAX_SERIES_PER_PANEL;
    }
    groupsToShow.push({ name: grp.name, keys: renderKeys, totalKeys: numericKeys.length, spill });
  }

  // diff: remove panels no longer present
  const wantNames = new Set(groupsToShow.map(g => g.name));
  for (const name of Object.keys(S.charts)) {
    if (!wantNames.has(name)) {
      S.charts[name].destroy();
      delete S.charts[name];
      const el = grid.querySelector(`[data-panel="${cssEsc(name)}"]`);
      if (el) el.remove();
    }
  }

  // create/update panels
  for (const grp of groupsToShow) {
    let panel = grid.querySelector(`[data-panel="${cssEsc(grp.name)}"]`);
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'metric-panel';
      panel.dataset.panel = grp.name;
      panel.innerHTML = `
        <div class="metric-panel-head">
          <div class="metric-panel-title">${esc(grp.name)}</div>
          <div class="metric-panel-meta">${grp.totalKeys} series${grp.spill ? ` <span class="spill">· top ${grp.keys.length} shown</span>` : ''}</div>
        </div>
        <div class="metric-panel-canvas"><canvas></canvas></div>
        <div class="metric-panel-legend"></div>`;
      grid.appendChild(panel);
    } else {
      panel.querySelector('.metric-panel-meta').innerHTML =
        `${grp.totalKeys} series${grp.spill ? ` <span class="spill">· top ${grp.keys.length} shown</span>` : ''}`;
    }

    const datasets = grp.keys.map((k, i) => {
      const ys = ema(data.series[k] || [], S.smoothing);
      const color = SERIES_PALETTE[i % SERIES_PALETTE.length];
      return {
        label: shortLabel(k, grp.name),
        data: xs.map((x, idx) => ({ x, y: ys[idx] })),
        borderColor: color,
        backgroundColor: color + '14',
        borderWidth: 1.2,
        tension: 0.18,
        pointRadius: data.rows < 10 ? 2 : 0,
        pointHoverRadius: 4,
        _fullKey: k,
      };
    });

    const canvas = panel.querySelector('canvas');
    const cfg = panelChartConfig(datasets, xKey === '_index' ? 'row' : xKey);
    if (S.charts[grp.name]) {
      S.charts[grp.name].data = cfg.data;
      S.charts[grp.name].options = cfg.options;
      S.charts[grp.name].update('none');
    } else {
      S.charts[grp.name] = new Chart(canvas, cfg);
    }

    // legend below canvas
    const legend = panel.querySelector('.metric-panel-legend');
    legend.innerHTML = datasets.map(ds => `
      <span class="metric-panel-legend-item" title="${esc(ds._fullKey)}">
        <span class="metric-panel-legend-swatch" style="background:${ds.borderColor}"></span>
        ${esc(ds.label)}
      </span>`).join('');
  }
}

/* shorten "actor/perf/max_memory_allocated_gb" → "max_memory_allocated_gb"
   for legend readability; full path on hover via title=. */
function shortLabel(key, groupName) {
  // drop the first slug if it equals the group name's first word
  const head = groupName.split(/\s*[·/]\s*/)[0];        // e.g. "actor" from "actor · loss"
  if (head && key.startsWith(head + '/')) return key.slice(head.length + 1);
  return key;
}

function cssEsc(s) { return s.replace(/[^\w]/g, '\\$&'); }

function panelChartConfig(datasets, xLabel) {
  // Claude design.md tokens — kept in sync with app.css :root vars.
  const tick = '#6c6a64';                                         /* --muted     */
  const grid = '#ebe6df';                                         /* --hairline-soft */
  const tickFont = { family: "'JetBrains Mono', monospace", size: 10 };
  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { type: 'linear',
             title: { display: true, text: xLabel, color: tick, font: tickFont, padding: { top: 2 } },
             ticks: { color: tick, font: tickFont, maxTicksLimit: 6 },
             grid: { color: grid, drawBorder: false, lineWidth: 0.5 } },
        y: { type: 'linear',
             ticks: { color: tick, font: tickFont, maxTicksLimit: 5 },
             grid: { color: grid, drawBorder: false, lineWidth: 0.5 } },
      },
      plugins: {
        legend: { display: false },        // custom legend rendered separately
        tooltip: {
          backgroundColor: '#181715',                              /* --surface-dark */
          borderColor: '#252320',                                  /* --surface-dark-elevated */
          borderWidth: 1,
          titleColor: '#faf9f5',                                   /* --on-dark   */
          bodyColor:  '#a09d96',                                   /* --on-dark-soft */
          titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
          bodyFont:  { family: "'JetBrains Mono', monospace", size: 11 },
          padding: 10,
          cornerRadius: 8,                                         /* --r-md      */
          callbacks: {
            label(ctx) {
              const v = ctx.parsed.y;
              const str = (typeof v === 'number' && isFinite(v))
                ? (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3) ? v.toExponential(3) : v.toPrecision(5))
                : '—';
              return `${ctx.dataset._fullKey || ctx.dataset.label}: ${str}`;
            },
          },
        },
      },
    },
  };
}
/* ── reflect (closed-loop refinement) card ────────────────────────────────
   Only rendered when the run activated the reflect loop (workspace/reflect/
   exists). Shows iter counter, target status, per-iter delta rows, and
   the plan/report artefact contents behind a details toggle. */
function renderReflectCard(r) {
  const el = $('#reflect-card');
  if (!el) return;
  if (!r || !r.present) { el.hidden = true; el.innerHTML = ''; return; }

  const s = r.state || {};
  const history = Array.isArray(s.history) ? s.history : [];
  const iter = s.iteration ?? history.length;
  const cap = s.max_iterations ?? '?';

  // Pill classification: target_met | budget_exhausted | in-progress
  let pillLabel, pillClass;
  if (s.success === true) {
    pillLabel = 'target met';
    pillClass = 'ok';
  } else if (s.stop_reason) {
    pillLabel = String(s.stop_reason).replace(/_/g, ' ');
    pillClass = 'stop';
  } else {
    pillLabel = 'in progress';
    pillClass = 'live';
  }

  const targetLine = s.target_metric
    ? `target · <code>${esc(s.target_metric)}</code> ≥ <code>${esc(s.target_value ?? '?')}</code>`
    : '';

  const rows = history.map(h => {
    const dstr = h.delta === null || h.delta === undefined
      ? '<span class="reflect-muted">no delta (final iter)</span>'
      : Object.entries(h.delta).map(([k, v]) => {
          const oldv = v && v.old !== undefined ? esc(v.old) : '?';
          const newv = v && v.new !== undefined ? esc(v.new) : '?';
          return `<code>${esc(k)}</code>: ${oldv} → ${newv}`;
        }).join('<br>');
    const metric = h.metric_final_step ?? h.metric_last_third_mean ?? '—';
    const metricStr = typeof metric === 'number' ? metric.toFixed(4) : esc(metric);
    return `
      <tr>
        <td class="reflect-iter">${esc(h.iteration ?? '?')}</td>
        <td class="reflect-diag">${esc(h.diagnosis || '—')}</td>
        <td class="reflect-metric">${metricStr}</td>
        <td class="reflect-delta">${dstr}</td>
        <td class="reflect-job">${esc((h.job || '—').split(',')[0])}</td>
      </tr>`;
  }).join('');

  const bestLine = s.best_checkpoint
    ? `<div class="reflect-foot-row">best iter <em>${esc(s.best_iteration ?? '?')}</em> · <code>${esc(s.best_checkpoint)}</code></div>`
    : '';

  const artefacts = [];
  if (r.refinement_plan) {
    artefacts.push(`<details class="reflect-artefact"><summary>refinement_plan.md</summary><div class="markdown">${marked.parse(r.refinement_plan)}</div></details>`);
  }
  if (r.reflect_report) {
    artefacts.push(`<details class="reflect-artefact"><summary>reflect_report.md</summary><div class="markdown">${marked.parse(r.reflect_report)}</div></details>`);
  }

  el.innerHTML = `
    <header class="reflect-head">
      <div class="reflect-title">reflect · closed-loop refinement</div>
      <div class="reflect-head-meta">
        <span class="reflect-iter-pill">iter <em>${esc(iter)}</em> / ${esc(cap)}</span>
        <span class="reflect-pill sev-${pillClass}">${esc(pillLabel)}</span>
      </div>
    </header>
    ${targetLine ? `<div class="reflect-target">${targetLine}</div>` : ''}
    <table class="reflect-table">
      <thead><tr>
        <th>iter</th><th>diagnosis</th><th>metric</th><th>delta applied</th><th>job</th>
      </tr></thead>
      <tbody>${rows || `<tr><td colspan="5" class="reflect-empty">no iterations recorded yet</td></tr>`}</tbody>
    </table>
    ${bestLine}
    ${artefacts.length ? `<div class="reflect-artefacts">${artefacts.join('')}</div>` : ''}
  `;
  el.hidden = false;
}

function renderAnomalies(rows) {
  const wrap = $('#anomalies-strip');
  if (!rows.length) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = rows.slice(-20).reverse().map(r => `
    <div class="anomaly-row sev-${esc(r.severity)}">
      <span class="ts">${esc(r.timestamp || '—')}</span>
      <span class="sev">${esc(r.severity)}</span>
      <span class="body">${esc(r.body)}</span>
    </div>`).join('');
}

/* ════════════════════════════ JOB ════════════════════════════ */
async function refreshJob() {
  try {
    const data = await getJSON(scoped('/api/job'));
    const info = data.info || {};
    const status = data.status || {};
    const job = info.slurm_jobid || info.pid || '—';
    $('#meta-job').textContent = job;
    if (status.final_step) $('#meta-step').textContent = `${status.final_step}`;
  } catch (_) {}
}

/* ════════════════════════════ MAP / mermaid ════════════════════════════ */
function initMermaid() {
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'base',
    flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis' },
    fontFamily: "'Inter', sans-serif",
    themeVariables: {
      // Claude design.md tokens (kept in sync with app.css :root vars)
      fontFamily: "'Inter', sans-serif",
      fontSize: '12px',
      background: 'transparent',
      primaryColor: '#efe9de',        // --surface-card (default node fill)
      primaryBorderColor: '#6c6a64',  // --muted (default node stroke)
      primaryTextColor: '#141413',    // --ink
      lineColor: '#8e8b82',           // --muted-soft (edges)
      mainBkg: '#efe9de',
      nodeBorder: '#6c6a64',
      edgeLabelBackground: '#faf9f5', // --canvas
    },
  });
}
/* (a) Synthesize a 5-node phase-view mermaid graph for goal=train. Each phase
       aggregates several FSM states; its status is derived from its members.
       Click a phase → spec view jumps to the first member state. */
function synthesizePhaseMermaid() {
  const phases = PHASES.train;
  const lines = ['flowchart TD'];
  // node decls — use a slab label so they read like UI tiles
  for (const p of phases) lines.push(`  ${p.id}["${p.label}"]`);
  // linear edges, unlabelled — the order itself is self-evident
  for (let i = 0; i < phases.length - 1; i++) {
    lines.push(`  ${phases[i].id} --> ${phases[i+1].id}`);
  }
  return lines.join('\n');
}

/* (a) filter mermaid source down to states relevant for the active goal,
       unless mode === 'all'. */
function filterMermaidByTrack(src) {
  if (S.mapMode === 'all') return src;
  const keep = new Set(TRACK_STATES[S.goal] || TRACK_STATES.train);
  keep.add(S.harness?.overview_node);
  // Plus always include any state actually visited this run (defensive — covers
  // cross-track entries like resume_train hitting monitor_training).
  (S.run?.visited || []).forEach(s => keep.add(s));
  if (S.run?.current) keep.add(S.run.current);

  const lines = src.split('\n');
  const out = [];
  // first line is `flowchart TD`
  out.push(lines[0]);
  // simple line regexes
  const NODE_RE = /^\s*(\w+)\["?[^"]*"?\]/;
  const EDGE_RE = /^\s*(\w+)\s*-->.*?\|\s*(\w+)$|^\s*(\w+)\s*--\.[^.]*\.->\s*(\w+)$|^\s*(\w+)\s*-->\s*(\w+)\s*$/;
  for (let i = 1; i < lines.length; i++) {
    const ln = lines[i];
    if (!ln.trim()) continue;
    const node = ln.match(NODE_RE);
    if (node) {
      if (keep.has(node[1])) out.push(ln);
      continue;
    }
    // edge: extract both endpoints
    const e1 = ln.match(/^\s*(\w+)\s*(?:-->|-\.-?>|-{2,}>?)\|?[^|]*\|?\s*(\w+)\s*$/);
    if (e1) {
      const [, a, b] = e1;
      if (keep.has(a) && keep.has(b)) out.push(ln);
      continue;
    }
    // edge with label `a -->|"label"| b`
    const e2 = ln.match(/^\s*(\w+)\s*-->\|[^|]*\|\s*(\w+)\s*$/);
    if (e2) {
      const [, a, b] = e2;
      if (keep.has(a) && keep.has(b)) out.push(ln);
      continue;
    }
    // edge `__overview__ -.start.-> intake`
    const e3 = ln.match(/^\s*(\w+)\s*-\.[^.]+\.->\s*(\w+)\s*$/);
    if (e3) {
      const [, a, b] = e3;
      if (keep.has(a) && keep.has(b)) out.push(ln);
      continue;
    }
    // anything else (classDef etc.) — keep
    out.push(ln);
  }
  return out.join('\n');
}

async function renderGraph() {
  if (!S.harness) return;
  const visited = new Set(S.run?.visited || []);
  const current = S.run?.current;
  const terminals = new Set(S.harness.terminal_states || []);

  // ── PHASE MODE (default): synthesize 5 nodes, aggregate status from members ──
  // Only available for the standard train goal; the smaller resume/generate/eval
  // tracks already have ≤5 states so they fall through to stage mode.
  const canPhase = S.mapMode === 'phase' && S.goal === 'train' && PHASES.train;
  if (canPhase) {
    const phases = PHASES.train;
    let src = synthesizePhaseMermaid();
    const classes = [];
    for (const p of phases) {
      const mems = p.members;
      const live   = mems.some(m => m === current);
      const allVis = mems.every(m => visited.has(m));
      const anyVis = mems.some(m => visited.has(m));
      const isTerm = mems.some(m => terminals.has(m));
      if      (live)             classes.push(`class ${p.id} fhLive`);
      else if (isTerm && anyVis) classes.push(`class ${p.id} fhTerminal`);
      else if (allVis)           classes.push(`class ${p.id} fhVisited`);
      // partial visit shows default cream — honest about "in progress"
    }
    src = src + '\n' + classes.join('\n');

    const trackLbl = $('#map-track-label');
    if (trackLbl) trackLbl.innerHTML = `<em>phases</em> · 5 · <span style="color:var(--muted-soft)">${PHASES.train.flatMap(p=>p.members).length} states</span>`;

    try {
      const { svg } = await mermaid.render(`g${Date.now()}`, src);
      const wrap = $('#graph');
      wrap.innerHTML = svg;
      wrap.querySelectorAll('.node').forEach(n => {
        const id = n.id?.replace(/^flowchart-/, '').split('-')[0];
        const phase = phases.find(p => p.id === id);
        if (phase) {
          n.addEventListener('click', () => {
            // jump to the first member; user can navigate from there in spec nav
            setView('spec'); selectState(phase.members[0]);
          });
        }
      });
      applyZoom();
    } catch (e) {
      $('#graph').textContent = 'mermaid render failed';
      console.error(e);
    }
    return;
  }

  // ── STAGE / ALL MODE: existing per-state FSM render ──────────────────────────
  let src = filterMermaidByTrack(S.harness.mermaid);
  const trackLbl = $('#map-track-label');
  if (trackLbl) {
    const visibleCount = (TRACK_STATES[S.goal] || []).length;
    trackLbl.innerHTML = S.mapMode === 'all'
      ? `<em>all states</em> · 16`
      : `track <em>${esc(S.goal)}</em> · ${visibleCount} states`;
  }
  const overview = S.harness.overview_node;
  const classes = [];
  if (overview) classes.push(`class ${overview} fhOverview`);
  for (const name of S.harness.states) {
    if (name === current) classes.push(`class ${name} fhLive`);
    else if (S.current?.kind === 'state' && S.current?.id === name) classes.push(`class ${name} fhSel`);
    else if (visited.has(name)) classes.push(`class ${name} fhVisited`);
    else if (terminals.has(name)) classes.push(`class ${name} fhTerminal`);
  }
  src = src + '\n' + classes.join('\n');
  try {
    const { svg } = await mermaid.render(`g${Date.now()}`, src);
    const wrap = $('#graph');
    wrap.innerHTML = svg;
    wrap.querySelectorAll('.node').forEach(n => {
      const id = n.id?.replace(/^flowchart-/, '').split('-')[0];
      if (id) n.addEventListener('click', () => {
        if (id === S.harness.overview_node) { setView('spec'); selectOverview(); }
        else { setView('spec'); selectState(id); }
      });
    });
    applyZoom();
  } catch (e) {
    $('#graph').textContent = 'mermaid render failed';
    console.error(e);
  }
}
function applyZoom() {
  const g = $('#graph');
  if (g) g.style.transform = `scale(${S.zoom})`;
  const lvl = $('#zoom-level');
  if (lvl) lvl.textContent = `${Math.round(S.zoom * 100)}`;
}

/* ════════════════════════════ SPEC nav + inspector ════════════════════════════ */
function renderSpecNav() {
  if (!S.harness) return;
  const wrap = $('#spec-nav');
  if (!wrap) return;
  const states = S.harness.states || [];
  const skills = S.harness.skills || [];

  const curKind = S.current?.kind;
  if (curKind === 'overview') S.navExpanded.overview = true;
  if (curKind === 'state')    S.navExpanded.states   = true;
  if (curKind === 'skill')    S.navExpanded.skills   = true;

  const openAttr = (k) => S.navExpanded[k] ? ' open' : '';
  const html = [];

  html.push(`<details class="spec-nav-group" data-group="overview"${openAttr('overview')}>`);
  html.push(`<summary class="spec-nav-summary">overview</summary>`);
  html.push(`<div class="spec-nav-rows">`);
  html.push(`<div class="spec-nav-row" data-kind="overview" data-id="${esc(S.harness.overview_node)}">task-overview.md</div>`);
  html.push(`</div></details>`);

  html.push(`<details class="spec-nav-group" data-group="states"${openAttr('states')}>`);
  html.push(`<summary class="spec-nav-summary">states</summary>`);
  html.push(`<div class="spec-nav-rows">`);
  for (const name of states) {
    const live = (name === S.run?.current) ? ' <span class="row-sub">live</span>' : '';
    html.push(`<div class="spec-nav-row" data-kind="state" data-id="${esc(name)}">${esc(name)}${live}</div>`);
  }
  html.push(`</div></details>`);

  html.push(`<details class="spec-nav-group" data-group="skills"${openAttr('skills')}>`);
  html.push(`<summary class="spec-nav-summary">skills</summary>`);
  html.push(`<div class="spec-nav-rows">`);
  for (const path of skills) {
    const used = S.activeStateSkills.has(path) ? ' <span class="row-sub">·</span>' : '';
    html.push(`<div class="spec-nav-row" data-kind="skill" data-id="${esc(path)}">${esc(path)}${used}</div>`);
  }
  html.push(`</div></details>`);

  wrap.innerHTML = html.join('');
  wrap.querySelectorAll('details.spec-nav-group').forEach(d =>
    d.addEventListener('toggle', () => { S.navExpanded[d.dataset.group] = d.open; }));
  wrap.querySelectorAll('.spec-nav-row').forEach(row =>
    row.addEventListener('click', () => {
      const { kind, id } = row.dataset;
      if (kind === 'overview') selectOverview();
      else if (kind === 'state') selectState(id);
      else if (kind === 'skill') selectSkill(id);
    }));
  paintSpecNavActive();
}
function paintSpecNavActive() {
  $$('.spec-nav-row').forEach(r => {
    const k = r.dataset.kind, i = r.dataset.id;
    r.classList.toggle('active',
      S.current && S.current.kind === k && S.current.id === i);
  });
}
async function selectOverview() {
  S.current = { kind: 'overview', id: S.harness.overview_node, editPath: 'task-overview.md' };
  exitEditor();
  const data = await getJSON(`/api/state/${S.harness.overview_node}`);
  paintInspector('overview', S.harness.title, data.compiled, data.editable, data.file);
  paintSpecNavActive();
}
async function selectState(name) {
  S.current = { kind: 'state', id: name, editPath: `states/${name}.md` };
  exitEditor();
  const data = await getJSON(`/api/state/${name}`);
  S.activeStateSkills = new Set(data.skills || []);
  paintInspector(data.is_terminal ? 'terminal state' : 'state', name, data.compiled, data.editable, data.file);
  paintSpecNavActive();
}
async function selectSkill(path) {
  S.current = { kind: 'skill', id: path };
  exitEditor();
  const data = await getJSON(`/api/skill?path=${encodeURIComponent(path)}`);
  paintInspector('skill', path, data.compiled, false, null);
  paintSpecNavActive();
}
function paintInspector(eyebrow, title, md, editable, file) {
  $('#spec-title').innerHTML = `${esc(eyebrow)} <em>${esc(title || '')}</em>`;
  const rendered = $('#rendered');
  rendered.innerHTML = marked.parse(md || '');
  rendered.hidden = false;
  $('#editor-host').hidden = true;
  const eb = $('#edit-btn');
  if (editable && file) { eb.hidden = false; eb.dataset.path = file; }
  else { eb.hidden = true; delete eb.dataset.path; }
}
function reselect() {
  if (!S.current) return;
  if (S.current.kind === 'overview') selectOverview();
  else if (S.current.kind === 'state') selectState(S.current.id);
  else if (S.current.kind === 'skill') selectSkill(S.current.id);
}

/* ════════════════════════════ editor ════════════════════════════ */
let cm = null;
async function openEditor() {
  const path = $('#edit-btn').dataset.path;
  if (!path) return;
  const data = await getJSON(`/api/file?path=${encodeURIComponent(path)}`);
  $('#rendered').hidden = true;
  $('#edit-btn').hidden = true;
  $('#save-btn').hidden = false;
  $('#cancel-btn').hidden = false;
  $('#save-status').textContent = '';
  const host = $('#editor-host');
  host.hidden = false;
  host.innerHTML = '<textarea></textarea>';
  cm = CodeMirror.fromTextArea(host.querySelector('textarea'), {
    mode: 'markdown', lineNumbers: true, lineWrapping: true,
    indentUnit: 2, tabSize: 2,
  });
  cm.setValue(data.content || '');
  cm.on('change', () => {
    $('#save-status').textContent = '— unsaved';
    $('#save-status').className = 'save-status dirty';
  });
}
async function saveEdit() {
  const path = $('#edit-btn').dataset.path;
  if (!path || !cm) return;
  try {
    await putJSON('/api/file', { path, content: cm.getValue() });
    $('#save-status').textContent = 'saved';
    $('#save-status').className = 'save-status saved';
    toast('saved');
  } catch (e) { toast(e.message, 'err'); }
}
function exitEditor() {
  $('#save-btn').hidden = true;
  $('#cancel-btn').hidden = true;
  const eb = $('#edit-btn');
  if (eb.dataset.path) eb.hidden = false;
  $('#editor-host').hidden = true;
  $('#editor-host').innerHTML = '';
  $('#rendered').hidden = false;
  cm = null;
}

/* ════════════════════════════ toast ════════════════════════════ */
let toastTimer;
function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = 'toast'; t.hidden = true; }, 2400);
}

/* ════════════════════════════ SSE ════════════════════════════ */
function connectSSE() {
  const es = new EventSource('/events');
  es.addEventListener('hello',   () => pulseSignal());
  es.addEventListener('changed', async () => { pulseSignal(); await refreshAll(); });
  es.addEventListener('error',   () => { setTimeout(connectSSE, 3000); es.close(); });
}

/* ════════════════════════════ refresh aggregator ════════════════════════════ */
async function refreshAll() {
  await reloadRun();
  paintBanner();
  await refreshJob();
  await refreshStream();
  await refreshHitl();
  if (S.view === 'metrics') await refreshMetrics();
  if (S.view === 'map')     await renderGraph();
  if (S.view === 'spec')    renderSpecNav();
  if (S.view === 'submit')  await refreshSubmitTasks();
}

/* ════════════════════════════ HITL — approvals ════════════════════════════ */
async function refreshHitl() {
  let requests = [];
  try {
    const data = await getJSON(scoped('/api/hitl/pending'));
    requests = Array.isArray(data.requests) ? data.requests : [];
  } catch (_) {
    // Endpoint may briefly 5xx during server reload; silent retry next tick.
    return;
  }
  paintHitlBanner(requests.length);
  paintHitlBadge(requests.length);
  if (S.view === 'hitl') renderHitlCards(requests);
}

function paintHitlBanner(count) {
  const banner = $('#hitl-banner');
  if (!banner) return;
  banner.hidden = count === 0;
  const c = $('#hitl-banner-count'); if (c) c.textContent = String(count);
  const p = $('#hitl-banner-plural'); if (p) p.textContent = count === 1 ? '' : 's';
}
function paintHitlBadge(count) {
  const b = $('#hitl-badge');
  if (!b) return;
  b.hidden = count === 0;
  b.textContent = String(count);
}
function renderHitlCards(requests) {
  const host = $('#hitl-cards');
  if (!host) return;
  if (!requests.length) {
    host.innerHTML = `<p class="empty-prose"><em>no requests — nothing waiting on you</em></p>`;
    return;
  }
  host.innerHTML = requests.map(r => {
    const cls = r.is_always_on ? 'hitl-card always-on' : 'hitl-card';
    const tag = r.is_always_on ? `<span class="hitl-card-tag">always-on</span>` : '';
    return `
      <article class="${cls}" data-req-id="${esc(r.id)}">
        <div class="hitl-card-head">
          <span class="hitl-card-state">${esc(r.state || '')}</span>
          ${tag}
        </div>
        <h3 class="hitl-card-title">${esc(r.title || '')}</h3>
        <p class="hitl-card-desc">${esc(r.description || '')}</p>
        <div class="hitl-card-actions">
          <button class="hitl-btn primary" data-decision="approve">Approve</button>
          <button class="hitl-btn deny"    data-decision="deny">Deny</button>
          <button class="hitl-btn"         data-decision="skip">Skip</button>
        </div>
      </article>`;
  }).join('');
  host.querySelectorAll('.hitl-btn[data-decision]').forEach(btn => {
    btn.addEventListener('click', onHitlDecision);
  });
}
async function onHitlDecision(ev) {
  const btn = ev.currentTarget;
  const decision = btn.dataset.decision;
  const card = btn.closest('.hitl-card');
  const reqId = card?.dataset.reqId;
  if (!reqId) return;
  card.querySelectorAll('.hitl-btn').forEach(b => { b.disabled = true; });
  try {
    const r = await fetch(scoped(`/api/hitl/${encodeURIComponent(reqId)}`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `${r.status}`);
    toast(`${decision} · ${reqId.slice(0, 8)}…`);
    // Card will vanish on next refresh; trigger immediately.
    await refreshHitl();
  } catch (e) {
    toast(`error: ${e.message}`);
    card.querySelectorAll('.hitl-btn').forEach(b => { b.disabled = false; });
  }
}

/* ════════════════════════════ SUBMIT ════════════════════════════ */
function initSubmit() {
  refreshSubmitTasks();
  if (S.submitReady) return;
  S.submitReady = true;
  const form = $('#submit-form');
  if (!form) return;
  form.addEventListener('submit', onSubmitForm);
  const fields = ['#f-algo', '#f-model', '#f-dataset', '#f-extra', '#f-hitl'];
  fields.forEach(sel => { const el = $(sel); if (el) el.addEventListener('input', renderPromptPreview); });
  $('#tasks-refresh')?.addEventListener('click', refreshSubmitTasks);
  renderPromptPreview();
}

function composePromptPreview() {
  const algorithm = $('#f-algo')?.value?.trim() || '';
  const model     = $('#f-model')?.value?.trim() || '';
  const dataset   = $('#f-dataset')?.value?.trim() || '';
  const extra     = $('#f-extra')?.value?.trim() || '';
  const hitl      = $('#f-hitl')?.checked;
  const hitlFlag  = hitl ? 'on' : '--no-hitl';
  const extraLine = extra ? ` ${extra}` : '';
  return `Intent: Train ${algorithm} on ${dataset} using ${model}.${extraLine}\nHITL: ${hitlFlag}`;
}
function renderPromptPreview() {
  const el = $('#submit-preview-body');
  if (el) el.textContent = composePromptPreview();
}

async function onSubmitForm(ev) {
  ev.preventDefault();
  const btn = $('#submit-btn');
  const status = $('#submit-status');
  const body = {
    algorithm: $('#f-algo').value,
    model:     $('#f-model').value,
    dataset:   $('#f-dataset').value,
    extra:     $('#f-extra').value,
    hitl:      $('#f-hitl').checked,
  };
  btn.disabled = true;
  status.textContent = 'submitting…';
  try {
    const r = await fetch('/api/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      status.textContent = `error: ${data.error || r.status}`;
      toast(data.error || 'submit failed', 'err');
      return;
    }
    status.textContent = `launched · ${data.task_id} · pid ${data.pid}`;
    toast(`launched ${data.task_id}`, 'ok');
    await refreshSubmitTasks();
  } catch (e) {
    status.textContent = `error: ${e.message || e}`;
    toast('submit failed', 'err');
  } finally {
    btn.disabled = false;
  }
}

async function refreshSubmitTasks() {
  const wrap = $('#submit-tasks');
  if (!wrap) return;
  try {
    const data = await getJSON('/api/tasks');
    const tasks = data.tasks || [];
    if (!tasks.length) {
      wrap.innerHTML = `<p class="empty-prose"><em>no tasks yet</em></p>`;
      return;
    }
    const html = tasks.map(t => {
      const f = t.form || {};
      const desc = `${esc(f.algorithm || '?')} · ${esc(f.dataset || '?')} · ${esc(f.model || '?')}`;
      const sev = {
        running:  'live',
        sleeping: 'wait',
        done:     'ok',
        failed:   'err',
        killed:   'stop',
      }[t.status] || 'stop';
      const age = t.started_ts ? relTime(t.started_ts) : '—';
      const deleteBtn = `<button class="task-kill-btn" data-task="${esc(t.task_id)}" data-status="${esc(t.status)}" title="remove this task" aria-label="remove">×</button>`;
      return `<div class="task-row" data-task="${esc(t.task_id)}">
        <div class="task-row-head">
          <span class="task-pill sev-${sev}">${esc(t.status)}</span>
          <span class="task-id">${esc(t.task_id)}</span>
          <span class="task-age">${esc(age)}</span>
          ${deleteBtn}
        </div>
        <div class="task-desc">${desc}</div>
      </div>`;
    }).join('');
    wrap.innerHTML = html;
    wrap.querySelectorAll('.task-kill-btn').forEach(btn =>
      btn.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        const tid = btn.dataset.task;
        const status = btn.dataset.status;
        const msg = (status === 'running')
          ? `Task ${tid} is still running. Kill it and remove?`
          : `Remove task ${tid}?`;
        if (!confirm(msg)) return;
        btn.disabled = true;
        try {
          const r = await fetch(`/api/task/${encodeURIComponent(tid)}`, { method: 'DELETE' });
          const data = await r.json();
          if (!r.ok || data.error) { toast(data.error || 'remove failed', 'err'); return; }
          toast(data.was_running ? `killed + removed ${tid}` : `removed ${tid}`, 'ok');
        } catch (e) {
          toast(`remove failed: ${e.message || e}`, 'err');
        } finally {
          refreshSubmitTasks();
        }
      }));
  } catch (e) {
    wrap.innerHTML = `<p class="empty-prose"><em>failed to load: ${esc(e.message || e)}</em></p>`;
  }
}

function relTime(ts) {
  const sec = Math.max(0, Math.floor(Date.now()/1000 - ts));
  if (sec < 60)     return `${sec}s ago`;
  if (sec < 3600)   return `${Math.floor(sec/60)}m ago`;
  if (sec < 86400)  return `${Math.floor(sec/3600)}h ago`;
  return `${Math.floor(sec/86400)}d ago`;
}

/* ════════════════════════════ go ════════════════════════════ */
boot();

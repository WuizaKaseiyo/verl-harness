/* verl-harness-web — dashboard logic.
   Mirrors fastharness-web's structure (no SPA framework) but adds the
   training-specific panels: progress chart, anomalies, log tail, job card. */

import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs';

const $ = (s) => document.querySelector(s);
const on = (sel, ev, fn) => { const el = $(sel); if (el) el.addEventListener(ev, fn); };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
));

const S = {
  config: { live: true },
  harness: null,
  run: null,
  current: null,        // { kind, id, editPath }
  expanded: new Set(),
  skillFiles: {},
  activeStateSkills: new Set(),
  editing: false,
  renderSeq: 0,
  zoom: 1,
  chart: null,
  logOffset: 0,
};

const isDark = () => document.documentElement.dataset.theme !== 'light';

/* ---------- Mermaid theme ---------- */
function initMermaid() {
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'base',
    flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis' },
    fontFamily: "'JetBrains Mono', monospace",
    themeVariables: {
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: '12px',
      background: 'transparent',
      primaryColor: '#131a37',
      primaryBorderColor: '#2c3a70',
      primaryTextColor: '#e6ecff',
      lineColor: '#2c3a70',
      defaultLinkColor: '#2c3a70',
    },
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('vh-theme', theme);
  const btn = $('#theme-btn');
  if (btn) btn.textContent = theme === 'light' ? '☾' : '☼';
  if (S.harness) renderGraph();
  if (S.chart) styleChart();
}

/* ---------- API helpers ---------- */
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

/* ---------- toast ---------- */
let toastTimer;
function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = 'toast'; }, 2400);
}

/* ============================================================
   BOOT
   ============================================================ */
async function boot() {
  try {
    S.config = await getJSON('/api/config');
  } catch (e) {
    $('#harness-title').textContent = 'API unreachable';
    return;
  }
  $('#mode-chip').textContent = S.config.live ? 'live' : 'static';

  applyTheme(localStorage.getItem('vh-theme') || 'dark');
  initMermaid();

  await reloadHarness();
  await reloadRun();
  await renderGraph();
  renderSkills();
  selectOverview();

  initZoom();
  if (S.config.live) connectSSE();

  on('#theme-btn', 'click', () => applyTheme(isDark() ? 'light' : 'dark'));
  on('#edit-btn',  'click', openEditor);
  on('#save-btn',  'click', saveEdit);
  on('#cancel-btn','click', () => { exitEditor(); reselect(); });

  // Periodic training-panel refresh — keeps the chart, log tail, anomalies fresh
  setInterval(refreshTrainingPanels, S.config.live ? 5000 : 10000);
  refreshTrainingPanels();
}

/* ---------- load harness + run ---------- */
async function reloadHarness() {
  S.harness = await getJSON('/api/harness');
  $('#harness-title').textContent = S.harness.title;
  const cap = (S.harness.required_capabilities || []).length;
  $('#harness-sub').textContent =
    `${S.harness.state_count} states · ${cap} capabilities · HITL ${S.harness.hitl || '—'}`;
}

async function reloadRun() {
  try { S.run = await getJSON('/api/run'); }
  catch { S.run = { run_id: null, status: 'idle', entries: [], current: null }; }
  const badge = $('#run-badge');
  const status = S.run.status || 'idle';
  badge.dataset.status = status;
  badge.querySelector('.run-label').textContent = status;
  S.activeStateSkills = new Set();
  if (S.run.current && S.harness && S.harness.mermaid) {
    // Find the state's skills from the parsed harness — fetch the state.
    try {
      const stateData = await getJSON(`/api/state/${S.run.current}`);
      (stateData.skills || []).forEach(s => S.activeStateSkills.add(s));
    } catch (_) {}
  }
}

/* ============================================================
   GRAPH (Mermaid)
   ============================================================ */
async function renderGraph() {
  if (!S.harness) return;
  let src = S.harness.mermaid;

  // Append classDef assignments based on runtime state.
  const visited = new Set(S.run?.visited || []);
  const current = S.run?.current;
  const terminals = new Set(S.harness.terminal_states || []);
  const overview = S.harness.overview_node;

  const classes = [];
  if (overview) classes.push(`class ${overview} fhOverview`);
  for (const name of S.harness.states) {
    if (name === current) classes.push(`class ${name} fhLive`);
    else if (S.current?.kind === 'state' && S.current?.id === name)
      classes.push(`class ${name} fhSel`);
    else if (visited.has(name)) classes.push(`class ${name} fhVisited`);
    else if (terminals.has(name)) classes.push(`class ${name} fhTerminal`);
  }
  src = src + '\n' + classes.join('\n');

  const seq = ++S.renderSeq;
  try {
    const { svg } = await mermaid.render(`g${Date.now()}`, src);
    if (seq !== S.renderSeq) return;
    const wrap = $('#graph');
    wrap.innerHTML = svg;
    // Click handlers
    wrap.querySelectorAll('.node').forEach(n => {
      const id = n.id?.replace(/^flowchart-/, '').split('-')[0];
      if (id) {
        n.addEventListener('click', () => {
          if (id === S.harness.overview_node) selectOverview();
          else selectState(id);
        });
      }
    });
    applyZoom();
  } catch (e) {
    $('#graph').textContent = 'mermaid render failed';
    console.error(e);
  }
}

/* ============================================================
   SKILLS PANEL
   ============================================================ */
function renderSkills() {
  if (!S.harness) return;
  const wrap = $('#skills-tree');
  wrap.innerHTML = '';
  const skills = S.harness.skills || [];
  $('#skills-hint').textContent = `${skills.length} folders`;
  if (skills.length === 0) {
    wrap.innerHTML = '<div class="empty-hint">No skill folders.</div>';
    return;
  }
  for (const path of skills) {
    const row = document.createElement('div');
    row.className = 'skill-row';
    if (S.activeStateSkills.has(path)) row.classList.add('in-active-state');
    if (S.current?.kind === 'skill' && S.current?.id === path) row.classList.add('active');
    row.innerHTML = `
      <span class="skill-name">${esc(path)}</span>
      <span class="skill-files">→</span>`;
    row.addEventListener('click', () => selectSkill(path));
    wrap.appendChild(row);
  }
}

/* ============================================================
   DETAIL / INSPECTOR
   ============================================================ */
async function selectOverview() {
  S.current = { kind: 'overview', id: S.harness?.overview_node, editPath: 'task-overview.md' };
  exitEditor();
  const data = await getJSON(`/api/state/${S.harness.overview_node}`);
  renderDetail('Overview', S.harness.title, data.compiled, data.editable, data.file);
  await renderGraph();
  renderSkills();
}

async function selectState(name) {
  S.current = { kind: 'state', id: name, editPath: `states/${name}.md` };
  exitEditor();
  const data = await getJSON(`/api/state/${name}`);
  S.activeStateSkills = new Set(data.skills || []);
  renderDetail(data.is_terminal ? 'Terminal state' : 'State',
               name, data.compiled, data.editable, data.file);
  await renderGraph();
  renderSkills();
}

async function selectSkill(path) {
  S.current = { kind: 'skill', id: path };
  exitEditor();
  const data = await getJSON(`/api/skill?path=${encodeURIComponent(path)}`);
  const subtitle = data.used_by?.length ?
    `used by ${data.used_by.join(', ')}` : 'unused';
  renderDetail(`Skill · ${subtitle}`, path, data.compiled, false, null);
  renderSkills();
}

function renderDetail(eyebrow, title, markdownText, editable, file) {
  $('#ctx-tag').textContent = '[03]';
  $('#ctx-title').textContent = title || 'Inspector';
  const rendered = $('#rendered');
  rendered.innerHTML = marked.parse(markdownText || '');
  rendered.hidden = false;
  $('#editor-host').hidden = true;
  const eb = $('#edit-btn');
  if (editable && file) {
    eb.hidden = false; eb.dataset.path = file;
  } else {
    eb.hidden = true; delete eb.dataset.path;
  }
}

function reselect() {
  if (!S.current) return;
  if (S.current.kind === 'overview') selectOverview();
  else if (S.current.kind === 'state') selectState(S.current.id);
  else if (S.current.kind === 'skill') selectSkill(S.current.id);
}

/* ---------- editor ---------- */
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
    mode: 'markdown',
    lineNumbers: true,
    lineWrapping: true,
    indentUnit: 2,
    tabSize: 2,
  });
  cm.setValue(data.content || '');
  cm.on('change', () => {
    $('#save-status').textContent = '● unsaved';
    $('#save-status').className = 'save-status dirty';
  });
  S.editing = true;
}

async function saveEdit() {
  const path = $('#edit-btn').dataset.path;
  if (!path || !cm) return;
  try {
    await putJSON('/api/file', { path, content: cm.getValue() });
    $('#save-status').textContent = '✓ saved';
    $('#save-status').className = 'save-status saved';
    toast('saved');
  } catch (e) {
    toast(e.message, 'err');
  }
}

function exitEditor() {
  S.editing = false;
  $('#save-btn').hidden = true;
  $('#cancel-btn').hidden = true;
  const eb = $('#edit-btn');
  if (eb.dataset.path) eb.hidden = false;
  const host = $('#editor-host');
  host.hidden = true;
  host.innerHTML = '';
  $('#rendered').hidden = false;
  cm = null;
}

/* ============================================================
   TRAINING PANELS — progress chart, anomalies, job card, log tail
   ============================================================ */
async function refreshTrainingPanels() {
  if (!S.harness) return;
  // Show the panels whenever a run exists (even if idle), so the
  // user can browse a finished run's chart + summary.
  const hasRun = !!S.run?.run_id;
  $('#training-panels').hidden = !hasRun;
  if (!hasRun) return;

  const [progress, anomalies, job] = await Promise.allSettled([
    getJSON('/api/progress'),
    getJSON('/api/anomalies'),
    getJSON('/api/job'),
  ]);

  if (progress.status === 'fulfilled') renderProgress(progress.value);
  if (anomalies.status === 'fulfilled') renderAnomalies(anomalies.value.anomalies || []);
  if (job.status === 'fulfilled') renderJob(job.value);

  await refreshLogTail();
}

/* ---------- progress chart ---------- */
const SERIES_META = [
  { key: 'mean_ep_return_100', label: 'mean_ep_return_100', color: '#aaff00' },
  { key: 'reward',             label: 'reward',             color: '#aaff00' },
  { key: 'train_loss',         label: 'train_loss',         color: '#00e5ff' },
  { key: 'loss',               label: 'loss',               color: '#00e5ff' },
  { key: 'policy_entropy',     label: 'policy_entropy',     color: '#9b5dff' },
  { key: 'kl',                 label: 'kl',                 color: '#ff2a87' },
  { key: 'entropy_coef',       label: 'entropy_coef',       color: '#ffb20a' },
];

function renderProgress(data) {
  const empty = !data || !data.rows || !data.columns?.length;
  $('#progress-empty').hidden = !empty;
  const canvas = $('#chart-progress');
  $('#progress-hint').textContent = empty ? 'no data' :
    `${data.rows} rows · ${data.columns.length} cols`;
  if (empty) {
    if (S.chart) { S.chart.destroy(); S.chart = null; }
    canvas.style.display = 'none';
    return;
  }
  canvas.style.display = '';
  // Pick an x-axis: prefer `env_steps`, fall back to step / update / row index.
  const cols = data.columns;
  const xCol = cols.find(c => /env_steps|^step$|update/i.test(c));
  const xs = xCol ? data.series[xCol] : data.series[cols[0]].map((_, i) => i);

  // Choose series that have a non-trivial numeric range.
  const datasets = [];
  for (const meta of SERIES_META) {
    if (!cols.includes(meta.key)) continue;
    const ys = data.series[meta.key];
    if (!ys || !ys.length) continue;
    if (!ys.some(v => typeof v === 'number' && isFinite(v))) continue;
    datasets.push({
      label: meta.label,
      data: xs.map((x, i) => ({ x, y: ys[i] })),
      borderColor: meta.color,
      backgroundColor: hexToRgba(meta.color, 0.15),
      borderWidth: 1.6,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      yAxisID: /loss|kl|entropy_coef/.test(meta.key) ? 'yRight' : 'yLeft',
    });
  }

  if (datasets.length === 0) {
    if (S.chart) { S.chart.destroy(); S.chart = null; }
    $('#progress-empty').hidden = false;
    $('#progress-empty').textContent = 'progress.csv has rows but no plottable numeric series.';
    canvas.style.display = 'none';
    return;
  }

  const cfg = chartConfig(xCol || 'index', datasets);
  if (S.chart) {
    S.chart.data = cfg.data;
    S.chart.options = cfg.options;
    S.chart.update('none');
  } else {
    S.chart = new Chart(canvas, cfg);
  }
}

function chartConfig(xLabel, datasets) {
  const dark = isDark();
  const grid = dark ? 'rgba(44, 58, 112, 0.4)' : 'rgba(204, 213, 236, 0.7)';
  const tick = dark ? '#adb6d8' : '#5d6a93';
  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: xLabel, color: tick, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          ticks: { color: tick, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid:  { color: grid, drawBorder: false },
        },
        yLeft: {
          type: 'linear', position: 'left',
          ticks: { color: tick, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid:  { color: grid, drawBorder: false },
          title: { display: true, text: 'reward / return', color: '#aaff00',
                   font: { family: "'JetBrains Mono', monospace", size: 10 } },
        },
        yRight: {
          type: 'linear', position: 'right',
          ticks: { color: tick, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid:  { drawOnChartArea: false },
          title: { display: true, text: 'loss / kl / coef', color: '#00e5ff',
                   font: { family: "'JetBrains Mono', monospace", size: 10 } },
        },
      },
      plugins: {
        legend: {
          labels: { color: tick, font: { family: "'JetBrains Mono', monospace", size: 11 },
                    usePointStyle: true, pointStyle: 'rectRounded', padding: 14 },
        },
        tooltip: {
          backgroundColor: dark ? '#0c1126' : '#ffffff',
          borderColor: '#2c3a70',
          borderWidth: 1,
          titleColor: '#00e5ff',
          bodyColor: dark ? '#e6ecff' : '#0a1233',
          titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
          bodyFont:  { family: "'JetBrains Mono', monospace", size: 11 },
        },
      },
    },
  };
}

function styleChart() {
  if (!S.chart) return;
  const cfg = chartConfig(S.chart.options.scales.x.title.text, S.chart.data.datasets);
  S.chart.options = cfg.options;
  S.chart.update('none');
}

function hexToRgba(hex, a) {
  const m = hex.replace('#', '').match(/.{2}/g);
  if (!m) return hex;
  const [r, g, b] = m.map(x => parseInt(x, 16));
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

/* ---------- anomalies ---------- */
function renderAnomalies(rows) {
  const wrap = $('#anomalies-body');
  $('#anomalies-hint').textContent =
    rows.length ? `${rows.length} entries` : '';
  if (!rows.length) {
    wrap.innerHTML = '<div class="empty-hint">No anomalies detected.</div>';
    return;
  }
  wrap.innerHTML = rows.map(r => `
    <div class="anomaly-row sev-${esc(r.severity)}">
      <span class="anomaly-ts">${esc(r.timestamp || '—')}</span>
      <span class="anomaly-body">${esc(r.body)}</span>
    </div>
  `).join('');
}

/* ---------- job card ---------- */
function renderJob(data) {
  const info = data.info || {};
  const status = data.status || {};
  const wrap = $('#job-body');
  const empty = !Object.keys(info).length && !Object.keys(status).length;
  $('#job-hint').textContent = status.status ? `→ ${status.status}` :
                              (info.target || '');
  if (empty) {
    wrap.innerHTML = '<div class="empty-hint">No job info yet.</div>';
    return;
  }
  const cells = [];
  const statusCls = status.status ? `status-${(status.status).toLowerCase()}` : '';
  if (status.status) cells.push(['status', status.status, statusCls]);
  if (info.target) cells.push(['target', info.target]);
  if (info.pid)            cells.push(['pid', info.pid]);
  if (info.slurm_jobid)    cells.push(['slurm jobid', info.slurm_jobid]);
  if (info.remote_alias)   cells.push(['remote', info.remote_alias]);
  if (info.started_at)     cells.push(['started', info.started_at]);
  if (status.final_step)   cells.push(['final step', status.final_step]);
  if (status.final_epoch)  cells.push(['final epoch', status.final_epoch]);
  if (status.final_loss)   cells.push(['final loss', status.final_loss]);
  if (status.final_reward) cells.push(['final reward', status.final_reward]);
  if (status.last_checkpoint) cells.push(['last ckpt', status.last_checkpoint]);
  if (info.output_dir)     cells.push(['output dir', info.output_dir]);

  wrap.innerHTML = cells.map(([k, v, cls]) => `
    <div class="job-cell ${cls || ''}">
      <span class="job-key">${esc(k)}</span>
      <span class="job-val">${esc(v)}</span>
    </div>
  `).join('');
}

/* ---------- log tail ---------- */
async function refreshLogTail() {
  try {
    const data = await getJSON(`/api/logs?since=${S.logOffset}`);
    if (data.size === undefined) return;
    if (data.size < S.logOffset) S.logOffset = 0;       // file truncated
    if (data.content) {
      appendLog(data.content);
    }
    S.logOffset = data.size;
    $('#logs-hint').textContent = `${humanBytes(data.size)} · offset ${S.logOffset}`;
  } catch (_) {}
}

function appendLog(chunk) {
  const pre = $('#log-tail');
  const lines = chunk.split('\n').map(line => {
    let cls = '';
    if (/error|cuda out of memory|nan|inf|traceback/i.test(line)) cls = 'log-line-error';
    else if (/warn|preempt|timeout/i.test(line)) cls = 'log-line-warn';
    else if (/^step:\s*\d+|^update=\d+|env_steps?=/.test(line)) cls = 'log-line-step';
    else if (/training (finished|complete)|saved/i.test(line)) cls = 'log-line-ok';
    return cls ? `<span class="${cls}">${esc(line)}</span>` : esc(line);
  });
  pre.insertAdjacentHTML('beforeend', lines.join('\n'));
  // Cap rendered output at ~400 KB
  if (pre.innerHTML.length > 400_000) {
    pre.innerHTML = pre.innerHTML.slice(-200_000);
  }
  pre.scrollTop = pre.scrollHeight;
}

function humanBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
  return `${(n/1024/1024).toFixed(2)} MB`;
}

/* ============================================================
   SSE
   ============================================================ */
function connectSSE() {
  const es = new EventSource('/events');
  es.addEventListener('changed', async () => {
    await reloadRun();
    await renderGraph();
    renderSkills();
    refreshTrainingPanels();
  });
  es.addEventListener('error', () => {
    // Soft reconnect after a backoff
    setTimeout(connectSSE, 3000);
    es.close();
  });
}

/* ============================================================
   Zoom
   ============================================================ */
function applyZoom() {
  const g = $('#graph');
  if (g) g.style.transform = `scale(${S.zoom})`;
  $('#zoom-level').textContent = `${Math.round(S.zoom * 100)}%`;
}
function initZoom() {
  on('#zoom-in',  'click', () => { S.zoom = Math.min(S.zoom + 0.1, 2.0); applyZoom(); });
  on('#zoom-out', 'click', () => { S.zoom = Math.max(S.zoom - 0.1, 0.5); applyZoom(); });
  on('#zoom-level', 'click', () => { S.zoom = 1.0; applyZoom(); });
}

/* ============================================================
   GO
   ============================================================ */
boot();

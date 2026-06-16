/* ═══════════════════════════════════════════════════════════════════════════
   app.js — wires the dashboard to the local backend (server.py).
     GET  /api/field        → grid + facts  → Plotly contour + readouts
     GET  /api/feasibility  → sweep verdict
     POST /api/caption      → "Explain this graph"
     POST /api/ask          → chat
   All same-origin; the OpenAI key never leaves the server.
   ═══════════════════════════════════════════════════════════════════════════ */
'use strict';

const $ = (id) => document.getElementById(id);

// matplotlib "inferno" sampled to stops, so the web map matches the saved PNG.
const INFERNO = [
  [0.000, '#000004'], [0.111, '#1b0c42'], [0.222, '#4b0c6b'], [0.333, '#781c6d'],
  [0.444, '#a52c60'], [0.556, '#cf4446'], [0.667, '#ed6925'], [0.778, '#fb9b06'],
  [0.889, '#f7d13d'], [1.000, '#fcffa4'],
];
const MONO = "'Fira Code', ui-monospace, monospace";
const C = { amber: '#f0883e', green: '#3fb950', blue: '#58a6ff', dim: '#8b949e', grid: '#161b22', line: '#2a3240' };

const STABILITY = ['', 'A', 'B', 'C', 'D', 'E', 'F'];
const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const compass = (deg) => COMPASS[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];

const ppmStr = (v) => v == null || isNaN(v) ? null
  : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ─── boot ────────────────────────────────────────────────────────────────────
loadField();
loadFeasibility();
wireExplain();
wireChat();

// ─── /api/field ───────────────────────────────────────────────────────────────
async function loadField() {
  try {
    const d = await (await fetch('/api/field')).json();
    renderScenario(d.scenario);
    renderPlot(d);
    renderMetrics(d.facts);
    renderDecay(d.facts);
    $('keyDetect').textContent = `${d.detect_level.toFixed(2)} ppm`;
    $('plot').closest('.plot-frame').classList.add('ready');
  } catch (e) {
    $('plotScrim').innerHTML = `<span style="color:#f85149">FIELD ERROR · ${esc(String(e.message))}</span>`;
  }
}

function renderScenario(s) {
  const rail = $('scenarioRail');
  const chips = [
    ['SOURCE Q', `${s.Q.toFixed(1)}<small>g/s</small>`],
    ['WIND', `${s.u.toFixed(1)}<small>m/s</small>`],
    ['HEIGHT', `${s.H.toFixed(1)}<small>m</small>`],
    ['STABILITY', `${STABILITY[s.stability_class] || s.stability_class}<small>cls ${s.stability_class}</small>`],
    ['BEARING', `${s.wind_dir_deg.toFixed(0)}°<small>from ${compass(s.wind_dir_deg)}</small>`],
  ];
  rail.innerHTML = chips.map(([k, v]) =>
    `<div class="chip"><span class="chip__k">${k}</span><span class="chip__v">${v}</span></div>`).join('');
}

function renderPlot(d) {
  const { xs, ys, ppm } = d;
  let pmax = 0;
  for (const row of ppm) for (const v of row) if (v > pmax) pmax = v;
  const z = ppm.map((row) => row.map((v) => Math.log10(v)));
  const zmin = Math.log10(d.background * 1.04);
  const zmax = Math.log10(pmax);
  const detectLog = Math.log10(d.detect_level);

  const ppmTicks = [2, 3, 5, 10, 20, 50, 100, 200, 300].filter((t) => t >= d.background && t <= pmax);

  const heat = {
    type: 'heatmap', x: xs, y: ys, z, zmin, zmax, colorscale: INFERNO, zsmooth: 'best',
    customdata: ppm,
    hovertemplate: '%{x:.0f} m downwind · %{y:.0f} m crosswind<br><b>%{customdata:.2f} ppm</b><extra></extra>',
    colorbar: {
      title: { text: 'ppm', font: { family: MONO, size: 10, color: C.dim }, side: 'top' },
      tickvals: ppmTicks.map(Math.log10), ticktext: ppmTicks.map(String),
      tickfont: { family: MONO, size: 9, color: C.dim },
      outlinecolor: C.line, outlinewidth: 1, thickness: 12, len: 0.92, x: 1.004,
    },
  };

  // single dashed iso-line at the detection limit (background + 3σ)
  const detect = {
    type: 'contour', x: xs, y: ys, z, showscale: false, hoverinfo: 'skip', autocontour: false,
    contours: { coloring: 'none', start: detectLog, end: detectLog, size: 1 },
    line: { color: C.green, width: 2, dash: 'dash' },
  };

  const source = {
    type: 'scatter', x: [0], y: [0], mode: 'markers', text: [`${d.scenario.Q} g/s`],
    marker: { symbol: 'triangle-up', size: 15, color: C.blue, line: { color: '#0a0d12', width: 1.5 } },
    hovertemplate: 'SOURCE · Q=%{text}<extra></extra>',
  };

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#0a0d12',
    font: { family: MONO, size: 11, color: C.dim },
    margin: { l: 62, r: 18, t: 14, b: 46 }, showlegend: false,
    xaxis: {
      title: { text: 'EASTING FROM SOURCE (m) — DOWNWIND ▸', font: { size: 10, color: C.dim } },
      range: [-4, 201], gridcolor: C.grid, zeroline: false, color: C.dim,
      linecolor: C.line, ticks: 'outside', tickcolor: C.line,
    },
    yaxis: {
      title: { text: 'CROSSWIND (m)', font: { size: 10, color: C.dim } },
      range: [-150, 150], gridcolor: C.grid, zeroline: false, color: C.dim,
      linecolor: C.line, ticks: 'outside', tickcolor: C.line,
    },
    shapes: [
      { type: 'line', x0: 100, x1: 100, yref: 'paper', y0: 0, y1: 1,
        line: { color: C.amber, width: 1, dash: 'dot' }, opacity: 0.5 },
    ],
    annotations: [
      { x: 188, y: 128, ax: 142, ay: 128, xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true, arrowhead: 2, arrowsize: 1.3, arrowwidth: 2, arrowcolor: C.blue, text: '' },
      { x: 165, y: 139, xref: 'x', yref: 'y', text: 'WIND ▸', showarrow: false,
        font: { family: MONO, size: 11, color: C.blue } },
      { x: 100, y: -150, xref: 'x', yref: 'y', yanchor: 'bottom', xanchor: 'left',
        text: ' 100 m · σ extrapolated ▸', showarrow: false, opacity: 0.85,
        font: { family: MONO, size: 9.5, color: C.amber } },
      { x: 0, y: 0, xref: 'x', yref: 'y', yshift: 24, xanchor: 'left', text: 'SOURCE',
        showarrow: false, font: { family: MONO, size: 9.5, color: C.blue } },
    ],
  };

  Plotly.newPlot('plot', [heat, detect, source], layout, {
    responsive: true, displaylogo: false, displayModeBar: 'hover',
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
  });
}

function renderMetrics(f) {
  const unit = (v, u) => v == null ? '—' : `${v}<small>${u}</small>`;
  $('mPeak').innerHTML = unit(ppmStr(f.peak_ppm), 'ppm');
  $('mPeakHint').textContent = f.peak_x != null ? `strongest reading, ~${f.peak_x.toFixed(0)} m out` : 'strongest reading';
  $('mReach').innerHTML = f.detect_reach_m == null ? '<small>&lt; grid</small>' : unit(f.detect_reach_m.toFixed(0), 'm');
  $('mWidth').innerHTML = f.half_width_at_50_m == null ? '<small>n/a</small>' : unit(f.half_width_at_50_m.toFixed(1), 'm');
  $('mThresh').innerHTML = unit(f.threshold.toFixed(2), 'ppm');
}

function renderDecay(f) {
  const pts = [[50, f.ppm_at_50], [100, f.ppm_at_100], [200, f.ppm_at_200]];
  const ref = f.ppm_at_50 || 1;
  $('decay').innerHTML = pts.map(([d, v]) => {
    const w = v == null ? 0 : Math.max(4, Math.min(100, (v / ref) * 100));
    const val = v == null ? 'n/a' : `${ppmStr(v)}<small>ppm</small>`;
    return `<div class="decay__cell"><div class="decay__d">${d} m</div>` +
           `<div class="decay__v">${val}</div>` +
           `<div class="decay__bar"><i style="--w:${w}%"></i></div></div>`;
  }).join('');
}

// ─── /api/feasibility ──────────────────────────────────────────────────────────
async function loadFeasibility() {
  try {
    const d = await (await fetch('/api/feasibility')).json();
    $('feasText').innerHTML = esc(d.verdict).replace(/(\d+(?:\.\d+)?\s*g\/s)/g, '<b>$1</b>');
  } catch (e) {
    $('feasText').textContent = 'Feasibility sweep unavailable.';
  }
}

// ─── /api/caption ("Explain this graph") ───────────────────────────────────────
function wireExplain() {
  const btn = $('explainBtn'), cap = $('captionText'), err = $('captionErr'), badge = $('aiBadge');
  btn.addEventListener('click', async () => {
    btn.setAttribute('aria-busy', 'true');
    btn.querySelector('span').textContent = 'Reading the field…';
    err.hidden = true;
    try {
      const d = await (await fetch('/api/caption', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_ai: true }),
      })).json();
      cap.textContent = d.caption;
      cap.classList.add('has-text');
      setBadge(badge, d.source);
      if (d.ai_error) {
        err.hidden = false;
        err.textContent = `AI unavailable: ${shorten(d.ai_error)} — showing the built-in template.`;
      }
    } catch (e) {
      err.hidden = false;
      err.textContent = `Could not reach the server: ${e.message}`;
    } finally {
      btn.removeAttribute('aria-busy');
      btn.querySelector('span').textContent = 'Explain this graph';
    }
  });
}

function setBadge(el, source) {
  el.classList.remove('badge--idle', 'badge--ai', 'badge--tmpl');
  if (source === 'openai') { el.classList.add('badge--ai'); el.textContent = 'AI · OpenAI'; }
  else { el.classList.add('badge--tmpl'); el.textContent = 'Template'; }
}
const shorten = (s) => s.length > 90 ? s.slice(0, 90) + '…' : s;

// ─── /api/ask (chat) ───────────────────────────────────────────────────────────
const SEEDS = [
  'What does the green dashed line mean?',
  'Why is the plume so narrow?',
  'How far can we still detect this leak?',
  'Why does the graph start at 10 m?',
];

function wireChat() {
  const seeds = $('chatSeeds'), form = $('chatForm'), input = $('chatInput');
  seeds.innerHTML = SEEDS.map((q) => `<button type="button" class="seed">${esc(q)}</button>`).join('');
  seeds.querySelectorAll('.seed').forEach((b) =>
    b.addEventListener('click', () => { input.value = b.textContent; ask(b.textContent); }));
  form.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
}

async function ask(question) {
  question = (question || '').trim();
  if (!question) return;
  const input = $('chatInput'), send = $('chatSend');
  input.value = ''; send.disabled = true;

  addMsg('user', '>', question);
  const pending = addMsg('ai', 'AI', '', true);

  try {
    const d = await (await fetch('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })).json();
    finishMsg(pending, d.answer, isErrorAnswer(d.answer));
    setBadge($('aiBadge'), isErrorAnswer(d.answer) ? 'template' : 'openai');
  } catch (e) {
    finishMsg(pending, `Could not reach the server: ${e.message}`, true);
  } finally {
    send.disabled = false; input.focus();
  }
}

const isErrorAnswer = (a) =>
  /^(couldn'?t reach|no openai_api_key|the 'openai')/i.test(a) || /api key is invalid/i.test(a);

function addMsg(role, tag, text, pending) {
  const log = $('chatLog');
  const el = document.createElement('div');
  el.className = `msg msg--${role}${pending ? ' msg--pending' : ''}`;
  el.innerHTML = `<span class="msg__tag">${tag}</span><span class="msg__body">${esc(text)}</span>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function finishMsg(el, text, isErr) {
  el.classList.remove('msg--pending');
  if (isErr) el.classList.add('msg--err');
  el.querySelector('.msg__body').textContent = text;
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
}

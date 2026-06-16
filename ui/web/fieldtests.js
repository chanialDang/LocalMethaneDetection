/* ═══════════════════════════════════════════════════════════════════════════
   fieldtests.js — upload / store / process / interpret / compare real readings.
     POST /api/fieldtests              upload a CSV   → store raw, return processed
     POST /api/fieldtests/sample       synthetic test → same shape
     GET  /api/fieldtests              list + summaries
     GET  /api/fieldtests/<id>         detail + full series
     GET  /api/fieldtests/compare      overlay + across-test spread
     POST /api/fieldtests/<id>/interpret | /ask   AI (or template) read + chat
     DELETE /api/fieldtests/<id>
   Same-origin; the OpenAI key never leaves the server.
   ═══════════════════════════════════════════════════════════════════════════ */
'use strict';

const $ = (id) => document.getElementById(id);
const C = CH4.COLOR;
const MONO = CH4.MONO;
const PLOT_CFG = {
  responsive: true, displaylogo: false, displayModeBar: 'hover',
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
};

const state = { tests: [], currentId: null, compareSel: new Set() };

// ─── boot ────────────────────────────────────────────────────────────────────
loadList();
wireUpload();
wireSample();
wireInterpret();
$('compareBtn').addEventListener('click', () => loadCompare([...state.compareSel]));
CH4.mountChat({
  logEl: $('ftChatLog'), formEl: $('ftChatForm'), inputEl: $('ftChatInput'),
  sendEl: $('ftChatSend'), seedsEl: $('ftChatSeeds'),
  seeds: ['Was methane detected, and how confident?',
          'How big was the event versus the noise floor?',
          'What could cause a false negative here?',
          'How should I read the baseline line?'],
  endpoint: () => `/api/fieldtests/${state.currentId}/ask`,
  buildBody: (q) => ({ question: q }),
  onSource: setBadge,
});

// ─── list ──────────────────────────────────────────────────────────────────────
async function loadList() {
  try {
    const d = await (await fetch('/api/fieldtests')).json();
    if (d.backend) {
      $('backendTxt').textContent =
        d.backend.includes('Postgres') ? 'RAILWAY POSTGRES' : 'LOCAL SQLITE';
    }
    renderList(d.tests || []);
  } catch (e) {
    $('testList').innerHTML =
      `<p class="testlist__empty">Could not load tests: ${CH4.esc(e.message)}</p>`;
  }
}

function renderList(tests) {
  state.tests = tests;
  const list = $('testList');
  if (!tests.length) {
    list.innerHTML = '<p class="testlist__empty">No saved tests yet — upload a CSV ' +
      'or hit <b>Generate sample test</b>.</p>';
    return;
  }
  list.innerHTML = tests.map((t) => {
    const id = t.meta.id, f = t.facts;
    const sel = id === state.currentId ? ' is-selected' : '';
    const chk = state.compareSel.has(id) ? ' checked' : '';
    const date = t.meta.test_date ? ` · ${CH4.esc(t.meta.test_date)}` : '';
    const site = t.meta.site ? CH4.esc(t.meta.site) : 'unspecified site';
    const det = f.detected
      ? `<span class="test-row__badge is-ok" title="${f.confidence_sigmas.toFixed(1)}σ">DET</span>`
      : '<span class="test-row__badge is-no" title="no sustained event">—</span>';
    return `<div class="test-row${sel}" data-id="${id}">
        <input type="checkbox" class="test-row__check"${chk} aria-label="select for compare" />
        <button type="button" class="test-row__main">
          <span class="test-row__name">${CH4.esc(t.meta.name || 'untitled')}</span>
          <span class="test-row__meta">${site}${date} · ${f.n_samples} pts</span>
        </button>
        ${det}
        <button type="button" class="test-row__del" title="Delete test" aria-label="Delete">✕</button>
      </div>`;
  }).join('');

  list.querySelectorAll('.test-row').forEach((row) => {
    const id = Number(row.dataset.id);
    row.querySelector('.test-row__main').addEventListener('click', () => selectTest(id));
    row.querySelector('.test-row__check').addEventListener('change', (e) => toggleCompare(id, e.target.checked));
    row.querySelector('.test-row__del').addEventListener('click', () => deleteTest(id));
  });
}

function toggleCompare(id, on) {
  if (on) state.compareSel.add(id); else state.compareSel.delete(id);
  const n = state.compareSel.size;
  const btn = $('compareBtn');
  btn.textContent = `Compare (${n})`;
  btn.disabled = n < 2;
}

// ─── selection / single-test render ─────────────────────────────────────────────
async function selectTest(id) {
  try {
    const d = await (await fetch(`/api/fieldtests/${id}`)).json();
    if (d.error) return;
    state.currentId = id;
    renderSingle(d);
  } catch (e) { /* ignore transient */ }
}

function selectFromDetail(d) {           // d from upload/sample (has id/meta/facts/series)
  state.currentId = d.id;
  renderSingle(d);
}

function renderSingle(d) {
  $('readoutLabel').textContent = 'READOUT';
  $('readoutSub').textContent =
    `${CH4.esc(d.meta.name || 'test')} · ${d.series && d.series.weather_corrected ? 'weather-corrected' : 'no weather correction'}`;
  $('ftEmpty').hidden = true;
  $('seriesKey').hidden = false;
  $('ftMetrics').hidden = false;
  $('spreadPanel').hidden = true;

  renderMetrics(d.facts);
  if (d.series) renderSinglePlot(d.series);
  renderList(state.tests);             // refresh selection highlight
  enableAssistant(true);
  resetAssistant();
}

function renderMetrics(f) {
  const unit = (v, u) => v == null ? '—' : `${v}<small>${u}</small>`;
  const det = !!f.detected;
  $('cardDetect').dataset.accent = det ? 'green' : 'red';
  $('mDetected').textContent = det ? 'DETECTED' : 'NONE';
  $('mDetectedHint').textContent = det
    ? `${f.confidence_sigmas.toFixed(1)}σ above the noise floor`
    : 'no sustained event above 3σ';
  $('mExcess').innerHTML = unit(CH4.ppmStr(f.excess_peak_ppm), 'ppm');
  $('mConf').innerHTML = f.confidence_sigmas ? `${f.confidence_sigmas.toFixed(1)}<small>σ</small>` : '—';
  $('mDur').textContent = det ? CH4.fmtSeconds(f.event_duration_s) : '—';
}

function axisStyle(extra) {
  return Object.assign({
    gridcolor: C.grid, zeroline: false, color: C.dim, linecolor: C.line,
    ticks: 'outside', tickcolor: C.line, tickfont: { family: MONO, size: 9 },
  }, extra);
}

function renderSinglePlot(s) {
  const t = s.time;
  const raw = {
    type: 'scatter', mode: 'lines', x: t, y: s.raw, xaxis: 'x', yaxis: 'y',
    line: { color: C.amber, width: 1.5 },
    hovertemplate: '%{x:.0f}s · %{y:.2f} ppm<extra>raw</extra>',
  };
  const base = {
    type: 'scatter', mode: 'lines', x: t, y: s.baseline, xaxis: 'x', yaxis: 'y',
    line: { color: C.faint, width: 1, dash: 'dot' }, hoverinfo: 'skip',
  };
  const exc = {
    type: 'scatter', mode: 'lines', x: t, y: s.excess, xaxis: 'x', yaxis: 'y2',
    line: { color: C.green, width: 1.6 }, fill: 'tozeroy', fillcolor: 'rgba(63,185,80,0.08)',
    hovertemplate: '%{x:.0f}s · +%{y:.2f} ppm<extra>excess</extra>',
  };

  const shapes = [{
    type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2',
    y0: s.threshold, y1: s.threshold, line: { color: C.green, width: 1.5, dash: 'dash' },
  }];
  const annotations = [{
    xref: 'paper', x: 0.006, yref: 'y2', y: s.threshold, yanchor: 'bottom', xanchor: 'left',
    text: `3σ · ${s.threshold.toFixed(2)} ppm`, showarrow: false,
    font: { family: MONO, size: 9, color: C.green },
  }];
  const win = s.detection && s.detection.window;
  if (win) {
    shapes.push({
      type: 'rect', xref: 'x', x0: win.start_s, x1: win.end_s, yref: 'paper', y0: 0, y1: 1,
      fillcolor: 'rgba(63,185,80,0.10)', line: { width: 0 }, layer: 'below',
    });
    annotations.push({
      xref: 'x', x: (win.start_s + win.end_s) / 2, yref: 'paper', y: 1.0, yanchor: 'bottom',
      text: 'DETECTED EVENT', showarrow: false, font: { family: MONO, size: 9, color: C.green },
    });
  }

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#0a0d12',
    font: { family: MONO, size: 11, color: C.dim },
    margin: { l: 60, r: 16, t: 20, b: 42 }, showlegend: false,
    xaxis: axisStyle({ domain: [0, 1], anchor: 'y2', title: { text: 'TIME (s)', font: { size: 10, color: C.dim } } }),
    yaxis: axisStyle({ domain: [0.56, 1.0], title: { text: 'ppm', font: { size: 10, color: C.dim } } }),
    yaxis2: axisStyle({ domain: [0, 0.44], title: { text: 'excess', font: { size: 10, color: C.dim } } }),
    shapes, annotations,
  };
  Plotly.newPlot('ftPlot', [raw, base, exc], layout, PLOT_CFG);
}

// ─── compare ────────────────────────────────────────────────────────────────────
async function loadCompare(ids) {
  if (ids.length < 2) return;
  try {
    const d = await (await fetch(`/api/fieldtests/compare?ids=${ids.join(',')}`)).json();
    renderCompare(d);
  } catch (e) { /* ignore */ }
}

function renderCompare(d) {
  $('readoutLabel').textContent = 'COMPARE';
  $('readoutSub').textContent = `${d.spread.n_tests} tests overlaid · ${d.spread.n_detected} detected`;
  $('ftEmpty').hidden = true;
  $('seriesKey').hidden = true;
  $('ftMetrics').hidden = true;
  $('spreadPanel').hidden = false;

  const ramp = [C.amber, C.green, C.blue, '#d2a8ff', '#56d4dd', '#ff9a52', '#f778ba'];
  const traces = [];
  d.tests.forEach((t, i) => {
    if (!t.series) return;
    traces.push({
      type: 'scatter', mode: 'lines', x: t.series.time, y: t.series.excess,
      name: t.meta.name, line: { color: ramp[i % ramp.length], width: 1.5 },
      hovertemplate: `%{x:.0f}s · +%{y:.2f} ppm<extra>${CH4.esc(t.meta.name)}</extra>`,
    });
  });
  const thrTest = d.tests.find((t) => t.series);
  const shapes = thrTest ? [{
    type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y',
    y0: thrTest.series.threshold, y1: thrTest.series.threshold,
    line: { color: C.dim, width: 1, dash: 'dash' },
  }] : [];

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#0a0d12',
    font: { family: MONO, size: 11, color: C.dim }, margin: { l: 60, r: 16, t: 20, b: 42 },
    showlegend: true,
    legend: { font: { family: MONO, size: 9, color: C.dim }, bgcolor: 'rgba(0,0,0,0)', x: 1, xanchor: 'right', y: 1 },
    xaxis: axisStyle({ title: { text: 'TIME (s)', font: { size: 10, color: C.dim } } }),
    yaxis: axisStyle({ title: { text: 'excess (ppm above baseline)', font: { size: 10, color: C.dim } } }),
    shapes,
  };
  Plotly.newPlot('ftPlot', traces, layout, PLOT_CFG);
  renderSpread(d);
}

function spreadStat(label, o, unit) {
  if (!o) return `<div class="spread__stat"><span class="spread__k">${label}</span><span class="spread__v">—</span></div>`;
  return `<div class="spread__stat"><span class="spread__k">${label}</span>
      <span class="spread__v">${CH4.ppmStr(o.mean)}<small>${unit}</small></span>
      <span class="spread__range">${CH4.ppmStr(o.min)}–${CH4.ppmStr(o.max)} · Δ${CH4.ppmStr(o.spread)}</span></div>`;
}

function renderSpread(d) {
  const sp = d.spread;
  const strip = `<div class="spread__stats">
      ${spreadStat('peak excess', sp.excess_peak_ppm, ' ppm')}
      ${spreadStat('confidence', sp.confidence_sigmas, 'σ')}
      <div class="spread__stat"><span class="spread__k">detected</span>
        <span class="spread__v">${sp.n_detected}<small>/ ${sp.n_tests}</small></span>
        <span class="spread__range">tests with an event</span></div>
    </div>`;
  const rows = d.tests.map((t) => `<tr>
      <td class="spread__name">${CH4.esc(t.meta.name)}</td>
      <td>${t.facts.detected ? '<span class="dot dot--ok"></span>yes' : '<span class="dot dot--no"></span>no'}</td>
      <td>${CH4.ppmStr(t.facts.excess_peak_ppm) ?? '—'}</td>
      <td>${t.facts.confidence_sigmas ? t.facts.confidence_sigmas.toFixed(1) + 'σ' : '—'}</td>
    </tr>`).join('');
  $('spreadBody').innerHTML = strip +
    `<table class="spread__table">
       <thead><tr><th>test</th><th>det</th><th>peak excess</th><th>conf</th></tr></thead>
       <tbody>${rows}</tbody></table>`;
}

// ─── delete ─────────────────────────────────────────────────────────────────────
async function deleteTest(id) {
  const t = state.tests.find((x) => x.meta.id === id);
  if (!confirm(`Delete “${t ? t.meta.name : 'test ' + id}”? This removes its readings too.`)) return;
  try {
    await fetch(`/api/fieldtests/${id}`, { method: 'DELETE' });
  } catch (e) { /* fall through to refresh */ }
  state.compareSel.delete(id);
  toggleCompare(id, false);
  if (state.currentId === id) { state.currentId = null; showEmpty(); }
  await loadList();
}

function showEmpty() {
  $('ftEmpty').hidden = false;
  $('seriesKey').hidden = true;
  $('ftMetrics').hidden = true;
  $('spreadPanel').hidden = true;
  $('readoutSub').textContent = 'select or upload a field test';
  Plotly.purge('ftPlot');
  enableAssistant(false);
}

// ─── upload / sample ────────────────────────────────────────────────────────────
function busy(btn, on, busyText) {
  if (!btn) return;
  if (on) { btn.dataset._t = btn.querySelector('span') ? btn.querySelector('span').textContent : btn.textContent; btn.setAttribute('aria-busy', 'true'); }
  else btn.removeAttribute('aria-busy');
  const target = btn.querySelector('span') || btn;
  if (on && busyText) target.textContent = busyText;
  if (!on && btn.dataset._t) target.textContent = btn.dataset._t;
  btn.disabled = on;
}

function wireUpload() {
  const form = $('uploadForm'), msg = $('acqMsg'), btn = $('uploadBtn');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    if (!(fd.get('name') || '').trim()) { msg.className = 'acq__msg is-err'; msg.textContent = 'Please name the test.'; return; }
    if (!fd.get('file') || !fd.get('file').name) { msg.className = 'acq__msg is-err'; msg.textContent = 'Choose a CSV file.'; return; }
    busy(btn, true, 'Analysing…'); msg.textContent = '';
    try {
      const r = await fetch('/api/fieldtests', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) { msg.className = 'acq__msg is-err'; msg.textContent = d.error || 'Upload failed.'; return; }
      msg.className = 'acq__msg is-ok';
      msg.textContent = d.facts.detected
        ? `Stored. Methane event detected (${d.facts.confidence_sigmas.toFixed(1)}σ).`
        : 'Stored. No sustained event detected.';
      form.reset();
      await loadList();
      selectFromDetail(d);
    } catch (err) {
      msg.className = 'acq__msg is-err'; msg.textContent = `Could not reach the server: ${err.message}`;
    } finally { busy(btn, false); }
  });
}

function wireSample() {
  const btn = $('sampleBtn'), msg = $('acqMsg');
  btn.addEventListener('click', async () => {
    busy(btn, true, 'Generating…'); msg.textContent = '';
    try {
      const r = await fetch('/api/fieldtests/sample', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      });
      const d = await r.json();
      msg.className = 'acq__msg is-ok'; msg.textContent = 'Synthetic test created.';
      await loadList();
      selectFromDetail(d);
    } catch (err) {
      msg.className = 'acq__msg is-err'; msg.textContent = `Could not reach the server: ${err.message}`;
    } finally { busy(btn, false); }
  });
}

// ─── assistant (interpret + chat) ───────────────────────────────────────────────
function enableAssistant(on) {
  $('assistant').classList.toggle('is-locked', !on);
  $('interpretBtn').disabled = !on;
  $('ftChatInput').disabled = !on;
  $('ftChatSend').disabled = !on;
}

function resetAssistant() {
  $('ftChatLog').innerHTML = '';
  const cap = $('interpretText');
  cap.classList.remove('has-text');
  cap.textContent = 'Press “Interpret this test” for a plain-English read, or ask a question below.';
  $('interpretErr').hidden = true;
  setBadge('idle');
}

function setBadge(source) {
  const el = $('ftBadge');
  el.classList.remove('badge--idle', 'badge--ai', 'badge--tmpl');
  if (source === 'openai') { el.classList.add('badge--ai'); el.textContent = 'AI · OpenAI'; }
  else if (source === 'template') { el.classList.add('badge--tmpl'); el.textContent = 'Template'; }
  else { el.classList.add('badge--idle'); el.textContent = 'offline'; }
}

function wireInterpret() {
  const btn = $('interpretBtn'), cap = $('interpretText'), err = $('interpretErr');
  btn.addEventListener('click', async () => {
    if (!state.currentId) return;
    busy(btn, true, 'Reading…'); err.hidden = true;
    try {
      const r = await fetch(`/api/fieldtests/${state.currentId}/interpret`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_ai: true }),
      });
      const d = await r.json();
      cap.textContent = d.text; cap.classList.add('has-text');
      setBadge(d.source);
      if (d.ai_error) {
        err.hidden = false;
        err.textContent = `AI unavailable: ${shorten(d.ai_error)} — showing the built-in read.`;
      }
    } catch (e) {
      err.hidden = false; err.textContent = `Could not reach the server: ${e.message}`;
    } finally { busy(btn, false); }
  });
}

const shorten = (s) => s && s.length > 90 ? s.slice(0, 90) + '…' : s;

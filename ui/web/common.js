/* ═══════════════════════════════════════════════════════════════════════════
   common.js — shared helpers + a reusable chat widget for every dashboard page.

   Everything lives under window.CH4 so it can never clash with a page script's
   own top-level declarations (app.js keeps its own locals untouched).
   ═══════════════════════════════════════════════════════════════════════════ */
'use strict';

window.CH4 = (function () {
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  // Compact ppm string: 2 dp under 10, 1 dp under 100, integer above.
  const ppmStr = (v) => (v == null || isNaN(v)) ? null
    : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

  const fmtSeconds = (s) => {
    if (s == null || isNaN(s)) return '—';
    if (s < 90) return `${s.toFixed(0)} s`;
    const m = s / 60;
    return m < 90 ? `${m.toFixed(1)} min` : `${(m / 60).toFixed(1)} h`;
  };

  // Same heuristic the model chat uses to colour an answer as an error.
  const isErrorAnswer = (a) =>
    /^(couldn'?t reach|no openai_api_key|the 'openai'|no question)/i.test(a) ||
    /api key is invalid/i.test(a);

  // Instrument palette — single source of truth shared with the Plotly charts.
  const COLOR = {
    amber: '#f0883e', green: '#3fb950', blue: '#58a6ff', red: '#f85149',
    dim: '#8b949e', faint: '#5a6473', grid: '#161b22', line: '#2a3240', bg: '#0a0d12',
  };
  const MONO = "'Fira Code', ui-monospace, monospace";

  /* Wire a chat widget to a POST endpoint that returns { answer }.
     opts = { logEl, formEl, inputEl, sendEl, seedsEl?, seeds?, endpoint, buildBody, onSource? } */
  function mountChat(opts) {
    const { logEl, formEl, inputEl, sendEl, seedsEl, seeds = [],
            endpoint, buildBody, onSource } = opts;

    if (seedsEl && seeds.length) {
      seedsEl.innerHTML = seeds.map((q) => `<button type="button" class="seed">${esc(q)}</button>`).join('');
      seedsEl.querySelectorAll('.seed').forEach((b) =>
        b.addEventListener('click', () => { inputEl.value = b.textContent; ask(b.textContent); }));
    }
    formEl.addEventListener('submit', (e) => { e.preventDefault(); ask(inputEl.value); });

    async function ask(question) {
      question = (question || '').trim();
      if (!question) return;
      inputEl.value = '';
      if (sendEl) sendEl.disabled = true;
      addMsg('user', '>', question);
      const pending = addMsg('ai', 'AI', '', true);
      try {
        const r = await fetch(endpoint(), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildBody(question)),
        });
        const d = await r.json();
        const answer = d.answer || '(the server returned no answer)';
        const err = isErrorAnswer(answer);
        finishMsg(pending, answer, err);
        if (onSource) onSource(err ? 'template' : 'openai');
      } catch (e) {
        finishMsg(pending, `Could not reach the server: ${e.message}`, true);
      } finally {
        if (sendEl) sendEl.disabled = false;
        inputEl.focus();
      }
    }
    function addMsg(role, tag, text, pending) {
      const el = document.createElement('div');
      el.className = `msg msg--${role}${pending ? ' msg--pending' : ''}`;
      el.innerHTML = `<span class="msg__tag">${tag}</span><span class="msg__body">${esc(text)}</span>`;
      logEl.appendChild(el);
      logEl.scrollTop = logEl.scrollHeight;
      return el;
    }
    function finishMsg(el, text, isErr) {
      el.classList.remove('msg--pending');
      if (isErr) el.classList.add('msg--err');
      el.querySelector('.msg__body').textContent = text;
      logEl.scrollTop = logEl.scrollHeight;
    }
    return { ask };
  }

  return { esc, ppmStr, fmtSeconds, isErrorAnswer, mountChat, COLOR, MONO };
})();

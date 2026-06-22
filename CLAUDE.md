# LocalMethaneDetection — CLAUDE.md

## What this project is

A **Gaussian plume forward model** for methane (CH₄) detection near the
Custer Road Transfer Station in Allen, TX. Built by a high school student
as Week 2 of an 8-week research project:

- **Week 2 (this code)** — Forward model: given a source, predict sensor readings.
- **Week 3** — Inversion: given sensor readings, find the source (`scipy.optimize`).
- **Week 4** — Feasibility sweep: determine whether detection is physically possible.

The sensor is a **Figaro TGS 2611-E00 MOX sensor**, characterised by the
manufacturer at 500–12,500 ppm. Using it at 2–40 ppm above a 1.9 ppm background
is the research contribution.

**Deployment / data sequence (keep in mind for the inversion):** real inversion
data arrives in two stages — **(1) Custer Road Transfer Station (Allen, TX) first**,
then **(2) a Melissa, TX landfill second** as the follow-on test site. The Week-3
inversion + the whole pipeline must run unchanged on both. Melissa is expected to be
the harder, lower-ppm case, so the optimizer and detection floor have to hold up at
**very small ppm above the 1.9 ppm background** — design every guard rail for that
worst case, not just the stronger Custer signal.

---

## Current status (Week 2 — validated green)

- `python3 -m pytest tests/ -v` → **148/148 PASS**. `python3 -m misc.demo` now runs validation,
  saves `plume_contour.png`, then **serves the interactive web dashboard** at
  `http://127.0.0.1:5050` and opens the browser (Ctrl+C to stop). For a non-blocking,
  exit-0 run (tests/CI) use **`MPLBACKEND=Agg python3 -m misc.demo --check`** — it validates +
  writes the PNG and exits without serving. (Entry points run as modules from the project
  root — `python3 -m misc.demo` / `-m misc.server` / `-m misc.ask` — so the `physics`/`ui`/
  `misc` packages import cleanly.) Use `python3` (the numpy/matplotlib interpreter
  here is `/usr/bin/python3`, 3.9.6). The old blocking `plt.show()` is gone — matplotlib runs
  headless (Agg) for the PNG and the live, interactive graph is the browser dashboard.
  (Port 5050, not 5000: macOS AirPlay Receiver squats on 5000.)
- **σ table** — Briggs (1973) open-country formulas, 6 classes A–F, tabulated
  1–200 m by `generate_sigma_table.py`. The committed `briggs_dispersion_sigma.csv`
  is byte-identical to the generator output (verified reproducible provenance).
- **Ground reflection — consistent (not an open issue).** `concentration_gm3` uses a
  2π denominator with the real+image vertical term (which equals 2 at the ground), so
  it reduces to the standard `C = Q/(π σ_y σ_z u)` at H=z=0 — the correct
  ground-level-with-reflection form.
- **Feasibility (Briggs Class-D σ):** smallest detectable leak at the 50 m fence
  ≈ **0.05 g/s** (threshold 0.90 ppm = 3× the 0.30 ppm noise floor) — but that is a
  *lab-floor* figure. `verdict()` now reframes it as an **upper bound** carrying a
  wind/stability error bar (`implied_Q_range`: the same fenceline signal implies Q over a
  ~160× range across plausible wind 0.5–5 m/s and classes B–F), and can report a
  field-realistic floor (random noise averages as √N; calibration bias does not).
- **Field-test data collection (NEW).** Real CSV readings can be uploaded, cleaned (the
  `processing.py` pipeline), detection-checked, interpreted (AI or template), stored, and
  compared across runs — **one ppm-vs-time graph per field test**. Storage is `db.py`:
  Railway Postgres when `DATABASE_URL` is set, else a local SQLite file (offline/tests).
  Three pages now: `/` (Model contour), `/fieldtests.html` (data collection),
  `/explain.html` (how it works). See "Field-test data collection" below.

**Before trusting any "detectable?" verdict** — the noise floor (0.30 ppm) and the
temperature/humidity coefficients are **ASSUMED placeholders**; measure them on the real
Figaro first (see the Constants table). Replacing the linear weather correction with a
nonlinear form (adding a T·H interaction term) is a candidate future improvement.

---

## Automated improvement log (self-paced /loop)

A recursive-improvement loop made these **verified** changes — full test suite green
(now 35 tests) and `demo.py` exit 0 after each step; physics behaviour preserved unless
noted.

1. **Vectorized `predict_ppm`** — replaced the per-receptor Python loop and the per-call
   σ-table re-sort with cached numpy arrays + `np.interp`; factored the Gaussian profile
   into one shared `_plume_shape`. 50k-point grid **2822 ms → ~2 ms (~1280×)**, output
   numerically identical (max Δ 1.7e-13). This is the hot path Week-3 inversion will hammer.
2. **σ-table provenance verified** — `generate_sigma_table.py` output is byte-identical to
   the committed CSV; the six Briggs formulas + anchors hand-checked; an unverified
   "Table 4-4" citation softened.
3. **Context trimmed** — the dated "Next session" / blocker-history block became the
   compact status above; the π-vs-2π "open item" resolved in-doc (the code's form is the
   standard correct one).
4. **Test coverage 5 → 35** — `test_plume.py` (physics core + a vectorized-vs-scalar
   equivalence guard), `test_feasibility.py` (pins the 0.05 g/s fenceline verdict),
   `test_explain.py` (template + clean AI-absent fallback).
5. **Security** — added `.gitignore`; `.env` (OPENAI_API_KEY) is no longer one `git add .`
   from being committed.
6. **Web dashboard (replaces blocking `plt.show()`).** Added `server.py` (localhost Flask) +
   `ui/web/` (plain HTML/CSS/JS + vendored Plotly) so `python3 -m misc.demo` now opens an interactive
   browser graph with an "Explain this graph" caption and a chat widget; the OpenAI key stays
   server-side. The duplicated 250×200 grid construction (demo.py + scenario_facts) was
   factored into one `explain.compute_field()` reused everywhere (numerically identical;
   `test_server.py` pins it). `demo.py --check` keeps the non-blocking exit-0 path.

**Boundary decision — RESOLVED (optimizer-safe mode).** `predict_ppm` now takes
`clamp_to_table: bool = False`. Default (strict) still raises for any downwind
distance outside the σ table's 1–200 m band, so demo/validation runs surface bad
geometry. The Week-3 inversion calls `predict_ppm(..., clamp_to_table=True)`, which
**never raises on geometry**: sub-1 m downwind points clamp up to the 1 m floor, and
>200 m points fall back to background only (a source that far is effectively
undetectable here). This is the guard rail that keeps `scipy.optimize` alive while it
probes source positions on top of / far from a sensor. Locked by 6 tests in
`test_plume.py`. The optimizer still owns three bounds the forward model does NOT
auto-fix: `u ≥ 0.5 m/s`, integer `stability_class` (fix or loop, never vary
continuously), and `Q ≥ 0`. See `predict_ppm`'s docstring and memory
`plume-predict-ppm-boundary`.

---

## File layout

The code is grouped into three packages — **`physics/`** (the numerical core,
the "math"), **`ui/`** (explanation + browser dashboard), and **`misc/`**
(runnable entry points + storage plumbing) — plus `tests/` and root config.
Each package is a real Python package (`__init__.py`), so imports are absolute:
`from physics.plume import …`, `from ui.explain import …`, `from misc import db`.

```
LocalMethaneDetection/
├── physics/                       # ── "math": numerical core ──────────────────
│   ├── plume.py                   #   Core physics — Gaussian plume + Briggs σ
│   ├── sensor_sim.py              #   Synthetic sensor data: truth + noise + weather
│   ├── processing.py              #   Signal cleaning: baseline, averaging, T-H, detection
│   ├── feasibility.py             #   Forward-model sweep + detectability verdict
│   ├── accuracy.py                #   Accuracy framework: self-measured noise, recovery, LOD/Q, CRB, grade
│   ├── fieldtest.py               #   REAL readings: CSV parse + clean + sample gen + aggregate_for_inversion
│   ├── weather.py                 #   Real wind + Pasquill stability from Open-Meteo archive (offline-safe)
│   ├── inversion.py               #   Week-3 source inversion: readings → (x, y, Q); scipy-free WLS + CRB
│   ├── generate_sigma_table.py    #   Regenerates the σ table from Briggs (1973) formulas
│   └── briggs_dispersion_sigma.csv#   Briggs σ lookup table (1–200 m, 6 classes A–F)
├── ui/                            # ── explanation + dashboard front-end ────────
│   ├── explain.py                 #   Caption + Q&A + field-test interpret; compute_field
│   └── web/                       #   Browser dashboard (no build step)
│       ├── index.html             #     Model: contour + explain/chat (+ shared nav)
│       ├── fieldtests.html        #     Field Tests: upload, per-test time series, interpret, compare
│       ├── explain.html           #     How it works: plain-English guide + chat
│       ├── common.js              #     shared helpers + reusable chat widget (window.CH4)
│       ├── app.js                 #     Model page: contour + explain/chat
│       ├── fieldtests.js          #     Field Tests page: upload/list/detail/compare logic
│       ├── styles.css             #     dark instrument-readout theme (Fira Code/Sans)
│       └── vendor/plotly.min.js   #     Plotly cartesian bundle, vendored for offline use
├── misc/                          # ── runnable entry points + storage plumbing ─
│   ├── demo.py                    #   Validation + sweep + PNG, then launches the web dashboard
│   ├── server.py                  #   Flask backend (localhost): model API + /api/fieldtests/* + OpenAI proxy
│   ├── ask.py                     #   Single-shot CLI Q&A about the model/graph
│   └── db.py                      #   Field-test storage: Railway Postgres (DATABASE_URL) | local SQLite
├── tests/test_processing.py       # Pytest — signal processing (Tests 1–5)
├── tests/test_plume.py            # Pytest — physics core + vectorization equivalence
├── tests/test_feasibility.py      # Pytest — feasibility sweep + verdict
├── tests/test_explain.py          # Pytest — caption template + AI-absent fallback
├── tests/test_server.py           # Pytest — compute_field equivalence + model JSON API
├── tests/test_db.py               # Pytest — storage layer (SQLite roundtrip, cascade delete)
├── tests/test_fieldtest.py        # Pytest — CSV parse + cleaning pipeline + interpret
├── tests/test_server_fieldtests.py# Pytest — field-test API (upload/detail/compare/degrade)
├── tests/test_accuracy.py         # Pytest — accuracy framework (noise/recovery/LOD/CRB/grade)
├── tests/test_weather.py          # Pytest — Open-Meteo wind/Pasquill (fixture-based, offline)
├── tests/test_inversion.py        # Pytest — source inversion: recovery, linear-Q closed form, CRB compare
├── tests/fixtures/openmeteo_archive.json # Committed Open-Meteo response for hermetic weather tests
├── conftest.py                    # Pytest config — puts the project root on sys.path
├── samples/custer_sample.csv      # Example readings CSV (from sensor_sim) — try the upload flow
├── plume_contour.png              # Static figure from demo.py (with caption box)
├── Procfile                       # OPTIONAL — deploy the whole app to Railway (gunicorn misc.server:app)
├── requirements.txt               # numpy, matplotlib, rich, flask, pytest; openai/psycopg/gunicorn (opt)
├── CLAUDE.md                      # This file (architecture & API docs)
├── EXPLANATION.md                 # Plain-English project overview
└── ACCURACY.md                    # Accuracy framework: protocol, math, grading, worked example
```

**Path notes (so moves don't break):** `physics/plume.py` and
`physics/generate_sigma_table.py` resolve `briggs_dispersion_sigma.csv` next to
themselves (inside `physics/`); `ui/explain.py` and `misc/db.py` read `.env` from
the **project root** (one level up); `misc/server.py` resolves the dashboard at
`ui/web/` absolutely, so it serves correctly from any working directory.

---

## Key physics and formulas

### Gaussian plume equation (Pasquill 1961, Gifford 1961)

```
C = Q / (2π σ_y σ_z u)
  · exp(−y²/2σ_y²)
  · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
```

- The second bracket is the **ground reflection** term — models the ground
  as a perfect mirror (no gas absorbed by soil).
- At H=0, y=0, z=0 this simplifies to `C = Q / (π σ_y σ_z u)`.

### Dispersion coefficients — Briggs (1973) open-country formulas

`σ_y` and `σ_z` (metres) depend on downwind distance `x` (metres) and the
Pasquill stability class (A–F, unstable to stable). See `plume.py:100–206` for
the exact expressions. **Key validity note:** these formulas were validated at
≥100 m over flat open terrain. Sub-100 m results and rough-terrain sites
(structures, equipment) are order-of-magnitude estimates only.

### Wind-frame coordinate rotation

`wind_dir_deg` is the **meteorological from-direction** (clockwise from North).
270° = wind blows FROM the west, TOWARD the east. The downwind unit vector is:

```
û_down = (−sin φ, −cos φ)   [East, North components]
x_wind = dx·(−sin φ) + dy·(−cos φ)    # positive = downwind
y_wind = dx·(cos φ)  + dy·(−sin φ)    # crosswind
```

---

## Constants (single source of truth)

| Symbol | Value | Unit | Source | Where |
|--------|-------|------|--------|-------|
| `R_GAS` | 8.314 | J/(mol·K) | NIST CODATA 2018 | `plume.py` |
| `M_CH4` | 16.04 | g/mol | IUPAC 2021 | `plume.py` |
| `CH4_BACKGROUND` | 1.9 | ppm | NOAA GML 2023 | `plume.py`, `sensor_sim.py` |
| `U_MIN` | 0.5 | m/s | Model undefined below this | `plume.py` |
| `RELEASE_HEIGHT_M` | 1.0 | m | **ASSUMED** ~1 m — H; set measured value before a real run | `plume.py` |
| `SENSOR_HEIGHT_M` | 1.0 | m | **ASSUMED** ~1 m — z (fence mount); `feasibility` + `inversion` read it | `plume.py` |
| `SENSOR_NOISE_PPM` | 0.30 | ppm (1σ) | **ASSUMED** — RANDOM jitter; averages down as √N | `sensor_sim.py` |
| `BIAS_FLOOR_PPM` | 0.00 | ppm (1σ) | **ASSUMED** — non-averageable bias/drift; `effective_noise_floor` combines it with the random part in quadrature | `sensor_sim.py` |
| `DETECT_K` | 3.0 | sigmas | Detection threshold (k·noise_std) | `sensor_sim.py` (imported by `feasibility.py`, `explain.py`) |
| `BASELINE_DRIFT_PPM` | 1.00 | ppm | **ASSUMED** — slow sensor zero wander | `sensor_sim.py` |
| `TEMP_COEFF_PPM_PER_C` | 0.05 | ppm/°C | **ASSUMED** — MOX sensitivity to temperature | `sensor_sim.py` |
| `HUMID_COEFF_PPM_PER_PCT` | 0.02 | ppm/%RH | **ASSUMED** — MOX sensitivity to humidity | `sensor_sim.py` |

**Note:** `DETECT_K` and `SENSOR_NOISE_PPM` live in `sensor_sim.py` and are imported
by both `feasibility.py` and `explain.py`, ensuring the plot's detection-limit line,
the processing tests' threshold, and the feasibility verdict all use the same number.

---

## Public API (`plume.py`)

| Function | Returns | Purpose |
|----------|---------|---------|
| `sigma_y(x, stability_class)` | float (m) | Crosswind dispersion coefficient |
| `sigma_z(x, stability_class)` | float (m) | Vertical dispersion coefficient |
| `rotate_to_wind_frame(x_r, y_r, src_x, src_y, wind_dir_deg)` | (float, float) | Map → wind frame |
| `concentration_gm3(x, y, Q, u, H, sy, sz, z=0)` | float (g/m³) | Raw Gaussian plume concentration |
| `gm3_to_ppm_methane(C_gm3, T_K, P_Pa)` | float (ppm) | Unit conversion only — no background added |
| `predict_ppm(src_pos, Q, u, wind_dir_deg, H, stability_class, receptors, ...)` | ndarray (ppm) | Full pipeline including background |

**`predict_ppm` is the function you'll call in downstream work.** It handles
coordinate rotation, upwind masking, σ lookup, unit conversion, and background
addition in one call.

---

## Simulation and signal processing (`sensor_sim.py`, `processing.py`)

### Synthetic sensor data (`sensor_sim.py`)

Since the real Figaro sensor is not yet deployed, we generate fake-but-realistic
readings where we know the ground truth. This allows us to test the processing
pipeline and verify it recovers the known signal.

**What a raw reading includes:**
```
raw = CH4_BACKGROUND (1.9 ppm)
    + true_plume (the signal we want to recover)
    + slow baseline drift (sensor zero wanders over hours)
    + temperature effect (TEMP_COEFF × [T − T0_REF])
    + humidity effect (HUMID_COEFF × [H − H0_REF])
    + random noise (Gaussian, 1σ = SENSOR_NOISE_PPM)
```

**Key functions:**
- `synthetic_timeseries(true_ppm, noise_ppm=0.30, drift_ppm=1.00, with_weather=True)`
  → returns `SensorData` with aligned arrays of time, raw reading, temperature,
  humidity, and ground-truth plume (for testing).
- `make_plume_event(n, event_start_s, event_duration_s, event_ppm)` 
  → builds a ground-truth signal (flat background, then a plume bump, then background again).
- `make_weather(n, temp_amp_c, humid_amp_pct)` → slowly-varying temperature and
  humidity tracks (sine waves, realistic phase).
- `make_baseline_drift(n, drift_ppm)` → slow sensor zero wander.

**⚠ Important:** The noise size (0.30 ppm), drift size (1.00 ppm), and temperature/
humidity coefficients below are *assumptions* chosen to be realistic. Once real
sensor data exists, replace the constants in the "ASSUMED SENSOR CHARACTERISTICS"
block with measured values — nothing else needs to change.

### Signal processing (`processing.py`)

Four tools to pull the tiny methane signal out of the messy raw reading. Apply them
in order: correct weather → remove baseline → smooth noise → detect plume event.

**1. Temperature/humidity correction:**
```python
corrected, coeffs = temp_humidity_correct(signal, temperature, humidity, ref_mask=None)
```
Fits `signal ≈ a·T + b·H + c` on a no-plume reference window (or the whole record
if `ref_mask=None`), then subtracts the fitted weather-varying part. Removes the
part of the reading that just tracks air conditions.

**2. Baseline subtraction:**
```python
excess, baseline = subtract_baseline(signal, window=601, method="percentile", percentile=25.0)
```
Estimates the slowly-varying baseline (rolling low percentile, median, or minimum)
and subtracts it. What remains is the *excess* above the local floor — the plume.

**3. Moving average:**
```python
smoothed = moving_average(signal, window=61)
```
Centred rolling mean over `window` samples. Shrinks random noise by √N while
preserving real, steady signals. **Implementation note:** O(n) cumsum form, not
O(n·window) convolution — essential for ESP32 portability.

**4. Pattern detection:**
```python
detection = detect_pattern(excess, noise_std=0.30, k=3.0, min_run=5)
```
Flags a plume event where the processed excess climbs above `k·noise_std` and
stays there for ≥`min_run` consecutive samples. Returns a `Detection` with:
- `detected`: boolean
- `start_idx, end_idx`: event window (−1 if not detected)
- `confidence`: mean excess / noise_std (in "sigmas")
- `threshold`: the k·noise_std cutoff used

---

## Feasibility sweep and verdict (`feasibility.py`)

Runs the forward model across many combinations of leak size, distance, and weather
to answer: **"Is this plume model + sensor combination physically able to detect
the leak we care about?"**

**Key functions:**
- `sweep(Q_list, distance_list, stability_list, ...)` → computes `predict_ppm`
  for every (Q, distance, stability) combination; classifies each as
  "detectable" (excess ≥ 3σ noise), "marginal", or "undetectable".
- `implied_Q_range(observed_excess_ppm, distance, u_list, stability_list, ...)` → back-solves
  the source strength Q for a fixed observed signal across a wind × stability grid (exploiting
  the plume's linearity in Q — one forward eval per cell, no optimizer). Returns min/median/max
  Q plus a `swing_factor` (the multiplier by which Q is uncertain when wind/stability are not
  known). This is the **quantified error bar** behind the upper-bound verdict.
- `verdict(cells, fence_distance=50.0, stability=4, noise_floor=0.30, k=3.0, bias_floor=0.0, n_avg=1)`
  → plain-English bottom line. Keeps the headline minimum detectable leak, then reframes it as an
  **upper bound** via `implied_Q_range`; if `bias_floor`/`n_avg` are supplied it also reports the
  field-realistic floor (`sensor_sim.effective_noise_floor`: random noise ÷√n_avg ⊕ non-averageable
  bias). Defaults reproduce the original single-line verdict.

**Classification threshold:** A signal is **detectable** if it exceeds
`k × SENSOR_NOISE_PPM` (default: 3 × 0.30 = 0.90 ppm). This is the same
threshold the processing tests use, so the "is it detectable?" answer is
consistent everywhere in the project.

---

## Accuracy framework (`accuracy.py`)

Answers **"how accurate is this, and how accurate could it ever be?"** as one
ordered protocol. Reuses `processing.py`, `sensor_sim.py`, `feasibility.py`, and
`plume.py` — no physics is reimplemented. Full method + math + worked example live
in **`ACCURACY.md`**; this is the API summary.

| Function | Returns | Purpose |
|----------|---------|---------|
| `estimate_noise_floor(excess_unsmoothed, detection)` | `NoiseEstimate` | **(1) Characterize** — measure real noise from the quiet samples: `random_ppm` (successive-diff, averages as 1/√N) + `bias_ppm` (slow-residual **proxy**). Feed it `raw − baseline`. |
| `recovery_metrics(true_ppm, recovered_excess, detection, time)` | `RecoveryReport` | **(2) Validate** — RMSE/bias/%recovery/R²/correlation + detection outcome + timing error vs known synthetic truth. |
| `detection_limit(random_ppm, bias_ppm, distance, …)` | `DetectionLimit` | **(4a) Bound, empirical** — LOD/LOQ + minimum detectable Q with a wind/stability `swing_factor` error bar (back-solved via `feasibility`), plus the averaging curve + bias crossover. |
| `crb_source_bound(src_pos, Q, …, receptors, sigma_ppm, params)` | `CRBound` | **(4b) Bound, theoretical** — Cramér-Rao lower bound on a source fix from the Fisher information (numerical Jacobian of `predict_ppm`). Standalone now; Week-3 compares its scatter to it. |
| `accuracy_report(result, meta, true_ppm=None)` | `dict` (JSON-safe) | **(3+5) Quantify+Grade** — measured-vs-assumed floor, SNR, detection limit, optional recovery, a **0–100 score + letter grade + one recommendation**. Shaped to drop into `summarize_fieldtest` facts. |

**Two-part error model (single source of truth):** random noise averages down as
1/√N; bias/drift does not. `accuracy.py` reuses `sensor_sim.effective_noise_floor`
(random ÷√N ⊕ bias, in quadrature) so the "best you can do" floor is consistent
with the feasibility verdict. **Honesty caveat:** in-record `bias_ppm` is a proxy;
the rigorous value needs a lab zero-air run — until then it tracks residual drift.

---

## Explanation and visualization (`explain.py`)

Turns the plume field into a human-readable caption. Two paths:

**Template (always available):**
```python
text = template_explanation(facts)
```
Built from summary numbers (peak, detection reach, decay, half-width). Plain
English, no API, no dependencies.

**AI-enriched (optional, with OpenAI API key):**
```python
text = ai_explanation(facts, model="gpt-4o-mini")
```
Sends just the summary numbers to OpenAI; returns a richer paragraph. Any failure
(missing key, network error, API error) silently falls back to the template.
Key is read from the `OPENAI_API_KEY` environment variable, never hard-coded.

**Orchestrator:**
```python
caption, source = explain_field(ppm_grid, xs, ys, params, noise_floor, use_ai=True)
# source is "openai" or "template"
```

### Ask the AI a question (`ask.py`)

Single-shot CLI — the way to ask questions about the model/graph:
```bash
python3 -m misc.ask "what does the green dashed line on the graph mean?"
python3 -m misc.ask "why does the inversion need clamp_to_table?"
```
It takes the question as an argument (no stdin), so it works in any context with
internet — there is **no interactive chat loop** (the old `chat_session()` was removed
because it needs a real controlling terminal, which many launch contexts don't provide).
`ask_once(question, facts)` in `explain.py` does one OpenAI call grounded in
`_CHAT_SYSTEM_PROMPT` (full project knowledge) plus the current graph's numbers from
`scenario_facts()` (built from the shared `DEFAULT_SCENARIO`). On failure it returns a
**plain-English error** (no key / no internet / bad key) rather than failing silently.

**Requires a VALID `OPENAI_API_KEY` in `.env`.** A malformed-but-present key returns
OpenAI 401 "Incorrect API key"; the demo's caption-source line and `ask.py` both surface
that reason (via `explain.LAST_AI_ERROR`) instead of telling you to set a key you already
set. The plot caption falls back to the template and never breaks regardless.

---

## Field-test data collection (real readings)

The forward model PREDICTS readings; this is where REAL ones come in. A **field test** is one
deployment of the Figaro sensor — a ppm-vs-time CSV at a single location. Each test gets its own
time-series graph (raw + estimated baseline + cleaned excess + 3σ threshold + detected-event
window), an AI/template interpretation, and a chat grounded in its numbers. Multiple tests can be
overlaid to see run-to-run variance.

**Key idea — two shapes of data:** the model is a *field over space* (the contour); a real sensor
is a *series over time* at one point. So real data lives on the per-test time-series graph, not
the contour — the contour stays the model/reference. (Week-3 inversion will later estimate the
source from real data and put it back on the map.)

**Pieces (all reuse existing, tested code):**
- `fieldtest.parse_csv(data)` — flexible CSV reader (needs a methane column; time/temperature/
  humidity optional). `fieldtest.process_fieldtest(...)` runs the SAME `processing.py` pipeline
  (temp/humidity correct → baseline subtract → smooth → detect) and returns plottable arrays in
  the raw frame (`raw ≈ baseline + excess`). `fieldtest.make_sample_readings(...)` builds a
  synthetic test via `sensor_sim` so the flow works before the real sensor exists.
- `explain.summarize_fieldtest / template_fieldtest_explanation / interpret_fieldtest` mirror the
  model-side trio (template always available; OpenAI when a key works; `LAST_AI_ERROR` surfaced).
  Chat reuses `explain.ask_once(question, facts)` unchanged.
- `db.py` stores **only the raw readings** (`field_tests` ──< `readings`); the processed view is
  recomputed on read, so cleaning/noise-floor can be retuned without re-uploading.

**Storage backend (`db.py`):**
- `DATABASE_URL` set (Postgres) → **Railway Postgres** via `psycopg` (lazy import).
- `DATABASE_URL` unset → **local SQLite** file `fieldtests.db` (stdlib; offline + tests).
- `db.backend_label()` reports which is active; `db.init_schema()` is idempotent (the API calls
  it lazily, re-initialising automatically if the backend changes — that is how the tests isolate
  a temp DB per case).

**API (added to `server.py`; same-origin, OpenAI key stays server-side):**
- `GET    /api/fieldtests` → list + per-test summary + `backend`.
- `POST   /api/fieldtests` (multipart CSV + metadata) → store raw → `{id, meta, facts, series}`.
- `GET    /api/fieldtests/<id>` → meta + full series + facts.
- `POST   /api/fieldtests/<id>/interpret` `{use_ai}` → `{text, source, ai_error}`.
- `POST   /api/fieldtests/<id>/ask` `{question}` → `{answer}` (grounded in the test).
- `GET    /api/fieldtests/compare?ids=…` → overlay series + across-test spread (variance).
- `DELETE /api/fieldtests/<id>`, `POST /api/fieldtests/sample`, `GET /api/sample.csv`.

**Railway setup (local app + remote DB — the chosen mode):**
1. Create a Railway project and add a **Postgres** database (dashboard, or `railway add`).
2. Copy its **public** connection string (Railway → Postgres → *Connect* → Public Network, i.e.
   `DATABASE_PUBLIC_URL`) into the local `.env` as one line (it is a secret; `.env` is gitignored):
   `DATABASE_URL=postgresql://user:pass@host.proxy.rlwy.net:PORT/railway`
3. `pip install -r requirements.txt` (pulls `psycopg[binary]`), then `python3 -m misc.server`. The
   schema is created automatically on first request. The app still binds **127.0.0.1 only** —
   nothing is exposed; only the DB connection leaves your machine (over TLS).
4. With `DATABASE_URL` unset it transparently uses local `fieldtests.db` instead — no Railway
   needed to develop. (Deploying the *whole* app to Railway later is the optional `Procfile`/
   `gunicorn` path; that makes the endpoints public, so add auth first.)

---

## Running the demo and tests

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the forward model + open the interactive dashboard:**
```bash
python3 -m misc.demo        # run as a module from the project root
```
- Validation (Gaussian plume, unit conversion, wind rotation), feasibility sweep (24 cases),
  and `plume_contour.png` (static figure with caption).
- Then it serves the **web dashboard** at `http://127.0.0.1:5050` and opens your browser:
  an interactive Plotly contour, metric cards, centreline decay, the feasibility verdict, an
  "Explain this graph" caption, and a chat widget. Ctrl+C to stop. Exit code 1 if validation fails.

**Validation only (non-blocking — tests/CI):**
```bash
MPLBACKEND=Agg python3 -m misc.demo --check    # validates + writes the PNG, exits 0, no server
```

**Run the web dashboard on its own:**
```bash
python3 -m misc.server        # → http://127.0.0.1:5050
```
`misc/server.py` is a tiny Flask app bound to **127.0.0.1 only** (the OpenAI key stays server-side,
never reaches the browser). It reuses the existing functions unchanged:
- `GET  /api/field` → grid + facts JSON (via `explain.compute_field`).
- `POST /api/caption` `{use_ai}` → `{caption, source, ai_error}` (via `explain.explain_field`).
- `POST /api/ask` `{question}` → `{answer}` (via `explain.ask_once`).
- `GET  /api/feasibility` → sweep + verdict (via `feasibility.sweep`/`verdict`).
- Field-test endpoints under `/api/fieldtests/*` (see "Field-test data collection" above).

Pages served: `/` (Model contour), `/fieldtests.html` (Field Tests — data collection),
`/explain.html` (How it works). A shared top nav links them; `web/common.js` holds the shared
helpers + chat widget under `window.CH4`.

**Run the tests:**
```bash
pytest tests/ -v        # 148 tests
```
`test_processing.py` (Tests 1–5) validates baseline subtraction, noise averaging,
temperature/humidity correction, pattern detection, and the full pipeline end-to-end.
`test_plume.py` covers the physics core (σ lookup, ground-reflection form, upwind/calm guards,
vectorized-vs-scalar equivalence). `test_feasibility.py` pins the fenceline verdict.
`test_explain.py` pins the caption template + clean AI-absent fallback. `test_server.py` pins
the shared `compute_field` grid, the JSON API shapes, and graceful degradation when the key
is absent. `test_db.py` pins the storage layer (SQLite roundtrip, column defaults, cascade
delete). `test_fieldtest.py` pins CSV parsing, the cleaning pipeline on a known synthetic event,
and the template interpretation. `test_server_fieldtests.py` pins the field-test API end to end
(upload, detail, compare, delete, and template/error degradation with the key removed).
`test_accuracy.py` pins the accuracy framework (self-measured noise recovers the injected
0.30 ppm, recovery metrics on a known event, the LOD→min-detectable-Q back-solve, the
averaging crossover, clean-vs-noisy grading, and the Cramér-Rao bound's monotonicity + σ
scaling). `test_weather.py` pins the Open-Meteo layer (archive parse from a committed fixture,
Pasquill day/night cases, log-law height adjustment, offline fallback). `test_fieldtest.py`
also pins `aggregate_for_inversion` (event-window mean + σ-of-the-mean weight, σ shrinking with
window, the quiet "saw-nothing" null constraint, and the fit-background raw/baseline exposure).
`test_inversion.py` pins the source inversion (recovers a known (x, y, Q) from synthetic
multi-sensor readings, the closed-form linear-in-Q amplitude, stability-class selection, the
weak-signal "unconstrained" guard, and the CRB comparison).
All new tests run on SQLite with no network.

**Optional: AI-enriched captions + chat (requires a valid OpenAI key):**
```bash
echo 'OPENAI_API_KEY=sk-...' > .env        # server-side only — never sent to the browser
python3 -m misc.demo
```
With a valid key, the "Explain this graph" caption and the chat widget call GPT-4o-mini.
Without it (or with an invalid/truncated key → OpenAI 401), the caption falls back to the
template and the chat returns a clear, human-readable reason — the dashboard never breaks.

---

## Constraints and caveats

### Forward model (plume.py)
1. **Steady-state only.** The model gives a time-mean picture. Individual
   sensor readings fluctuate far more due to real turbulence.
2. **u ≥ 0.5 m/s required.** `concentration_gm3` raises `ValueError` below
   this. Calm-wind readings must be handled separately.
3. **Open-country σ underestimates dilution** at a rough site (structures,
   berm, equipment increase mixing beyond open-field levels).
4. **Sub-100 m predictions are order-of-magnitude only.** Briggs formulas were
   validated at ≥100 m over flat, uniform terrain.
5. `gm3_to_ppm_methane` is a pure unit converter — it does NOT add background.
   Background is added in `predict_ppm` only.

### Simulation (sensor_sim.py)
6. **Assumed sensor characteristics are placeholders.** The noise floor (0.30 ppm),
   drift size (1.00 ppm), and temperature/humidity coefficients are plausible
   assumptions chosen so the simulation behaves realistically. Replace them with
   measured values once the real Figaro sensor is characterized.
7. **The synthetic model is linear in noise and weather.** Real MOX sensors may
   have nonlinear responses. Use synthetic data for algorithm validation only;
   compare results against real data once available.

### Processing (processing.py)
8. **Baseline subtraction assumes the plume is a minority of the window.** If the
   plume lasts as long as the baseline window, the rolling percentile will miss it.
   Adjust the window size if you change the expected plume duration.
9. **Temperature/humidity correction is a simple linear fit.** Real MOX drift may
   be nonlinear. Pre-fit the coefficients offline and ship them as constants for
   on-device use.

### Feasibility assessment (feasibility.py)
10. **The sweep tests the centreline downwind sensor only** (worst case = most generous
    for detectability). Off-centreline receptors will see smaller signals.

---

## Week-3 prep: averaging reframe + real wind (Open-Meteo)

Two fixes that make the upcoming inversion *well-posed* before the optimizer is written.
They are independent problems but **compose**: 2 sensors at different downwind distances
make the ratio `C₁/C₂` independent of Q (it constrains position + stability); a known wind
then turns the absolute level into Q. Together they collapse the **163× Q-uncertainty
swing** (the spread `implied_Q_range` reports when wind/stability are unknown).

**Problem 1 — averaging reframe.** "Averaging" was doing two unrelated jobs; they are now
separated (full table in `ACCURACY.md` → "Two averaging roles"):
- *Noise √N* (`processing.moving_average`) — bias-limited and for **display/detection only**.
  The inversion does NOT pre-smooth: a weighted least-squares fit already averages optimally.
- *Meander time-mean* (`fieldtest.aggregate_for_inversion`) — required by the steady-state
  model. It returns, per sensor, the **time-mean of the *unsmoothed* excess** over an event/
  meander window plus the **σ of that mean** (`effective_noise_floor` with `n_avg`=window) as
  the WLS weight — i.e. an `InversionPoint(mean_excess_ppm, sigma_ppm, …)`. Bias never averages
  away, so both framings are always returned: subtract the baseline as a known zero
  (`mean_excess_ppm`), or fit background as a free parameter (`mean_raw_ppm`+`baseline_ppm`). A
  quiet record yields a legitimate "saw-nothing" null constraint, not an error.

**Problem 2 — real wind (`physics/weather.py`).** Replaces the hardcoded `u`/stability guess
with the wind that was actually blowing, from the **Open-Meteo historical archive** (free, no
API key, stdlib `urllib` — no new dependency):
- `fetch_wind_archive(lat, lon, start, end)` is the ONLY networked call; `parse_archive`,
  `summarize_archive`, `pasquill_class`, `adjust_wind_to_height` are pure and fixture-tested.
- Derives a **Pasquill class** from sun/cloud/wind (⚠ cutoffs are the standard scheme but must
  be **verified against a cited reference** — coded as named constants, treated as a first guess),
  and **log-law-adjusts** the 10 m archive wind toward the ~2 m release height (using 10 m raw
  would overstate u and understate Q).
- **Offline-safe** like the OpenAI path: failures return None and set `weather.LAST_WEATHER_ERROR`.
- Wired into `misc/server.py` upload (`_enrich_meta_with_weather`): when wind isn't hand-entered,
  the `site` resolves (Custer/Melissa in `weather.SITES`), and a `test_date` is present, it
  auto-fills `wind_speed`/`wind_dir_deg`/`stability_class`. The `test_date` gate keeps tests
  network-free.

Out of scope here (still Week 3): the `scipy.optimize` inversion itself. This only makes its
two inputs — clean aggregated readings and real wind — correct.

## Project roadmap

**Week 2 (complete):** Forward model + simulation + processing toolkit
- Physics engine (`plume.py`): steady-state Gaussian plume with Briggs dispersion.
- Synthetic sensor (`sensor_sim.py`): realistic fake readings with known ground truth.
- Signal processing (`processing.py`): baseline subtraction, smoothing, weather correction, pattern detection.
- Processing tests (`tests/test_processing.py`): validation against synthetic data (Tests 1–5).
- Feasibility assessment (`feasibility.py`): sweep forward model, classify detectable/marginal/undetectable.
- Visualization (`explain.py`, `demo.py`): contour plot with detection-limit line and plain-English caption.

**Week 3 (built — `physics/inversion.py`; pending real-data validation):** Inversion — given sensor readings, find the source
- *Inputs prepped* (see "Week-3 prep" above): `fieldtest.aggregate_for_inversion` yields
  per-sensor weighted means + σ; `physics/weather.py` supplies real wind + stability.
- `invert(...)` / `invert_field_tests(...)` recover source (x, y) + emission rate Q by
  weighted least squares — **scipy-free, by design**: the plume is linear in Q, so Q has a
  closed form at every trial (x, y) and only a 2-D position search remains (shrinking-grid,
  multi-start). Stability class is fit by trying all six and keeping the best.
- Accepts real (noisy) readings or synthetic test data; single source only.
- Reports per-parameter uncertainty by comparing the fit residual scatter to `crb_source_bound`.
- *Remaining:* validate end-to-end on real Custer readings, then Melissa; wire `invert` into
  the dashboard/map so a recovered source draws back onto the contour.

**Week 4 (follow-on):** Feasibility sweep across all scenarios
- Repeat Week 2 sweep with varied sensor noise, wind, weather, site geometry.
- Quantify minimum detectable leak as a function of distance and conditions.
- Final bottom-line verdict: can this project work at the fenceline distance?

## Project context

- Operating range: 10–200 m; release height H ≈ 0–2 m.
- Site: Custer Road Transfer Station (weak, intermittent methane source).
- Target signal: 2–40 ppm above 1.9 ppm background.
- Sensor: Figaro TGS 2611-E00 MOX (rated 500–12,500 ppm — using it at 2–40 ppm
  is the research contribution).
- Week 3 inversion will consume `predict_ppm` and `processing.py` output to locate
  and quantify sources from deployed readings.

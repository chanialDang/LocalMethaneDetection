# LocalMethaneDetection — CLAUDE.md

## What this project is

A **Gaussian plume forward model** for methane (CH₄) detection at a fenceline,
built by a high school student as an 8-week research project. **Week 2 (done):**
forward model — source → predicted readings. **Week 3 (built, `inversion.py`):**
inversion — readings → source (x, y, Q). **Week 4:** feasibility sweep.

Sensor: **Figaro TGS 2611-E00 MOX**, rated 500–12,500 ppm; using it at **2–40 ppm
above a 1.9 ppm background** is the research contribution. Range 10–200 m, H ≈ 0–2 m.

**Deployment sequence (matters for the inversion):** real data comes from **Custer
Road Transfer Station (Allen, TX) first**, then a **Melissa, TX landfill second**.
The same pipeline must run unchanged on both. **Melissa is the harder, lower-ppm
case — design every guard rail for very small ppm above background**, not just the
stronger Custer signal.

---

## Current status — Week 2 green

- `python3 -m pytest tests/ -v` → **148/148 PASS**. Use `python3` (`/usr/bin/python3`,
  3.9.6, has numpy/matplotlib). Entry points run as modules from the repo root:
  `python3 -m misc.demo` / `-m misc.server` / `-m misc.ask`.
- **Run + dashboard:** `python3 -m misc.demo` validates, writes `plume_contour.png`,
  then serves the dashboard at `http://127.0.0.1:5050` (port 5050 — macOS AirPlay
  squats on 5000) and opens the browser. **CI/non-blocking:** `MPLBACKEND=Agg
  python3 -m misc.demo --check` (validates + PNG, exits 0, no server). matplotlib is
  headless (Agg); the live graph is the browser.
- **σ table:** Briggs (1973) open-country formulas, classes A–F (ints 1–6), tabulated
  1–200 m by `generate_sigma_table.py`; the committed CSV is byte-identical to the
  generator (reproducible provenance).
- **Feasibility (Class-D):** smallest detectable leak at the 50 m fence ≈ **0.05 g/s**
  (0.90 ppm = 3× the 0.30 ppm floor) — a *lab-floor*. `verdict()` reframes it as an
  **upper bound** with a wind/stability error bar: the same signal implies Q over a
  **~160× range** (`implied_Q_range`, wind 0.5–5 m/s × classes B–F).
- **Field-test collection:** real CSVs upload → clean (`processing.py`) → detect →
  interpret → store → compare; one ppm-vs-time graph per test. Storage `db.py`:
  Railway Postgres if `DATABASE_URL` set, else local SQLite.
- **Known issues / test gaps:** tracked in **`BUGS.md`** (F1–F10) — read it before
  editing physics.

**⚠ Before trusting any "detectable?" verdict:** the 0.30 ppm noise floor and the
temperature/humidity coefficients are **ASSUMED placeholders** — measure them on the
real Figaro first (see Constants). A nonlinear weather correction (T·H term) is a
candidate improvement.

**Optimizer-safe boundary (key for the inversion):** `predict_ppm(..., clamp_to_table
=True)` never raises on geometry — sub-1 m downwind clamps to the 1 m floor, >200 m
falls back to background. Default (`False`/strict) raises, so demo/validation surface
bad geometry. The optimizer still owns three bounds the model does NOT auto-fix:
`u ≥ 0.5 m/s`, integer `stability_class` (fix or loop, never vary continuously),
and `Q ≥ 0`.

---

## File layout

Three packages — **`physics/`** (numerical core), **`ui/`** (explanation + dashboard),
**`misc/`** (entry points + storage) — plus `tests/`. Real packages, so imports are
absolute (`from physics.plume import …`).

```
physics/  plume.py (Gaussian plume + Briggs σ) · sensor_sim.py (synthetic data) ·
          processing.py (baseline/avg/T-H/detect) · feasibility.py (sweep+verdict) ·
          accuracy.py (noise/recovery/LOD/CRB/grade) · fieldtest.py (CSV→clean→
          aggregate_for_inversion) · weather.py (Open-Meteo wind+Pasquill, offline-safe) ·
          inversion.py (readings→x,y,Q; scipy-free WLS+CRB) · generate_sigma_table.py ·
          briggs_dispersion_sigma.csv
ui/       explain.py (caption/Q&A/interpret; compute_field) · web/ (index/fieldtests/
          explain .html + common/app/fieldtests .js + styles.css + vendor/plotly.min.js)
misc/     demo.py (validate+sweep+PNG+dashboard) · server.py (localhost Flask API) ·
          ask.py (one-shot CLI Q&A) · db.py (Postgres | SQLite)
tests/    one per module (plume, processing, feasibility, explain, server, db, fieldtest,
          server_fieldtests, accuracy, weather, inversion) + fixtures/openmeteo_archive.json
root      conftest.py · samples/custer_sample.csv · Procfile (optional Railway) ·
          requirements.txt · CLAUDE.md · BUGS.md · EXPLANATION.md · ACCURACY.md
```

**Path notes:** `plume.py`/`generate_sigma_table.py` resolve the CSV next to themselves
(in `physics/`); `ui/explain.py` and `misc/db.py` read `.env` from the repo root;
`misc/server.py` resolves `ui/web/` absolutely (serves from any cwd).

---

## Key physics

**Gaussian plume (Pasquill 1961, Gifford 1961):**
```
C = Q / (2π σ_y σ_z u) · exp(−y²/2σ_y²) · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
```
The second bracket is **ground reflection** (perfect mirror). At H=y=z=0 it collapses
to the standard `C = Q/(π σ_y σ_z u)` — the correct ground-level-with-reflection form.

**Dispersion σ_y, σ_z:** Briggs (1973) open-country, by Pasquill class (1=A unstable …
6=F stable), read from the CSV with linear interpolation. **Validity:** fit at ≥100 m
over flat open terrain — sub-100 m and rough-terrain (built fenceline) results are
order-of-magnitude only.

**Wind-frame rotation:** `wind_dir_deg` is the meteorological **from-direction**
(clockwise from N). 270° = from west, toward east.
```
x_wind = dx·(−sin φ) + dy·(−cos φ)    # + = downwind
y_wind = dx·( cos φ) + dy·(−sin φ)    # crosswind
```
⚠ Every test uses 270°, where this is the **identity** (`xw=dx, yw=dy`) — `BUGS.md` F1.

---

## Constants (single source of truth)

| Symbol | Value | Unit | Source / note | Where |
|--------|-------|------|--------|-------|
| `R_GAS` | 8.314 | J/(mol·K) | NIST CODATA 2018 | `plume.py` |
| `M_CH4` | 16.04 | g/mol | IUPAC 2021 | `plume.py` |
| `CH4_BACKGROUND` | 1.9 | ppm | NOAA GML 2023 | `plume.py`, `sensor_sim.py` |
| `U_MIN` | 0.5 | m/s | model undefined below this | `plume.py` |
| `RELEASE_HEIGHT_M` | 1.0 | m | **ASSUMED** — H; set measured before a real run | `plume.py` |
| `SENSOR_HEIGHT_M` | 1.0 | m | **ASSUMED** — z (fence mount); read by feasibility + inversion | `plume.py` |
| `SENSOR_NOISE_PPM` | 0.30 | ppm 1σ | **ASSUMED** — RANDOM jitter; averages as √N | `sensor_sim.py` |
| `BIAS_FLOOR_PPM` | 0.00 | ppm 1σ | **ASSUMED** — non-averageable bias; combined in quadrature | `sensor_sim.py` |
| `DETECT_K` | 3.0 | σ | detection threshold (k·noise_std) | `sensor_sim.py` |
| `BASELINE_DRIFT_PPM` | 1.00 | ppm | **ASSUMED** — slow zero wander | `sensor_sim.py` |
| `TEMP_COEFF_PPM_PER_C` | 0.05 | ppm/°C | **ASSUMED** — MOX temp sensitivity | `sensor_sim.py` |
| `HUMID_COEFF_PPM_PER_PCT` | 0.02 | ppm/%RH | **ASSUMED** — MOX humidity sensitivity | `sensor_sim.py` |

`DETECT_K` and `SENSOR_NOISE_PPM` live in `sensor_sim.py`, imported by `feasibility.py`
+ `explain.py`, so the detection line, processing threshold, and verdict use one number.

---

## Public API (`plume.py`)

| Function | Returns | Purpose |
|----------|---------|---------|
| `sigma_y/sigma_z(x, stability_class)` | float (m) | Crosswind / vertical dispersion coeff |
| `rotate_to_wind_frame(x_r, y_r, src_x, src_y, wind_dir_deg)` | (float, float) | Map → wind frame |
| `concentration_gm3(x, y, Q, u, H, sy, sz, z=0)` | float (g/m³) | Raw Gaussian plume concentration |
| `gm3_to_ppm_methane(C_gm3, T_K, P_Pa)` | float (ppm) | Unit conversion only — no background |
| `predict_ppm(src_pos, Q, u, wind_dir_deg, H, stability_class, receptors, …, clamp_to_table=False)` | ndarray (ppm) | Full pipeline incl. background |
| `predict_excess_grid(source_grid, receptors, Q, …)` | ndarray (K,N) | Batched excess for many trial sources (built for the inversion; **unused** — `BUGS.md` F3) |

`predict_ppm` is the main downstream call (rotation + upwind mask + σ lookup + unit
conversion + background). `gm3_to_ppm_methane` is a pure converter. Use
`clamp_to_table=True` for the inversion (boundary note above).

---

## Modules in brief

**`sensor_sim.py`** — synthetic readings with known truth: `raw = 1.9 + true_plume +
drift + temp + humid + noise(1σ=SENSOR_NOISE_PPM)`. `synthetic_timeseries`,
`make_plume_event`, `make_weather`, `make_baseline_drift`. `effective_noise_floor(
random, bias, n_avg) = √[(random/√n)² + bias²]` — random averages, bias doesn't.

**`processing.py`** — clean in order: `temp_humidity_correct` (fit a·T+b·H+c on a
no-plume window, subtract the varying part) → `subtract_baseline` (rolling
low-percentile floor; assumes the plume is a window minority) → `moving_average`
(O(n) cumsum centred mean, ESP32-portable; **display/detect only**) → `detect_pattern`
(excess > k·noise for ≥min_run → `Detection{detected, start_idx, end_idx (exclusive),
confidence, threshold}`).

**`feasibility.py`** — `sweep(Q, dist, stability)` classifies each cell detectable
(≥k·noise) / marginal / undetectable; `implied_Q_range` back-solves Q across
wind×stability (linearity → one eval/cell) → the `swing_factor` error bar;
`verdict(...)` is the plain-English bottom line (headline floor → upper bound →
optional field-realistic floor).

**`accuracy.py`** — ordered protocol (full math in **`ACCURACY.md`**):
`estimate_noise_floor` (random via successive-diff; bias via slow-residual *proxy*) ·
`recovery_metrics` (vs known truth) · `detection_limit` (LOD/LOQ + min-Q + swing) ·
`crb_source_bound` (Cramér-Rao floor from a numerical Jacobian of `predict_ppm`) ·
`accuracy_report` (0–100 score + grade + one recommendation). ⚠ in-record bias is a
proxy; the rigorous value needs a zero-air run.

**`explain.py` / `ask.py`** — caption + chat. `template_explanation` always works;
`ai_explanation`/`ask_once` use OpenAI (`OPENAI_API_KEY` in `.env`, server-side only)
and fall back to the template with a plain-English reason on any failure
(`LAST_AI_ERROR`). `python3 -m misc.ask "question"` is one-shot (no chat loop).

**`fieldtest.py`** — bridge from a real CSV to the per-test graph and the inversion.
`parse_csv` (flexible columns; needs methane, time/T/H optional) → `process_fieldtest`
(same `processing.py` pipeline, raw frame: `raw ≈ baseline + excess`).
`aggregate_for_inversion` → `InversionPoint(mean_excess_ppm, sigma_ppm, …)`: time-mean
of the **unsmoothed** excess over a meander window + the σ-of-the-mean weight (NOT
pre-smoothed — WLS averages optimally). `make_sample_readings` for pre-sensor demos.

**`weather.py`** — real wind + Pasquill class from the Open-Meteo archive (free, no
key, stdlib `urllib`). `fetch_wind_archive` is the only networked call;
`parse_archive`/`summarize_archive`/`pasquill_class`/`adjust_wind_to_height` are pure
+ fixture-tested. Log-law-adjusts 10 m wind toward the ~2 m release height. Offline-safe
(None + `LAST_WEATHER_ERROR`). ⚠ Pasquill cutoffs are a first guess — verify vs a cited
reference (`BUGS.md` F8).

**`inversion.py`** (Week 3) — `invert(sensor_positions, points, u, wind_dir_deg,
stability_class=None, …)` → `SourceEstimate(x, y, Q, …, crb_std, converged, n_snapshots)`.
**scipy-free:** the plume is linear in Q, so Q has a closed form `Q* = Σwgd/Σwg²` at
every trial (x, y); only a 2-D position search remains (shrinking-grid, multi-start).
Stability fit by trying all six. `invert_field_tests(results, …)` is the raw-CSV→source
light switch. Compares scatter to the CRB; flags `converged=False` when no sensor beats 3σ.
**Multi-snapshot fusion (F7 fix):** a single wind on a hard fenceline is under-determined
(only ~2 sensors in the plume for 3 unknowns) and silently lands ~28 m off. `invert_multi(
sensor_positions, [Snapshot(points, u, wind_dir_deg, stability_class=None)], …)` /
`invert_field_tests_multi(snapshots, …)` fuse the SAME source seen under several winds —
Q shared closed-form across snapshots (steady leak), joint 2-D search — triangulating it
to sub-metre (2–3 winds). Fisher info adds → `accuracy.crb_source_bound_multi` is the
combined (tighter) bound. `invert` is the K=1 case of `invert_multi`. Denser default grid
(`coarse=60, n_seeds=12`) since each snapshot sharpens the joint basin. See `BUGS.md` F7.

**`db.py`** — `DATABASE_URL` set → Railway Postgres (`psycopg`, lazy); unset → local
SQLite `fieldtests.db` (offline + tests). Stores **only raw readings**; the processed
view recomputes on read. `init_schema()` idempotent.

---

## Server + storage

`misc/server.py` — tiny Flask bound to **127.0.0.1 only** (OpenAI key stays
server-side). Reuses existing functions unchanged:
- `GET /api/field` (grid+facts via `compute_field`) · `POST /api/caption` ·
  `POST /api/ask` · `GET /api/feasibility`.
- Field tests: `GET/POST /api/fieldtests`, `GET /api/fieldtests/<id>`,
  `POST .../interpret`, `POST .../ask`, `GET .../compare?ids=…`, `DELETE .../<id>`,
  `POST .../sample`, `GET /api/sample.csv`.

Pages: `/` (contour), `/fieldtests.html`, `/explain.html`; shared nav + chat widget in
`web/common.js` (`window.CH4`).

**Railway (local app + remote DB):** add a Postgres DB; put its **public** URL in
`.env` as `DATABASE_URL=…` (secret, gitignored); `pip install -r requirements.txt`;
`python3 -m misc.server` (schema auto-creates; still binds 127.0.0.1 only). Unset
`DATABASE_URL` → local SQLite. Deploying the whole app via `Procfile`/`gunicorn` makes
the endpoints public — add auth first.

---

## Proactive code reading & bug-hunting (physics + tests)

Live backlog of known issues + test gaps: **`BUGS.md`** (F1–F10). Read it before
editing physics; add what you find.

**Before touching physics:** read the function, its callers (`grep` the symbol), and
its test. Decide which "right" you need — *coded-correct* vs *physics-valid* (model
matches reality); don't conflate them.

**Failure taxonomy (where each lives):**
- **Units** — `gm3_to_ppm_methane`; g↔kg / Pa↔hPa / ppb↔ppm / g·s↔kg·hr. Off by a
  round factor → suspect this; catch with dimensional analysis + a magnitude check.
- **Angle/sign + 270° identity** — `rotate_to_wind_frame` / inline rotation; bugs are
  invisible at 270° (every test). Exercise a non-cardinal wind. (F1)
- **Normalization** — the `2π` + ×2 reflection; a factor error keeps the shape and
  only scales Q. Only **conservation** catches it cleanly. (F4)
- **Singularities** — `u→0`, `σ→0`, WLS denom→0, zero σ weight: guarded today, don't
  remove a guard.
- **Boundary / silent clamp** — `np.interp` clamps outside 1–200 m; `clamp_to_table`
  makes that a deliberate decision.
- **Inversion local-minimum / ill-posedness** — wrong basin (F7), or collinear sensors
  → clean fit + huge `crb_std` (not a bug; add/spread sensors).

**A green suite is necessary, not sufficient.** Beware *circular tests* (same function
both sides — F2) and *trivial-fixture tests* (the 270° identity — F1). Expected values
must come from an **independent** route (hand calc, a different formula, a conservation
law), never the code under test.

**Detection toolkit (ranked):** magnitude check → conservation (`u·∫∫C dy dz = Q`) →
forward↔inverse round-trip → analytic limits (centreline `Q/(π σ_y σ_z u)`, upwind/far
= background) → symmetry/monotonicity → adversarial + finiteness → two-implementations
agree → plot it and look.

**2-step removal (per `BUGS.md` item):** (1) reproduce/lock with a test first (failing
for a bug; characterization at the untested condition for a gap); (2) then change code,
confirm that test **and** full `pytest tests/` are green; tick it off. Never fix silently.

**Coded-right ≠ physics-valid:** steady-state, open-country σ at a built site, linear
weather, and the Pasquill cutoffs are tracked (`BUGS.md` C + caveats) — removable only
by real/controlled-release data + a citation, not a test.

---

## Constraints & caveats

- **Forward (plume.py):** steady-state (time-mean only; real readings fluctuate more) ·
  `u ≥ 0.5 m/s` (raises below) · open-country σ under-predicts dilution at a rough site ·
  sub-100 m is order-of-magnitude · `gm3_to_ppm_methane` adds no background.
- **Simulation (sensor_sim.py):** assumed sensor constants are placeholders; the model
  is linear in noise/weather (real MOX may be nonlinear) — synthetic is for algorithm
  validation only.
- **Processing (processing.py):** baseline assumes the plume is a window minority; T/H
  correction is a linear fit (real drift may be nonlinear — pre-fit + ship as constants
  on-device).
- **Feasibility (feasibility.py):** sweeps the centreline downwind sensor only (most
  generous for detection).

---

## Week-3 prep + roadmap

**Two inputs make the inversion well-posed (they compose):** (1) **averaging done
right** — `processing.moving_average` is √N noise smoothing for display/detect only;
the inversion uses `fieldtest.aggregate_for_inversion`'s meander time-mean + σ-of-the-
mean weight and does NOT pre-smooth (WLS averages optimally; full table in `ACCURACY.md`
"two averaging roles"). (2) **real wind** — `weather.py` replaces the u/stability guess
with Open-Meteo data. Together they collapse the ~160× Q swing.

**Roadmap.** Week 2 — forward model + simulation + processing + feasibility + viz:
**done**. Week 3 — inversion (`inversion.py`): **built; pending real-data validation**
on Custer then Melissa, then wire a recovered source back onto the contour/map. Week 4
— feasibility sweep across noise/wind/weather/geometry; final fenceline verdict.

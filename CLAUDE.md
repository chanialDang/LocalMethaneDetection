# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Single source of truth.** Consolidates the former `EXPLANATION.md`, `ACCURACY.md`,
> `BUGS.md`, `TOFIX.md`, and `REVIEW_2026-06-25_bug-campaign.md`. Long-form tutorial
> prose and worked examples were compressed to keep this file context-cheap; the
> operational facts and every conceptual distinction are preserved.

## What this project is

A **Gaussian plume forward model** for methane (CH₄) detection at a fenceline, built by
a high-school student as an 8-week research project. The headline question: *if a leak
at (x, y) emits Q g/s and wind u blows from φ, what ppm does a fenceline sensor read?* —
then **invert** it: readings → source (x, y, Q). The inversion is only meaningful if the
forward direction is correct.

Sensor: **Figaro TGS 2611-E00 MOX**, rated 500–12,500 ppm; the research contribution is
running it at **2–40 ppm above a 1.9 ppm background** — 2–3 orders of magnitude below its
characterised range, so signal extraction needs statistical baseline subtraction, not
threshold crossing. Operating range 10–200 m, H ≈ 0–2 m.

**Deployment sequence (matters for the inversion):** real data comes from **Custer Road
Transfer Station (Allen, TX) first**, then a **Melissa, TX landfill second**. The same
pipeline must run unchanged on both. **Melissa is the harder, lower-ppm case — design
every guard rail for very small ppm above background**, not just the stronger Custer signal.

**Weeks:** 2 — forward model + simulation + processing + feasibility + viz (**done**) ·
3 — inversion (`inversion.py`, **built; pending real-data validation** on Custer then
Melissa, then wire a recovered source back onto the contour/map) · 4 — feasibility sweep
across noise/wind/weather/geometry + final fenceline verdict.

## Commands

`python3` = `/usr/bin/python3` (3.9.6; has numpy/matplotlib). Packages are real, so imports
are absolute (`from physics.plume import …`) and entry points run as modules from repo root.

- **Tests:** `python3 -m pytest tests/ -v` → **265/265 PASS**. One file:
  `python3 -m pytest tests/test_inversion.py -v`. One test:
  `… tests/test_inversion.py::test_name`. CI/headless: `MPLBACKEND=Agg python3 -m pytest
  tests/ -q` (~4 s).
- **Run + dashboard:** `python3 -m misc.demo` — validates, writes `plume_contour.png`,
  serves the dashboard at `http://127.0.0.1:5050`, opens the browser. **Port 5050**
  (macOS AirPlay squats on 5000). matplotlib is headless (Agg); the live graph is the browser.
- **CI/non-blocking:** `MPLBACKEND=Agg python3 -m misc.demo --check` (validate + PNG,
  exit 0, no server).
- **Other entry points:** `python3 -m misc.server` (localhost Flask API) ·
  `python3 -m misc.ask "question"` (one-shot CLI Q&A, no chat loop).
- **CSV preflight:** `python3 -m misc.preflight yourfile.csv` — field self-check on a real
  upload BEFORE trusting it: reports detected columns, inferred time units/cadence, ppm
  sanity, every data warning, detection outcome, and a GO/CHECK/STOP verdict (exit 0/1/2).
- **Dress rehearsal:** `python3 -m misc.rehearse` — generate a realistic multi-sensor, multi-wind
  scenario with a KNOWN hidden source, write it as CSVs, run the full parse→process→fuse pipeline,
  and print recovered-vs-true source. A pre-field dry run of the whole deployment workflow.
- **σ table:** `python3 physics/generate_sigma_table.py` regenerates the CSV; the committed
  file is byte-identical to the generator (reproducible provenance).

## File layout

Three packages — **`physics/`** (numerical core), **`ui/`** (explanation + dashboard),
**`misc/`** (entry points + storage) — plus `tests/`.

```
physics/  plume.py (Gaussian plume + Briggs σ) · sensor_sim.py (synthetic data) ·
          sensor_frontend.py (voltage→ppm front-end: divider+power-law+T/RH; gated) ·
          processing.py (baseline/avg/T-H/detect) · feasibility.py (sweep+verdict) ·
          accuracy.py (noise/recovery/LOD/CRB/grade) · fieldtest.py (CSV→clean→
          aggregate_for_inversion) · weather.py (Open-Meteo wind+Pasquill, offline-safe) ·
          inversion.py (readings→x,y,Q; scipy-free WLS+CRB) · generate_sigma_table.py ·
          briggs_dispersion_sigma.csv
ui/       explain.py (caption/Q&A/interpret; compute_field) · web/ (index/fieldtests/
          explain .html + common/app/fieldtests .js + styles.css + vendor/plotly.min.js)
misc/     demo.py (validate+sweep+PNG+dashboard) · server.py (localhost Flask API) ·
          ask.py (one-shot CLI Q&A) · db.py (Postgres | SQLite) ·
          preflight.py (real-CSV field self-check: GO/CHECK/STOP) ·
          rehearse.py (full-pipeline dress rehearsal with a known source)
tests/    one per module + fixtures/openmeteo_archive.json (plus test_fieldtest_transferability.py,
          test_preflight.py, test_inversion_anchors.py, test_pipeline_e2e.py, test_sigma_table.py,
          test_rehearse.py)
root      conftest.py · samples/custer_sample.csv · Procfile (optional Railway) ·
          requirements.txt · CLAUDE.md · docs/DEPLOYMENT.md (field runbook: bench→CSV→invert) ·
          docs/CALIBRATION.md (bench runbook: volts→ppm front-end constants)
```

**Path notes:** `plume.py`/`generate_sigma_table.py` resolve the CSV next to themselves
(in `physics/`); `ui/explain.py` and `misc/db.py` read `.env` from the repo root;
`misc/server.py` resolves `ui/web/` absolutely (serves from any cwd).

## Key physics

**Gaussian plume (Pasquill 1961, Gifford 1961):**
```
C = Q / (2π σ_y σ_z u) · exp(−y²/2σ_y²) · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
```
- `Q / (2π σ_y σ_z u)` — mass conservation: flux through any cross-section = Q.
- `exp(−y²/2σ_y²)` — crosswind bell curve, peak on the centreline.
- Second bracket — **ground reflection** (perfect mirror; image source at depth −H).
  At H=y=z=0 it collapses to the standard `C = Q/(π σ_y σ_z u)`.

**Dispersion σ_y, σ_z:** Briggs (1973) open-country, by Pasquill class (1=A unstable …
6=F stable), read from the CSV with linear interpolation. **Validity:** fit at ≥100 m over
flat open terrain — sub-100 m and rough-terrain (a built fenceline) results are
order-of-magnitude only.

**Wind-frame rotation:** `wind_dir_deg` is the meteorological **from-direction** (clockwise
from N). 270° = from west, toward east.
```
x_wind = dx·(−sin φ) + dy·(−cos φ)    # + = downwind
y_wind = dx·( cos φ) + dy·(−sin φ)    # crosswind
```
⚠ Every test uses 270°, where this is the **identity** (`xw=dx, yw=dy`) — see Backlog F1.
Upwind receptors (negative downwind coordinate) get background only.

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
| `SUPPLY_VOLTAGE_V` | None | V | **UNSET** — V_c across divider; set from circuit (gates conversion) | `sensor_frontend.py` |
| `LOAD_RESISTANCE_OHM` | None | Ω | **UNSET** — R_L over-amplification resistor; set from circuit | `sensor_frontend.py` |
| `R0_OHM` | None | Ω | **UNSET** — R_s in clean air; **measure on bench** | `sensor_frontend.py` |
| `POWERLAW_A` | 1.0 | — | **DATASHEET-TYPICAL** placeholder (R_s/R₀=A·C^−m); refit w/ span gas | `sensor_frontend.py` |
| `POWERLAW_M` | 0.35 | — | **DATASHEET-TYPICAL** — ⚠ UNVERIFIED at 2–40 ppm (curve is 500–12,500) | `sensor_frontend.py` |

`DETECT_K` and `SENSOR_NOISE_PPM` live in `sensor_sim.py`, imported by `feasibility.py`
and `explain.py`, so the detection line, processing threshold, and verdict use one number.

**⚠ Before trusting any "detectable?" verdict:** the 0.30 ppm noise floor and the
T/H coefficients are **ASSUMED placeholders** — measure them on the real Figaro first
(see Backlog → `BIAS_FLOOR_PPM`, the item that can sink the project).

## Public API (`plume.py`)

| Function | Returns | Purpose |
|----------|---------|---------|
| `sigma_y/sigma_z(x, stability_class)` | float (m) | Crosswind / vertical dispersion coeff |
| `rotate_to_wind_frame(x_r, y_r, src_x, src_y, wind_dir_deg)` | (float, float) | Map → wind frame |
| `concentration_gm3(x, y, Q, u, H, sy, sz, z=0)` | float (g/m³) | Raw Gaussian plume concentration |
| `gm3_to_ppm_methane(C_gm3, T_K, P_Pa)` | float (ppm) | Unit conversion only (ideal gas; no background) |
| `predict_ppm(src_pos, Q, u, wind_dir_deg, H, stability_class, receptors, …, clamp_to_table=False)` | ndarray (ppm) | Full pipeline incl. background |
| `predict_excess_grid(source_grid, receptors, Q, …)` | ndarray (K,N) | Batched excess for many trial sources (inversion grid) |

`predict_ppm` is the main downstream call (rotation + upwind mask + σ lookup + unit
conversion + background). `gm3_to_ppm_methane` is a pure converter. Use
`clamp_to_table=True` for the inversion (boundary note below).

**Optimizer-safe boundary (key for the inversion):** `predict_ppm(..., clamp_to_table=True)`
never raises on geometry — sub-1 m downwind clamps to the 1 m floor, >200 m falls back to
background. Default (`False`/strict) raises, so demo/validation surface bad geometry. The
optimizer still owns three bounds the model does NOT auto-fix: `u ≥ 0.5 m/s`, integer
`stability_class` (fix or loop, never vary continuously), and `Q ≥ 0`.

## Modules in brief

**`sensor_sim.py`** — synthetic readings with known truth: `raw = 1.9 + true_plume + drift
+ temp + humid + noise(1σ=SENSOR_NOISE_PPM)`. `synthetic_timeseries`, `make_plume_event`,
`make_weather`, `make_baseline_drift`. `effective_noise_floor(random, bias, n_avg) =
√[(random/√n)² + bias²]` — random averages, bias doesn't.

**`sensor_frontend.py`** — the missing FIRST stage: raw ADC voltage → ppm.
`voltage_to_ppm(v_out, T, RH)` runs `R_s = R_L·(V_c−V_out)/V_out` → `R_s/R₀ = A·C^(−m)`
(inverted to `C = (A/ratio)^(1/m)`) → multiplicative T/RH factor. Outputs **total ppm incl.
background** (adds none of its own — that's what `parse_csv`'s ppm already means).
`ppm_to_voltage` is the exact inverse (round-trip test + synthetic voltage). Constants live in
one block: circuit/bench values (`SUPPLY_VOLTAGE_V`, `LOAD_RESISTANCE_OHM`, `R0_OHM`) start
`None` and `voltage_to_ppm` **refuses** until set (`calibration_status()` gate) so it never
fabricates a ppm; `POWERLAW_A/M` are datasheet-typical placeholders (⚠ unverified at 2–40 ppm);
T/RH coeffs default to a no-op. Bench runbook: `docs/CALIBRATION.md`. Per-sample bad voltage
(`v_out ≤ 0` or `≥ V_c`, non-finite) → NaN (skipped), not a crash.

**`processing.py`** — clean in order: `temp_humidity_correct` (fit a·T+b·H+c on a no-plume
window, subtract the varying part) → `subtract_baseline` (rolling low-percentile floor;
assumes the plume is a window minority) → `moving_average` (O(n) cumsum centred mean,
ESP32-portable; **display/detect only**) → `detect_pattern` (excess > k·noise for ≥min_run
→ `Detection{detected, start_idx, end_idx (exclusive), confidence, threshold}`).

**`feasibility.py`** — `sweep(Q, dist, stability)` classifies each cell detectable
(≥k·noise) / marginal / undetectable; `implied_Q_range` back-solves Q across wind×stability
(linearity → one eval/cell) → the `swing_factor` error bar; `verdict(...)` is the
plain-English bottom line. **Class-D headline:** smallest detectable leak at the 50 m fence
≈ **0.05 g/s** (0.90 ppm = 3× the 0.30 ppm floor) — a *lab-floor*; `verdict()` reframes it
as an **upper bound** with a ~160× `implied_Q_range`. Sweeps the centreline downwind sensor
only (most generous for detection).

**`fieldtest.py`** — bridge from a real CSV to the per-test graph and the inversion.
`parse_csv` (**real-world-robust**, F16: `#`/blank preamble, `;`/tab/decimal-comma delimiters,
token-matched header spellings with `ppm` beating generic `raw`/`value`, header-less positional
fallback, `millis()`/epoch/ISO/clock time → seconds, unit-sanity flags; never silently misreads)
→ `process_fieldtest` (same `processing.py` pipeline, raw frame: `raw ≈ baseline + excess`).
**Voltage entry point:** a CSV with a raw `voltage`/`adc` column (`_VOLT_NAMES`) and no ppm
column is converted via `sensor_frontend.voltage_to_ppm` INSIDE `parse_csv` — gated on
`calibration_status()`, so an uncalibrated front-end raises (STOP) rather than emitting a
fabricated ppm; a real `ppm` column always wins.
`aggregate_for_inversion` → `InversionPoint(mean_excess_ppm, sigma_ppm, …)`: time-mean of the
**unsmoothed** excess over a meander window + the σ-of-the-mean weight (NOT pre-smoothed — WLS
averages optimally). `make_sample_readings` for pre-sensor demos. Run `misc.preflight` on a real
upload first to see exactly what was detected.

**`weather.py`** — real wind + Pasquill class from the Open-Meteo archive (free, no key,
stdlib `urllib`). `fetch_wind_archive` is the only networked call; `parse_archive` /
`summarize_archive` / `pasquill_class` / `adjust_wind_to_height` are pure + fixture-tested.
Log-law-adjusts 10 m wind toward the ~2 m release height. Offline-safe (None +
`LAST_WEATHER_ERROR`). Daytime Pasquill cutoffs match EPA Table 6-7 (closed F8);
the night path still uses cloud cover instead of ΔT — a documented deviation.

**`inversion.py` (Week 3)** — `invert(sensor_positions, points, u, wind_dir_deg,
stability_class=None, …)` → `SourceEstimate(x, y, Q, …, crb_std, converged, n_snapshots)`.
**scipy-free:** the plume is linear in Q, so Q has a closed form `Q* = Σwgd/Σwg²` at every
trial (x, y); only a 2-D position search remains (shrinking-grid, multi-start). Stability
fit by trying all six. Compares scatter to the CRB; flags `converged=False` when no sensor
beats 3σ. **Multi-snapshot fusion (F7):** a single wind on a hard fenceline is
under-determined (~2 sensors in the plume for 3 unknowns) and silently lands ~28 m off.
`invert_multi(sensor_positions, [Snapshot(points, u, wind_dir_deg, stability_class=None)], …)`
fuses the SAME source under several winds — Q shared closed-form across snapshots (steady
leak), joint 2-D search — triangulating to sub-metre (2–3 winds). `invert` is the K=1 case.
`invert_field_tests` / `invert_field_tests_multi` are the raw-CSV→source light switches.
Fisher info adds → `accuracy.crb_source_bound_multi` is the combined (tighter) bound.
Denser default grid (`coarse=60, n_seeds=12`) since each snapshot sharpens the joint basin.

**`explain.py` / `ask.py`** — caption + chat. `template_explanation` always works;
`ai_explanation` / `ask_once` use OpenAI (`OPENAI_API_KEY` in `.env`, server-side only) and
fall back to the template with a plain-English reason on any failure (`LAST_AI_ERROR`).

**`db.py`** — `DATABASE_URL` set → Railway Postgres (`psycopg`, lazy); unset → local SQLite
`fieldtests.db` (offline + tests). Stores **only raw readings**; the processed view
recomputes on read. `init_schema()` idempotent.

## Accuracy framework (`physics/accuracy.py`)

A detection is only worth our confidence in it — critical when running a 500–12,500 ppm
sensor at 2–40 ppm, fainter still at Melissa. One **ordered protocol**:

1. **CHARACTERIZE** (`estimate_noise_floor`) — measure noise from the record's *quiet*
   samples, not the assumed 0.30 ppm. **Random** from successive differences (robust
   MAD ÷ √2; slow drift cancels) — averages as 1/√N. **Bias** from the heavily-smoothed
   quiet signal — does *not* average away. ⚠ in-record bias is only a **proxy**; a constant
   calibration offset is invisible within one record and needs a **zero-air bench run**.
2. **VALIDATE** (`recovery_metrics`) — vs known synthetic truth: RMSE, mean bias, %
   recovery, R², detection outcome, event-timing error. `null` for real uploads (no truth)
   → cleanly separates algorithm-validation from runtime self-assessment.
3. **BOUND** — *empirical* (`detection_limit`): `LOD = DETECT_K·floor`, `LOQ = 10·floor`,
   back-solved to a min detectable Q (linear in Q → no optimizer), reported **with a
   wind×stability error bar** (`swing_factor`, ~160×). *Theoretical* (`crb_source_bound`):
   the Cramér-Rao floor `√diag((JᵀJ/σ²)⁻¹)` from a numerical Jacobian of `predict_ppm` —
   the best **any unbiased estimator** could do (Gaussian iid noise, single source, local
   linearity; optimistic if any break). The inversion compares its achieved scatter to this.
4. **GRADE** (`accuracy_report`) — one JSON-safe dict: measured-vs-assumed floor, SNR,
   confidence, the limit + error bar, optional recovery, and **0–100 + letter + one
   recommendation**. Levers: Confidence 0–50 (detection margin, full marks at 2× threshold),
   Stability 0–30 (low *bias fraction* — error averaging can beat), Recovery 0–20 (synthetic
   only). 90=A…60=D. The recommendation attacks the weakest lever.

**Combined floor:** `floor(N) = √[(random/√N)² + bias²]` (`effective_noise_floor`). You
**cannot average to zero** — past N ≈ (random/bias)² the bias term wins.

**Two averaging roles (don't conflate — this is what makes the inversion honest):** noise
averaging (√N, `processing.moving_average`, display/detect only) vs meander time-mean
(`fieldtest.aggregate_for_inversion`, matches the steady-state model). The inversion does
**not** pre-smooth — a WLS fit over raw samples already does the √N reduction *optimally*
(it is the MLE under Gaussian noise); smoothing first is redundant and distorts the noise model.

## Server + storage

`misc/server.py` — tiny Flask bound to **127.0.0.1 only** (OpenAI key stays server-side).
Reuses existing functions unchanged:
- `GET /api/field` (grid+facts via `compute_field`) · `POST /api/caption` · `POST /api/ask`
  · `GET /api/feasibility`.
- Field tests: `GET/POST /api/fieldtests`, `GET /api/fieldtests/<id>`, `POST .../interpret`,
  `POST .../ask`, `GET .../compare?ids=…`, `DELETE .../<id>`, `POST .../sample`,
  `GET /api/sample.csv`.

Pages: `/` (contour), `/fieldtests.html`, `/explain.html`; shared nav + chat widget in
`web/common.js` (`window.CH4`). The model draws a **map** (concentration everywhere at
once); a real sensor gives a **graph over time** at one spot (methane = a bump rising out
of the 1.9 ppm background as the plume passes) — hence one ppm-vs-time graph per field test.

**Railway (local app + remote DB):** add a Postgres DB; put its **public** URL in `.env` as
`DATABASE_URL=…` (secret, gitignored); `pip install -r requirements.txt`; `python3 -m
misc.server` (schema auto-creates; still binds 127.0.0.1 only). Unset `DATABASE_URL` → local
SQLite. Deploying the whole app via `Procfile`/`gunicorn` makes the endpoints public — add
auth first.

## Bug-hunting method + backlog

(Former `BUGS.md` + `TOFIX.md` + the 2026-06-25 campaign review.)

**Standing practice — this loop runs every time, not just during a campaign.** Any change
that touches `physics/` (or the request that prompted it) ends with this, before calling
the work done:
1. **Generate adversarial fake data for the specific change** — not the existing fixtures.
   Pick from the failure taxonomy below for what could break (e.g. touched rotation → test
   a non-cardinal wind; touched the inversion → feed pure noise / collinear sensors /
   autocorrelated drift; touched normalization → integrate and check conservation).
   Independent expected values only (hand calc / different formula / conservation law) —
   never assert against the function being tested.
2. **Run the Detection toolkit (below) against it**, ranked — cheapest/highest-yield first.
3. **Lock whatever it finds as a test** before fixing (2-step removal, next paragraph), so
   the next change can't silently reintroduce it.
4. **Re-run the full suite** (`MPLBACKEND=Agg python3 -m pytest tests/ -q`) — green is
   necessary, not sufficient on its own (see *verification ≠ validation* below), so step 1's
   adversarial case is what actually earns the "done."
If a change doesn't touch `physics/` logic (docs, UI, plumbing), this loop doesn't apply —
don't manufacture adversarial data for its own sake.

**2-step removal:** (1) reproduce/lock with a test FIRST — *failing* for an active bug (✗),
*characterization* at the untested condition for a gap (⚠); (2) then change code, confirm
that test **and** full `pytest tests/` are green; tick it off. **Never fix silently** — a
closed item is the proof the guard exists. Expected values must come from an **independent**
route (hand calc, a different formula, a conservation law), never the code under test.
Beware *circular tests* (same function both sides — F2) and *trivial-fixture tests* (the
270° identity — F1). A green suite is necessary, not sufficient. *Verification* ("built it
right" — the test passes) ≠ *validation* ("built the right thing" — the model is correct):
F6 and F10's documented diagnoses were both wrong, and only measurement caught it.

**Failure taxonomy:** units (`gm3_to_ppm_methane`; g↔kg, Pa↔hPa, ppb↔ppm) · angle/sign +
270° identity (rotation; invisible at 270°, exercise a non-cardinal wind) · normalization
(the 2π + ×2 reflection; a factor error keeps the shape and only scales Q — only
**conservation** catches it) · singularities (u→0, σ→0, WLS denom→0; guarded, don't remove a
guard) · boundary/silent clamp (`np.interp` clamps outside 1–200 m; `clamp_to_table` makes
it deliberate) · inversion local-minimum / ill-posedness (wrong basin F7; collinear sensors
→ clean fit + huge `crb_std`, not a bug).
**Detection toolkit (ranked):** magnitude check → conservation (`u·∫∫C dy dz = Q`) →
forward↔inverse round-trip → analytic limits (centreline `Q/(π σ_y σ_z u)`, upwind/far =
background) → symmetry/monotonicity → adversarial + finiteness → two implementations agree
→ plot it and look.

### Open — highest value first (triage priority: bench › cite › desk › bound)

- **🔬 `BIAS_FLOOR_PPM = 0.00`** (`sensor_sim.py`) — the one item that can sink the project.
  At low-ppm Melissa, *drift* (not random noise) sets the detection floor and averaging
  can't touch it. Currently `0.00` = optimistic → every "detectable?" verdict is a best
  case until measured (hours of slow drift on the real Figaro).
  **📚 LIT (2026-07-09) — external confirmation this is THE governing floor:** Shah et al.
  2023 (AMT 16:3391) saw field baseline excursions of **+78% / −20**%; Furuta et al. 2024
  (AMT 17:2103) had to fit the baseline *piecewise in time*; Honeycutt et al. 2019 (Sensors
  19:3157) rate TGS-2611 the most baseline-stable low-cost CH₄ sensor yet still requiring
  "dynamic background subtraction." Non-averageable drift is the documented failure mode of
  this sensor class. See memory `literature-validation`.
- **🔬 Assumed sensor constants** (`sensor_sim.py`) — `SENSOR_NOISE_PPM 0.30`,
  `TEMP_COEFF 0.05`, `HUMID_COEFF 0.02`, `BASELINE_DRIFT 1.00`. Placeholders that drive
  every detectability/accuracy number. Measure on a zero-air/clean-air bench run; then every
  limit, error bar, and grade re-computes automatically — no other change.
  **📚 LIT (2026-07-09):** peer-reviewed characterization of the EXACT sensor puts the
  realistic resolution at **~1–2 ppm, not 0.30** — Shah et al. 2023 (RMSE <±1 ppm for CH₄
  ≤28 ppm; 1 ppm → only 1.4–2% resistance drop) and Furuta et al. 2024 ("can distinguish 2
  from 10 ppm, but not 2 from 3 ppm"; field RMSE ≈0.5–0.65 ppm after heavy correction). So
  `SENSOR_NOISE_PPM=0.30` is optimistic ~3–6× → it *worsens* every verdict, esp. Melissa.
  Humidity DOMINATES the raw signal at ~2 ppm (Shah 2023) → supports the nonlinear T·H item.
- **F6 residual** (`inversion.py`) — strongly autocorrelated *background* (ρ≈0.8–0.95) still
  fabricates a source ~40% (the detector fires on sustained wander) — that fabrication-rate
  residual remains open. The σ-overconfidence half is CLOSED (see F12): `sigma_ppm` now uses
  the record's own measured ρ̂ via `effective_noise_floor(rho=)`; MC coverage 1.03–1.29×
  (slightly conservative) at a 120-sample window across ρ 0–0.95. Remaining edge: SHORT
  windows (~30 samples) at ρ≈0.9 still under-cover (~0.74×, vs 0.64× before the fix).
  White-noise fabrication is already fixed (54% → 0%).
- **F7 residual** (`inversion.py`) — a single-snapshot fit on an under-determined layout
  still reports `converged=True` with an optimistic CRB (false confidence). Cured
  *operationally* by fusion; deployment guidance: **fuse ≥2 winds, or place ≥3 sensors that
  all sit in the plume.** The honest-flag path (widen CRB / `converged=False`) was
  deprioritized in favour of the fusion fix.
- **📚 Cite** — Briggs σ validity sub-100 m at a cluttered site (order-of-magnitude only);
  roughness `DEFAULT_Z0 = 0.1` (open country; a built fenceline is 0.3–1.0 m → changes the
  wind-height adjustment and implied stability — pick a site-specific z0).
- **⚖️ Quantify-only (by design)** — rough-terrain dilution (open-country σ *under*-predicts
  mixing → may **over**-predict ppm; note the bias direction when reporting); the in-record
  bias proxy; the ~160× Q-swing until real wind pins u + class (so `weather.py` must actually
  be wired per real run, not left default).
- **🔬 Sensor front-end BUILT — constants pending bench** (`sensor_frontend.py`, 2026-07-12).
  The voltage→ppm stage (`R_s = R_L·(V_c−V_out)/V_out` → `R_s/R₀ = A·C^(−m)` → T/RH) now
  exists and is gated-wired into `parse_csv`, so a raw-voltage CSV can drive the whole
  pipeline. What remains is **measurement, not code**: `SUPPLY_VOLTAGE_V`, `LOAD_RESISTANCE_OHM`,
  `R0_OHM` are `None` (converter refuses until set — the gate); `POWERLAW_A/M` are
  datasheet-typical and **⚠ unverified at 2–40 ppm** (datasheet curve is 500–12,500), so every
  ppm is provisional until refit against certified span gas. Bench runbook + indoor guidance:
  `docs/CALIBRATION.md`. Follow-ups: fit A/m from span gas (one-constant edit); derive the
  noise floor from ADC bits / V_ref / R_L instead of the `sensor_sim.py` assumption; emit
  voltage columns from `misc/rehearse.py` (uses the `ppm_to_voltage` inverse).
- **🔧 / enhance** — pressure still fixed at 1 atm in `gm3_to_ppm_methane` (temperature is
  now wired: CSV T column → `aggregate_for_inversion.mean_temperature_c` → inversion + CRB
  `T_K`; P_Pa remains default); nonlinear T·H weather-correction term; multi-source
  inversion (sum of plumes). The over-amplification method (large load resistor + precision
  ADC — the experimental justification for the whole project) is now documented in
  `docs/CALIBRATION.md`.

### Closed ledger (guard exists; proof is the named test)

- **F1** — rotation pinned off-cardinal (hand-calc 30°/90°/180° + vectorized-vs-scalar 200°
  + round-trip; 30° not 45°, since at 45° `sin=cos` hides a sin↔cos swap). No code change.
- **F2** — `gm3_to_ppm_methane` anchored to an independent ideal-gas `V_m = RT/P`
  (≈1499.6 ppm per g/m³ at STP) + ∝C, ∝T checks. Catches a ×1000 / kg↔g slip. No code change.
- **F3** — wrote the missing `predict_excess_grid` equivalence test, then wired the batched
  kernel into the inversion via `_eval_cells` (one plume pass per lattice). Suite 18 s → 6 s.
- **F4** — mass-conservation: `u·∫∫C dy dz ≈ Q` across x ∈ {50,100,150} × classes {1,4,6}
  (holding across x is the signature of correct, x-independent normalization). No code change.
- **F5** — `aggregate_for_inversion` off-by-one: dropped the `+1` (`end_idx` already
  exclusive); window == the detected event. Was biasing `mean_excess` (and Q) low.
- **F6** — white-noise fabrication 54% → 0% by gating on the per-sample detection floor
  `DETECT_K·√(random²+bias²)`, not the σ-of-the-mean (which the positive baseline bias +
  √N shrink fooled). Residual above.
- **F7** — single-wind ill-posedness cured by `invert_multi` multi-snapshot fusion
  (~28 m → ~0.3 m at 2 winds, ~0.5 m at 3). NOT an optimizer local-minimum — genuinely
  under-determined; the cure is more information, not cleverer post-processing.
- **F8** — daytime Pasquill cutoffs fixed to EPA-454/R-99-005 Table 6-7: `925/675/175`
  + the `<175→D` near-neutral band (was `700/350`, biased too unstable). Locked by
  `test_pasquill_*boundary*` incl. exact-boundary inclusivity (≥925/≥675/≥175). NOTE:
  CLAUDE.md had reserved this as a user-owned decision (it shifts the real-site class);
  the change shipped in the 2026-06-30 working tree — flag to the user before Custer.
  Night path (cloud-cover proxy for ΔT) unchanged and still a documented deviation.
- **F9** — `summarize_archive` guarded against a direction-less archive (`direction is None`
  / all-NaN → default 0°).
- **F10** — `temp_humidity_correct` drops a (near-)constant regressor before the fit and
  reports its coefficient as 0 (the min-norm `lstsq` had invented a phantom `b_humid=0.058`).
  The *corrected signal* was fine; the reported *coefficient* was the defect.
- **F11** (2026-07-07 review) — weather-fit fabrication via interpolated T/H: a gap
  spanning the plume event, filled by parse-time interpolation, fed the WHOLE-record fit
  and could teach it to subtract real methane. Cure at one altitude: `parse_csv` reports
  reality (NaN gaps, no invented values — also persisted as NULL, so stored tests round-trip
  identically), `process_fieldtest._densify` owns drop-vs-interpolate, and the fit trains
  only on REAL samples (`ref_mask`). Locked by
  `test_weather_fit_ignores_interpolated_gap_through_event` + the nasty-CSV test.
- **F12** (2026-07-07 review) — F6 σ-overconfidence: AR(1) effective-N now lives INSIDE
  `sensor_sim.effective_noise_floor(rho=)` (one "how noise averages" story), fed by the
  record's measured ρ̂; `detection_limit`'s averaging curve takes the same `rho`. Guard:
  ρ̂ is NOT measured on event-dominated records (corrcoef of the plume bump ≈0.98 would
  collapse that sensor's WLS weight → returns 0 = no correction).
  `test_effective_noise_floor_rho_reduces_effective_n`,
  `test_lag1_autocorrelation_refuses_event_dominated_record`.
- **F13** (2026-07-07 review) — `ill_posed` flag made unit-safe: raw `cond(F)` mixes ppm/m
  with ppm/(g/s) and flips with leak size; now = correlation-normalized Fisher cond
  (rank deficiency, e.g. 1 sensor × 2 winds) OR the geometric mirror test
  `_wind_collinear_blind` (all sensors on one wind-parallel line → crosswind position is
  reflection-ambiguous; invisible to any local Fisher analysis at the fitted point).
  `test_ill_posed_flag_is_scale_invariant`.
- **F14** (2026-07-07 review) — data-quality warnings actually reach the user: parse +
  process warnings ride the create response, process warnings are recomputed on every
  detail read (storage keeps raw gaps), and the fieldtests page renders a DATA WARNINGS
  panel. `/api/fieldtests/invert` refuses tests whose stored `sensor_distance_m` disagree.
  Accuracy report rides the detail/create responses (pipeline runs once per view, not
  twice). `test_upload_surfaces_parse_and_process_warnings`,
  `test_detail_recomputes_process_warnings`,
  `test_invert_endpoint_rejects_mismatched_sensor_distances`.
- **F15** (2026-07-07 review) — `accuracy_report` crashed (and 500'd the upload/detail
  endpoints) on a stored calm-wind test (`u < U_MIN`, where `predict_ppm` raises).
  Now degrades honestly: `detection_limit: null` + the reason in the recommendation.
  `test_accuracy_report_survives_calm_wind_metadata`.
- **F16** (2026-07-09 hardening) — **real-world CSV transferability** for the imminent
  Custer/Melissa uploads. `parse_csv`/`process_fieldtest` were fitted to the clean sample and
  hard-failed or *silently misread* a student logger's real output. Now handled (policy: never
  silently misinterpret — auto-fix only when unambiguous, else warn loud): leading `#`/blank
  preamble skipped; `;`/tab delimiters + decimal-comma sniffed; header spellings matched by
  token containment (`CH4 (ppm)`) with a real `ppm` column always beating a generic
  `raw`/`value`; header-less numeric dumps assigned positional columns; `millis()`/epoch time
  auto-scaled to seconds and ISO-8601/clock strings parsed (was the worst silent bug — it
  collapsed the meander window to n=1); ADC/ppb/% unit mismatches flagged; MAX_ROWS truncation
  warned; Flask `MAX_CONTENT_LENGTH` cap; str-BOM stripped. Then an **adversarial fuzz sweep**
  (~100 cases, 4 rounds, loop-until-dry) hardened the survivors: NUL bytes stripped (were a
  `csv.Error` crash), non-`#` prose banners skipped, ragged rows (extra fields) warned (catches a
  decimal-comma-in-a-comma-file silent misread), ISO-tz/AM-PM time parsed, zero-width unicode
  stripped, and `preflight.run_preflight` made never-raise. 34 tests in
  `tests/test_fieldtest_transferability.py` (incl. a capstone "every quirk at once"). New field
  tool `misc/preflight.py` (GO/CHECK/STOP), `tests/test_preflight.py`. F11/F14 + DB NaN-gap
  round-trip preserved.
- **Math re-verification (2026-07-09)** — a 3-agent independent audit re-derived the forward
  (mass conservation → Q<0.001%; units 1499.6 ppm/(g/m³); off-cardinal rotation) and inversion
  (closed-form Q; CRB vs a from-scratch FD-Jacobian; fusion) math against *independent* routes:
  **no coding errors** — only the documented validity limits (F6/F7). Five independent anchors
  the suite lacked are now locked in `tests/test_inversion_anchors.py`: absolute CRB vs a
  from-scratch FD-Jacobian, closed-form Q = Σwgd/Σwg², the `effective_noise_floor` quadrature +
  its bias/random limits, `detection_limit` LOD/LOQ + linear-Q back-solve, and multi-snapshot
  fusion recovery + Fisher-addition CRB tightening. Plus `tests/test_pipeline_e2e.py` (CSV text →
  recovered source) and `tests/test_sigma_table.py` (σ-table byte-identical to its generator). A
  literature pass confirmed the modeling caveats — see the 📚 LIT notes above and memory
  `literature-validation`.
- **F17** (2026-07-09) — `weather.py` offline-safe contract generalized. `summarize_archive`
  guarded `cloud`/`direction` (F9) but `np.nanmedian(None)` on a missing `wind_speed_10m` or
  `shortwave_radiation` column **crashed** (TypeError) — a partial Open-Meteo response would
  take down `wind_for_site_date`. New `_nanmedian_or` makes every column None/all-NaN-safe;
  `wind_for_site_date` now returns None (+ reason) when wind is all-NaN rather than feeding a
  NaN `u` (C∝1/u) into the inversion. `test_summarize_archive_missing_wind_or_shortwave_does_not_crash`,
  `test_wind_for_site_date_returns_none_on_unusable_wind`.

## Constraints & caveats (coded-right ≠ physics-valid)

These are tracked validity limits, removable only by real / controlled-release data + a
citation, not by a test:

- **Forward (`plume.py`):** steady-state (time-mean only; real readings fluctuate far more,
  expect large scatter) · `u ≥ 0.5 m/s` (raises below; calm wind is both the most detectable
  case and where the model is undefined — flag those field measurements) · open-country σ
  under-predicts dilution at a rough site · sub-100 m is order-of-magnitude ·
  `gm3_to_ppm_methane` adds no background.
- **Simulation (`sensor_sim.py`):** the assumed sensor constants are placeholders; the model
  is linear in noise/weather (real MOX may be nonlinear) — synthetic data is for **algorithm
  validation only**.
- **Processing (`processing.py`):** baseline assumes the plume is a window minority; the T/H
  correction is a linear fit (real drift may be nonlinear — pre-fit and ship as constants
  on-device).
- **Feasibility (`feasibility.py`):** sweeps the centreline downwind sensor only (most
  generous for detection).

## Before editing physics

Read the function, its callers (`grep` the symbol), and its test. Decide which "right" you
need — *coded-correct* vs *physics-valid* — and don't conflate them. **Two inputs make the
inversion well-posed (they compose):** (1) averaging done right (meander time-mean + WLS,
not pre-smoothing); (2) real wind from `weather.py` replacing the u/stability guess.
Together they collapse the ~160× Q swing.

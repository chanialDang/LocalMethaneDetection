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

---

## File layout

```
LocalMethaneDetection/
├── plume.py                   # Core physics module — Gaussian plume + Briggs σ
├── sensor_sim.py              # Synthetic sensor data: truth + noise + weather
├── processing.py              # Signal cleaning: baseline, averaging, T-H, detection
├── feasibility.py             # Forward-model sweep + detectability verdict
├── explain.py                 # Caption generation: template or OpenAI-enriched
├── demo.py                    # Validation, sweep, contour plot with caption
├── tests/test_processing.py   # Pytest suite (Tests 1–5)
├── conftest.py                # Pytest configuration (makes project importable)
├── otm33a_dispersion_sigma.csv # Briggs σ lookup table (1–200 m, 6 classes)
├── plume_contour.png          # Output from demo.py (with caption box)
├── requirements.txt           # numpy, matplotlib, rich, pytest, openai (opt)
├── CLAUDE.md                  # This file (architecture & API docs)
└── EXPLANATION.md             # Plain-English project overview
```

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
| `SENSOR_NOISE_PPM` | 0.30 | ppm (1σ) | **ASSUMED** — replace with measured value | `sensor_sim.py` |
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
- `verdict(cells, fence_distance=50.0, stability=4, noise_floor=0.30, k=3.0)` 
  → returns a plain-English sentence: the minimum detectable leak at the fenceline,
  or whether detection is infeasible.

**Classification threshold:** A signal is **detectable** if it exceeds
`k × SENSOR_NOISE_PPM` (default: 3 × 0.30 = 0.90 ppm). This is the same
threshold the processing tests use, so the "is it detectable?" answer is
consistent everywhere in the project.

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

---

## Running the demo and tests

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the forward model validation and feasibility assessment:**
```bash
python demo.py
```
- Validation: Gaussian plume equation, unit conversion, wind rotation checks.
- Feasibility sweep: 24 cases (4 leak sizes, 3 distances, 2 stability classes).
- Contour plot: `plume_contour.png` with detection-limit line and plain-English caption.
- Exit code 1 if any validation fails.

**Run the processing tests:**
```bash
pytest tests/ -v
```
Tests 1–5 validate baseline subtraction, noise averaging, temperature/humidity correction,
pattern detection, and the full pipeline end-to-end. Each test uses synthetic sensor data
with a known-answer ground truth (see `sensor_sim.py`).

**Optional: AI-enriched captions (requires OpenAI API key):**
```bash
export OPENAI_API_KEY=sk-...
python demo.py
```
The caption in `plume_contour.png` will be generated by GPT-4o-mini. Without the key,
captions use the template automatically.

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

## Project roadmap

**Week 2 (complete):** Forward model + simulation + processing toolkit
- Physics engine (`plume.py`): steady-state Gaussian plume with Briggs dispersion.
- Synthetic sensor (`sensor_sim.py`): realistic fake readings with known ground truth.
- Signal processing (`processing.py`): baseline subtraction, smoothing, weather correction, pattern detection.
- Processing tests (`tests/test_processing.py`): validation against synthetic data (Tests 1–5).
- Feasibility assessment (`feasibility.py`): sweep forward model, classify detectable/marginal/undetectable.
- Visualization (`explain.py`, `demo.py`): contour plot with detection-limit line and plain-English caption.

**Week 3 (next):** Inversion — given sensor readings, find the source
- Use the forward model as an objective function in `scipy.optimize`.
- Accept real (noisy) sensor readings or synthetic test data.
- Return estimated source position (x, y) and emission rate Q.
- Quantify uncertainty in the estimate.

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

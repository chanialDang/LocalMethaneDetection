# Reviewer's Packet — Methane Fenceline Detection Project

**Purpose.** This document lets an independent reviewer (a teacher, mentor, or academic) audit
the scientific claims of this project in roughly an hour, without reading the source code. It
states every governing equation with its primary source, separates what the software *proves*
from what only measurement can prove, lists every assumption and its status, and gives commands
a reviewer can run to reproduce the core checks. It doubles as the Methods section of the writeup.

**One-sentence scope.** This is a **provisional fenceline leak-detection and localization tool** —
a forward Gaussian-plume model plus an inversion — built to run a Figaro TGS 2611-E00 metal-oxide
sensor at 2–40 ppm, far below its rated 500–12,500 ppm range. It is **not** a certified
concentration instrument; absolute ppm is provisional until calibration-gas characterization.

**Honesty posture (read first).** The project's credibility rests on stating limits *before* a
reviewer finds them. Numbers that are measured are labeled measured; numbers that are assumed are
labeled ASSUMED; extrapolations and provisional scales are flagged at the point of use, in code and
in output. Nothing in the physics is invented — every equation traces to a cited primary source.

---

## 0. How to read the citation discipline

Citations below are given by author/year/venue as recorded in the project's working notes and the
Mitchell extraction. **Before publication, the student should confirm each primary source directly**
(that the reference exists, and that the value/claim attributed to it is correct). This packet
deliberately does not assert page numbers or precise figures it cannot reproduce; where a number is
reproduced by independent calculation, that is stated explicitly and the command is given.

Two classes of claim are kept strictly separate throughout:
- **Verification** ("built it right") — the code correctly implements the stated math. Provable by tests.
- **Validation** ("built the right thing") — the math matches physical reality. Provable **only** by
  bench/field measurement, never by a passing test. (Distinction after Tegmark; see §4.)

---

## 1. Governing equations, with sources and independent checks

### 1.1 Gaussian plume (the forward model)
```
C(x,y,z) = Q / (2π σ_y σ_z u) · exp(−y²/2σ_y²)
                              · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
```
- **Source:** Pasquill (1961); Gifford (1961) — standard steady-state Gaussian plume with a
  ground-reflection image term. Any air-dispersion textbook reproduces this form.
- **Meaning:** `Q/(2πσ_yσ_z u)` is mass conservation; the first exponential is crosswind spread; the
  bracket is perfect ground reflection (image source at depth −H).
- **Independent check (reproducible):** integrating C over a downwind y–z plane and multiplying by u
  must return the emitted Q, at every downwind distance. Reproduced here with a *separately written*
  numpy integration: recovered **Q = 5.01 g/s vs. true 5.0 (≈0.25% error, constant across x = 50/100/150 m)**.
  The error is pure grid discretization; that it does **not** drift with x is the signature of correct,
  distance-independent normalization. Locked as a regression test (`tests/`, "mass conservation").

### 1.2 Dispersion coefficients σ_y, σ_z
- **Source:** Briggs (1973), open-country parameterization, indexed by Pasquill stability class A–F.
  Read from `physics/briggs_dispersion_sigma.csv`, generated reproducibly by
  `physics/generate_sigma_table.py` (the committed table is byte-identical to the generator, locked by
  `tests/test_sigma_table.py`).
- **Stated validity (a limitation, not a bug):** fit at ≥100 m over flat open terrain. At a sub-100 m,
  built-up fenceline the coefficients are order-of-magnitude only, and open-country σ *under*-predicts
  mixing at a rough site, which can **over**-predict ppm. This bias direction is disclosed wherever
  results are reported.

### 1.3 Wind-frame rotation
```
x_wind = dx·(−sin φ) + dy·(−cos φ)     (downwind, + = downwind)
y_wind = dx·( cos φ) + dy·(−sin φ)     (crosswind)
```
- `wind_dir_deg` (φ) is the meteorological **from-direction** (clockwise from N). Verified off-cardinal
  (30°, 90°, 180°, 200°) against hand calculation and a vectorized-vs-scalar cross-check — deliberately
  *not* only at 270°, where the rotation degenerates to the identity and would hide a sin/cos swap.

### 1.4 Unit conversion (g/m³ → ppmv)
- **Basis:** ideal gas, molar volume V_m = RT/P; ppmv = (1/M_CH4)·V_m·10⁶.
- **Constants:** R = 8.314 J/(mol·K) (NIST CODATA); M_CH4 = 16.04 g/mol (IUPAC); background CH₄ = 1.9 ppm
  (NOAA GML). *Student to confirm each primary value.*
- **Independent check (reproducible):** hand-derived **1397.3 ppmv per g/m³ at 0 °C**, matches the code
  exactly. Note this value is temperature-dependent (≈1499.7 at 20 °C) — code and hand-calc agree at the
  *same* T, which is itself a correctness check. No background is added by this converter.

### 1.5 Sensor front-end — voltage → ppm (the calibration)
**Voltage divider (Ohm's law):**
```
V_out = V_c · R_L / (R_L + R_s)     ⇒     R_s = R_L · (V_c − V_out) / V_out
```
**Power-law characteristic (Figaro TGS 2611-E00 datasheet):**
```
R_s / R_0 = A · C^(−m)     ⇒     C = ( A / (R_s/R_0) )^(1/m)
```
**Gas-free background anchor (the project's method for calibrating without span gas):** at clean air
R_s = R_0 ⇒ ratio = 1 ⇒ C = A^(1/m); forcing that to equal the known 1.9 ppm background gives
**A = 1.9^m** (derived, not stored). Independent check: on the real Node 1 baseline the pipeline returns
**median 1.90 ppm** in clean air (the anchor holds by construction), reproducible via `misc.calibrate`.
- **Disclosed limitation:** the datasheet A/m are referenced to the datasheet's own R_0 (measured at
  5000 ppm), while we anchor to a clean-air R_0 (at 1.9 ppm). This re-referencing makes the **absolute**
  ppm scale *provisional* — harmless for locating a leak (C is linear in Q, so a scale error rescales the
  recovered leak *rate*, not its *position*) but not a certified concentration. And m is a 500–12,500 ppm
  slope extrapolated down to 2–40 ppm.

### 1.6 Mitchell Eq. 16 (built but DORMANT — disclosed as not-yet-usable)
```
M = C1 + C2·exp(C3·V − C4·ln(T+65) − C5·ln(H)) − C7·ln((T+65)·V)
```
- **Source:** Mitchell, Cox & Lewis (2024), *Sensors* 24(4):1066, DOI 10.3390/s24041066. Full extraction
  in `calibration/docs/mitchell2024_eq16_extraction.md`.
- **Why it is gated off, not used:** (a) it ingests V_out directly and does **not** use R_0; (b) its
  coefficients are hardware-specific and require calibration gas to fit — the paper's own coefficients
  produce **≈ −165 ppm** on our hardware (physically impossible; reproduced and locked as a test as the
  concrete proof coefficients cannot be transported); (c) even correctly fit, its reported RMSE ≈ 5.1 ppm.
  The kernel therefore refuses to run until gas-fit coefficients exist. This is the honest handling of a
  model we cannot yet legitimately apply.

### 1.7 Inversion (readings → source)
- **Closed-form Q:** because C is linear in Q, at any trial (x,y) the optimal Q has a closed form
  `Q* = Σ w g d / Σ w g²` — no iterative solver for Q; only a 2-D position search remains. Verified against
  a from-scratch derivation (`tests/test_inversion_anchors.py`).
- **Multi-snapshot fusion:** a single wind on a fenceline is under-determined (~2 sensors in the plume for
  3 unknowns) and lands ~28 m off; fusing the same source under 2–3 winds (Fisher information adds)
  triangulates to sub-metre. Demonstrated end-to-end on a *known* synthetic source: recovered position
  error **≈ 0.5 m** (`misc.rehearse`).
- **Cramér–Rao bound:** the theoretical best-possible 1σ any unbiased estimator could achieve, from the
  Fisher information of a numerical Jacobian of the forward model. The inversion compares its achieved
  scatter to this floor. Verified against an independent finite-difference Jacobian.

### 1.8 Honest noise floor
```
total floor = √( random² + bias² + model² )        LOD = 3σ · total floor
```
- **random** — fast electrical/turbulent jitter; averages as 1/√N (measured from successive differences).
- **bias** — non-averageable slow baseline drift (an in-record *proxy*; a true constant offset needs a
  zero-air bench run).
- **model** — datasheet-extrapolation uncertainty; entered in quadrature at ±1.7 ppm (Van den Bossche
  2017 benchmark on this sensor — *student to confirm*).
- **Why this matters:** the fast electrical jitter alone converts to a flattering ~0.018 ppm, which is
  **not** the detection limit. The floor is dominated by drift and model error, which averaging cannot
  reduce. On the real Node 1 baseline the honest floor is **≈ 1.70 ppm, LOD ≈ 5.10 ppm**.

---

## 2. Where the project stands: code vs. hardware

- **The code is mature and ahead of the data.** Forward model, inversion, calibration front-end, and the
  accuracy framework are built and internally cross-checked (305 automated tests pass).
- **The binding constraint is now measurement, not software.** The items that would most improve the
  science are physical: the sensor's temperature/humidity coefficients, its real multi-hour drift,
  calibration gas for an absolute ppm scale, per-sensor calibration, and actual multi-sensor field data.
- **Consequence for a reviewer:** software correctness (verification) is well-evidenced here; physical
  validity (validation) is explicitly *pending data* and is not claimed.

---

## 3. Test methodology — why these are not circular ("AI-slop") tests

The project's standing rule (predating any AI involvement) is that **every expected value in a test comes
from an independent route** — a hand calculation, a conservation law, a second formula, or a round-trip —
**never from the function under test asserting against itself.** Representative examples:

| Check | Independent route used | What a failure would catch |
|---|---|---|
| Mass conservation | separate numpy integration of the plume, compared to Q | a normalization / 2π / reflection-factor error |
| Unit conversion | hand-derived ideal-gas V_m = RT/P | a g↔kg or ×1000 slip |
| Rotation | hand calc at 30°/90°/180° + vectorized-vs-scalar | a sin/cos swap (invisible at 270°) |
| Calibration anchor | algebra A = 1.9^m, and R_s(baseline) ≈ R_0 by the divider | a mis-anchored ppm scale |
| Calibration slope | analytic derivative vs. finite difference of the real function | a wrong sensitivity → wrong ppm floor |
| Mitchell transport | hand-evaluated Eq. 16 giving −165 ppm | pretending paper coefficients are usable |
| Cramér–Rao bound | from-scratch finite-difference Jacobian | an optimistic/incorrect precision claim |
| Front-end round-trip | ppm→voltage→ppm via an independently written inverse | any asymmetry in the conversion |

A passing suite is **necessary but not sufficient**: it proves the code matches the math, not that the
math matches the sensor. That second step is §4.

---

## 4. Verification ≠ Validation (the crux of the credibility question)

- **Verified (evidenced by tests):** the equations of §1 are implemented correctly; conservation holds;
  units are consistent; the calibration anchor is self-consistent; the inversion recovers a *known*
  synthetic source to sub-metre.
- **NOT yet validated (requires measurement):** that the power-law slope m is correct at 2–40 ppm for this
  physical sensor; that the humidity/temperature response is as modeled; that real field drift matches the
  assumed floor; that the absolute ppm scale is accurate without calibration gas.
- **Two documented cases where a *desk* diagnosis was wrong and only measurement/re-derivation caught it**
  (F6, F10 in the project's bug ledger) are retained as evidence that the project does not treat "the code
  runs" as "the science is right."

---

## 5. Assumptions & limitations register

| Item | Status | Impact if wrong | How to remove |
|---|---|---|---|
| Power-law slope m = 0.35 | DATASHEET-TYPICAL placeholder | wrong ppm scale + wrong noise floor | digitize the real datasheet curve; fit m (`calibration/datasheet/`) |
| Absolute ppm scale | PROVISIONAL (no gas) | leak *rate* Q rough (location OK) | certified span-gas calibration |
| Temperature coeff | ASSUMED 0 (no-op) | drift with air temperature mis-attributed | bench T sweep |
| Humidity coeff | ASSUMED 0 (no-op) | **largest error at low ppm**; humidity fakes methane | bench RH sweep (highest priority) |
| Baseline drift / bias floor | in-record PROXY | detection limit optimistic on long field runs | multi-hour zero-air bench run |
| Dispersion σ (Briggs open-country) | model choice | order-of-magnitude at rough sub-100 m fenceline | site-specific σ / controlled-release validation |
| Release height H, sensor height z | ASSUMED (1 m) | scales predicted ppm | measure on site |
| Single-source assumption | model scope | multi-leak sites mis-fit | multi-source inversion (future) |
| Wind u, stability class | must be wired per run | C ∝ 1/u; class sets σ | real wind via `physics/weather.py` per site/date |
| Mitchell Eq. 16 | DORMANT (needs gas) | n/a — refuses to run | fit coefficients with span gas |

---

## 6. Agreement with independent literature (a validation signal)

The pipeline's honest floor (≈1.7 ppm) and LOD (≈5.1 ppm) were **not tuned** to match anything, yet land
in the range independent peer-reviewed work reports for this exact sensor class:
- ~1–2 ppm realistic resolution, drift-dominated (Shah et al. 2023; Furuta et al. 2024; Honeycutt et al. 2019);
- ~5 ppm RMSE for the best low-cost calibration model (Mitchell et al. 2024);
- ±1.7 ppm variable error benchmark (Van den Bossche 2017).

*These references are transcribed from the project's working notes and MUST be confirmed by the student
against the primary sources before publication.* Agreement with work the project did not author is
evidence the model reflects reality rather than being fit to a desired answer.

---

## 7. How a reviewer can reproduce the core claims (commands)

```
# Full automated suite (verification): expect all pass, ~4 s
MPLBACKEND=Agg python3 -m pytest tests/ -q

# The physics anchors specifically
MPLBACKEND=Agg python3 -m pytest tests/ -q -k "conserv or anchor or round_trip or rotation or crb"

# Calibration on the REAL Node 1 baseline: expect median ~1.9 ppm, honest floor ~1.7 ppm, LOD ~5.1 ppm
python3 -m misc.calibrate calibration/nodes/node1.py "Claude Chat -> Code/node1_clean_baseline.csv"

# End-to-end inversion on a KNOWN synthetic source: expect sub-metre recovery
python3 -m misc.rehearse

# Datasheet-slope fit workflow
python3 calibration/datasheet/fit_powerlaw.py
```
A reviewer who wants to check the plume normalization by hand can integrate C(x,y,z) over a y–z plane and
confirm `u·∫∫C dydz ≈ Q` at several x — the derivation is in §1.1 and the result is ~0.25% (grid-limited).

---

## 8. Self-disclosed open issues (stated before a reviewer finds them)

- **Absolute concentration is provisional** until span-gas calibration (§1.5).
- **Humidity/temperature correction is off** (coefficients 0) pending a bench sweep — the most likely source
  of field error, especially at the low-ppm Melissa site.
- **Autocorrelated-drift edge case (F6 residual):** strongly autocorrelated background can still fabricate a
  weak source ~40% of the time on short windows; the σ-overconfidence half is fixed and locked by test.
- **Single-wind ill-posedness (F7):** a single-snapshot fit on an under-determined layout can report false
  confidence; mitigated operationally by requiring ≥2 winds or ≥3 in-plume sensors.
- **Night-time Pasquill class** uses a cloud-cover proxy rather than ΔT — a documented deviation from the
  EPA daytime table.

---

## 9. References (to be confirmed against primary sources before publication)

- Pasquill, F. (1961). Atmospheric diffusion / stability classification.
- Gifford, F.A. (1961). Gaussian plume dispersion.
- Briggs, G.A. (1973). Diffusion estimation for small emissions (open-country σ).
- U.S. EPA-454/R-99-005, Table 6-7 (Pasquill class cutoffs).
- Figaro Engineering, TGS 2611-E00 datasheet (sensitivity characteristic; rated 500–12,500 ppm).
- Mitchell, H.L., Cox, S.J., Lewis, H.G. (2024). *Sensors* 24(4):1066. DOI 10.3390/s24041066.
- Shah et al. (2023); Furuta et al. (2024); Honeycutt et al. (2019) — TGS-2611 low-cost sensor characterization.
- Van den Bossche, M. et al. (2017). Low-end linear fit; ±1.7 ppm benchmark.
- NIST CODATA (R); IUPAC (M_CH4); NOAA GML (CH₄ background).

---

*This packet describes the state of the software and its evidentiary basis. It makes no claim of physical
validity beyond what independent measurement supports, and it invites — rather than resists — external
review.*

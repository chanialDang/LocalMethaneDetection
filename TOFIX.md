# TOFIX — bugs, gaps, and things to improve

A running list of things that may need fixing, verifying, or improving. Add to it
freely as we go. Nothing here is a crash bug right now (the suite is 148/148 green);
most items are **assumptions to replace with data** or **caveats to verify** — which
is exactly the honest accounting this project needs.

**How to use**
- Tick `- [x]` when done and move the line to **Resolved** at the bottom (keep the date).
- Each item: what + where (clickable `file:line`) + why it matters / what to do.
- Tags: `[measure]` needs lab/field data · `[verify]` needs a cited reference ·
  `[risk]` could make results wrong/optimistic · `[code]` hygiene/double-check ·
  `[enhance]` nice-to-have. `(by-design)` = documented limitation, decide if your
  site needs action — not a defect.

Created 2026-06-19.

---

## Triage — what each item actually needs (read this first)

Every open item below is one of four "fix paths." This tells you at a glance whether
a problem is a keyboard fix, a measurement you can't fake, or a permanent caveat:

| Path | Meaning | Items | Do it now? |
|------|---------|-------|------------|
| 🔧 **desk** | code/doc fix, no new data | §4; §3 z-height, §3 T/P | Yes — just do it |
| 📚 **cite** | look up + verify a number | §2 (Pasquill, Briggs, z0) | Yes — needs a reference, not hardware |
| 🔬 **bench** | measure on the real Figaro | §1 (noise, bias, T/H coeffs) | No — gated on the sensor |
| ⚖️ **bound** | can't remove, only quantify | §3 rough-terrain, bias-proxy, Q-swing | N/A — report the error bar honestly |

**Priority: `bench › cite › desk › bound`.** The item that can actually sink the
project is **§1 `BIAS_FLOOR_PPM`** (🔬 bench, currently `0.00` = optimistic): at
low-ppm Melissa, *drift* — not random noise — sets the detection floor, and averaging
can't touch it. Until it's measured, every "detectable?" verdict is a best case.

---

## 1. Measure before trusting (assumed sensor constants)

These are placeholders in `sensor_sim.py`. They drive every detectability and
accuracy number, so replacing them with measured values is the highest-value work.

- [ ] **`SENSOR_NOISE_PPM = 0.30`** — `physics/sensor_sim.py:54` `[measure]`
  The random floor. Measure from a zero-air / clean-air bench run (sample-to-sample
  jitter). Over-amplification + averaging lower this; it's the *averageable* half.
- [ ] **`BIAS_FLOOR_PPM = 0.00`** — `physics/sensor_sim.py:60` `[measure][risk]`
  Currently zero, so the model is optimistic. This is the **real detection limiter**
  (drift that averaging can't remove), especially at low-ppm Melissa. Measure the
  slow drift over hours and set it. Until then every "detectable?" verdict is a
  best case.
- [ ] **`TEMP_COEFF_PPM_PER_C = 0.05`** — `physics/sensor_sim.py:78` `[measure]`
  Assumed MOX temperature sensitivity. Characterize on the real Figaro.
- [ ] **`HUMID_COEFF_PPM_PER_PCT = 0.02`** — `physics/sensor_sim.py:81` `[measure]`
  Assumed humidity sensitivity. Characterize on the real Figaro.
- [ ] **`BASELINE_DRIFT_PPM = 1.00`** — `physics/sensor_sim.py:75` `[measure]`
  Assumed amplitude of slow zero-wander used in synthetic data.

## 2. Verify against a cited reference (literature gaps)

- [ ] **Pasquill stability cutoffs** — `physics/weather.py` `pasquill_class` `[verify]`
  **Checked 2026-06-21 — scope narrowed:** the wind×insolation→class LETTER GRID is the
  standard Pasquill–Gifford–Turner table (5 wind bands × 3 day insolation × 2 night cloud)
  — structure verified. The only open part is keying day insolation off W/m²
  (`_INSOL_STRONG/_MODERATE` = 700/350): that's the EPA **SRDT** method (legitimate), not
  the original solar-elevation-angle scheme, so the specific 700/350 numbers still need
  checking against the EPA SRDT day table (**EPA-454/R-99-005**). Night uses cloud ≥50% as
  the overcast split (spirit-consistent; SRDT uses ΔT instead).
  Refs: Wikipedia "Turner stability class"; WebMET §6.4.2 (SRDT).
- [ ] **Briggs σ validity at this site** — `physics/plume.py:222-223` `[verify][risk]`
  Briggs formulas were validated at ≥100 m over flat open terrain. Sub-100 m and the
  cluttered transfer-station fenceline are order-of-magnitude only. Decide how much
  to trust sub-100 m predictions.
- [ ] **Roughness length `DEFAULT_Z0 = 0.1`** — `physics/weather.py:77` `[verify][risk]`
  0.1 m = open country. A structure-cluttered fenceline is rougher (0.3–1.0 m), which
  changes the wind height-adjustment and the implied stability. Pick a site-specific z0.

## 3. Accuracy risks (could make results optimistic/wrong)

- [ ] **Rough-terrain dilution** — `physics/plume.py` (`concentration_gm3`) `[risk]` (by-design)
  Open-country σ *underestimates* mixing at a structure-filled site → may **over**-predict
  concentration → detectability looks better than reality. Note the bias direction when
  reporting.
- [ ] **In-record bias is a proxy** — `physics/accuracy.py:31-34`, `:87-92` `[measure]` (by-design)
  A constant calibration offset is invisible within one record; the measured bias only
  tracks residual drift. Pin the true offset with a zero-air run.
- [ ] **Sensor height `z` / release height `H` consistency** `[code][risk]`
  **Progress 2026-06-21:** deployment geometry is now centralized in `physics/plume.py`
  as `SENSOR_HEIGHT_M` / `RELEASE_HEIGHT_M`; `feasibility` and `inversion` read those
  instead of hardcoding `1.0`, so the real fence-mounted height (~1–1.5 m) is set in ONE
  place. Still open: (1) set the **measured** value before a real run (still the assumed
  1.0 m); (2) `compute_field` draws the contour at `z=0.0` (the textbook ground-level
  reference) while the detection verdict + inversion use `SENSOR_HEIGHT_M` — intended,
  but if you want the picture to match the verdict, point `compute_field` at
  `SENSOR_HEIGHT_M` too (that changes the contour and its pinned test).
- [ ] **Fixed T/P in unit conversion** — `gm3_to_ppm_methane` defaults 20 °C / 1 atm
  `physics/plume.py:494` `[code]`
  Field temperature/pressure differ; effect is small but real. Consider passing actual
  T/P from the weather/field metadata.
- [ ] **Wind/stability Q swing (~160×)** — `physics/feasibility.py` `implied_Q_range`
  `[risk]` (by-design)
  Recovered Q is only pinned to ~160× until real wind pins u + class. Tracked here as a
  reminder that `weather.py` must actually be wired/used per real run, not left default.

## 4. Code hygiene / double-check

- [ ] **Linear weather correction** — `physics/processing.py:172`
  (`temp_humidity_correct`) `[enhance]` (by-design)
  Simple `a·T + b·H + c` fit. Real MOX drift may be nonlinear; CLAUDE.md suggests a
  T·H interaction term as a candidate improvement.

## 5. Enhancements / future

- [ ] **Document the over-amplification method** — `EXPLANATION.md` / `CLAUDE.md`
  `[enhance]`
  Large load resistor + precision ADC is the *how* behind using a 500–12,500 ppm
  sensor at 2–40 ppm. Add a paragraph; it's the experimental justification for the
  whole project. (Level 0 from our discussion.)
- [ ] **Sensor front-end model** — new `physics/sensor_frontend.py` `[enhance]`
  Forward chain ppm → sensor resistance → R_L divider → ADC quantization+noise → ppm,
  so the noise floor is *derived* from circuit specs (ADC bits, V_ref, R_L) instead of
  assumed. Plugs in below `sensor_sim`, changes nothing downstream. Label it a design
  calculator built on extrapolated datasheet curves until benched. (Level 1.)
- [ ] **Field re-zeroing plan** — `[enhance]`
  Once amplification + averaging crush random noise, drift dominates. Plan periodic
  baseline/zero checks in deployment and a way to log them.
- [ ] **Multi-source inversion** — `physics/inversion.py:54` `[enhance]` (by-design)
  Currently single source only. Multiple simultaneous leaks would need a sum of plumes.

---

## Resolved

_(move finished items here with the date, e.g. `- [x] 2026-06-20 — fixed X`)_

- [x] 2026-06-21 — **Test count in docs** corrected (CLAUDE.md 138 → **148**, both the
  status line and the `pytest` comment). Verified via `pytest --collect-only` (148).
- [x] 2026-06-21 — **`inversion.py` documented**: added to the CLAUDE.md file-layout tree
  and the test-description prose, and the Week-3 roadmap reframed from "(next) / use
  `scipy.optimize`" to "(built) — scipy-free closed-form-Q WLS + grid search", matching the
  actual `invert(...)` implementation. (Remaining inversion work is real-data validation,
  still tracked under §5 / roadmap.)

---

## Template for new items

```
- [ ] **Short title** — `path/to/file.py:line` `[tag]`
  One line on why it matters / what to do.
```

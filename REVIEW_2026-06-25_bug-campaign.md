# Briefing — BUGS.md F-item eradication campaign (2026-06-25)

A durable record of what was changed, why, how it was verified, and what is still
open. Companion to `BUGS.md` (the live backlog). Reviewed with `/code-review` and
`/simplify` after the changes.

**Status at write time:** all changes **UNCOMMITTED** in the working tree. Full test
suite **160 → 165 passing** (`MPLBACKEND=Agg python3 -m pytest tests/ -q`, ~6 s).

**How to inspect the actual changes:**
```bash
git diff HEAD --stat          # the 10 touched files
git diff HEAD physics/inversion.py   # (or any file below)
MPLBACKEND=Agg python3 -m pytest tests/ -q
```

**Governing principle (Tegmark, Life 3.0):** *verification* = "built it right" (the
test passes) vs *validation* = "built the right thing" (the model/test is correct).
Two BUGS.md diagnoses (F6, F10) turned out to be the wrong story — only measurement,
not a passing test, caught that. Every new test takes its expected value from an
**independent route** (a different formula, conservation, or the physics), never the
code under test.

---

## What changed, item by item

### F5 — off-by-one in the aggregation window ✅ CLOSED
- **Was:** `aggregate_for_inversion` used `lo, hi = det.start_idx, det.end_idx + 1`,
  but `Detection.end_idx` is already *exclusive* → one sub-threshold sample was folded
  into the mean → `mean_excess` (and recovered Q) biased low.
- **Fix:** dropped the `+1` (`physics/fieldtest.py`).
- **Verified:** `tests/test_fieldtest.py::test_aggregate_window_matches_detected_event_exactly`
  asserts the window equals the detector's own reported `(start_idx, end_idx)`. RED→GREEN
  confirmed (window was `(235, 367)`, should be `(235, 366)`).

### F6 — fabricated source on a no-source record ⚠️ DOMINANT CAUSE FIXED (residual open)
- **BUGS.md said:** autocorrelation → underestimated σ → false convergence.
- **What measurement actually showed:** the dangerous failure happens even on pure
  *white* noise. The gate `signal_present = np.any(d > 3·sig)` compared `mean_excess`
  against the **σ-of-the-mean**. The baseline (a rolling *low-percentile* floor) biases
  `mean_excess` POSITIVE (~**+0.19 ppm** on 0.3-ppm white noise) while the σ-of-the-mean
  shrinks ~√N → pure noise fabricated a source **~50%** of the time.
- **Fix:** gate on the **per-sample detection floor** `DETECT_K·√(random²+bias²)` (the
  same threshold feasibility uses), not the σ-of-the-mean (`physics/inversion.py`,
  `_datum` + the `signal_present` line). This also sidesteps the autocorrelation σ issue
  for the gate. White-noise fabrication **54% → 0%**.
- **Verified:** `tests/test_inversion.py::test_pure_noise_does_not_fabricate_a_source`
  (RED at 27/50 fabricated, GREEN after).
- **RESIDUAL (still open):** strongly autocorrelated *background* (ρ≈0.8–0.95) still
  fabricates ~40% (the detector itself fires on sustained wander), and `sigma_ppm` is
  still ~2× overconfident for autocorrelated noise (so the CRB is optimistic when a real
  source *is* present). Correctly fixing this needs the **real Figaro noise
  autocorrelation** — an assumed placeholder today — so it was deliberately NOT fixed
  against a guessed AR model.

### F3 — unused batched kernel + missing test ✅ CLOSED
- **Was:** `predict_excess_grid` existed but was unused, and the test its docstring cited
  (`test_predict_excess_grid_matches_predict_ppm`) did not exist → adopting it would be
  unverified.
- **Fix:** wrote the equivalence test (kernel == `predict_ppm(…, clamp_to_table=True) −
  CH4_BACKGROUND` per receptor, exercised at 200° and 270° to dodge the rotation
  identity), then wired the kernel into the optimizer via a new `_eval_cells` that
  evaluates a whole lattice in one batched plume pass (replacing the per-cell
  `predict_ppm`). `physics/inversion.py`, `tests/test_plume.py`.
- **Payoff:** full suite **18 s → 6 s** (the inversion's per-cell setup is now paid once
  per lattice). Behavior unchanged — guaranteed identical per-receptor values, so the
  recovery tests stayed green.

### F8 — Pasquill insolation cutoffs 📚 VERIFIED (cutoffs diverge; doc updated, numeric fix recommended)
- **Verified against the primary source:** EPA-454/R-99-005 *Meteorological Monitoring
  Guidance for Regulatory Modeling Applications* (Feb 2000), Table 6-7 "Key to the SRDT
  Method" (p.6-15).
- **Finding:** EPA SRDT daytime solar-radiation bands are **≥925 / 925–675 / 675–175 /
  <175 W/m²** (four levels). The code uses **700/350** (three levels, no `<175→D`
  near-neutral band) → biased too *unstable* (it calls 700 W/m² "strong"; EPA's "strong"
  is ≥925). Daytime *wind* bands match EPA. EPA night uses **ΔT**, the code uses cloud
  cover (Turner's method) because Open-Meteo supplies cloud, not ΔT.
- **Action taken:** recorded the verified discrepancy + citation in the `physics/weather.py`
  validation caveat. **Not a code change** — the numeric correction (set 925/675/175 and
  add a `<175→D` band) shifts the stability class at the real deployment sites, so it is
  left as a user-owned decision (one-liner, teed up in BUGS.md F8).

### F9 — `summarize_archive` crash on a direction-less archive ✅ CLOSED
- **Was:** `archive.direction[~np.isnan(archive.direction)]` raised when the direction
  column was absent (`parse_archive` returns `None`) — broke the "never crash on a
  degraded archive" contract.
- **Fix:** guard `direction is None` (and all-NaN) → default 0° (`physics/weather.py`).
- **Verified:** `tests/test_weather.py::test_summarize_archive_without_direction_does_not_crash`.

### F10 — `temp_humidity_correct` rank-deficient on a constant regressor ✅ CLOSED
- **BUGS.md said:** "corrupts excess." **Measurement showed** the *corrected signal* was
  actually fine (the varying part is identifiable); what was wrong was the **reported
  coefficient** — the min-norm `lstsq` invented a phantom `b_humid = 0.058` for a
  genuinely unidentifiable (constant) humidity column.
- **Fix:** drop a (near-)constant regressor before the fit and report its coefficient as
  0 (`physics/processing.py`).
- **Verified:** `tests/test_processing.py::test_3b_constant_humidity_is_rank_safe`
  (asserts `b_humid == 0`, temp correction still exact, varying part fully removed). The
  old phantom 0.058 was reproduced to confirm it was a real defect.

---

## `/code-review` result — no correctness bugs in the changes

The 165 green tests + the equivalence test + the targeted RED→GREEN tests back this.
Findings were all minor; the two actionable ones were fixed in the `/simplify` pass.

1. **(Fixed)** `floor1` re-implemented `effective_noise_floor` → now calls the canonical
   helper (single source of truth, per CLAUDE.md's noise-floor convention).
2. **(Noted, edge case)** if `random_ppm`/`bias_ppm` were ever `NaN`, the gate would
   silently flag every sensor not-converged — fails *safe* (won't fabricate), not seen
   in tests, consistent with the rest of the code's no-NaN-guard style.
3. **(Behavior change, by design)** the new gate trades away sub-noise sensitivity: a
   real source whose mean excess is below `DETECT_K·single-sample-σ` is now flagged
   not-converged even though averaging could resolve its mean. Conservative on purpose
   for Melissa ("don't fabricate" beats "detect the faintest thing").
4. **(Fixed)** redundant `getattr(...) or 0.0` defaulting — dissolved by fix #1.

> Reviewed inline rather than via the skill's multi-agent fan-out, to save tokens (every
> line was authored this session). For maximally independent eyes, `/code-review ultra`
> runs it multi-agent in the cloud.

## `/simplify` result — one fix applied, rest already clean

- **Reuse (applied):** `physics/inversion.py` `_datum` now uses
  `effective_noise_floor(random, bias, n_avg=1)` instead of an inline `np.hypot`.
- **Simplification / Efficiency / Altitude:** nothing to change — the F10 variable-column
  fit is the minimal correct form, `_eval_cells` is the right batched generalization (not
  a bandaid), and the per-class closures capture only small arrays.

---

## Files touched (10)

| File | Item(s) |
|------|---------|
| `physics/fieldtest.py` | F5 |
| `physics/inversion.py` | F6, F3, simplify-reuse |
| `physics/weather.py` | F8 (caveat), F9 |
| `physics/processing.py` | F10 |
| `tests/test_fieldtest.py` | F5 |
| `tests/test_inversion.py` | F6 |
| `tests/test_plume.py` | F3 |
| `tests/test_weather.py` | F9 |
| `tests/test_processing.py` | F10 |
| `BUGS.md` | all items marked closed/verified, original findings preserved |

## Still open (your call)

- **F6 residual** — autocorrelated-background fabrication + ~2× σ overconfidence; needs
  real Figaro noise characterization before an effective-N correction.
- **F7** — local-minimum / collinear-sensor guard tests (you deferred this; never in scope).
- **F8 numeric** — change the daytime cutoffs to 925/675/175 + add `<175→D` (shifts
  real-site stability classes, so it's a deliberate decision).
- **Commit** — nothing is committed yet.

## Sources (F8)
- EPA-454/R-99-005, Table 6-7, p.6-15 — https://www.epa.gov/sites/default/files/2020-10/documents/mmgrma_0.pdf
- WebMET §6.4.2 (SRDT method) — http://www.webmet.com/met_monitoring/642.html

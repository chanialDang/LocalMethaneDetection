# BUGS.md — physics / inversion bug & test-gap backlog

A trackable, **removable** checklist of issues found by reviewing `physics/` and
`tests/` against the forward-model + inversion bug taxonomy (see the "Proactive
code reading & bug-hunting" section in `CLAUDE.md`). The code is **largely correct
today** — most entries are *missing guards*, not active bugs. Close items here as
they are fixed; this file should shrink over time.

**Severity legend:** ✗ active bug · ⚠ correct now but unguarded (test gap) ·
ℹ validity / robustness caveat.

## How to remove an item — the 2-step protocol

1. **Reproduce / lock it with a test FIRST.**
   - Active bug (✗) → write a test that **fails** on today's code.
   - Test gap (⚠) → write a *characterization* test exercising the **untested
     condition** (passes today; will fail if the behaviour regresses).
2. **Then change the code** (if any) and confirm the new test **and** the full
   `pytest tests/` suite are green.

Then tick the box, and move the entry down to **Section D — Closed** with a note.
Never delete a finding silently — a closed item is the proof the guard exists.

---

## Section A — Active bugs (✗)

### F5 — off-by-one in `aggregate_for_inversion` event window
- [ ] **✗ MED.** `physics/fieldtest.py:333-337` sets `lo, hi = det.start_idx,
  det.end_idx + 1`, but `Detection.end_idx` is **already exclusive**
  (`processing.py:243` documents "index just past where it ends"; `detect_pattern`
  computes its own mean with `excess[start_idx:end_idx]`, `processing.py:304`). The
  `+1` folds one extra **sub-threshold** sample into the averaging window.
  - **Impact (inversion):** `mean_excess_ppm` biased slightly **low** → recovered
    `Q` biased low. Worst for short events near `min_run`.
  - **Trigger:** any detected event; e.g. a 5-sample event averages 6 samples, the
    6th below threshold → mean pulled down ~1/6.
  - **Remove:** test that the aggregated window length == the detected event length
    (`hi - lo == det.end_idx - det.start_idx`), then drop the `+1`.

### F6 — overconfident inversion weights (the Melissa low-ppm trap)
- [ ] **✗ / risk MED.** `physics/fieldtest.py:348-349` builds the σ-of-the-mean
  with `n_avg = n_window`, i.e. treats every in-window sample as **independent**.
  Turbulent meander samples are autocorrelated, so the effective N is smaller and
  `sigma_ppm` is **underestimated**. This couples with `inversion.py:307`
  (`signal_present = bool(np.any(d > 3.0 * sig))`).
  - **Impact (inversion):** WLS weights too confident; at very low ppm a pure-noise
    record can cross `3·sig` and flip the fit to `converged=True`, **fabricating a
    source** — exactly the worst case for Melissa.
  - **Trigger:** a quiet multi-sensor record where the per-sensor σ is computed
    over a long window.
  - **Remove:** test that a pure-noise multi-sensor record returns
    `converged=False`; then apply an effective-N / autocorrelation correction to
    `sigma_ppm` (inflate σ toward the true degrees of freedom).

---

## Section B — Correct now but unguarded (⚠ test gaps)

### F3 — `predict_excess_grid` is unused and its cited test does not exist
- [ ] **⚠ / efficiency MED-HIGH.** `predict_excess_grid` (`physics/plume.py:689`) was
  built explicitly for the inversion grid search, but the inversion loops
  `predict_ppm` instead (`inversion.py:102-117`, `_per_unit_excess`). Its docstring
  cites `test_predict_excess_grid_matches_predict_ppm` (`plume.py:720`), which
  **does not exist** (grep-confirmed).
  - **Impact (inversion):** wasted compute now (thousands of `predict_ppm` calls per
    fit); **unverified** the moment anyone wires the kernel in for speed.
  - **Remove:** write the equivalence test first — `predict_excess_grid(...)` must
    equal `predict_ppm(..., clamp_to_table=True) − CH4_BACKGROUND` receptor-by-
    receptor — then adopt the kernel in the inversion behind that test.

### F7 — local-minimum / ill-posedness is unguarded
- [ ] **⚠ MED.** `tests/test_inversion.py:129` (`test_wide_search_box_is_clamp_safe`)
  asserts only **finiteness** with a huge box — not that the **true** source is
  recovered — and nothing tests answer stability across `coarse`/`grid`/`n_seeds`,
  nor a degenerate (collinear) sensor layout.
  - **Impact (inversion):** a confidently-wrong basin, or an under-determined layout
    where Q/position trade off, passes silently as a clean answer.
  - **Remove:** a wide-box test that still recovers the true source within tolerance;
    plus a collinear-sensor test asserting a large CRB (`crb_std`) and/or honest
    `converged` behaviour rather than false precision.

---

## Section C — Validity & robustness caveats (ℹ)

### F8 — `pasquill_class` insolation cutoffs unverified
- [ ] **ℹ.** The day W/m² cutoffs (`physics/weather.py:76-78`, `700/350`) are an EPA
  SRDT variant, not the original solar-angle scheme — already self-flagged in the
  `weather.py` validation caveat (`weather.py:32-43`). Not a code fix: **verify
  against EPA-454/R-99-005** and cite it. Mitigated because the inversion can try all
  classes and feasibility reports the swing across neighbours.

### F9 — `summarize_archive` crashes on an archive with no direction column
- [ ] **ℹ LOW.** `weather.py:228-230` does `archive.direction[~np.isnan(archive.direction)]`;
  if the column is absent, `parse_archive` returns `None`, so `~np.isnan(None)`
  raises — not caught by the offline-safe path. (In practice the fetch always
  requests direction.) **Remove:** guard `direction is None`; test with a
  direction-less fixture.

### F10 — `temp_humidity_correct` rank-deficient when temp or humidity is constant
- [ ] **ℹ LOW.** `processing.py:222` (`np.linalg.lstsq`) is rank-deficient if
  temperature or humidity is constant/collinear with the intercept; the min-norm
  solution can give meaningless coefficients that corrupt `excess` → `mean_excess` →
  `Q`. `process_fieldtest` only guards "either column varies", not collinearity.
  **Remove:** test a constant-humidity record; fall back to an intercept-only / single-
  regressor fit when a column doesn't vary.

---

## Section D — Closed

### F1 — rotation now pinned at non-cardinal winds
- [x] **⚠ HIGH → CLOSED (2026-06-23).** Rotation was only ever exercised at the 270°
  identity (`xw=dx, yw=dy`), hiding any sign flip / `sin`↔`cos` swap / `x`↔`y`
  transpose. Now locked three independent ways, all green:
  - hand-computed known-answer rotations at 90° / 180° / **30°** + a non-origin source
    offset (`test_rotate_known_answers_off_cardinal`, `tests/test_plume.py`). Used 30°
    rather than 45° on purpose: at 45° `sin = cos`, so 45° would NOT catch a sin↔cos swap.
  - a vectorized-vs-scalar equivalence at **200°** tying `predict_ppm`'s inline rotation
    to `rotate_to_wind_frame` (`test_vectorized_matches_scalar_off_cardinal`, same file).
  - a forward→inverse round-trip at 200° that recovers the source
    (`test_round_trip_recovers_source_off_cardinal`, `tests/test_inversion.py`) — an
    integration guard (shares `predict_ppm` both ways; the two above are the anchors).
  Verified correct — **no production code change needed.**

### F2 — unit conversion has an independent known-value test
- [x] **⚠ MED-HIGH → CLOSED (2026-06-23).** `gm3_to_ppm_methane` was only checked
  against itself. Now anchored to an **independent ideal-gas molar-volume** calc
  (`V_m = RT/P`): 1 g/m³ → ≈1499.6 ppm at STP, plus an order-of-magnitude guard and
  proportionality checks (∝ C, ∝ T) — `test_gm3_to_ppm_known_value`, `tests/test_plume.py`.
  Catches a ×1000 / kg↔g / Pa↔hPa slip. Verified correct — **no code change needed.**

### F4 — mass-conservation test on the normalization
- [x] **⚠ MED → CLOSED (2026-06-23).** The `2π` + ×2-reflection normalization was pinned
  only at one ground point. `test_plume_conserves_mass` (`tests/test_plume.py`) now
  numerically integrates `concentration_gm3` over a crosswind×vertical plane and asserts
  `u · ∫∫ C dy dz ≈ Q` (rel 5e-3) across x ∈ {50,100,150} m × classes {1,4,6} — holding
  across x is the signature of correct, x-independent normalization. Independent of the
  factors themselves. Verified correct — **no code change needed.**

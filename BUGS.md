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
- [x] **✗ → CLOSED (2026-06-25).** Dropped the `+1`; window is now exactly the detected
  event. Locked by `test_aggregate_window_matches_detected_event_exactly` (window ==
  `(det.start_idx, det.end_idx)`). Original finding below. `physics/fieldtest.py:333-337` set `lo, hi = det.start_idx,
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

### F6 — fabricated source on a no-source record (the Melissa low-ppm trap)
- [~] **✗ PARTIALLY CLOSED (2026-06-25) — diagnosis corrected by measurement.**
  Characterized first (per request); the documented cause (autocorrelation →
  underestimated σ) was NOT the dominant one.
  - **Dominant cause (FIXED):** the gate `signal_present = np.any(d > 3·sig)` compared
    `mean_excess` against the σ-OF-THE-MEAN. The baseline (rolling low-percentile floor)
    biases `mean_excess` POSITIVE (~+0.19 ppm on 0.3-ppm white noise) while the σ-of-the-
    mean shrinks ~√N — so pure WHITE noise fabricated a source ~50% of the time, even
    where σ is honest. Fix: gate on the per-sample detection floor
    `DETECT_K·√(random²+bias²)` (= the feasibility threshold), not the σ-of-the-mean
    (`inversion.py` `_datum` + gate). White-noise fabrication **54% → 0%**; locked by
    `test_pure_noise_does_not_fabricate_a_source`.
  - **Residual (OPEN):** strongly autocorrelated BACKGROUND (ρ≈0.8–0.95) still fabricates
    ~40% (detect_pattern fires on sustained wander), and `sigma_ppm` stays ~2×
    overconfident for autocorrelated noise (the `bias_ppm` proxy caps the worst case but
    not fully), so the CRB is optimistic when a real source IS present. Both need the REAL
    Figaro noise autocorrelation (CLAUDE.md flags the 0.3-ppm floor as ASSUMED) before an
    effective-N correction can be tuned to data rather than a guessed AR model.

---

## Section B — Correct now but unguarded (⚠ test gaps)

### F3 — `predict_excess_grid` is unused and its cited test does not exist
- [x] **⚠ → CLOSED (2026-06-25).** Wrote the missing equivalence test
  `test_predict_excess_grid_matches_predict_ppm` (kernel == `predict_ppm(…,
  clamp_to_table=True) − CH4_BACKGROUND` per receptor, exercised off-cardinal at 200°
  and 270°), then wired the kernel into the inversion via `_eval_cells` (one batched
  plume pass per lattice, replacing the per-cell `predict_ppm`). Full suite **18s → 6s**.
  Original finding below. `predict_excess_grid` (`physics/plume.py:689`) was
  built explicitly for the inversion grid search, but the inversion loops
  `predict_ppm` instead (`inversion.py:102-117`, `_per_unit_excess`). Its docstring
  cites `test_predict_excess_grid_matches_predict_ppm` (`plume.py:720`), which
  **does not exist** (grep-confirmed).
  - **Impact (inversion):** wasted compute now (thousands of `predict_ppm` calls per
    fit); **unverified** the moment anyone wires the kernel in for speed.
  - **Remove:** write the equivalence test first — `predict_excess_grid(...)` must
    equal `predict_ppm(..., clamp_to_table=True) − CH4_BACKGROUND` receptor-by-
    receptor — then adopt the kernel in the inversion behind that test.

### F7 — single-wind ill-posedness → CURED by multi-snapshot fusion
- [x] **✅ RESOLVED (2026-06-29) via `invert_multi`. Single-wind localisation on a hard
  fenceline is INHERENTLY under-determined; fusing a few wind directions fixes it.**
  An earlier pass (2026-06-26) called this "partially closed" claiming only *extreme user
  bounds* trigger a wrong basin and the **default** `_default_bounds` is safe. A realistic
  hard dataset disproved that, and probing it further reframed the bug entirely:
  - **It is NOT an optimiser local-minimum, and NOT fixable by a post-hoc uncertainty.**
    On the fenceline geometry where one near sensor sees a strong plume and the rest read
    ~background, only ~2 sensors carry real signal for 3 unknowns (x, y, Q) → genuinely
    under-determined. Worse, leftover baseline-drift bias on the quiet sensors makes a
    *wrong* location the legitimate maximum-likelihood fit (cost at the wrong basin < cost
    at truth), so the single-wind fit lands ~28 m off while `converged=True`. Measured: no
    local CRB, Birge-ratio (√reduced-χ²) inflation, profile-likelihood cost-surface spread,
    or signal-count heuristic reliably separates the confidently-wrong cases from good ones
    — the corrupted data really does prefer the wrong spot with low residuals.
  - **The cure is more INFORMATION, not cleverer post-processing (user chose "actually
    reduce the error").** Watching the same source under several wind directions
    triangulates it: each wind sweeps the plume across a different sensor subset, and the
    snapshots only agree at the true source. Q is shared (a steady leak emits the same g/s
    under every wind) so it stays closed-form — now summed across snapshots — leaving the
    same scipy-free 2-D grid search. Fisher information adds across snapshots
    (`crb_source_bound_multi`), so the combined CRB can only tighten.
  - **Measured (full pipeline, across seeds):** single wind ~28 m → **2 winds ~0.3 m, 3
    winds ~0.5 m** (max 0.6 m), ~100 ms/fit. Note: each added snapshot SHARPENS the joint
    basin, so `invert_multi` uses a denser default grid (`coarse=60, n_seeds=12`) than
    `invert` (28/6); raise `coarse` further if you fuse many snapshots.
  - **Shipped:** `physics/inversion.py` `invert_multi(sensor_positions, [Snapshot…])` +
    `invert_field_tests_multi(snapshots, …)`; `Snapshot(points, u, wind_dir_deg,
    stability_class=None)`; `physics/accuracy.py` `crb_source_bound_multi`. `invert` is now
    the K=1 special case of `invert_multi` (pinned by
    `test_invert_multi_equals_invert_for_single_snapshot`).
  - **Tests:** `test_f7_single_snapshot_is_under_determined` (documents the ~28 m single-wind
    limit), `test_f7_multi_snapshot_fusion_recovers_source` (3 winds → <5 m, the cure),
    `test_f7_multi_snapshot_crb_tightens`, `test_multi_snapshot_tolerates_a_blind_snapshot`
    (a wind that misses every sensor must not divide-by-zero). Builder
    `_realistic_field_results(…, wind_dir, seed0)` makes the hard dataset through the full
    deployment pipeline (what Custer/Melissa data will look like).
  - **GOOD-case characterisation kept:** `test_collinear_sensors_crb_flags_ill_posedness` —
    sensors on y=0 return `converged=True` with correct x/Q but `crb_std["y"] >
    crb_std["x"]×3` (the CRB is honest there: ∂ppm/∂y_src ≈ 0 → tiny Fisher info for y).
  - **Residual caveat (deprioritised, not lost):** a SINGLE-snapshot fit on an
    under-determined layout still reports `converged=True` with an optimistic CRB (false
    confidence). The honest-flag path (widen CRB / `converged=False` when a layout is
    under-determined) was deprioritised in favour of the fusion fix; deployment guidance is
    "fuse ≥2 winds, or place ≥3 sensors that all sit in the plume."

---

## Section C — Validity & robustness caveats (ℹ)

### F8 — `pasquill_class` insolation cutoffs unverified
- [x] **ℹ VERIFIED (2026-06-25) — cutoffs DIVERGE; caveat updated, numeric fix recommended.**
  Checked against EPA-454/R-99-005 Table 6-7 "Key to the SRDT Method" (p.6-15): EPA daytime
  bands are **≥925 / 925–675 / 675–175 / <175 W/m²** (4 levels), so the code's `700/350`
  (3 levels, no `<175→D` band) is biased too UNSTABLE (it calls 700 "strong"; EPA "strong"
  is ≥925). Daytime WIND bands match EPA; EPA night uses ΔT, code uses cloud (Turner) because
  Open-Meteo lacks ΔT. `weather.py` caveat now records this with the citation. **Recommended
  follow-up (user-owned, shifts real-site classes):** set cutoffs to 925/675/175 and add a
  `<175→D` daytime band. Original note below. The day W/m² cutoffs (`physics/weather.py:76-78`, `700/350`) are an EPA
  SRDT variant, not the original solar-angle scheme — already self-flagged in the
  `weather.py` validation caveat (`weather.py:32-43`). Not a code fix: **verify
  against EPA-454/R-99-005** and cite it. Mitigated because the inversion can try all
  classes and feasibility reports the swing across neighbours.

### F9 — `summarize_archive` crashes on an archive with no direction column
- [x] **ℹ LOW → CLOSED (2026-06-25).** Guarded `direction is None` (and all-NaN) → default
  0°; `test_summarize_archive_without_direction_does_not_crash`. Original finding below.
  `weather.py:228-230` did `archive.direction[~np.isnan(archive.direction)]`;
  if the column is absent, `parse_archive` returns `None`, so `~np.isnan(None)`
  raises — not caught by the offline-safe path. (In practice the fetch always
  requests direction.) **Remove:** guard `direction is None`; test with a
  direction-less fixture.

### F10 — `temp_humidity_correct` rank-deficient when temp or humidity is constant
- [x] **ℹ LOW → CLOSED (2026-06-25).** Now drops a (near-)constant regressor before the fit
  and reports its coefficient as 0; `test_3b_constant_humidity_is_rank_safe` (old min-norm
  invented a phantom `b_humid=0.058`). Original finding below.
  `processing.py:222` (`np.linalg.lstsq`) is rank-deficient if
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

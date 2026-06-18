"""
accuracy.py — How accurate is this, and how accurate could it ever be?

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS (plain English)
───────────────────────────────────────────────────────────────────────────────
Detecting a 2-40 ppm plume on top of a 1.9 ppm background — with a sensor the
maker rated for 500-12,500 ppm — only means something if we can also say HOW SURE
we are. This module is the project's "accuracy brain". It answers four questions,
in order (the Accuracy Protocol):

  1. CHARACTERIZE — measure the sensor's real noise from the data itself, instead
     of trusting the assumed 0.30 ppm placeholder. Splits it into the part that
     averages away (random) and the part that does NOT (bias/drift).
  2. VALIDATE   — on synthetic data where we KNOW the true plume, score how well
     the cleaning pipeline recovered it (RMSE, % recovery, R², detection hits).
  3. BOUND      — the best we can do: the smallest leak detectable now (LOD/LOQ +
     a minimum detectable Q with a wind/stability error bar), AND the theoretical
     best-possible precision of a source fix (the Cramér-Rao bound).
  4. GRADE      — fold it all into one 0-100 score, a letter, and the single most
     useful next action.

Everything reuses the already-tested core (processing.py, sensor_sim.py,
feasibility.py, plume.py) — no physics or signal processing is reimplemented here.

⚠ HONESTY NOTE. The RANDOM noise is genuinely measurable from any record. The
  BIAS term measured in-record is only a PROXY (residual slow drift); a true,
  constant calibration offset can only be pinned with a known reference — a
  zero-air bench run. See ACCURACY.md.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from physics.processing import Detection, moving_average
from physics.sensor_sim import (
    SENSOR_NOISE_PPM,
    DETECT_K,
    effective_noise_floor,
)
from physics import feasibility
from physics.plume import predict_ppm


# ─────────────────────────────────────────────────────────────────────────────
# Small robust-statistics helper
# ─────────────────────────────────────────────────────────────────────────────
def _robust_std(x) -> float:
    """
    1σ estimate that ignores outliers: 1.4826 × the median absolute deviation.

    For clean Gaussian data this equals the ordinary standard deviation, but a few
    spikes (a passing puff, an electrical glitch) don't inflate it. Returns 0.0 for
    an empty or constant input.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARACTERIZE — measure the sensor's real noise from the quiet data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NoiseEstimate:
    """The sensor's measured precision, split by how averaging affects it."""
    random_ppm: float    # averages down as 1/√N  (electrical / turbulent jitter)
    bias_ppm: float      # survives averaging      (residual slow drift — a PROXY)
    n_quiet: int         # samples used (the no-event part of the record)
    method: str          # short description of how it was measured


def estimate_noise_floor(
    excess_unsmoothed,
    detection: Detection | None = None,
    guard: int = 5,
    slow_window: int | None = None,
) -> NoiseEstimate:
    """
    Measure the sensor's real noise from the QUIET part of a record (Protocol 1).

    Feed this the UN-smoothed, baseline-removed excess — i.e. ``raw − baseline``
    from process_fieldtest (that difference cancels the weather + baseline and
    leaves noise plus any plume). The smoothed ``excess`` would hide the very
    jitter we want to size, so do NOT pass that.

    "Quiet" = the samples OUTSIDE any detected event (the whole record if nothing
    was detected), padded by ``guard`` samples so the event's edges don't leak in.

      • random_ppm — from successive DIFFERENCES of the quiet signal. Differencing
        cancels slow drift, so what's left is the sample-to-sample jitter; a robust
        (MAD) spread ÷ √2 turns that back into a 1σ noise. This is the part that
        averaging beats down as 1/√N.
      • bias_ppm — a robust spread of the heavily-smoothed quiet signal: the slow
        residual wander that a single averaging window does NOT remove. PROXY only
        (see the module note) — reported so the floor isn't optimistically low.

    Returns a NoiseEstimate.
    """
    x = np.asarray(excess_unsmoothed, dtype=float)
    n = x.size

    quiet = np.ones(n, dtype=bool)
    if detection is not None and detection.detected and detection.start_idx >= 0:
        lo = max(0, detection.start_idx - guard)
        hi = min(n, detection.end_idx + guard)
        quiet[lo:hi] = False

    q = x[quiet]
    note = "all samples" if q.size == n else "no-event samples"
    if q.size < 8:                       # too little quiet data → use everything
        q = x
        note = "whole record (too little quiet data)"

    # random: robust σ from successive differences (immune to slow drift).
    if q.size >= 2:
        diffs = np.diff(q)
        random_ppm = _robust_std(diffs) / np.sqrt(2.0)
    else:
        random_ppm = 0.0

    # bias proxy: the slow residual left after heavy smoothing.
    if slow_window is None:
        slow_window = max(11, q.size // 8)
    slow = moving_average(q, slow_window)
    bias_ppm = _robust_std(slow)

    return NoiseEstimate(
        random_ppm=float(random_ppm),
        bias_ppm=float(bias_ppm),
        n_quiet=int(q.size),
        method=f"successive-diff random + slow-residual bias ({note})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATE — score recovery against a KNOWN truth (synthetic only)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RecoveryReport:
    """How well the pipeline recovered a KNOWN truth (synthetic validation)."""
    rmse_ppm: float
    bias_ppm: float
    pct_recovery: float | None     # recovered peak ÷ true peak × 100
    r2: float | None
    correlation: float | None
    true_detected: bool            # was there really an event?
    detected: bool                 # did we flag one?
    outcome: str                   # "true positive" | "false positive" | ...
    start_err_s: float | None
    end_err_s: float | None


def recovery_metrics(
    true_ppm,
    recovered_excess,
    detection: Detection | None = None,
    time=None,
) -> RecoveryReport:
    """
    Score recovered signal against the known truth (Protocol 2 — synthetic only).

    Only meaningful when the truth is known (synthetic data from sensor_sim). For
    real uploads there is no truth, so accuracy_report skips this and leaves the
    recovery fields null.
    """
    true = np.asarray(true_ppm, dtype=float)
    rec = np.asarray(recovered_excess, dtype=float)
    n = min(true.size, rec.size)
    true, rec = true[:n], rec[:n]

    resid = rec - true
    rmse = float(np.sqrt(np.mean(resid ** 2))) if n else 0.0
    bias = float(np.mean(resid)) if n else 0.0

    true_peak = float(np.max(true)) if n else 0.0
    pct = float(np.max(rec) / true_peak * 100.0) if true_peak > 0 else None

    ss_tot = float(np.sum((true - np.mean(true)) ** 2)) if n else 0.0
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else None

    if n > 1 and np.std(true) > 0 and np.std(rec) > 0:
        correlation = float(np.corrcoef(true, rec)[0, 1])
    else:
        correlation = None

    true_event = np.where(true > 0)[0]
    true_detected = true_event.size > 0
    detected = bool(detection.detected) if detection is not None else False

    if detected and true_detected:
        outcome = "true positive"
    elif detected and not true_detected:
        outcome = "false positive"
    elif not detected and true_detected:
        outcome = "false negative"
    else:
        outcome = "true negative"

    start_err = end_err = None
    if (time is not None and detection is not None and detection.detected
            and true_detected):
        t = np.asarray(time, dtype=float)
        start_err = abs(float(t[detection.start_idx]) - float(t[true_event[0]]))
        end_i = min(detection.end_idx, n) - 1
        end_err = abs(float(t[end_i]) - float(t[true_event[-1]]))

    return RecoveryReport(
        rmse_ppm=rmse, bias_ppm=bias, pct_recovery=pct, r2=r2,
        correlation=correlation, true_detected=true_detected, detected=detected,
        outcome=outcome, start_err_s=start_err, end_err_s=end_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4a. BOUND (empirical) — smallest leak detectable now, and how averaging helps
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectionLimit:
    """The smallest leak we can claim — now, and how averaging moves it."""
    floor_ppm: float          # effective 1σ floor at n_avg (random ⊕ bias)
    lod_ppm: float            # limit of detection      = k · floor
    loq_ppm: float            # limit of quantification = 10 · floor
    min_detectable_Q: float   # g/s producing lod_ppm at the nominal (u, stability)
    q_min: float              # error bar across plausible wind / stability …
    q_max: float              # … (upper bound on the implied leak)
    swing_factor: float       # q_max / q_min — how uncertain Q is
    n_avg: int
    crossover_n: float | None # N beyond which bias dominates (averaging stalls)
    curve: list               # [{n, floor_ppm, lod_ppm, min_Q}] averaging sweep


def detection_limit(
    random_ppm: float,
    bias_ppm: float = 0.0,
    distance: float = 50.0,
    k: float = DETECT_K,
    u: float = 2.0,
    stability: int = 4,
    n_avg: int = 1,
    n_avg_curve=(1, 4, 16, 64, 256),
) -> DetectionLimit:
    """
    Best-achievable detection NOW, from a measured/assumed floor (Protocol 4a).

    Converts a ppm noise floor into the smallest LEAK we could honestly claim, by
    reusing the feasibility back-solver (the plume is linear in Q). Reports it at
    the nominal (u, stability) plus the wind/stability error bar, and traces how
    averaging lowers the limit until the non-averageable bias takes over.
    """
    floor = effective_noise_floor(random_ppm, bias_ppm, n_avg)
    lod = k * floor
    loq = 10.0 * floor

    per_unit_Q = feasibility._excess_at(distance, 1.0, u, stability, clamp=True)
    min_Q = float(lod / per_unit_Q) if per_unit_Q > 1e-12 else float("inf")

    if lod > 0:
        qr = feasibility.implied_Q_range(lod, distance)
        q_min, q_max, swing = qr.q_min, qr.q_max, qr.swing_factor
    else:
        q_min = q_max = float("nan")
        swing = float("inf")

    # Past N ≈ (random/bias)² the bias term dominates and more averaging is futile.
    crossover = float((random_ppm / bias_ppm) ** 2) if bias_ppm > 0 else None

    curve = []
    for nv in n_avg_curve:
        f = effective_noise_floor(random_ppm, bias_ppm, nv)
        l = k * f
        mq = float(l / per_unit_Q) if per_unit_Q > 1e-12 else float("inf")
        curve.append({"n": int(nv), "floor_ppm": float(f),
                      "lod_ppm": float(l), "min_Q": mq})

    return DetectionLimit(
        floor_ppm=float(floor), lod_ppm=float(lod), loq_ppm=float(loq),
        min_detectable_Q=min_Q, q_min=float(q_min), q_max=float(q_max),
        swing_factor=float(swing), n_avg=int(n_avg), crossover_n=crossover,
        curve=curve,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3 + 5. QUANTIFY + GRADE — one self-assessing report for a processed field test
# ─────────────────────────────────────────────────────────────────────────────
def _grade_letter(score: float) -> str:
    """Map a 0-100 score to a school letter (90 A, 80 B, 70 C, 60 D, else F)."""
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


def accuracy_report(result: dict, meta: dict | None = None,
                    true_ppm=None) -> dict:
    """
    The whole Accuracy Protocol for one processed field test → one JSON-safe dict.

    Combines the measured floor (1), optional recovery vs truth (2), the detection
    limit + error bar (4a), and a 0-100 score, letter grade and recommendation
    (3 + 5). Shaped to drop straight into summarize_fieldtest's facts, so the
    chatbot can ground answers on it unchanged.

    Parameters
    ----------
    result   : dict from fieldtest.process_fieldtest (raw/baseline/excess/detection…).
    meta     : optional field-test metadata (sensor_distance_m, wind_speed,
               stability_class) — supplies the geometry for the detection limit.
    true_ppm : optional ground truth (synthetic only) → adds recovery metrics.
    """
    meta = dict(meta or {})
    raw = np.asarray(result["raw"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    excess = np.asarray(result["excess"], dtype=float)
    det: Detection = result["detection"]
    assumed_floor = float(result.get("noise_ppm", SENSOR_NOISE_PPM))

    # Geometry — fall back to the canonical fenceline case when metadata is absent.
    distance = float(meta.get("sensor_distance_m") or 50.0)
    u = float(meta.get("wind_speed") or 2.0)
    stability = int(meta.get("stability_class") or 4)

    # (1) Measure the floor from the un-smoothed, baseline-removed excess.
    ne = estimate_noise_floor(raw - baseline, det)
    meas_floor = effective_noise_floor(ne.random_ppm, ne.bias_ppm, 1)
    total_noise = ne.random_ppm + ne.bias_ppm
    bias_fraction = (ne.bias_ppm / total_noise) if total_noise > 0 else 0.0

    excess_peak = float(np.max(excess)) if excess.size else 0.0
    snr = float(excess_peak / meas_floor) if meas_floor > 0 else 0.0
    confidence = float(det.confidence)

    # (4a) Detection limit from the MEASURED floor at this geometry.
    dl = detection_limit(ne.random_ppm, ne.bias_ppm, distance=distance,
                         u=u, stability=stability)

    # (2) Recovery vs known truth (synthetic only).
    recovery = None
    if true_ppm is not None:
        recovery = asdict(recovery_metrics(true_ppm, excess, det,
                                           time=result.get("time")))

    # (5) Score, 0-100, from three levers.
    conf = confidence if det.detected else snr
    conf_pts = float(np.clip(50.0 * conf / (2.0 * DETECT_K), 0.0, 50.0))
    stab_pts = float(30.0 * (1.0 - bias_fraction))
    if recovery is not None and recovery["pct_recovery"] is not None:
        true_peak = float(np.max(np.asarray(true_ppm, dtype=float)))
        acc = 1.0 - (recovery["rmse_ppm"] / true_peak) if true_peak > 0 else 0.0
        rec_pts = float(np.clip(20.0 * acc, 0.0, 20.0))
        score = conf_pts + stab_pts + rec_pts
    else:
        rec_pts = None
        score = (conf_pts + stab_pts) * 100.0 / 80.0      # rescale to 0-100
    score = float(np.clip(score, 0.0, 100.0))

    # Recommendation — attack the weakest lever first.
    if bias_fraction > 0.5:
        rec_msg = (
            "Bias-limited: the non-averageable drift dominates — recalibrate or "
            "run zero-air; averaging past N≈"
            f"{dl.crossover_n:.0f} buys almost nothing."
            if dl.crossover_n is not None else
            "Bias-limited: recalibrate the sensor; averaging will not help."
        )
    elif conf < DETECT_K:
        rec_msg = ("Signal sits near the noise — average more samples or move the "
                   "sensor closer; only larger leaks are reliable here.")
    elif distance > 100:
        rec_msg = ("Good margin, but the sensor is far — moving it closer would "
                   "lower the smallest detectable leak.")
    else:
        rec_msg = "Signal-rich: a comfortable detection margin at this geometry."

    return {
        "assumed_floor_ppm": assumed_floor,
        "measured_random_ppm": ne.random_ppm,
        "measured_bias_ppm": ne.bias_ppm,
        "measured_floor_ppm": float(meas_floor),
        "bias_fraction": float(bias_fraction),
        "n_quiet": ne.n_quiet,
        "noise_method": ne.method,
        "snr": snr,
        "confidence_sigmas": confidence,
        "detected": bool(det.detected),
        "detection_limit": asdict(dl),
        "recovery": recovery,
        "score": round(score, 1),
        "grade": _grade_letter(score),
        "score_breakdown": {
            "confidence": round(conf_pts, 1),
            "stability": round(stab_pts, 1),
            "recovery": round(rec_pts, 1) if rec_pts is not None else None,
        },
        "recommendation": rec_msg,
        "geometry": {"distance_m": distance, "u": u, "stability": stability},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4b. BOUND (theoretical) — Cramér-Rao lower bound on a source fix
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CRBound:
    """Theoretical best-possible 1σ on a source estimate (Cramér-Rao)."""
    params: tuple
    std: dict            # param → best-possible 1σ (m for x/y, g/s for Q)
    cov: list            # full covariance matrix (list of lists)
    n_receptors: int
    sigma_ppm: float
    condition_number: float
    note: str


def crb_source_bound(
    src_pos,
    Q: float,
    u: float,
    wind_dir_deg: float,
    H: float,
    stability_class: int,
    receptors,
    sigma_ppm: float,
    params=("x", "y", "Q"),
    rel_step: float = 1e-3,
) -> CRBound:
    """
    Cramér-Rao lower bound on a source fix (Protocol 4b — the theoretical best).

    No unbiased estimator can beat this: given Gaussian measurement noise of 1σ
    ``sigma_ppm`` independent at each receptor, the best-possible variance of ANY
    unbiased estimate of the source parameters is the inverse Fisher information.
    We build it from a numerical Jacobian of predict_ppm (clamp_to_table=True so a
    probe never raises), so when the Week-3 inversion exists it can check its
    achieved scatter against this floor — if the optimizer's spread is near the
    CRB, it is as good as the physics allows; if far above, there is room to improve.

    Parameters
    ----------
    src_pos         : (x, y) true source location (m).
    Q, u, …         : forward-model parameters (see predict_ppm).
    receptors       : (N, 2) sensor positions (m).
    sigma_ppm       : measurement noise 1σ at each receptor (ppm).
    params          : which parameters to bound — any subset of ("x", "y", "Q").
    rel_step        : relative finite-difference step.

    Returns
    -------
    CRBound with the per-parameter best-possible 1σ and the full covariance.

    Caveats: assumes Gaussian iid noise, a SINGLE source, and that the forward
    model is locally linear over the step — a best CASE, optimistic if any of
    those break.
    """
    receptors = np.asarray(receptors, dtype=float)
    base = {"x": float(src_pos[0]), "y": float(src_pos[1]), "Q": float(Q)}

    def predict(p):
        return predict_ppm(
            src_pos=(p["x"], p["y"]), Q=p["Q"], u=u, wind_dir_deg=wind_dir_deg,
            H=H, stability_class=stability_class, receptors=receptors,
            clamp_to_table=True,
        )

    # Numerical Jacobian: column j = ∂(ppm at receptors)/∂param_j (central diff).
    cols = []
    for name in params:
        val = base[name]
        if name in ("x", "y"):
            h = max(rel_step * abs(val), 0.1)        # ≥ 0.1 m even near the origin
        else:
            h = max(rel_step * abs(val), 1e-3)
        hi, lo = dict(base), dict(base)
        hi[name] += h
        lo[name] -= h
        cols.append((predict(hi) - predict(lo)) / (2.0 * h))
    J = np.column_stack(cols)                          # (n_receptors, n_params)

    F = J.T @ J / (sigma_ppm ** 2)                     # Fisher information
    cond = float(np.linalg.cond(F))
    try:
        cov = np.linalg.inv(F)
        note = "ok"
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F)
        note = "Fisher matrix singular (too few / degenerate receptors) — pinv used"

    std = {name: float(np.sqrt(abs(cov[i, i]))) for i, name in enumerate(params)}

    return CRBound(
        params=tuple(params), std=std, cov=np.asarray(cov).tolist(),
        n_receptors=int(receptors.shape[0]), sigma_ppm=float(sigma_ppm),
        condition_number=cond, note=note,
    )

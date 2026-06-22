"""
feasibility.py — "Will this even work?" sanity check for the forward model.

═══════════════════════════════════════════════════════════════════════════════
WHAT A FEASIBILITY SWEEP IS (plain English)
───────────────────────────────────────────────────────────────────────────────
Before anyone buys hardware or stands at a fenceline collecting data, we want to
know: given a realistically weak leak, does the plume model predict a signal big
enough for the sensor to actually notice?

A "sweep" just means: run the forward model many times, changing one knob at a
time (leak size Q, distance, weather stability), and look at the pattern of
answers. For each case we ask a single, honest question:

    Is the predicted methane ABOVE background bigger than the sensor's noise?

We answer it by comparing the predicted excess (ppm above the 1.9 ppm
background) against a detection threshold built from the SAME noise floor the
processing tests use (SENSOR_NOISE_PPM, imported from sensor_sim.py). Using one
shared noise floor means "detectable" means the same thing everywhere in the
project.

    excess ≥ k · noise        →  DETECTABLE   (clearly above the noise)
    noise ≤ excess < k · noise →  MARGINAL     (real but needs averaging/stats)
    excess < noise            →  UNDETECTABLE (lost in the noise)

The headline output is a plain-English VERDICT: the smallest leak the sensor
could catch at the fenceline under typical wind — i.e. whether the project is
physically feasible at all.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.plume import (
    CH4_BACKGROUND, RELEASE_HEIGHT_M, SENSOR_HEIGHT_M, predict_ppm,
)
from physics.sensor_sim import (
    SENSOR_NOISE_PPM, DETECT_K, BIAS_FLOOR_PPM, effective_noise_floor,
)

# Detection strictness lives in sensor_sim.py (single source of truth). Kept
# under the local name DEFAULT_K for the sweep's signature/defaults.
DEFAULT_K = DETECT_K

# Plausible field ranges the sensitivity error bar sweeps over. Defined once here
# (not inline) so implied_Q_range's defaults and verdict's prose can never drift
# apart — change the spread in one place and both the math and the sentence follow.
SENSITIVITY_U_RANGE = (0.5, 1.0, 2.0, 3.0, 5.0)   # m/s (0.5 floor = plume.U_MIN)
SENSITIVITY_STABILITY = (2, 3, 4, 5, 6)            # Pasquill classes B–F
_PASQUILL_LETTERS = "ABCDEF"                        # class 1→A … 6→F


def _excess_at(
    distance: float,
    Q: float,
    u: float,
    stability: int,
    wind_dir_deg: float = 270.0,
    H: float = RELEASE_HEIGHT_M,
    z: float = SENSOR_HEIGHT_M,
    clamp: bool = False,
) -> float:
    """Plume excess (ppm above background) at one centreline-downwind receptor.

    The shared forward-model call behind both ``sweep`` and ``implied_Q_range``:
    one place that builds the (distance, 0) receptor and runs ``predict_ppm``, so
    the receptor convention and the optimizer-safe ``clamp`` flag live in one spot.
    """
    receptor = np.array([[float(distance), 0.0]])  # centreline, downwind
    total = float(predict_ppm(
        src_pos=(0.0, 0.0), Q=Q, u=float(u), wind_dir_deg=wind_dir_deg,
        H=H, stability_class=int(stability), receptors=receptor, z=z,
        clamp_to_table=clamp,
    )[0])
    return total - CH4_BACKGROUND


@dataclass
class SweepCell:
    """One result of the sweep: a single (Q, distance, stability) combination."""
    Q: float                 # emission rate (g/s)
    distance: float          # downwind distance to the sensor (m)
    stability: int           # Pasquill class 1–6 (1=A … 6=F)
    ppm_total: float         # predicted reading INCLUDING background (ppm)
    excess: float            # predicted plume contribution above background (ppm)
    verdict: str             # "detectable" | "marginal" | "undetectable"


def classify(excess: float, noise_floor: float, k: float = DEFAULT_K) -> str:
    """Label one excess-ppm value against the noise floor (see module header)."""
    if excess >= k * noise_floor:
        return "detectable"
    if excess >= noise_floor:
        return "marginal"
    return "undetectable"


def sweep(
    Q_list,
    distance_list,
    stability_list,
    u: float = 2.0,
    wind_dir_deg: float = 270.0,
    H: float = RELEASE_HEIGHT_M,
    z: float = SENSOR_HEIGHT_M,
    noise_floor: float = SENSOR_NOISE_PPM,
    k: float = DEFAULT_K,
) -> list[SweepCell]:
    """
    Run the forward model across every (Q, distance, stability) combination.

    The sensor is placed straight downwind on the plume centreline (worst case
    for the sensor = best case for detection, i.e. the most generous test). Wind
    is 270° (from the west), so "downwind" is due east and the centreline
    receptor sits at (distance, 0).

    Parameters
    ----------
    Q_list         : emission rates to test (g/s).
    distance_list  : downwind sensor distances to test (m).
    stability_list : Pasquill classes to test (ints 1–6).
    u              : wind speed (m/s).
    wind_dir_deg   : wind FROM-direction (° clockwise from N); 270 = from west.
    H              : release height (m).
    z              : sensor height (m).
    noise_floor    : sensor noise (1σ, ppm) — drives the detectable/marginal cut.
    k              : sigmas above noise required for "detectable".

    Returns
    -------
    list of SweepCell, one per combination.
    """
    cells: list[SweepCell] = []
    for stability in stability_list:
        for Q in Q_list:
            for dist in distance_list:
                excess = _excess_at(dist, Q, u, stability, wind_dir_deg, H, z)
                cells.append(SweepCell(
                    Q=Q, distance=float(dist), stability=int(stability),
                    ppm_total=excess + CH4_BACKGROUND, excess=excess,
                    verdict=classify(excess, noise_floor, k),
                ))
    return cells


def min_detectable_Q(
    cells: list[SweepCell],
    distance: float,
    stability: int,
    threshold: float | None = None,
) -> float | None:
    """
    Smallest tested Q that is detectable at a given distance and stability.

    By default (``threshold=None``) a cell counts as detectable via the verdict
    label baked in at sweep time (against the sweep's own lab noise floor). Pass an
    explicit ``threshold`` (ppm of excess) to re-judge detection against a DIFFERENT
    floor — e.g. a realistic field floor — without re-running the sweep, since
    ``excess`` is pure physics and independent of whatever floor we compare it to.
    Returns None if nothing qualifies there.
    """
    qs = [
        c.Q for c in cells
        if c.distance == distance and c.stability == stability
        and (c.excess >= threshold if threshold is not None
             else c.verdict == "detectable")
    ]
    return min(qs) if qs else None


@dataclass
class QRange:
    """
    How much the back-solved emission rate Q swings under wind/stability doubt.

    The headline field is ``swing_factor`` (= q_max / q_min): the multiplier by
    which Q is uncertain when you DON'T know the wind speed or stability class.
    """
    q_min: float             # smallest Q that fits (most dispersive air assumed)
    q_median: float
    q_max: float             # largest Q that fits (most confining air) → upper bound
    swing_factor: float      # q_max / q_min — the "error bar" multiplier
    n_unsolvable: int        # (u, stability) cells where the signal can't reach the sensor


def implied_Q_range(
    observed_excess_ppm: float,
    distance: float,
    u_list=SENSITIVITY_U_RANGE,
    stability_list=SENSITIVITY_STABILITY,
    wind_dir_deg: float = 270.0,
    H: float = RELEASE_HEIGHT_M,
    z: float = SENSOR_HEIGHT_M,
) -> QRange:
    """
    Given ONE observed excess-ppm reading at a downwind ``distance``, report the
    range of source strengths Q that could have produced it — once you admit that
    wind speed and stability class are not actually known precisely.

    Why this exists
    ───────────────
    The Gaussian plume is LINEAR in Q: excess = Q · (excess produced by Q = 1 g/s).
    So for a fixed observed excess,

        Q_implied = observed_excess / excess_per_unit_Q(u, stability, distance)

    and excess_per_unit_Q swings hard with wind speed (a clean 1/u term) and
    stability class (σ_z alone varies ~13× from class A to F over this site's
    distances). Sweeping the plausible (u, stability) grid turns "my wind data is
    coarse" into a QUANTIFIED error bar: Q is only pinned to within ``swing_factor``.
    No optimizer needed — one cheap forward evaluation per (u, stability) cell.

    Use it to report an honest UPPER BOUND ("source ≤ q_max g/s") instead of a
    false-precision point estimate. The defaults span a fenceline-plausible wind
    range (0.5–5 m/s, the 0.5 floor = plume.U_MIN) and stability classes B–F.
    """
    if observed_excess_ppm <= 0:
        raise ValueError("observed_excess_ppm must be > 0 to back-solve Q")

    qs: list[float] = []
    n_unsolvable = 0
    for u in u_list:
        for stability in stability_list:
            excess_per_unit_Q = _excess_at(
                distance, 1.0, u, stability, wind_dir_deg, H, z, clamp=True)
            if excess_per_unit_Q <= 1e-12:
                # Signal can't reach the sensor under this geometry (upwind / too
                # far): a finite leak of any size reads ~background, so Q is
                # unconstrained here. Skip rather than divide by ~0.
                n_unsolvable += 1
                continue
            qs.append(observed_excess_ppm / excess_per_unit_Q)

    if not qs:
        return QRange(float("nan"), float("nan"), float("nan"),
                      float("inf"), n_unsolvable)

    qs.sort()
    # qs are all > 0 (observed_excess > 0 ÷ positive per-unit-Q), so q_min > 0.
    return QRange(qs[0], float(np.median(qs)), qs[-1], qs[-1] / qs[0], n_unsolvable)


def verdict(
    cells: list[SweepCell],
    fence_distance: float,
    stability: int = 4,
    noise_floor: float = SENSOR_NOISE_PPM,
    k: float = DEFAULT_K,
    bias_floor: float = BIAS_FLOOR_PPM,
    n_avg: int = 1,
) -> str:
    """
    Build the plain-English bottom-line answer for the project.

    Focuses on the realistic case: a sensor at the fenceline distance under a
    neutral (Class D = 4) atmosphere, the most common daytime condition.

    Three honesty upgrades over a bare point estimate:

      1. The headline "smallest detectable leak" is kept (lab floor = ``noise_floor``).
      2. It is then reframed as an UPPER BOUND with a wind/stability error bar,
         via ``implied_Q_range`` — because the recovered Q scales directly with
         the (poorly known) wind speed and stability class.
      3. If a field-realistic floor is supplied (``bias_floor`` > 0 and/or
         ``n_avg`` > 1), it reports how the detectable leak grows once you account
         for averaging (which beats only random noise, ``noise_floor``) versus the
         non-averageable ``bias_floor``.

    Defaults (``bias_floor=0``, ``n_avg=1``) preserve the original headline + the
    new upper-bound sentence. Returns a human-readable paragraph.

    Geometry note: step (2) deliberately sweeps the FULL plausible wind range
    (independent of the single ``u`` the cells were swept at — that uncertainty is
    the whole point) but at the site's DEFAULT geometry (270°, H = 1 m, z = 1 m,
    via ``implied_Q_range``'s defaults). The headline (step 1) instead reflects
    whatever geometry produced ``cells``. They agree for the default fenceline
    setup every current caller uses; if you sweep ``cells`` at a non-default
    release/sensor height or wind direction, re-derive the spread with matching
    geometry rather than reading step (2)'s sentence literally.
    """
    q_min = min_detectable_Q(cells, fence_distance, stability)
    threshold = k * noise_floor
    if q_min is None:
        return (
            f"At {fence_distance:.0f} m (fenceline) under neutral Class-{stability} "
            f"wind, NONE of the tested leaks produced a signal above the "
            f"{threshold:.2f} ppm detection threshold ({k:g}× the {noise_floor:.2f} "
            f"ppm noise floor). Detection looks INFEASIBLE for this setup — move the "
            f"sensor closer, reduce its noise, or expect only larger leaks."
        )

    parts = [(
        f"At {fence_distance:.0f} m (fenceline) under neutral Class-{stability} wind, "
        f"the smallest detectable leak is Q ≈ {q_min:g} g/s — anything weaker stays "
        f"below the {threshold:.2f} ppm detection threshold ({k:g}× the "
        f"{noise_floor:.2f} ppm noise floor) and would need heavy averaging or be "
        f"lost in the noise."
    )]

    # ── (2) That single Q assumes you KNOW the wind. You don't — quantify it. ──
    # threshold = k·noise_floor is > 0 for any sane floor (so implied_Q_range's
    # observed_excess > 0 precondition holds); guard it explicitly all the same.
    if threshold > 0:
        qr = implied_Q_range(threshold, fence_distance)
        u_lo, u_hi = min(SENSITIVITY_U_RANGE), max(SENSITIVITY_U_RANGE)
        cls_lo = _PASQUILL_LETTERS[min(SENSITIVITY_STABILITY) - 1]
        cls_hi = _PASQUILL_LETTERS[max(SENSITIVITY_STABILITY) - 1]
        if np.isfinite(qr.swing_factor):
            parts.append(
                f"But that number assumes the wind is known. Across plausible wind "
                f"({u_lo:g}–{u_hi:g} m/s) and stability (classes {cls_lo}–{cls_hi}), the "
                f"SAME fenceline signal implies Q anywhere from {qr.q_min:.2g} to "
                f"{qr.q_max:.2g} g/s — a {qr.swing_factor:.0f}× spread — so treat it as an "
                f"upper bound (source ≤ {qr.q_max:.2g} g/s), not a precise figure."
            )

    # ── (3) Field-realistic floor: averaging beats random noise, never bias. ──
    if bias_floor > 0 or n_avg > 1:
        field_floor = effective_noise_floor(noise_floor, bias_floor, n_avg)
        field_threshold = k * field_floor
        q_field = min_detectable_Q(
            cells, fence_distance, stability, threshold=field_threshold)
        if n_avg > 1:
            avg_note = (
                f"averaging {n_avg} samples cuts the {noise_floor:.2f} ppm random noise "
                f"to {noise_floor / np.sqrt(n_avg):.2f} ppm, but the {bias_floor:.2f} ppm "
                f"bias survives"
            )
        else:
            avg_note = (
                f"the {noise_floor:.2f} ppm random noise plus a {bias_floor:.2f} ppm "
                f"bias that no amount of averaging removes"
            )
        if q_field is None:
            parts.append(
                f"With a realistic field floor of {field_floor:.2f} ppm ({avg_note}), "
                f"NONE of the tested leaks clear the {field_threshold:.2f} ppm threshold "
                f"here — detection would be infeasible in the field."
            )
        else:
            parts.append(
                f"With a realistic field floor of {field_floor:.2f} ppm ({avg_note}), the "
                f"smallest detectable leak rises to Q ≈ {q_field:g} g/s (threshold "
                f"{field_threshold:.2f} ppm)."
            )

    return " ".join(parts)

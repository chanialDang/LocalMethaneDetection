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

from plume import CH4_BACKGROUND, predict_ppm
from sensor_sim import SENSOR_NOISE_PPM, DETECT_K

# Detection strictness lives in sensor_sim.py (single source of truth). Kept
# under the local name DEFAULT_K for the sweep's signature/defaults.
DEFAULT_K = DETECT_K


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
    H: float = 1.0,
    z: float = 1.0,
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
                receptor = np.array([[float(dist), 0.0]])  # centreline, downwind
                ppm_total = float(predict_ppm(
                    src_pos=(0.0, 0.0), Q=Q, u=u, wind_dir_deg=wind_dir_deg,
                    H=H, stability_class=int(stability), receptors=receptor, z=z,
                )[0])
                excess = ppm_total - CH4_BACKGROUND
                cells.append(SweepCell(
                    Q=Q, distance=float(dist), stability=int(stability),
                    ppm_total=ppm_total, excess=excess,
                    verdict=classify(excess, noise_floor, k),
                ))
    return cells


def min_detectable_Q(
    cells: list[SweepCell],
    distance: float,
    stability: int,
) -> float | None:
    """
    Smallest tested Q that is "detectable" at a given distance and stability.

    Returns None if no tested leak was detectable there.
    """
    detectable_Qs = [
        c.Q for c in cells
        if c.distance == distance and c.stability == stability
        and c.verdict == "detectable"
    ]
    return min(detectable_Qs) if detectable_Qs else None


def verdict(
    cells: list[SweepCell],
    fence_distance: float,
    stability: int = 4,
    noise_floor: float = SENSOR_NOISE_PPM,
    k: float = DEFAULT_K,
) -> str:
    """
    Build the plain-English bottom-line answer for the project.

    Focuses on the realistic case: a sensor at the fenceline distance under a
    neutral (Class D = 4) atmosphere, the most common daytime condition.

    Returns a human-readable sentence.
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
    return (
        f"At {fence_distance:.0f} m (fenceline) under neutral Class-{stability} wind, "
        f"the smallest detectable leak is Q ≈ {q_min:g} g/s — anything weaker stays "
        f"below the {threshold:.2f} ppm detection threshold ({k:g}× the "
        f"{noise_floor:.2f} ppm noise floor) and would need heavy averaging or be "
        f"lost in the noise."
    )

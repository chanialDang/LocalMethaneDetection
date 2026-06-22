"""
inversion.py — Week-3 source inversion: readings in, leak (x, y, Q) out.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES (plain English)
───────────────────────────────────────────────────────────────────────────────
The forward model (plume.py) answers "given a leak, what will the sensors read?"
This module answers the REVERSE — the whole point of the project:

    "Given what the sensors actually read, WHERE is the leak and HOW BIG is it?"

You hand it a few sensors (their map positions + one aggregated reading each, from
fieldtest.aggregate_for_inversion) and the wind that was blowing, and it returns a
SourceEstimate: the best-fit source position (x, y) and emission rate Q, plus the
theoretical best-possible precision (the Cramér-Rao bound) so you immediately know
whether the fit is as good as the physics allows.

It is meant to be the "light switch": one call, ``invert(...)``, turns the whole
Week-2 forward model into a working Week-3 inversion.

═══════════════════════════════════════════════════════════════════════════════
WHY THERE IS NO scipy HERE (the key idea)
───────────────────────────────────────────────────────────────────────────────
The roadmap said "use scipy.optimize", but we don't need it — and the project has
a deliberate plain-numpy, no-new-dependency ethos (see weather.py / processing.py).
The reason we can drop scipy is a property of the physics:

    The Gaussian plume is LINEAR in Q:   excess_i(Q) = Q · excess_i(Q = 1 g/s).

So for ANY trial source position (x, y) and stability class, the best-fit Q is not
searched at all — it has a closed form (weighted least squares of a single
amplitude):

        Q* = Σ wᵢ gᵢ dᵢ / Σ wᵢ gᵢ²       (clamped to Q ≥ 0, which is physical)

    where  dᵢ = observed excess at sensor i (ppm above background)
           gᵢ = modelled excess at sensor i for a UNIT (1 g/s) source at (x, y)
           wᵢ = 1/σᵢ²  (the inverse-variance weight from aggregate_for_inversion)

That collapses a 3-parameter fit (x, y, Q) into a robust 2-D search over (x, y)
only, with Q solved exactly at every trial point. A 2-D smooth-ish surface is easy
to minimise without scipy: we use a shrinking-grid search (coarse grid → zoom in
on the best cell → repeat). It is dependency-free, has no convergence knobs to get
wrong, and is plenty precise for a source localisation.

The stability class is a DISCRETE integer (1–6), so we don't optimise it
continuously (predict_ppm forbids that anyway) — we simply try each class and keep
the one that fits best. That means the inversion can ESTIMATE the stability class
too when you don't trust the weather-derived guess.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS DOES NOT DO (honest scope)
───────────────────────────────────────────────────────────────────────────────
• Single source only. Multiple simultaneous leaks would need a sum of plumes.
• Wind (u, direction) and the geometry (H, z) are KNOWN inputs, not fit — pin them
  with weather.py, then the 160× Q-uncertainty swing collapses (see feasibility).
• It minimises a non-convex surface by grid zoom; with too few/too-collinear
  sensors the basin can be shallow. Compare the achieved scatter to ``crb_std``
  (returned) — if your error ≫ CRB, add a sensor or widen their spacing.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from physics.plume import (
    CH4_BACKGROUND, RELEASE_HEIGHT_M, SENSOR_HEIGHT_M, U_MIN, predict_ppm,
)
from physics.fieldtest import InversionPoint, aggregate_for_inversion
from physics.accuracy import crb_source_bound


@dataclass
class SourceEstimate:
    """
    The recovered leak. ``x``/``y`` are map coordinates (m, East/North), ``Q`` is
    the emission rate (g/s). ``crb_std`` is the theoretical best-possible 1σ on each
    parameter — compare your run-to-run scatter against it to know if the fit is as
    tight as the physics permits.
    """
    x: float                  # source East coordinate (m, map frame)
    y: float                  # source North coordinate (m, map frame)
    Q: float                  # emission rate (g/s) — closed-form WLS optimum, ≥ 0
    stability_class: int      # Pasquill class used/found (1=A … 6=F)
    u: float                  # wind speed assumed (m/s)
    wind_dir_deg: float       # wind FROM-direction assumed (° from N)
    cost: float               # weighted SSR at the optimum  Σ wᵢ(dᵢ − Q gᵢ)²
    rmse_ppm: float           # unweighted RMS residual (ppm) — human-readable fit
    residuals_ppm: list       # per-sensor (observed − modelled) excess (ppm)
    n_sensors: int
    crb_std: dict | None      # CRB best-possible 1σ {"x","y","Q"} (None if skipped)
    converged: bool           # False when the signal is too weak to localise
    note: str                 # short provenance / caveat


# ─────────────────────────────────────────────────────────────────────────────
# Core least-squares pieces (pure, no optimiser dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _per_unit_excess(x, y, sensors_xy, u, wind_dir_deg, H, z, stability,
                     T_K, P_Pa) -> np.ndarray:
    """
    Modelled excess (ppm above background) at each sensor for a UNIT 1 g/s source
    at map position (x, y). Because the plume is linear in Q, scaling this by Q
    gives the excess for any Q — that linearity is what removes Q from the search.

    Uses ``clamp_to_table=True`` so a trial position that puts a sensor < 1 m or
    > 200 m downwind never raises (the optimiser must be free to probe anywhere).
    """
    total = predict_ppm(
        src_pos=(float(x), float(y)), Q=1.0, u=u, wind_dir_deg=wind_dir_deg,
        H=H, stability_class=int(stability), receptors=sensors_xy,
        T_K=T_K, P_Pa=P_Pa, z=z, clamp_to_table=True,
    )
    return total - CH4_BACKGROUND


def _solve_Q(g: np.ndarray, d: np.ndarray, w: np.ndarray) -> float:
    """
    Closed-form non-negative weighted-least-squares amplitude.

    Minimises Σ wᵢ(dᵢ − Q gᵢ)² over the single scalar Q:
        dCost/dQ = 0  ⇒  Q = Σ wᵢ gᵢ dᵢ / Σ wᵢ gᵢ²
    then clamp to Q ≥ 0 (a negative emission rate is unphysical). When the modelled
    shape is ~0 at every sensor (source can't reach them), the denominator is ~0 and
    we return Q = 0 — i.e. "this position explains nothing", which the cost will then
    penalise for any real signal.
    """
    denom = float(np.sum(w * g * g))
    if denom <= 1e-30:
        return 0.0
    return max(0.0, float(np.sum(w * g * d) / denom))


class _Cell(NamedTuple):
    """One evaluated candidate: its weighted cost, position, closed-form Q, residuals."""
    cost: float
    x: float
    y: float
    Q: float
    resid: np.ndarray


def _cost_at(x, y, sensors_xy, d, w, u, wind_dir_deg, H, z, stability, T_K, P_Pa) -> _Cell:
    """Evaluate one trial (x, y): closed-form Q, then the weighted SSR. Returns a _Cell."""
    g = _per_unit_excess(x, y, sensors_xy, u, wind_dir_deg, H, z, stability, T_K, P_Pa)
    Q = _solve_Q(g, d, w)
    resid = d - Q * g
    cost = float(np.sum(w * resid * resid))
    return _Cell(cost, float(x), float(y), Q, resid)


def _default_bounds(sensors_xy: np.ndarray, pad: float = 150.0) -> tuple:
    """
    A generous search box: the sensors' bounding box, padded on every side.

    The source sits UPWIND of the sensors, so the box must extend beyond them in all
    directions; ``pad`` (default 150 m, ~the σ-table reach) covers a fenceline-scale
    site. clamp_to_table makes over-reach harmless, so erring large is safe.
    """
    xs, ys = sensors_xy[:, 0], sensors_xy[:, 1]
    return (float(xs.min() - pad), float(xs.max() + pad),
            float(ys.min() - pad), float(ys.max() + pad))


def _eval_grid(cost_fn, xs, ys) -> list:
    """Evaluate ``cost_fn`` at every point of the xs×ys lattice → list of _Cell."""
    return [cost_fn(x, y) for x in xs for y in ys]


def _fit_xy(cost_fn, bounds, grid: int, rounds: int, shrink: float,
            coarse: int, n_seeds: int) -> _Cell:
    """
    Global-ish 2-D minimiser of the cost surface — scipy-free, two stages.

    Stage 1 (find the basins): one coarse ``coarse``×``coarse`` scan over the whole
    box. Stage 2 (refine): from EACH of the best ``n_seeds`` coarse cells, run a
    local shrinking-grid zoom (re-centre on the best cell, shrink the box by
    ``shrink`` each of ``rounds`` rounds), then keep the global best.

    Why multi-start and not a single zoom: close, strong sensors make the cost basin
    razor-thin in the crosswind direction (its width ≈ σ_y, only a few metres at
    short range). A single coarse "best" can sit in a shallow WRONG basin while the
    true deep-but-narrow well is between coarse samples; committing to that one cell
    and shrinking would lock the answer in the wrong place. Keeping several seeds
    alive lets the fine refine discover the true global minimum. ``cost_fn(x, y)``
    returns a _Cell; this returns the single best _Cell found.
    """
    x0, x1, y0, y1 = bounds
    cells = sorted(_eval_grid(cost_fn, np.linspace(x0, x1, coarse),
                              np.linspace(y0, y1, coarse)), key=lambda c: c.cost)
    hw0 = 1.5 * (x1 - x0) / max(coarse - 1, 1)   # seed zoom half-width = 1.5 coarse cells
    hh0 = 1.5 * (y1 - y0) / max(coarse - 1, 1)

    best = cells[0]
    for seed in cells[:max(1, n_seeds)]:
        local, cx, cy, hw, hh = seed, seed.x, seed.y, hw0, hh0
        for _ in range(rounds):
            cand = min(_eval_grid(cost_fn, np.linspace(cx - hw, cx + hw, grid),
                                  np.linspace(cy - hh, cy + hh, grid)), key=lambda c: c.cost)
            if cand.cost < local.cost:
                local = cand
            cx, cy = local.x, local.y            # re-centre on the current best
            hw *= shrink
            hh *= shrink
        if local.cost < best.cost:
            best = local
    return best


# ─────────────────────────────────────────────────────────────────────────────
# THE LIGHT SWITCH — one call does the whole inversion
# ─────────────────────────────────────────────────────────────────────────────
def invert(
    sensor_positions,
    points,
    u: float,
    wind_dir_deg: float,
    stability_class: int | None = None,
    H: float = RELEASE_HEIGHT_M,
    z: float = SENSOR_HEIGHT_M,
    T_K: float = 293.15,
    P_Pa: float = 101325.0,
    bounds: tuple | None = None,
    coarse: int = 28,
    grid: int = 11,
    rounds: int = 6,
    shrink: float = 0.5,
    n_seeds: int = 6,
    with_crb: bool = True,
) -> SourceEstimate:
    """
    Recover the source (x, y, Q) from per-sensor aggregated readings. THE entry point.

    Parameters
    ----------
    sensor_positions : (N, 2) array of sensor map positions [x_east, y_north] (m).
    points           : list of N InversionPoint (from aggregate_for_inversion) — each
                       carries ``mean_excess_ppm`` (the datum) and ``sigma_ppm`` (its
                       weight). May also be (mean_excess_ppm, sigma_ppm) pairs.
    u, wind_dir_deg  : the KNOWN wind (m/s, ° FROM-direction). Pin these from weather.py.
    stability_class  : Pasquill 1–6 to FIX, or None to try all six and keep the best
                       fit (lets the inversion estimate stability too).
    H, z             : release height and sensor height (m) — fixed geometry.
    bounds           : (x0, x1, y0, y1) search box (m); default = sensors' bbox + 150 m.
    coarse, grid, rounds, shrink, n_seeds : two-stage optimiser controls (see _fit_xy):
                       ``coarse`` = full-box scan resolution; the best ``n_seeds``
                       basins are each refined with a ``grid``-cell local zoom over
                       ``rounds`` (shrinking by ``shrink``).
    with_crb         : also compute the Cramér-Rao best-possible 1σ at the solution.

    Returns
    -------
    SourceEstimate. ``converged`` is False when no sensor carries a real signal
    (Q≈0): then the position is unconstrained and only an upper bound on Q is meaningful.

    Notes
    -----
    • Q is never searched — it is solved in closed form at every trial (x, y), see
      the module header. That is why this is fast and scipy-free.
    • u must be ≥ U_MIN (0.5 m/s); the steady-state plume is undefined below that.
    """
    sensors_xy = np.atleast_2d(np.asarray(sensor_positions, dtype=float))
    n = sensors_xy.shape[0]
    if n == 0:
        raise ValueError("need at least one sensor position")
    if len(points) != n:
        raise ValueError(f"got {n} sensor positions but {len(points)} readings")
    if u < U_MIN:
        raise ValueError(f"wind u={u} m/s is below the model minimum {U_MIN} m/s")

    # Pull the datum + weight out of each point (InversionPoint or a plain pair).
    def _datum(p):
        if hasattr(p, "mean_excess_ppm"):
            return float(p.mean_excess_ppm), float(p.sigma_ppm)
        return float(p[0]), float(p[1])

    data = [_datum(p) for p in points]
    d = np.array([m for m, _ in data])
    sig = np.array([s for _, s in data])
    sig = np.where(sig > 1e-9, sig, 1e-9)        # guard a zero σ (infinite weight)
    w = 1.0 / (sig * sig)

    if bounds is None:
        bounds = _default_bounds(sensors_xy)

    classes = [int(stability_class)] if stability_class is not None else [1, 2, 3, 4, 5, 6]

    best, cls = None, classes[0]     # best _Cell across classes, and the winning class
    for trial_cls in classes:
        # Per-class cost closure: binds the fixed geometry/wind so the optimiser only
        # varies (x, y). Short-lived (dropped each iteration), so it pins no big scope.
        def cost_fn(xx, yy, _cls=trial_cls):
            return _cost_at(xx, yy, sensors_xy, d, w, u, wind_dir_deg, H, z,
                            _cls, T_K, P_Pa)
        cell = _fit_xy(cost_fn, bounds, grid, rounds, shrink, coarse, n_seeds)
        if best is None or cell.cost < best.cost:
            best, cls = cell, trial_cls

    cost, x, y, Q, resid = best.cost, best.x, best.y, best.Q, best.resid
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    # "Converged" only if there is real signal to localise: some sensor's reading
    # must stand clearly above its own noise AND the fit must explain a positive Q.
    signal_present = bool(np.any(d > 3.0 * sig))
    converged = bool(signal_present and Q > 0.0)

    crb_std = None
    note = "shrinking-grid WLS; Q closed-form (linear-in-Q)"
    if not converged:
        note = "no sensor shows signal above 3σ — Q≈0, position unconstrained (upper-bound only)"
    elif with_crb:
        # Representative iid σ for the CRB (it assumes equal independent noise); use
        # the RMS of the per-sensor σ. Approximate when σ varies a lot across sensors.
        sigma_rep = float(np.sqrt(np.mean(sig ** 2)))
        try:
            crb = crb_source_bound(
                src_pos=(x, y), Q=Q, u=u, wind_dir_deg=wind_dir_deg, H=H,
                stability_class=cls, receptors=sensors_xy, sigma_ppm=sigma_rep)
            crb_std = crb.std
            note += f"; CRB at σ≈{sigma_rep:.2f} ppm ({crb.note})"
        except Exception as exc:                 # never let the referee crash the fit
            note += f"; CRB unavailable ({type(exc).__name__})"

    return SourceEstimate(
        x=x, y=y, Q=Q, stability_class=cls, u=float(u),
        wind_dir_deg=float(wind_dir_deg), cost=cost, rmse_ppm=rmse,
        residuals_ppm=[float(r) for r in resid],
        n_sensors=n, crb_std=crb_std, converged=converged, note=note,
    )


def invert_field_tests(
    results,
    sensor_positions,
    u: float,
    wind_dir_deg: float,
    window_s: float | None = None,
    **kwargs,
) -> SourceEstimate:
    """
    Convenience light switch from RAW processed field tests straight to a source.

    ``results`` is a list of per-sensor dicts from fieldtest.process_fieldtest; this
    aggregates each one (aggregate_for_inversion → the time-mean + σ-of-the-mean) and
    forwards the InversionPoints to ``invert``. This is the call a deployment makes:
    upload N sensors' CSVs → process each → invert_field_tests → done.
    """
    pts = [aggregate_for_inversion(r, window_s=window_s) for r in results]
    return invert(sensor_positions, pts, u, wind_dir_deg, **kwargs)

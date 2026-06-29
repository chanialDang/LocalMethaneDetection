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
    RELEASE_HEIGHT_M, SENSOR_HEIGHT_M, U_MIN, predict_excess_grid,
)
from physics.fieldtest import InversionPoint, aggregate_for_inversion
from physics.accuracy import crb_source_bound_multi
from physics.sensor_sim import DETECT_K, effective_noise_floor


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
    n_snapshots: int = 1      # how many wind-snapshots were fused (1 = single)


@dataclass
class Snapshot:
    """
    One observation of the SAME source + sensors under ONE wind. A multi-hour field
    deployment yields several of these as the wind veers; fusing them triangulates an
    otherwise under-determined fenceline (see invert_multi / BUGS.md F7).

    ``points`` are this snapshot's per-sensor readings (InversionPoint objects or
    (mean_excess_ppm, sigma_ppm) pairs), in the same sensor order as every other
    snapshot. ``u``/``wind_dir_deg`` are the wind that blew DURING this snapshot.
    ``stability_class`` is this snapshot's Pasquill class if known (weather.py gives
    it per time); None lets the fit choose one shared class across snapshots.
    """
    points: list
    u: float
    wind_dir_deg: float
    stability_class: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Core least-squares pieces (pure, no optimiser dependency)
# ─────────────────────────────────────────────────────────────────────────────
class _Cell(NamedTuple):
    """One evaluated candidate: its weighted cost, position, closed-form Q, residuals."""
    cost: float
    x: float
    y: float
    Q: float
    resid: np.ndarray


class _Snap(NamedTuple):
    """A snapshot prepared for the optimiser: data, weights, and its resolved wind
    + stability. Same sensors as every other snapshot; only the wind/stability differ."""
    d: np.ndarray            # per-sensor mean excess (ppm)
    w: np.ndarray            # per-sensor WLS weight 1/σ²
    floor1: np.ndarray       # per-sensor single-sample detection floor (ppm)
    u: float                 # wind speed for this snapshot (m/s)
    wind_dir_deg: float      # wind FROM-direction for this snapshot (° from N)
    stability: int           # Pasquill class used for this snapshot (1–6)


def _prep_points(points, n) -> tuple:
    """Extract per-sensor (datum, σ-of-mean, single-sample floor) and the WLS weight.

    Accepts InversionPoint objects or plain (mean_excess, σ) pairs. Factored so the
    single- and multi-snapshot entry points share ONE definition of the datum and the
    F6 signal gate. ``floor1`` is the single-sample detection floor (NOT the √N-shrunk
    σ-of-the-mean): the right SIGNAL gate, immune to the baseline offset that made the
    σ-of-the-mean gate fire on pure noise (see F6)."""
    if len(points) != n:
        raise ValueError(f"got {n} sensor positions but {len(points)} readings")

    def datum(p):
        if hasattr(p, "mean_excess_ppm"):
            floor1 = float(effective_noise_floor(p.random_ppm, p.bias_ppm, n_avg=1))
            floor1 = floor1 if floor1 > 0.0 else float(p.sigma_ppm)
            return float(p.mean_excess_ppm), float(p.sigma_ppm), floor1
        return float(p[0]), float(p[1]), float(p[1])     # (mean, σ) tuple: σ is the floor

    data = [datum(p) for p in points]
    d = np.array([m for m, _, _ in data])
    sig = np.array([s for _, s, _ in data])
    floor1 = np.array([f for _, _, f in data])
    sig = np.where(sig > 1e-9, sig, 1e-9)                # guard a zero σ (infinite weight)
    return d, sig, floor1, 1.0 / (sig * sig)


def _eval_cells(xs, ys, sensors_xy, snaps, H, z, T_K, P_Pa) -> list:
    """Evaluate the whole xs×ys lattice across ALL snapshots in batched plume passes.

    A steady leak of strength Q seen under K winds produces K predicted patterns that
    SHARE the same Q (the source emits the same g/s whatever the wind). The plume is
    linear in Q, so the joint best-fit Q at each trial (x, y) stays closed-form, now
    summed over every snapshot's sensors:

        Q* = Σ_k Σ_i w g d  /  Σ_k Σ_i w g²        (clamped ≥ 0)

    K=1 reduces to the original per-cell WLS amplitude; K≥2 is what makes an
    under-determined single-wind fenceline identifiable (BUGS.md F7). One
    predict_excess_grid pass per snapshot pays the σ-table/rotation setup once per
    lattice (pinned vs predict_ppm by test_predict_excess_grid_matches_predict_ppm)."""
    src_grid = np.array([(float(x), float(y)) for x in xs for y in ys])
    gs = [predict_excess_grid(src_grid, sensors_xy, 1.0, sn.u, sn.wind_dir_deg, H,
                              int(sn.stability), T_K=T_K, P_Pa=P_Pa, z=z)
          for sn in snaps]
    num = sum(np.sum(sn.w * g * sn.d, axis=1) for sn, g in zip(snaps, gs))
    den = sum(np.sum(sn.w * g * g, axis=1) for sn, g in zip(snaps, gs))
    Q = np.zeros(len(src_grid))
    ok = den > 1e-30                       # ~0 everywhere → source reaches no sensor here
    Q[ok] = np.maximum(0.0, num[ok] / den[ok])
    # Weighted residuals + SSR over EVERY snapshot×sensor point (concatenated).
    resid_all = np.concatenate([sn.d[None, :] - Q[:, None] * g
                                for sn, g in zip(snaps, gs)], axis=1)    # (K_cells, K·N)
    w_all = np.concatenate([sn.w for sn in snaps])                        # (K·N,)
    cost = np.sum(w_all[None, :] * resid_all * resid_all, axis=1)
    return [_Cell(float(cost[i]), float(src_grid[i, 0]), float(src_grid[i, 1]),
                  float(Q[i]), resid_all[i]) for i in range(len(src_grid))]


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


def _fit_xy(batch_fn, bounds, grid: int, rounds: int, shrink: float,
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
    alive lets the fine refine discover the true global minimum. ``batch_fn(xs, ys)``
    evaluates a whole lattice → list of _Cell; this returns the single best _Cell found.
    """
    x0, x1, y0, y1 = bounds
    cells = sorted(batch_fn(np.linspace(x0, x1, coarse),
                            np.linspace(y0, y1, coarse)), key=lambda c: c.cost)
    hw0 = 1.5 * (x1 - x0) / max(coarse - 1, 1)   # seed zoom half-width = 1.5 coarse cells
    hh0 = 1.5 * (y1 - y0) / max(coarse - 1, 1)

    best = cells[0]
    for seed in cells[:max(1, n_seeds)]:
        local, cx, cy, hw, hh = seed, seed.x, seed.y, hw0, hh0
        for _ in range(rounds):
            cand = min(batch_fn(np.linspace(cx - hw, cx + hw, grid),
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
                       Avoid boxes >> the default: when a single near-sensor sees huge
                       excess, the coarse grid can lock onto a spurious basin with a
                       huge Q rather than the true source (see BUGS.md F7).
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
    return invert_multi(
        sensor_positions,
        [Snapshot(points=points, u=u, wind_dir_deg=wind_dir_deg)],
        stability_class=stability_class, H=H, z=z, T_K=T_K, P_Pa=P_Pa,
        bounds=bounds, coarse=coarse, grid=grid, rounds=rounds, shrink=shrink,
        n_seeds=n_seeds, with_crb=with_crb,
    )


def invert_multi(
    sensor_positions,
    snapshots,
    stability_class: int | None = None,
    H: float = RELEASE_HEIGHT_M,
    z: float = SENSOR_HEIGHT_M,
    T_K: float = 293.15,
    P_Pa: float = 101325.0,
    bounds: tuple | None = None,
    coarse: int = 60,
    grid: int = 11,
    rounds: int = 6,
    shrink: float = 0.5,
    n_seeds: int = 12,
    with_crb: bool = True,
) -> SourceEstimate:
    """
    Recover ONE steady source from the SAME sensors observed under several winds.

    This is the cure for the single-wind ill-posedness of BUGS.md F7. A fenceline where
    only a sensor or two sits in the plume is under-determined from one snapshot — many
    (x, y, Q) explain the readings equally, and the fit can land tens of metres away
    while looking confident. Watching the leak from several wind directions TRIANGULATES
    it: each wind sweeps the plume across a different subset of sensors, and the snapshots
    only agree at the true source. (Empirically, 2–3 winds collapse a ~28 m single-wind
    error to sub-metre.)

    The physics stays scipy-free: a steady leak emits the same Q under every wind, so Q
    is still solved in closed form — now jointly across all snapshots (see _eval_cells) —
    leaving the same 2-D (x, y) grid search. Fisher information adds across snapshots, so
    the returned CRB is the combined (tighter) bound (crb_source_bound_multi).

    Parameters
    ----------
    sensor_positions : (N, 2) sensor map positions [x_east, y_north] (m) — shared by all
                       snapshots (the array doesn't move; the wind does).
    snapshots        : list of Snapshot, one per wind. Each carries its own ``points``
                       (N readings, same sensor order), ``u``, ``wind_dir_deg`` and
                       optional ``stability_class``.
    stability_class  : shared Pasquill class to FIX, or None to fit one shared class. A
                       snapshot's own ``stability_class`` (if set) overrides this for that
                       snapshot — pin each from weather.py when you have per-time data.
    coarse, n_seeds  : denser defaults than ``invert`` (60/12 vs 28/6). Each snapshot you
                       add SHARPENS the joint cost basin (more constraints → narrower
                       well), so a coarse grid can step over it; raise ``coarse`` further
                       if you fuse many snapshots. (other args as in ``invert``.)

    Returns
    -------
    SourceEstimate with ``n_snapshots`` set. ``u``/``wind_dir_deg`` report the first
    snapshot's wind (representative); the full set is summarised in ``note``.
    """
    sensors_xy = np.atleast_2d(np.asarray(sensor_positions, dtype=float))
    n = sensors_xy.shape[0]
    if n == 0:
        raise ValueError("need at least one sensor position")
    if not snapshots:
        raise ValueError("need at least one snapshot")

    # Prepare every snapshot (datum/weights/floor + its own wind & known stability).
    base = []
    for sn in snapshots:
        if sn.u < U_MIN:
            raise ValueError(f"wind u={sn.u} m/s is below the model minimum {U_MIN} m/s")
        d, sig, floor1, w = _prep_points(sn.points, n)
        base.append((d, w, floor1, float(sn.u), float(sn.wind_dir_deg),
                     sn.stability_class))

    if bounds is None:
        bounds = _default_bounds(sensors_xy)

    # Try all six classes only when at least one snapshot's class is unknown AND no
    # shared class was fixed; otherwise a single pass (each snapshot keeps its own).
    need_fit = stability_class is None and any(st is None for *_, st in base)
    classes = [1, 2, 3, 4, 5, 6] if need_fit else [stability_class]

    best, best_snaps = None, None
    for trial in classes:
        resolved = [_Snap(d, w, floor1, u, wd, int(st if st is not None else trial))
                    for (d, w, floor1, u, wd, st) in base]
        # Cost closure binds the resolved snapshots so the optimiser only varies (x, y).
        def batch_fn(xs, ys, _snaps=resolved):
            return _eval_cells(xs, ys, sensors_xy, _snaps, H, z, T_K, P_Pa)
        cell = _fit_xy(batch_fn, bounds, grid, rounds, shrink, coarse, n_seeds)
        if best is None or cell.cost < best.cost:
            best, best_snaps = cell, resolved

    cost, x, y, Q, resid = best.cost, best.x, best.y, best.Q, best.resid
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    K = len(best_snaps)

    # "Converged" only with real signal to localise: some sensor in some snapshot must
    # stand clearly above its own per-sample floor (DETECT_K·single-sample σ, F6) AND
    # the fit must explain a positive Q.
    d_all = np.concatenate([s.d for s in best_snaps])
    floor1_all = np.concatenate([s.floor1 for s in best_snaps])
    n_signal = int(np.sum(d_all > DETECT_K * floor1_all))
    converged = bool(n_signal > 0 and Q > 0.0)
    cls = int(best_snaps[0].stability)

    crb_std = None
    note = f"shrinking-grid WLS; Q closed-form (linear-in-Q); {K} snapshot(s)"
    if not converged:
        note = "no sensor shows signal above 3σ — Q≈0, position unconstrained (upper-bound only)"
    else:
        if with_crb:
            # Representative iid σ (the CRB assumes equal independent noise): RMS of the
            # per-sensor σ across all snapshots. Fisher info adds → combined (tighter) bound.
            sig_all = np.concatenate([1.0 / np.sqrt(s.w) for s in best_snaps])
            sigma_rep = float(np.sqrt(np.mean(sig_all ** 2)))
            try:
                views = [(s.u, s.wind_dir_deg, int(s.stability)) for s in best_snaps]
                crb = crb_source_bound_multi(src_pos=(x, y), Q=Q, views=views, H=H,
                                             receptors=sensors_xy, sigma_ppm=sigma_rep)
                crb_std = crb.std
                note += f"; CRB at σ≈{sigma_rep:.2f} ppm ({crb.note})"
            except Exception as exc:             # never let the referee crash the fit
                note += f"; CRB unavailable ({type(exc).__name__})"

    return SourceEstimate(
        x=x, y=y, Q=Q, stability_class=cls, u=float(best_snaps[0].u),
        wind_dir_deg=float(best_snaps[0].wind_dir_deg), cost=cost, rmse_ppm=rmse,
        residuals_ppm=[float(r) for r in resid], n_sensors=n,
        crb_std=crb_std, converged=converged, note=note, n_snapshots=K,
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


def invert_field_tests_multi(
    snapshots,
    sensor_positions,
    window_s: float | None = None,
    **kwargs,
) -> SourceEstimate:
    """
    Multi-snapshot light switch from RAW processed field tests → one fused source.

    This is the real deployment call when the wind veered during the test: group the
    per-sensor records by wind episode, and pass one group per wind. Each ``snapshots``
    entry is ``(results, u, wind_dir_deg)`` or ``(results, u, wind_dir_deg,
    stability_class)``, where ``results`` is the per-sensor list of process_fieldtest
    dicts captured under that wind. Each is aggregated (aggregate_for_inversion) and the
    snapshots are fused by ``invert_multi`` — triangulating the source (BUGS.md F7).
    """
    snaps = []
    for snap in snapshots:
        results, u, wind_dir = snap[0], snap[1], snap[2]
        stability = snap[3] if len(snap) > 3 else None
        pts = [aggregate_for_inversion(r, window_s=window_s) for r in results]
        snaps.append(Snapshot(points=pts, u=u, wind_dir_deg=wind_dir,
                              stability_class=stability))
    return invert_multi(sensor_positions, snaps, **kwargs)

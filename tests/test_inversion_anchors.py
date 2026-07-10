"""
test_inversion_anchors.py — INDEPENDENT regression anchors for the inversion math.

The session audit verified the inversion/accuracy math against independent routes but
found two anchors the suite lacked: the CRB's ABSOLUTE value (only relative scaling
was pinned) and the closed-form Q against a hand-computed Σwgd/Σwg² (only linearity /
round-trip were pinned). These lock both so the verified-clean math can't silently
drift. Every expected value is re-derived here from `predict_ppm`/`predict_excess_grid`
alone — no accuracy.py / inversion.py internals are called on the expectation side.
"""
import numpy as np
import pytest

from physics.accuracy import crb_source_bound
from physics.inversion import invert
from physics.plume import (
    CH4_BACKGROUND,
    RELEASE_HEIGHT_M,
    SENSOR_HEIGHT_M,
    predict_excess_grid,
    predict_ppm,
)


def test_crb_matches_independent_finite_difference_jacobian():
    # Rebuild the Cramér-Rao bound from scratch: a central-difference Jacobian of
    # predict_ppm (z=0 default, matching crb_source_bound's internal probe), Fisher
    # F = JᵀJ/σ², std = √diag(F⁻¹). A DIFFERENT step from the code's (0.05 vs 0.1 m)
    # makes this a genuine re-derivation, not a copy — a missing σ², a forward
    # difference, or a JᵀJ transpose slip would all break the match.
    src = (5.0, 3.0)
    Q, u, wind, H, sc = 2.0, 3.0, 270.0, 1.0, 4
    receptors = np.array([[40, -8], [40, 0], [40, 8], [70, 0], [100, 4]], dtype=float)
    sigma = 0.2

    def predict(x, y, q):
        return predict_ppm((x, y), q, u, wind, H, sc, receptors, clamp_to_table=True)

    hx = hy = 0.05
    hQ = 5e-4
    J = np.column_stack([
        (predict(src[0] + hx, src[1], Q) - predict(src[0] - hx, src[1], Q)) / (2 * hx),
        (predict(src[0], src[1] + hy, Q) - predict(src[0], src[1] - hy, Q)) / (2 * hy),
        (predict(src[0], src[1], Q + hQ) - predict(src[0], src[1], Q - hQ)) / (2 * hQ),
    ])
    F = J.T @ J / sigma ** 2
    std_indep = np.sqrt(np.diag(np.linalg.inv(F)))

    cb = crb_source_bound(src, Q, u, wind, H, sc, receptors, sigma_ppm=sigma)
    for i, p in enumerate(("x", "y", "Q")):
        assert cb.std[p] == pytest.approx(std_indep[i], rel=0.03)

    # CRB is a variance bound → covariance ∝ σ², so std ∝ σ (independent scaling law).
    cb2 = crb_source_bound(src, Q, u, wind, H, sc, receptors, sigma_ppm=2 * sigma)
    for p in ("x", "y", "Q"):
        assert cb2.std[p] == pytest.approx(2 * cb.std[p], rel=1e-6)


def test_closed_form_Q_is_the_wls_optimum():
    # Round-trip on noiseless data, then confirm the recovered Q is EXACTLY the WLS
    # optimum Σ(w·g·d)/Σ(w·g²) with g = excess-per-unit-Q (predict_excess_grid at Q=1)
    # evaluated at the recovered position — the closed form the inversion claims.
    true_src = (10.0, -5.0)
    Q_true, u, wind, sc = 3.0, 2.5, 270.0, 4
    receptors = np.array([[45, -12], [45, 0], [45, 12], [75, -6], [90, 6]], dtype=float)
    sigma = 0.15

    excess = predict_ppm(true_src, Q_true, u, wind, RELEASE_HEIGHT_M, sc, receptors,
                         z=SENSOR_HEIGHT_M) - CH4_BACKGROUND
    pts = [(float(e), sigma) for e in excess]
    est = invert(receptors, pts, u=u, wind_dir_deg=wind, stability_class=sc,
                 with_crb=False)

    # Independent truth was chosen, so recovery is a real check (not circular).
    assert est.Q == pytest.approx(Q_true, rel=0.05)
    assert est.x == pytest.approx(true_src[0], abs=2.0)
    assert est.y == pytest.approx(true_src[1], abs=2.0)

    g = predict_excess_grid(np.array([[est.x, est.y]]), receptors, Q=1.0, u=u,
                            wind_dir_deg=wind, H=RELEASE_HEIGHT_M, stability_class=sc,
                            z=SENSOR_HEIGHT_M)[0]
    w = 1.0 / sigma ** 2
    Q_closed = float(np.sum(w * g * excess) / np.sum(w * g ** 2))
    assert est.Q == pytest.approx(Q_closed, rel=1e-6)


def test_effective_noise_floor_formula_and_limits():
    # Independent re-computation of the quadrature floor + its two physical limits.
    from physics.sensor_sim import effective_noise_floor
    for rnd, bias, n, rho in [(0.30, 0.0, 1, 0.0), (0.30, 0.10, 16, 0.0),
                              (0.30, 0.05, 64, 0.8), (0.5, 0.2, 100, 0.5),
                              (0.4, 0.0, 256, 0.9)]:
        n_eff = max(1.0, n * (1 - rho) / (1 + rho))
        want = np.sqrt((rnd / np.sqrt(n_eff)) ** 2 + bias ** 2)
        assert effective_noise_floor(rnd, bias, n, rho) == pytest.approx(want, rel=1e-12)
    # BIAS does not average away → floor → bias as n → ∞ (the un-averageable drift).
    assert effective_noise_floor(0.30, 0.10, 10 ** 9) == pytest.approx(0.10, rel=1e-4)
    # RANDOM does average away → floor = random/√n when bias = 0.
    assert effective_noise_floor(0.30, 0.0, 100) == pytest.approx(0.030, rel=1e-12)
    # Correlated samples (ρ>0) carry less info → a LARGER floor than the i.i.d. case.
    assert (effective_noise_floor(0.30, 0.0, 64, rho=0.8)
            > effective_noise_floor(0.30, 0.0, 64, rho=0.0))


def test_detection_limit_lod_loq_and_linear_backsolve():
    # LOD/LOQ are k· and 10· the floor; the min detectable Q is the LOD back-solved
    # through the (linear-in-Q) plume — all checked against independent routes.
    from physics import feasibility
    from physics.accuracy import detection_limit
    from physics.sensor_sim import effective_noise_floor
    dl = detection_limit(0.30, bias_ppm=0.0, distance=50.0, k=3.0, u=2.0, stability=4)
    floor = effective_noise_floor(0.30, 0.0, 1)
    assert dl.floor_ppm == pytest.approx(floor)
    assert dl.lod_ppm == pytest.approx(3.0 * floor)
    assert dl.loq_ppm == pytest.approx(10.0 * floor)
    # Back-solve arithmetic: min_Q × (excess per unit Q at 50 m) == LOD.
    per_unit = feasibility._excess_at(50.0, 1.0, 2.0, 4, clamp=True)
    assert dl.min_detectable_Q == pytest.approx(dl.lod_ppm / per_unit, rel=1e-9)
    # Linear-in-Q (what makes the back-solve exact): doubling Q doubles the excess.
    assert (feasibility._excess_at(50.0, 2.0, 2.0, 4, clamp=True)
            == pytest.approx(2.0 * per_unit, rel=1e-9))
    # Averaging lowers the limit until bias floors it: the curve's min_Q is monotone ↓.
    mins = [row["min_Q"] for row in dl.curve]
    assert all(b <= a + 1e-12 for a, b in zip(mins, mins[1:]))


def test_multi_snapshot_fusion_recovers_and_tightens_crb():
    # F7: a single wind on a fence-line (all sensors at one downwind distance) can't
    # pin the ALONG-wind position; fusing winds triangulates. Independent truth chosen,
    # noiseless data generated forward, recovered blind.
    from physics.accuracy import crb_source_bound, crb_source_bound_multi
    from physics.inversion import Snapshot, invert, invert_multi
    sensors = np.array([[50, -20], [50, -10], [50, 0], [50, 10], [50, 20]], dtype=float)
    true_src = (10.0, 3.0)
    Q, sc, sigma = 4.0, 4, 0.10
    winds = [(3.0, 270.0), (3.0, 240.0), (3.0, 300.0)]

    snaps = []
    for u, wd in winds:
        exc = predict_ppm(true_src, Q, u, wd, RELEASE_HEIGHT_M, sc, sensors,
                          z=SENSOR_HEIGHT_M) - CH4_BACKGROUND
        snaps.append(Snapshot(points=[(float(e), sigma) for e in exc],
                              u=u, wind_dir_deg=wd, stability_class=sc))

    est1 = invert(sensors, snaps[0].points, u=winds[0][0], wind_dir_deg=winds[0][1],
                  stability_class=sc)
    estM = invert_multi(sensors, snaps, stability_class=sc)

    err1 = float(np.hypot(est1.x - true_src[0], est1.y - true_src[1]))
    errM = float(np.hypot(estM.x - true_src[0], estM.y - true_src[1]))
    assert estM.n_snapshots == 3
    assert errM < 2.0                        # fusion pins it tightly…
    assert errM <= err1 + 1e-6               # …and never worse than the single wind

    # Fisher information ADDS → the fused CRB is ≤ any single-wind CRB on every axis
    # (F_multi ⪰ F_single ⇒ F_multi⁻¹ ⪯ F_single⁻¹ ⇒ smaller diagonal). Independent
    # of the estimator — a pure property of summing PSD Fisher matrices.
    views = [(u, wd, sc) for u, wd in winds]
    cbM = crb_source_bound_multi(true_src, Q, views, RELEASE_HEIGHT_M, sensors, sigma_ppm=sigma)
    cb1 = crb_source_bound(true_src, Q, winds[0][0], winds[0][1], RELEASE_HEIGHT_M, sc,
                           sensors, sigma_ppm=sigma)
    for p in ("x", "y", "Q"):
        assert cbM.std[p] <= cb1.std[p] * (1 + 1e-9)

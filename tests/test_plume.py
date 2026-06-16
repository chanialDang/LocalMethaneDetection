"""
test_plume.py — regression tests for the Gaussian plume forward model (plume.py).

demo.py validates the physics core *visually*; this suite locks it in for pytest so
Week-3's inversion work can refactor plume.py without silently breaking it. Coverage:

  - σ-table lookup reproduces the Briggs (1973) formula at an exact table row
  - σ monotonicity (distance, stability class) and range/class guards
  - the ground-level reflection form  C = Q/(π σ_y σ_z u)
  - upwind→background and calm-wind→ValueError guards
  - the vectorized predict_ppm matches the scalar pipeline receptor-by-receptor
    (this is the property the hot-path optimisation must preserve)
"""

import math

import numpy as np
import pytest

from physics.plume import (
    CH4_BACKGROUND,
    concentration_gm3,
    gm3_to_ppm_methane,
    predict_ppm,
    rotate_to_wind_frame,
    sigma_y,
    sigma_z,
)


def _briggs_D(x: float) -> tuple[float, float]:
    """Independent Briggs (1973) class-D σ_y, σ_z (computed here, not via the table)."""
    sy = 0.08 * x * (1.0 + 1.0e-4 * x) ** -0.5
    sz = 0.06 * x * (1.0 + 1.5e-3 * x) ** -0.5
    return sy, sz


# ── σ lookup ────────────────────────────────────────────────────────────────

def test_sigma_lookup_matches_briggs_at_table_row():
    # 100 m is an exact table row → lookup reproduces the formula (to 4-dp rounding).
    sy_ref, sz_ref = _briggs_D(100.0)
    assert sigma_y(100.0, 4) == pytest.approx(sy_ref, abs=5e-4)
    assert sigma_z(100.0, 4) == pytest.approx(sz_ref, abs=5e-4)


def test_sigma_interpolates_between_rows():
    # A non-integer distance must fall strictly between its bracketing table rows.
    lo, hi = sigma_y(37.0, 4), sigma_y(38.0, 4)
    assert lo < sigma_y(37.4, 4) < hi


def test_sigma_monotonic_in_distance_and_class():
    assert sigma_y(50.0, 4) < sigma_y(150.0, 4)        # grows downwind
    assert sigma_z(50.0, 1) > sigma_z(50.0, 6)         # class A spreads more than F


def test_sigma_out_of_range_raises():
    with pytest.raises(ValueError):
        sigma_y(0.5, 4)        # below the 1 m table floor
    with pytest.raises(ValueError):
        sigma_z(250.0, 4)      # above the 200 m table ceiling


@pytest.mark.parametrize("bad_class", [0, 7, 2.5, -1])
def test_sigma_invalid_class_raises(bad_class):
    with pytest.raises(ValueError):
        sigma_y(100.0, bad_class)


# ── concentration_gm3 ─────────────────────────────────────────────────────────

def test_concentration_ground_level_reflection_form():
    # At y=z=H=0 the 2π denominator + image term (=2) collapse to π:
    #   C = Q / (π σ_y σ_z u)
    sy, sz = _briggs_D(100.0)
    C = concentration_gm3(100.0, 0.0, Q=1.0, u=5.0, H=0.0, sy=sy, sz=sz, z=0.0)
    assert C == pytest.approx(1.0 / (math.pi * sy * sz * 5.0), rel=1e-12)


def test_concentration_upwind_returns_zero():
    assert concentration_gm3(-10.0, 0.0, 1.0, 5.0, 0.0, 8.0, 5.6) == 0.0


def test_concentration_calm_wind_raises():
    with pytest.raises(ValueError):
        concentration_gm3(100.0, 0.0, 1.0, u=0.1, H=0.0, sy=8.0, sz=5.6)


# ── predict_ppm guards ────────────────────────────────────────────────────────

def test_predict_ppm_upwind_is_exactly_background():
    rec = np.array([[-50.0, 0.0]])     # due west; wind from west (270°) → upwind
    out = predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 1.0, 4, rec)
    assert out[0] == pytest.approx(CH4_BACKGROUND, abs=1e-12)


def test_predict_ppm_calm_wind_raises_when_a_receptor_is_downwind():
    rec = np.array([[50.0, 0.0]])      # downwind under 270°
    with pytest.raises(ValueError):
        predict_ppm((0.0, 0.0), 1.0, 0.1, 270.0, 1.0, 4, rec)


def test_predict_ppm_calm_wind_ok_when_all_upwind():
    # Subtle invariant: u is never validated if no receptor is downwind, so an
    # all-upwind call with calm wind must return background, not raise.
    rec = np.array([[-50.0, 0.0]])
    out = predict_ppm((0.0, 0.0), 1.0, 0.1, 270.0, 1.0, 4, rec)
    assert out[0] == pytest.approx(CH4_BACKGROUND, abs=1e-12)


# ── optimizer-safe boundary mode (clamp_to_table) ───────────────────────────
# Week-3 inversion runs predict_ppm inside scipy.optimize, which probes source
# positions that put a sensor < 1 m or > 200 m downwind. Strict mode must raise
# there (so demo runs notice); clamp mode must NOT raise (so the optimizer lives).

def test_predict_ppm_strict_raises_below_table_floor():
    # Source ~0.4 m downwind of the sensor → inside the 1 m table floor.
    rec = np.array([[0.4, 0.0]])
    with pytest.raises(ValueError):
        predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 1.0, 4, rec)


def test_predict_ppm_strict_raises_beyond_table_ceiling():
    rec = np.array([[250.0, 0.0]])      # 250 m downwind, past the 200 m ceiling
    with pytest.raises(ValueError):
        predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 1.0, 4, rec)


def test_clamp_mode_does_not_raise_on_out_of_range_geometry():
    # The exact geometry that would crash the optimizer in strict mode.
    rec = np.array([[0.4, 0.0], [250.0, 0.0]])
    out = predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 1.0, 4, rec, clamp_to_table=True)
    assert np.all(np.isfinite(out))


def test_clamp_mode_far_field_is_background_only():
    # A sensor past the 200 m ceiling sees no plume → exactly background.
    rec = np.array([[250.0, 0.0]])
    out = predict_ppm((0.0, 0.0), 5.0, 2.0, 270.0, 1.0, 4, rec, clamp_to_table=True)
    assert out[0] == pytest.approx(CH4_BACKGROUND, abs=1e-12)


def test_clamp_mode_near_field_clamps_to_one_metre_floor():
    # A sub-1 m downwind point is evaluated as if at 1 m (the table floor), so it
    # equals the in-range prediction at exactly 1 m — finite, above background.
    # Ground-level source (H=0) so the clamped 1 m point reads clearly above
    # background — at H=1 the plume has not yet spread down to a z=0 sensor.
    near = np.array([[0.4, 0.0]])
    at_floor = np.array([[1.0, 0.0]])
    out_near  = predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 0.0, 4, near,     clamp_to_table=True)
    out_floor = predict_ppm((0.0, 0.0), 1.0, 2.0, 270.0, 0.0, 4, at_floor, clamp_to_table=True)
    assert out_near[0] > CH4_BACKGROUND
    assert out_near[0] == pytest.approx(out_floor[0], rel=1e-12)


def test_clamp_mode_agrees_with_strict_inside_valid_range():
    # Where geometry is legal, the optimizer-safe flag must change nothing.
    rec = np.array([[20.0, 5.0], [100.0, -30.0], [180.0, 0.0]])
    strict = predict_ppm((0.0, 0.0), 4.0, 3.0, 270.0, 1.0, 4, rec, z=1.0)
    clamp  = predict_ppm((0.0, 0.0), 4.0, 3.0, 270.0, 1.0, 4, rec, z=1.0,
                         clamp_to_table=True)
    assert np.allclose(strict, clamp, rtol=1e-12, atol=1e-12)


# ── the property the vectorization must preserve ─────────────────────────────

def test_vectorized_predict_ppm_matches_scalar_pipeline():
    """The fast array path must equal the documented scalar pipeline, point by point."""
    rec = np.array([
        [10.0, 0.0], [20.0, 10.0], [55.0, -20.0], [100.0, 30.0], [180.0, -5.0],
        [-50.0, 0.0], [0.0, 15.0], [4.0, -8.0],     # last three are upwind under 270°
    ])
    src, Q, u, wd, H, sc, z = (5.0, -3.0), 4.0, 3.0, 270.0, 1.0, 4, 1.0

    vec = predict_ppm(src, Q, u, wd, H, sc, rec, z=z)

    scalar = np.full(len(rec), CH4_BACKGROUND, dtype=float)
    for i, (rx, ry) in enumerate(rec):
        xw, yw = rotate_to_wind_frame(rx, ry, *src, wd)
        if xw <= 0.0:
            continue                                   # upwind → stays at background
        sy, sz = sigma_y(xw, sc), sigma_z(xw, sc)
        C = concentration_gm3(xw, yw, Q, u, H, sy, sz, z=z)
        scalar[i] = gm3_to_ppm_methane(C) + CH4_BACKGROUND

    assert np.allclose(vec, scalar, rtol=1e-9, atol=1e-12)

"""
test_feasibility.py — lock in the project's bottom-line detectability verdict.

feasibility.py turns the forward model into the headline answer ("the smallest leak
we can catch at the fence"). These tests pin that number and the classify/sweep/verdict
logic so Week-4's broader sweep — or any change to the σ table or noise floor — can't
silently move it without a test going red.
"""

import numpy as np
import pytest

from physics.feasibility import (
    SweepCell, classify, implied_Q_range, min_detectable_Q, sweep, verdict,
)
from physics.plume import CH4_BACKGROUND
from physics.sensor_sim import DETECT_K, SENSOR_NOISE_PPM, effective_noise_floor


def test_classify_boundaries():
    n = 0.30
    assert classify(3 * n, n, k=3.0) == "detectable"        # exactly k·noise → detectable
    assert classify(3 * n - 1e-9, n, k=3.0) == "marginal"
    assert classify(n, n, k=3.0) == "marginal"              # exactly noise → marginal
    assert classify(n - 1e-9, n, k=3.0) == "undetectable"
    assert classify(0.0, n, k=3.0) == "undetectable"


def test_sweep_cell_structure_and_excess():
    c = sweep([0.05], [50.0], [4])[0]
    assert isinstance(c, SweepCell)
    assert (c.Q, c.distance, c.stability) == (0.05, 50.0, 4)
    # excess is exactly ppm_total minus background, and verdict agrees with classify
    assert c.excess == pytest.approx(c.ppm_total - CH4_BACKGROUND)
    assert c.verdict == classify(c.excess, SENSOR_NOISE_PPM, DETECT_K)


def test_sweep_covers_every_combination():
    cells = sweep([0.1, 1.0], [25.0, 50.0], [3, 4])
    assert len({(c.Q, c.distance, c.stability) for c in cells}) == 2 * 2 * 2


def test_excess_falls_with_distance_and_rises_with_Q():
    cells = {(c.distance, c.Q): c for c in sweep([0.1, 0.5], [25.0, 100.0], [4])}
    assert cells[(100.0, 0.1)].excess < cells[(25.0, 0.1)].excess   # farther → weaker
    assert cells[(25.0, 0.1)].excess < cells[(25.0, 0.5)].excess    # bigger leak → stronger


def test_headline_min_detectable_Q_at_fence():
    # The documented bottom line: ~0.05 g/s at the 50 m fence under neutral Class D.
    Qs = [round(0.01 * i, 2) for i in range(1, 21)]   # 0.01 … 0.20 g/s
    cells = sweep(Qs, [50.0], [4])
    assert min_detectable_Q(cells, 50.0, 4) == pytest.approx(0.05)
    by_Q = {c.Q: c for c in cells}
    assert by_Q[0.05].verdict == "detectable"          # 0.924 ppm ≥ 0.90 threshold
    assert by_Q[0.04].verdict != "detectable"          # 0.739 ppm < 0.90 threshold


def test_min_detectable_Q_none_when_nothing_detectable():
    cells = sweep([1e-8], [200.0], [6])                # negligible leak, far, very stable
    assert min_detectable_Q(cells, 200.0, 6) is None


def test_verdict_infeasible_when_nothing_detectable():
    msg = verdict(sweep([1e-8], [200.0], [6]), 200.0, stability=6)
    assert "INFEASIBLE" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Two-part detection floor (Fix 2) — averaging beats random noise, not bias.
# ─────────────────────────────────────────────────────────────────────────────

def test_effective_noise_floor_defaults_are_backcompat():
    # bias 0, n_avg 1 → the raw random floor, so the 0.05 g/s headline is unchanged.
    assert effective_noise_floor(0.30, 0.0, 1) == pytest.approx(0.30)


def test_effective_noise_floor_random_part_averages_as_sqrt_n():
    assert effective_noise_floor(0.30, 0.0, 100) == pytest.approx(0.03)   # √100 = 10×
    # the critique's own example: 5 ppm RANDOM over 120 samples → ~0.46 ppm.
    assert effective_noise_floor(5.0, 0.0, 120) == pytest.approx(5.0 / np.sqrt(120))


def test_effective_noise_floor_bias_survives_averaging():
    # a 0.50 ppm bias is untouched by averaging and dominates at large N — this is the
    # whole point: you can't √N your way past a calibration bias.
    assert effective_noise_floor(0.30, 0.50, 100) == pytest.approx(np.hypot(0.03, 0.50))
    assert effective_noise_floor(0.30, 0.50, 10_000) == pytest.approx(0.50, abs=1e-3)


def test_effective_noise_floor_rejects_bad_inputs():
    with pytest.raises(ValueError):
        effective_noise_floor(0.30, 0.0, 0)


def test_effective_noise_floor_rho_reduces_effective_n():
    # AR(1) effective sample size: n_eff = n·(1−ρ)/(1+ρ). Hand-calc for ρ=0.5,
    # n=100 → n_eff = 100/3, so the random part shrinks by √(100/3), not √100.
    expected = np.hypot(0.30 / np.sqrt(100.0 / 3.0), 0.10)
    assert effective_noise_floor(0.30, 0.10, 100, rho=0.5) == pytest.approx(expected)
    # ρ=0 is the i.i.d. case — must match the two-arg behaviour exactly.
    assert (effective_noise_floor(0.30, 0.10, 100, rho=0.0)
            == pytest.approx(effective_noise_floor(0.30, 0.10, 100)))
    # ρ→1: n_eff floors at 1 — averaging buys nothing on a fully correlated record.
    assert effective_noise_floor(0.30, 0.0, 100, rho=0.99) <= 0.30
    assert effective_noise_floor(0.30, 0.0, 100, rho=0.99) >= 0.30 / np.sqrt(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity / implied-Q error bar (Fix 1) — wind & stability spread Q widely.
# ─────────────────────────────────────────────────────────────────────────────

def test_implied_Q_range_swings_with_wind_and_stability():
    qr = implied_Q_range(0.90, 50.0)
    assert qr.q_min < qr.q_max
    assert qr.q_min <= qr.q_median <= qr.q_max
    assert qr.swing_factor == pytest.approx(qr.q_max / qr.q_min)
    assert qr.swing_factor > 5.0          # wind + stability doubt alone is a big spread
    assert qr.n_unsolvable == 0           # every downwind cell is solvable at 50 m


def test_implied_Q_range_is_linear_in_observed_signal():
    a = implied_Q_range(0.90, 50.0)
    b = implied_Q_range(1.80, 50.0)
    # the plume is linear in Q: double the signal → double every implied Q, same spread.
    assert b.q_min == pytest.approx(2 * a.q_min)
    assert b.q_max == pytest.approx(2 * a.q_max)
    assert b.swing_factor == pytest.approx(a.swing_factor)


def test_implied_Q_range_rejects_nonpositive_signal():
    with pytest.raises(ValueError):
        implied_Q_range(0.0, 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict honesty upgrades — upper bound by default, field floor when asked.
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_default_keeps_headline_and_adds_upper_bound():
    cells = sweep([round(0.01 * i, 2) for i in range(1, 21)], [50.0], [4])
    msg = verdict(cells, 50.0)
    assert "0.05 g/s" in msg              # the documented headline is preserved …
    assert "0.90 ppm" in msg              # … quoting the 3 × 0.30 detection threshold …
    assert "upper bound" in msg           # … but now framed honestly as a bound
    assert "field floor" not in msg       # no field line unless asked for


def test_verdict_field_floor_line_appears_only_when_requested():
    cells = sweep([round(0.01 * i, 2) for i in range(1, 21)], [50.0], [4])
    field_msg = verdict(cells, 50.0, bias_floor=0.50, n_avg=10)
    assert "field floor" in field_msg
    # the field floor (random 0.30 over 10 samples + 0.50 bias) is higher than the lab
    # 0.30, so its 3σ threshold is higher and the detectable leak must be larger than 0.05.
    assert effective_noise_floor(0.30, 0.50, 10) > effective_noise_floor(0.30, 0.0, 1)


def test_verdict_field_floor_bias_only_has_no_phantom_averaging():
    # A bias floor with n_avg=1 (no averaging) must NOT emit the vacuous
    # "averaging 1 samples cuts ... to the same value" prose.
    cells = sweep([round(0.01 * i, 2) for i in range(1, 21)], [50.0], [4])
    msg = verdict(cells, 50.0, bias_floor=0.50)            # n_avg defaults to 1
    assert "field floor" in msg
    assert "averaging 1" not in msg
    assert "no amount of averaging removes" in msg

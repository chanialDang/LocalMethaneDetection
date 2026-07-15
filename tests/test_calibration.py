"""
test_calibration.py — Node-1 calibration pipeline: gas-free power-law kernel, the
dormant Mitchell kernel + its gate, and the mV→ppm slope.

Every expected value comes from an INDEPENDENT route — hand-written divider + power-law
algebra, an analytic derivative, a self-evident Mitchell reduction, or a conservation-style
round-trip — never from the function under test asserting against itself (CLAUDE.md's
bug-hunt doctrine). The Mitchell facts (−165 ppm on our hardware; needs gas) are the whole
reason the kernel ships dormant.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from physics import fieldtest, sensor_frontend as sf
from calibration.nodes.node1 import NODE1
from calibration.mitchell.paper_coeffs import PAPER_COEFFS_REFERENCE


# Node 1 circuit/bench truth, restated here so expectations don't read from the cfg loader.
VC, RL, R0, M = 5.0, 10_000.0, 79_960.0, 0.35
BG = 1.9
A_ANCHOR = BG ** M                       # gas-free anchor: A = background^m
V_BASE = NODE1["baseline_voltage_v"]     # 0.5558 V


@pytest.fixture(autouse=True)
def _restore_sf_globals():
    """apply_node writes module globals directly; snapshot + restore for test isolation."""
    names = ["SUPPLY_VOLTAGE_V", "LOAD_RESISTANCE_OHM", "R0_OHM", "POWERLAW_A",
             "POWERLAW_M", "BACKGROUND_PPM", "TCORR_PER_C", "RHCORR_PER_PCT",
             "KERNEL", "MITCHELL_COEFFS"]
    saved = {n: getattr(sf, n) for n in names}
    yield
    for n, v in saved.items():
        setattr(sf, n, v)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Anchor sanity + an independent power-law point
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_node_anchors_baseline_to_background():
    sf.apply_node(NODE1)
    # A is derived from the anchor, not stored.
    assert sf.POWERLAW_A == pytest.approx(A_ANCHOR, rel=1e-12)
    # Independent geometry: the divider at 0.5558 V yields Rs ≈ R0 (ratio ≈ 1).
    rs_base = RL * (VC - V_BASE) / V_BASE
    assert rs_base == pytest.approx(R0, rel=2e-4)
    # So the baseline reads the background, ~1.9 ppm.
    assert sf.voltage_to_ppm(V_BASE) == pytest.approx(BG, abs=2e-3)


def test_powerlaw_matches_independent_hand_calc_at_a_second_point():
    sf.apply_node(NODE1)
    v = 0.75                                     # a different voltage, in-range
    rs = RL * (VC - v) / v                       # divider solved for Rs (independent)
    ratio = rs / R0
    expected = (A_ANCHOR / ratio) ** (1.0 / M)   # invert Rs/R0 = A·C^(−m), hand-written
    assert sf.voltage_to_ppm(v) == pytest.approx(expected, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mitchell gate refuses without gas-fit coefficients
# ─────────────────────────────────────────────────────────────────────────────
def test_mitchell_gate_refuses_without_coeffs():
    sf.apply_node(NODE1)
    sf.KERNEL = "mitchell"
    sf.MITCHELL_COEFFS = None
    ready, missing = sf.calibration_status()
    assert ready is False and "MITCHELL_COEFFS" in missing
    with pytest.raises(ValueError) as exc:
        sf.voltage_to_ppm(V_BASE, 24.8, 48.8)
    assert "MITCHELL_COEFFS" in str(exc.value)


def test_parse_csv_refuses_voltage_when_mitchell_uncalibrated():
    sf.apply_node(NODE1)                          # circuit set, but…
    sf.KERNEL = "mitchell"
    sf.MITCHELL_COEFFS = None                      # …no gas-fit coeffs
    csv = "time,voltage,temp,humidity\n0,0.5558,24.8,48.8\n1,0.60,24.8,48.8\n"
    with pytest.raises(ValueError) as exc:
        fieldtest.parse_csv(csv)
    assert "not calibrated" in str(exc.value) and "MITCHELL_COEFFS" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Paper coefficients are unusable on our hardware, and never a default
# ─────────────────────────────────────────────────────────────────────────────
def test_paper_coeffs_give_impossible_negative_ppm():
    C = PAPER_COEFFS_REFERENCE
    v, T, H = V_BASE, 24.8, 48.8
    # Hand-evaluate Eq. 16 independently (M = C1 + C2·exp(...) − C7·ln((T+65)·V)).
    expo = C["C3"] * v - C["C4"] * math.log(T + 65) - C["C5"] * math.log(H)
    expected = C["C1"] + C["C2"] * math.exp(expo) - C["C7"] * math.log((T + 65) * v)
    got = float(sf._kernel_mitchell(v, T, H, C))
    assert got == pytest.approx(expected, rel=1e-9)
    assert expected < -100.0                       # ≈ −165 ppm: physically impossible


def test_sensor_frontend_never_imports_paper_coeffs():
    src = Path(sf.__file__).read_text()
    assert "paper_coeffs" not in src and "PAPER_COEFFS" not in src
    assert sf.MITCHELL_COEFFS is None              # no default coefficients baked in


# ─────────────────────────────────────────────────────────────────────────────
# 4. mV→ppm slope vs an independent analytic derivative
# ─────────────────────────────────────────────────────────────────────────────
def test_ppm_per_volt_matches_analytic_derivative():
    sf.apply_node(NODE1)
    # C(V) = u^(1/m) with u = (A·R0/R_L)·V/(V_c−V); dC/dV analytic (independent route).
    k = A_ANCHOR * R0 / RL
    u = k * V_BASE / (VC - V_BASE)
    dudV = k * VC / (VC - V_BASE) ** 2
    dCdV = (1.0 / M) * u ** (1.0 / M - 1.0) * dudV
    assert sf.ppm_per_volt(V_BASE) == pytest.approx(dCdV, rel=1e-4)
    # And the make-or-break conversion: 1.65 mV → ~0.018 ppm random-electrical floor.
    floor = abs(dCdV) * (NODE1["electrical_noise_mv"] / 1000.0)
    assert floor == pytest.approx(0.018, abs=0.005)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kernel-swap isolation — shared plumbing, not a fork
# ─────────────────────────────────────────────────────────────────────────────
def test_kernels_agree_when_made_to_agree():
    sf.apply_node(NODE1)
    v0 = 0.70
    p = float(sf.voltage_to_ppm(v0))               # power-law result at v0
    # A Mitchell coeff set that returns C1 for every valid sample (C2=0, C7=0 ⇒ M=C1).
    flat = dict(C1=p, C2=0.0, C3=0.0, C4=0.0, C5=0.0, C7=0.0)
    got = float(sf.voltage_to_ppm(v0, 24.8, 48.8, kernel="mitchell", coeffs=flat))
    assert got == pytest.approx(p, rel=1e-12)       # same surrounding plumbing → same answer
    # And both kernels honor the SAME validity mask: V ≥ V_c → NaN either way.
    assert math.isnan(sf.voltage_to_ppm(VC, 24.8, 48.8, kernel="mitchell", coeffs=flat))
    assert math.isnan(sf.voltage_to_ppm(VC))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Round-trip under Node 1 constants (conservation-style)
# ─────────────────────────────────────────────────────────────────────────────
def test_round_trip_under_node1_constants():
    sf.apply_node(NODE1)
    c0 = np.array([1.9, 2.0, 5.0, 12.0, 40.0])
    v = sf.ppm_to_voltage(c0)
    assert np.all((v > 0.0) & (v < VC))            # stays in the divider's valid region
    c1 = sf.voltage_to_ppm(v)
    assert np.allclose(c1, c0, rtol=1e-10, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 7. apply_node lands on the GATED globals parse_csv reads
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_node_unblocks_parse_csv_voltage_path():
    sf.apply_node(NODE1)
    assert sf.calibration_status() == (True, [])
    csv = "time,voltage\n0,0.5558\n1,0.5558\n2,0.5558\n"
    out = fieldtest.parse_csv(csv)
    assert out["columns"]["voltage"] == "voltage" and out["columns"]["ppm"] is None
    assert np.allclose(out["ppm"], BG, atol=2e-3)  # clean-air voltage → ~1.9 ppm

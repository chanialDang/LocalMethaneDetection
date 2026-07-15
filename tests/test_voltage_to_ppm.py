"""
test_voltage_to_ppm.py — the sensor front-end: raw ADC voltage → ppm.

Every expected value here comes from an INDEPENDENT route — a hand-computed
divider + power-law evaluation, a conservation-style round-trip, or a sign/
monotonicity argument — never from the function under test asserting against
itself (per CLAUDE.md's bug-hunt practice).

The clean anchor points all use v_c=5 V, R_L=R_0=10 kΩ, A=1, m=0.5, chosen so the
algebra lands on round numbers:

    V_out=2.500 V → R_s = 10k·(5−2.5)/2.5 = 10 kΩ → R_s/R_0 = 1.00 → C = 1^(1/0.5) = 1 ppm
    V_out=3.333 V → R_s =  5 kΩ            → ratio = 0.50 → C = 2^2  =  4 ppm
    V_out=4.000 V → R_s = 2.5 kΩ           → ratio = 0.25 → C = 4^2  = 16 ppm
"""
import math

import numpy as np
import pytest

from physics import fieldtest, sensor_frontend as sf

# A fully-specified calibration passed explicitly so these tests never depend on
# (or mutate) the module's None defaults.
CAL = dict(v_c=5.0, r_l=10_000.0, r0=10_000.0, a=1.0, m=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORWARD — hand-computed divider + power law
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v_out, expected_ppm", [
    (2.5, 1.0),
    (10.0 / 3.0, 4.0),
    (4.0, 16.0),
])
def test_forward_matches_hand_calc(v_out, expected_ppm):
    assert sf.voltage_to_ppm(v_out, **CAL) == pytest.approx(expected_ppm, rel=1e-9)


def test_forward_independent_general_formula():
    # A non-round point checked against the algebra written out separately.
    v_c, r_l, r0, a, m = 5.0, 8_000.0, 6_500.0, 1.3, 0.42
    v_out = 2.9
    rs = r_l * (v_c - v_out) / v_out          # divider solved for R_s
    ratio = rs / r0
    expected = (a / ratio) ** (1.0 / m)       # invert R_s/R_0 = A·C^(−m)
    got = sf.voltage_to_ppm(v_out, v_c=v_c, r_l=r_l, r0=r0, a=a, m=m)
    assert got == pytest.approx(expected, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUND-TRIP — ppm → voltage → ppm (conservation-style, independent inverse)
# ─────────────────────────────────────────────────────────────────────────────
def test_round_trip_recovers_ppm():
    c0 = np.array([1.9, 2.0, 5.0, 12.0, 40.0])
    v = sf.ppm_to_voltage(c0, **CAL)
    c1 = sf.voltage_to_ppm(v, **CAL)
    assert np.allclose(c1, c0, rtol=1e-10, atol=1e-10)


def test_round_trip_with_weather_correction():
    # A non-identity T/RH factor must cancel exactly across the round-trip.
    kw = dict(**CAL, tcorr=0.01, rhcorr=-0.005)
    c0 = np.array([2.0, 8.0, 25.0])
    T = np.array([10.0, 20.0, 35.0])
    RH = np.array([30.0, 55.0, 80.0])
    v = sf.ppm_to_voltage(c0, T, RH, **kw)
    c1 = sf.voltage_to_ppm(v, T, RH, **kw)
    assert np.allclose(c1, c0, rtol=1e-10, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MONOTONICITY / SIGN — physics direction, no reference values needed
# ─────────────────────────────────────────────────────────────────────────────
def test_more_methane_gives_higher_output_voltage():
    # n-type MOX: more CH4 → lower R_s → (divider) higher V_out.
    c = np.array([2.0, 5.0, 10.0, 40.0])
    v = sf.ppm_to_voltage(c, **CAL)
    assert np.all(np.diff(v) > 0)


def test_ppm_increases_monotonically_with_voltage():
    v = np.linspace(2.0, 4.5, 20)             # inside 0 < V_out < V_c
    ppm = sf.voltage_to_ppm(v, **CAL)
    assert np.all(np.diff(ppm) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GUARDS — bad samples → NaN (never crash); bad config → ValueError
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad_v", [0.0, -1.0, 5.0, 6.0, np.nan, np.inf, -np.inf])
def test_out_of_range_voltage_is_nan_not_crash(bad_v):
    # V_out ≤ 0 or ≥ V_c (=5) is outside the divider's valid region; non-finite too.
    assert math.isnan(sf.voltage_to_ppm(bad_v, **CAL))


def test_bad_sample_isolated_in_an_array():
    v = np.array([2.5, 0.0, 3.0, 5.0, 4.0])   # positions 1 and 3 are invalid
    ppm = sf.voltage_to_ppm(v, **CAL)
    assert np.isnan(ppm[1]) and np.isnan(ppm[3])
    assert np.all(np.isfinite(ppm[[0, 2, 4]]))


@pytest.mark.parametrize("bad_m", [0.0, -0.5])
def test_nonphysical_m_raises(bad_m):
    with pytest.raises(ValueError):
        sf.voltage_to_ppm(2.5, v_c=5.0, r_l=1e4, r0=1e4, a=1.0, m=bad_m)


def test_unset_constants_raise_naming_them(monkeypatch):
    monkeypatch.setattr(sf, "SUPPLY_VOLTAGE_V", None)
    monkeypatch.setattr(sf, "LOAD_RESISTANCE_OHM", None)
    monkeypatch.setattr(sf, "R0_OHM", None)
    with pytest.raises(ValueError) as exc:
        sf.voltage_to_ppm(2.5)                # no explicit constants → uses (None) globals
    msg = str(exc.value)
    assert "SUPPLY_VOLTAGE_V" in msg and "LOAD_RESISTANCE_OHM" in msg and "R0_OHM" in msg


def test_calibration_status_reflects_globals(monkeypatch):
    monkeypatch.setattr(sf, "SUPPLY_VOLTAGE_V", None)
    monkeypatch.setattr(sf, "LOAD_RESISTANCE_OHM", None)
    monkeypatch.setattr(sf, "R0_OHM", None)
    ready, missing = sf.calibration_status()
    assert ready is False and set(missing) == {
        "SUPPLY_VOLTAGE_V", "LOAD_RESISTANCE_OHM", "R0_OHM"}
    monkeypatch.setattr(sf, "SUPPLY_VOLTAGE_V", 5.0)
    monkeypatch.setattr(sf, "LOAD_RESISTANCE_OHM", 1e4)
    monkeypatch.setattr(sf, "R0_OHM", 1e4)
    assert sf.calibration_status() == (True, [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPERATURE / HUMIDITY correction factor
# ─────────────────────────────────────────────────────────────────────────────
def test_th_factor_is_identity_by_default():
    # Default coefficients are 0 → factor is exactly 1 regardless of T/RH.
    assert sf._th_correction_factor(temperature=35.0, humidity=90.0) == 1.0


def test_th_factor_matches_linear_form_when_set():
    # factor = (1 + tcorr·ΔT)·(1 + rhcorr·ΔRH), ΔX measured from the reference.
    f = sf._th_correction_factor(temperature=30.0, humidity=70.0,
                                 tcorr=0.01, rhcorr=-0.005, t0=20.0, h0=50.0)
    expected = (1 + 0.01 * (30 - 20)) * (1 + (-0.005) * (70 - 50))
    assert f == pytest.approx(expected, rel=1e-12)


def test_th_factor_is_nan_safe_per_sample():
    # A missing (NaN) T sample contributes no correction (factor 1) for that slot.
    T = np.array([20.0, np.nan, 40.0])
    f = sf._th_correction_factor(temperature=T, tcorr=0.02, t0=20.0)
    assert f[1] == pytest.approx(1.0)          # NaN → no shift
    assert f[2] == pytest.approx(1 + 0.02 * 20)


def test_at_reference_conditions_factor_is_one():
    # ΔT = ΔRH = 0 → identity even with nonzero coefficients.
    v = sf.voltage_to_ppm(2.5, temperature=20.0, humidity=50.0,
                          tcorr=0.05, rhcorr=0.03, **CAL)
    assert v == pytest.approx(1.0, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VECTORIZED shape / scalar-vs-array
# ─────────────────────────────────────────────────────────────────────────────
def test_scalar_in_scalar_out():
    out = sf.voltage_to_ppm(2.5, **CAL)
    assert isinstance(out, float)


def test_array_in_array_out():
    v = np.array([2.5, 3.0, 4.0])
    out = sf.voltage_to_ppm(v, **CAL)
    assert isinstance(out, np.ndarray) and out.shape == v.shape


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTEGRATION — parse_csv gated wire-in of a voltage column
# ─────────────────────────────────────────────────────────────────────────────
def _calibrate(monkeypatch):
    monkeypatch.setattr(sf, "SUPPLY_VOLTAGE_V", 5.0)
    monkeypatch.setattr(sf, "LOAD_RESISTANCE_OHM", 10_000.0)
    monkeypatch.setattr(sf, "R0_OHM", 10_000.0)
    monkeypatch.setattr(sf, "POWERLAW_A", 1.0)
    monkeypatch.setattr(sf, "POWERLAW_M", 0.5)


def test_parse_csv_converts_voltage_when_calibrated(monkeypatch):
    _calibrate(monkeypatch)
    # V_out 2.5 / 3.333 / 4.0 → 1 / 4 / 16 ppm by the anchor table above.
    csv = "time,voltage\n0,2.5\n1,3.3333333333\n2,4.0\n"
    out = fieldtest.parse_csv(csv)
    assert out["columns"]["voltage"] == "voltage"
    assert out["columns"]["ppm"] is None
    assert np.allclose(out["ppm"], [1.0, 4.0, 16.0], rtol=1e-6)
    assert any("voltage" in w.lower() and "ppm" in w.lower() for w in out["warnings"])


def test_parse_csv_voltage_column_variants(monkeypatch):
    _calibrate(monkeypatch)
    for header in ("vout", "ADC", "V_out", "Voltage (V)"):
        csv = f"{header}\n2.5\n4.0\n"
        out = fieldtest.parse_csv(csv)
        assert out["columns"]["voltage"] == header
        assert np.allclose(out["ppm"], [1.0, 16.0], rtol=1e-6)


def test_parse_csv_refuses_voltage_when_uncalibrated(monkeypatch):
    monkeypatch.setattr(sf, "SUPPLY_VOLTAGE_V", None)
    monkeypatch.setattr(sf, "LOAD_RESISTANCE_OHM", None)
    monkeypatch.setattr(sf, "R0_OHM", None)
    csv = "time,voltage\n0,2.5\n1,3.0\n"
    with pytest.raises(ValueError) as exc:
        fieldtest.parse_csv(csv)
    assert "not calibrated" in str(exc.value) and "SUPPLY_VOLTAGE_V" in str(exc.value)


def test_ppm_column_wins_over_voltage(monkeypatch):
    _calibrate(monkeypatch)
    csv = "ppm,voltage\n2.0,2.5\n3.0,4.0\n"
    out = fieldtest.parse_csv(csv)
    assert out["columns"]["ppm"] == "ppm"
    assert np.allclose(out["ppm"], [2.0, 3.0])   # the real ppm column, unconverted
    assert any("ignored" in w.lower() for w in out["warnings"])

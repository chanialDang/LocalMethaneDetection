"""
tests/test_processing.py — Proving the signal-processing code works.

═══════════════════════════════════════════════════════════════════════════════
THE IDEA (plain English)
───────────────────────────────────────────────────────────────────────────────
We have no real sensor yet, so we make FAKE data where we already know the right
answer (sensor_sim.py), run it through the cleaning code (processing.py), and
check the code recovers what we put in. If it does, we trust the code.

Each test below is one "Test N" with a known input and a known expected result.
Run them all with:   pytest tests/ -v
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np

from physics.processing import (
    moving_average,
    subtract_baseline,
    temp_humidity_correct,
    detect_pattern,
)
from physics.sensor_sim import synthetic_timeseries, make_plume_event


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Averaging makes noise smaller without moving the real value
# ─────────────────────────────────────────────────────────────────────────────

def test_1_averaging_reduces_noise():
    """A noisy flat signal: averaging should cut the noise but keep the mean."""
    rng = np.random.default_rng(1)
    n = 2000
    true_level = 5.0
    signal = true_level + rng.normal(0.0, 1.0, n)   # flat 5.0 buried in noise

    smoothed = moving_average(signal, window=51)

    # Noise (scatter) should drop a lot — averaging ~51 points cuts it by ~√51.
    assert smoothed.std() < 0.5 * signal.std()
    # The real underlying value must be preserved (not biased away).
    assert abs(smoothed.mean() - true_level) < 0.1
    # Output keeps the same length as the input.
    assert len(smoothed) == n


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Baseline subtraction removes slow drift, keeps the plume bump
# ─────────────────────────────────────────────────────────────────────────────

def test_2_baseline_subtraction_removes_drift():
    """A short plume pulse riding on a slow ramp: recover the pulse, lose the ramp."""
    n = 1000
    drift = np.linspace(0.0, 2.0, n)         # slow wandering baseline
    pulse = np.zeros(n)
    pulse[450:470] = 3.0                      # short plume event, height 3.0
    signal = drift + pulse

    # Baseline window much wider than the 20-sample pulse so the pulse is a
    # minority inside it and gets ignored by the median.
    excess, baseline = subtract_baseline(signal, window=151, method="median")

    # Away from the pulse, the drift should be fully removed (excess ≈ 0).
    away = np.r_[100:400, 600:900]
    assert np.abs(excess[away]).max() < 0.15
    # On the pulse, the recovered height should be close to the true 3.0.
    assert abs(excess[452:468].mean() - 3.0) < 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Temperature/humidity correction flattens weather-driven drift
# ─────────────────────────────────────────────────────────────────────────────

def test_3_temp_humidity_correction():
    """A reading that only tracks temp & humidity should go flat after correction."""
    rng = np.random.default_rng(3)
    n = 600
    t = np.arange(n)
    # Non-collinear temperature and humidity (different shapes + independent jitter)
    temperature = 20.0 + 5.0 * np.sin(2 * np.pi * t / n) + rng.normal(0, 0.5, n)
    humidity = 50.0 + 10.0 * np.cos(2 * np.pi * t / n) + rng.normal(0, 1.0, n)

    a, b, c = 0.05, 0.02, 2.0                 # true weather sensitivities + offset
    signal = a * temperature + b * humidity + c + rng.normal(0, 0.02, n)

    corrected, coeffs = temp_humidity_correct(signal, temperature, humidity)

    # The wobble that tracked weather should be almost entirely gone.
    assert corrected.std() < 0.2 * signal.std()
    # The fit should recover the true sensitivities reasonably well.
    assert abs(coeffs["a_temp"] - a) < 0.01
    assert abs(coeffs["b_humid"] - b) < 0.01


def test_3b_constant_humidity_is_rank_safe():
    # F10: a constant humidity column is collinear with the intercept, so the old
    # rank-deficient lstsq handed it a meaningless min-norm coefficient. The dropped
    # regressor must report 0, the temperature correction must still be exact, and the
    # varying weather part must be fully removed.
    n = 400
    t = np.arange(n)
    temperature = 20.0 + 3.0 * np.sin(2 * np.pi * t / n)   # varies
    humidity = np.full(n, 50.0)                            # constant → unidentifiable
    a, b0, c = 0.05, 0.02, 1.9
    signal = a * temperature + b0 * humidity + c           # noiseless

    corrected, coeffs = temp_humidity_correct(signal, temperature, humidity)

    assert coeffs["b_humid"] == 0.0            # dropped, not a min-norm artefact
    assert abs(coeffs["a_temp"] - a) < 1e-6    # temp sensitivity still exact
    assert corrected.std() < 1e-6              # all the varying part removed


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Pattern detection: finds a real event, ignores pure noise
# ─────────────────────────────────────────────────────────────────────────────

def test_4_pattern_detection_true_and_false():
    """Must flag a sustained plume AND not cry wolf on noise alone."""
    rng = np.random.default_rng(4)
    n = 600
    noise_std = 0.3

    # --- Real event present ---
    event = make_plume_event(n, event_ppm=2.0)        # 0, then 2.0 bump, then 0
    excess_with_event = event + rng.normal(0, 0.2, n)
    det_true = detect_pattern(excess_with_event, noise_std=noise_std, k=3.0, min_run=5)
    assert det_true.detected is True
    assert det_true.confidence > 3.0
    # detected window should overlap where the true event actually is
    true_region = np.where(event > 0)[0]
    assert det_true.start_idx >= true_region[0] - 5
    assert det_true.end_idx <= true_region[-1] + 5

    # --- No event, just noise ---
    excess_noise_only = rng.normal(0, noise_std, n)
    det_false = detect_pattern(excess_noise_only, noise_std=noise_std, k=3.0, min_run=5)
    assert det_false.detected is False


def test_4b_detect_pattern_zero_noise_is_clean():
    # σ→0 (a noiseless sensor) must NOT divide-by-zero: SNR is infinite, but the code
    # should return inf confidence cleanly with no RuntimeWarning and no NaN.
    import warnings
    excess = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any RuntimeWarning → test failure
        det = detect_pattern(excess, noise_std=0.0, k=3.0, min_run=3)
    assert det.detected is True
    assert det.confidence == float("inf")         # infinite SNR, not NaN
    assert not np.isnan(det.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Full pipeline recovers a known plume from messy raw data
# ─────────────────────────────────────────────────────────────────────────────

def test_5_full_pipeline_recovers_known_plume():
    """
    End-to-end closed loop: build messy raw data with a known 4 ppm plume, then
    run the whole cleaning chain and check we get the plume back.
    """
    n = 1200
    true = make_plume_event(n, event_ppm=4.0)
    data = synthetic_timeseries(true, seed=42)        # adds bkg, drift, weather, noise

    # 1) remove temperature/humidity influence
    corrected, _ = temp_humidity_correct(data.raw, data.temperature, data.humidity)
    # 2) remove the slow background/drift floor → leaves the plume excess
    excess, _ = subtract_baseline(corrected, window=601, method="percentile", percentile=25)
    # 3) smooth away noise
    smoothed = moving_average(excess, window=31)
    # 4) decide if a plume is present
    det = detect_pattern(smoothed, noise_std=data.noise_ppm, k=3.0, min_run=10)

    assert det.detected is True

    # Recovered plume height during the event should be near the true 4.0 ppm.
    event_region = np.where(true > 0)[0]
    recovered = smoothed[event_region].mean()
    assert 3.0 < recovered < 5.0

    # Away from the event (well clear of smoothing edges) excess should be ~0.
    quiet = smoothed[:150]
    assert np.abs(quiet).mean() < 0.6

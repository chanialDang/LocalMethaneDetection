"""
test_accuracy.py — pins the accuracy / self-assessment engine (physics/accuracy.py).

Covers the Accuracy Protocol end to end on known-answer synthetic data:
  1. CHARACTERIZE — the self-measured random noise recovers the injected 0.30 ppm.
  2. VALIDATE     — recovery_metrics scores a known event as a true positive.
  4a. BOUND (emp) — detection_limit reproduces the ~0.05 g/s fenceline figure and
                    its averaging curve bottoms out at the bias floor.
  4b. BOUND (CRB) — the Cramér-Rao bound tightens with more receptors and scales
                    linearly with the noise σ.
  3+5. GRADE      — a clean, strong test scores above a noisy, weak one.

All synthetic, no network, no DB.
"""
import numpy as np

from physics import accuracy, feasibility
from physics.fieldtest import make_sample_readings, process_fieldtest
from physics.sensor_sim import (
    SENSOR_NOISE_PPM,
    make_plume_event,
    synthetic_timeseries,
)


def _processed_sample(noise=SENSOR_NOISE_PPM, event_ppm=4.0, n=600, seed=0):
    """One synthetic field test run through the standard cleaning pipeline."""
    s = make_sample_readings(n=n, event_ppm=event_ppm, noise_ppm=noise, seed=seed)
    return process_fieldtest(
        s["ppm"], temperature=s["temperature"], humidity=s["humidity"],
        time=s["time"], noise_ppm=noise,
    )


# ── 1. CHARACTERIZE ──────────────────────────────────────────────────────────
def test_estimate_noise_recovers_injected_random():
    res = _processed_sample(noise=0.30)
    ne = accuracy.estimate_noise_floor(res["raw"] - res["baseline"], res["detection"])
    # The measured random noise should land near the injected 0.30 ppm (1σ).
    assert 0.20 <= ne.random_ppm <= 0.42
    assert ne.bias_ppm >= 0.0
    assert ne.n_quiet > 50


def test_random_noise_tracks_injection():
    lo = _processed_sample(noise=0.15)
    hi = _processed_sample(noise=0.60)
    ne_lo = accuracy.estimate_noise_floor(lo["raw"] - lo["baseline"], lo["detection"])
    ne_hi = accuracy.estimate_noise_floor(hi["raw"] - hi["baseline"], hi["detection"])
    assert ne_hi.random_ppm > ne_lo.random_ppm


# ── 2. VALIDATE ──────────────────────────────────────────────────────────────
def test_recovery_metrics_true_positive():
    n = 600
    true = make_plume_event(n, event_ppm=4.0)
    data = synthetic_timeseries(true, noise_ppm=0.30, seed=0)
    res = process_fieldtest(
        data.raw, temperature=data.temperature, humidity=data.humidity,
        time=data.time, noise_ppm=0.30,
    )
    rep = accuracy.recovery_metrics(true, res["excess"], res["detection"],
                                    time=res["time"])
    assert rep.outcome == "true positive"
    assert rep.detected and rep.true_detected
    assert rep.rmse_ppm < 4.0                 # recovered the event, not garbage
    assert rep.pct_recovery is not None
    assert 50.0 <= rep.pct_recovery <= 200.0


def test_recovery_metrics_true_negative_when_no_event():
    n = 400
    true = np.zeros(n)                          # no plume at all
    data = synthetic_timeseries(true, noise_ppm=0.30, seed=1)
    res = process_fieldtest(data.raw, temperature=data.temperature,
                            humidity=data.humidity, time=data.time, noise_ppm=0.30)
    rep = accuracy.recovery_metrics(true, res["excess"], res["detection"])
    assert not rep.true_detected
    assert rep.outcome in ("true negative", "false positive")
    assert rep.pct_recovery is None            # no true peak to divide by


# ── 4a. BOUND (empirical) ────────────────────────────────────────────────────
def test_detection_limit_matches_fenceline_figure():
    dl = accuracy.detection_limit(random_ppm=0.30, bias_ppm=0.0,
                                  distance=50.0, k=3.0, u=2.0, stability=4)
    assert abs(dl.lod_ppm - 0.90) < 1e-9                  # 3 × 0.30
    assert 0.02 <= dl.min_detectable_Q <= 0.08           # brackets the ~0.05 g/s figure
    assert dl.swing_factor > 1.0                          # there IS a wind/stability error bar
    assert dl.crossover_n is None                         # no bias → averaging always helps


def test_averaging_lowers_floor_until_bias():
    dl = accuracy.detection_limit(random_ppm=0.30, bias_ppm=0.10,
                                  distance=50.0, u=2.0, stability=4)
    floors = [c["floor_ppm"] for c in dl.curve]
    assert floors[0] > floors[-1]                         # averaging helps …
    assert floors[-1] >= 0.10 - 1e-9                      # … but never below the bias
    assert dl.crossover_n == (0.30 / 0.10) ** 2           # = 9 samples


def test_detection_limit_consistent_with_feasibility():
    # The empirical min detectable Q should agree with feasibility's own back-solve.
    dl = accuracy.detection_limit(random_ppm=0.30, bias_ppm=0.0,
                                  distance=50.0, k=3.0, u=2.0, stability=4)
    per_unit_Q = feasibility._excess_at(50.0, 1.0, 2.0, 4, clamp=True)
    assert np.isclose(dl.min_detectable_Q, 0.90 / per_unit_Q, rtol=1e-9)


# ── 3 + 5. GRADE ─────────────────────────────────────────────────────────────
def test_accuracy_report_grades_clean_above_noisy():
    meta = {"sensor_distance_m": 50.0, "wind_speed": 2.0, "stability_class": 4}
    clean = accuracy.accuracy_report(_processed_sample(noise=0.20, event_ppm=8.0), meta)
    noisy = accuracy.accuracy_report(_processed_sample(noise=1.50, event_ppm=1.0), meta)
    assert clean["score"] > noisy["score"]
    assert clean["grade"] in {"A", "B", "C", "D", "F"}
    # Report is self-contained and JSON-safe in shape.
    for key in ("score", "grade", "recommendation", "detection_limit",
                "measured_random_ppm", "measured_floor_ppm", "geometry"):
        assert key in clean


def test_accuracy_report_without_meta_or_truth():
    # No metadata, no ground truth → still produces a full report (recovery null).
    rep = accuracy.accuracy_report(_processed_sample())
    assert rep["recovery"] is None
    assert rep["score_breakdown"]["recovery"] is None
    assert 0.0 <= rep["score"] <= 100.0


def test_accuracy_report_includes_recovery_with_truth():
    n = 600
    true = make_plume_event(n, event_ppm=5.0)
    data = synthetic_timeseries(true, noise_ppm=0.30, seed=0)
    res = process_fieldtest(data.raw, temperature=data.temperature,
                            humidity=data.humidity, time=data.time, noise_ppm=0.30)
    rep = accuracy.accuracy_report(res, meta=None, true_ppm=true)
    assert rep["recovery"] is not None
    assert rep["recovery"]["outcome"] == "true positive"
    assert rep["score_breakdown"]["recovery"] is not None


# ── 4b. BOUND (Cramér-Rao) ───────────────────────────────────────────────────
_CRB_KW = dict(src_pos=(0.0, 0.0), Q=5.0, u=3.0, wind_dir_deg=270.0,
               H=1.0, stability_class=3)


def test_crb_tighter_with_more_receptors():
    few = np.array([[50.0, 0.0], [60.0, 8.0], [70.0, -8.0], [80.0, 4.0]])
    rng = np.random.default_rng(0)
    many = np.column_stack([rng.uniform(40.0, 120.0, 40),
                            rng.uniform(-25.0, 25.0, 40)])
    b_few = accuracy.crb_source_bound(receptors=few, sigma_ppm=0.30, **_CRB_KW)
    b_many = accuracy.crb_source_bound(receptors=many, sigma_ppm=0.30, **_CRB_KW)
    assert b_many.std["Q"] < b_few.std["Q"]
    assert b_many.std["x"] < b_few.std["x"]
    assert b_many.n_receptors == 40


def test_crb_scales_linearly_with_sigma():
    recs = np.column_stack([np.linspace(40.0, 120.0, 20), np.zeros(20)])
    b1 = accuracy.crb_source_bound(receptors=recs, sigma_ppm=0.30,
                                   params=("Q",), **_CRB_KW)
    b2 = accuracy.crb_source_bound(receptors=recs, sigma_ppm=0.60,
                                   params=("Q",), **_CRB_KW)
    # cov ∝ σ² ⇒ std ∝ σ: doubling the noise doubles the best-possible 1σ.
    assert np.isclose(b2.std["Q"], 2.0 * b1.std["Q"], rtol=1e-6)

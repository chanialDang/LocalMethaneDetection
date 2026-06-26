"""
test_fieldtest.py — CSV parsing + the field-test cleaning pipeline + interpretation.

No network: the only AI-touching tests run with OPENAI_API_KEY removed, exercising
the guaranteed template fallback. The detection test uses sensor_sim to build a
record with a KNOWN plume event, then checks the pipeline recovers it — the same
ground-truth trick the processing tests use.
"""
import numpy as np
import pytest

from physics import fieldtest
from ui.explain import (
    interpret_fieldtest,
    summarize_fieldtest,
    template_fieldtest_explanation,
)


# ─── parse_csv ──────────────────────────────────────────────────────────────────
def test_parse_full_csv():
    text = "time,ppm,temperature,humidity\n0,1.9,20,55\n1,2.4,20.1,54\n2,3.0,20.2,54\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 3
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.0])
    assert out["time"] is not None and out["temperature"] is not None
    assert out["humidity"] is not None


def test_parse_accepts_header_synonyms():
    # ch4 / t / temp / rh should all be recognised
    text = "t,ch4,temp,rh\n0,2.0,21,50\n1,2.2,21,50\n"
    out = fieldtest.parse_csv(text)
    assert np.allclose(out["ppm"], [2.0, 2.2])
    assert out["time"] is not None and out["temperature"] is not None
    assert out["humidity"] is not None
    assert out["columns"]["ppm"] == "ch4"


def test_parse_ppm_only():
    out = fieldtest.parse_csv("ppm\n1.9\n2.1\n2.0\n")
    assert out["n"] == 3
    assert out["time"] is None
    assert out["temperature"] is None and out["humidity"] is None


def test_parse_skips_nonnumeric_rows_and_partial_weather_is_dropped():
    # row "x" has no number → skipped; humidity has a gap → whole column dropped.
    text = "time,ppm,humidity\n0,1.9,55\n1,x,55\n2,2.5,\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2                       # the 'x' row was skipped
    assert np.allclose(out["ppm"], [1.9, 2.5])
    assert out["humidity"] is None             # partial column → unusable → None


def test_parse_no_ppm_column_raises():
    with pytest.raises(ValueError):
        fieldtest.parse_csv("time,temperature\n0,20\n1,21\n")


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        fieldtest.parse_csv("")


# ─── process_fieldtest ──────────────────────────────────────────────────────────
def test_raw_equals_baseline_plus_excess_without_smoothing():
    # With no weather correction and smooth_window=1, the frame identity is exact.
    rng = np.random.default_rng(0)
    ppm = 1.9 + rng.normal(0, 0.1, size=300)
    res = fieldtest.process_fieldtest(ppm, smooth_window=1)
    assert np.allclose(res["raw"], res["baseline"] + res["excess"])
    assert res["weather_corrected"] is False


def test_pipeline_detects_known_event():
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0, seed=0)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    det = res["detection"]
    assert det.detected is True
    assert det.confidence > 5.0                # 4 ppm event over a 0.30 floor
    assert res["weather_corrected"] is True
    assert res["threshold"] == pytest.approx(0.90)


def test_flat_record_detects_nothing():
    rng = np.random.default_rng(1)
    ppm = 1.9 + rng.normal(0, 0.05, size=400)  # pure background + tiny noise
    res = fieldtest.process_fieldtest(ppm, noise_ppm=0.30)
    assert res["detection"].detected is False


def test_process_empty_raises():
    with pytest.raises(ValueError):
        fieldtest.process_fieldtest(np.array([]))


# ─── summarize + interpret ──────────────────────────────────────────────────────
def _detected_facts():
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0, seed=0)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    meta = {"name": "Custer #1", "site": "Custer Road Transfer Station",
            "sensor_distance_m": 50.0}
    return summarize_fieldtest(res, meta)


def test_summarize_shapes_and_event_window():
    facts = _detected_facts()
    assert facts["kind"] == "fieldtest"
    assert facts["detected"] is True
    assert facts["name"] == "Custer #1"
    for k in ("raw_peak_ppm", "excess_peak_ppm", "baseline_mean_ppm",
              "confidence_sigmas", "threshold", "duration_s"):
        assert isinstance(facts[k], float)
    # The event is at ~40% of a 600 s record → starts in the low-mid hundreds.
    assert 150.0 < facts["event_start_s"] < 300.0
    assert facts["event_end_s"] > facts["event_start_s"]


def test_template_explains_detected_and_undetected():
    facts = _detected_facts()
    txt = template_fieldtest_explanation(facts)
    assert "detected" in txt.lower() and "Custer #1" in txt
    assert len(txt) > 120

    undetected = dict(facts, detected=False, excess_peak_ppm=0.2,
                      event_start_s=None, event_end_s=None, event_duration_s=None)
    txt2 = template_fieldtest_explanation(undetected)
    assert "threshold" in txt2.lower()


def test_interpret_falls_back_to_template_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text, source = interpret_fieldtest(_detected_facts(), use_ai=True)
    assert source == "template"
    assert isinstance(text, str) and len(text) > 120


# ─── aggregate_for_inversion (averaging reframe) ─────────────────────────────────
def test_aggregate_returns_weighted_mean_over_event():
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0, seed=0)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    pt = fieldtest.aggregate_for_inversion(res)
    assert pt.detected is True
    # Window is the detected event, and the mean excess is a real positive signal.
    assert pt.n_window > 1
    assert pt.window[1] > pt.window[0]
    assert pt.mean_excess_ppm > 1.0
    # σ of the mean must be below the single-sample floor (random part averaged down),
    # and never below the non-averageable bias.
    assert pt.bias_ppm <= pt.sigma_ppm < pt.random_ppm + pt.bias_ppm + 1e-9


def test_aggregate_sigma_shrinks_with_window():
    # A longer averaging window lowers the random part → smaller σ of the mean
    # (until bias dominates). Build a long quiet+event record and compare windows.
    sample = fieldtest.make_sample_readings(n=1200, event_ppm=4.0, seed=1)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    short = fieldtest.aggregate_for_inversion(res, window_s=10)
    longw = fieldtest.aggregate_for_inversion(res, window_s=200)
    # detection short-circuits the window when an event is found, so compare on a
    # flat record where no event exists (window_s actually drives n_window).
    flat = 1.9 + np.random.default_rng(2).normal(0, 0.3, size=1200)
    fres = fieldtest.process_fieldtest(flat, noise_ppm=0.30)
    s = fieldtest.aggregate_for_inversion(fres, window_s=10)
    l = fieldtest.aggregate_for_inversion(fres, window_s=400)
    assert l.n_window > s.n_window
    assert l.sigma_ppm <= s.sigma_ppm


def test_aggregate_quiet_record_is_a_null_constraint():
    flat = 1.9 + np.random.default_rng(3).normal(0, 0.3, size=400)
    res = fieldtest.process_fieldtest(flat, noise_ppm=0.30)
    pt = fieldtest.aggregate_for_inversion(res)
    assert pt.detected is False
    assert abs(pt.mean_excess_ppm) < 0.3        # "saw nothing" ≈ zero excess
    assert pt.n_window == 400                    # whole record when no event/window


def test_aggregate_window_matches_detected_event_exactly():
    # F5: the aggregation window must cover EXACTLY the detected event [start, end).
    # Independent route (not aggregate's own arithmetic): detect_pattern reports the
    # event bounds, and end_idx is exclusive (processing.py), so the aggregated window
    # must equal the detected window — no extra sub-threshold sample folded into the mean.
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0, seed=0)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    det = res["detection"]
    assert det.detected is True
    pt = fieldtest.aggregate_for_inversion(res)
    assert pt.window == (det.start_idx, det.end_idx)
    assert pt.n_window == det.end_idx - det.start_idx


def test_aggregate_exposes_both_bias_framings():
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0, seed=0)
    res = fieldtest.process_fieldtest(
        sample["ppm"], temperature=sample["temperature"],
        humidity=sample["humidity"], time=sample["time"], noise_ppm=0.30,
    )
    pt = fieldtest.aggregate_for_inversion(res)
    # Both framings always available: subtract baseline (mean_excess) OR fit it
    # (mean_raw + baseline). Raw-frame identity ties them together.
    assert pt.mean_raw_ppm == pytest.approx(pt.baseline_ppm + pt.mean_excess_ppm,
                                            abs=1e-6)
    assert pt.baseline_ppm > 1.0                 # background sits near 1.9 ppm


# ─── sample round-trip ──────────────────────────────────────────────────────────
def test_sample_to_csv_and_back():
    sample = fieldtest.make_sample_readings(n=120, seed=3)
    csv_text = fieldtest.to_csv_text(sample)
    parsed = fieldtest.parse_csv(csv_text)
    assert parsed["n"] == 120
    assert parsed["temperature"] is not None and parsed["humidity"] is not None
    assert np.allclose(parsed["ppm"], sample["ppm"], atol=1e-3)

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


def test_parse_skips_nonnumeric_rows_and_gappy_weather_keeps_nan():
    # row "x" has no number → skipped; a humidity gap stays as NaN — parse reports
    # reality, and process_fieldtest owns the drop-or-interpolate policy.
    text = "time,ppm,humidity\n0,1.9,55\n1,x,55\n2,2.5,\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2                       # the 'x' row was skipped
    assert np.allclose(out["ppm"], [1.9, 2.5])
    assert out["humidity"] is not None
    assert np.isnan(out["humidity"][1])        # the gap is preserved, not invented

    # Too few real values to trust an interpolation → process drops the column.
    res = fieldtest.process_fieldtest(out["ppm"], humidity=out["humidity"],
                                      temperature=np.array([20.0, np.nan]))
    assert res["humidity"] is None and res["temperature"] is None
    assert res["weather_corrected"] is False


def test_parse_no_ppm_column_raises():
    with pytest.raises(ValueError):
        fieldtest.parse_csv("time,temperature\n0,20\n1,21\n")


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        fieldtest.parse_csv("")


# ─── CSV hardening: NaN/Inf, unit sanity, timestamps, sample rate ───────────────
def test_parse_nan_and_inf_ppm_rows_are_skipped():
    # A failed analogRead is a real Arduino artifact — "nan"/"inf" cells must not
    # corrupt the array (they used to: float("nan") parses "successfully").
    text = "time,ppm\n0,1.9\n1,nan\n2,inf\n3,-inf\n4,2.1\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2                          # only the 2 finite rows kept
    assert np.allclose(out["ppm"], [1.9, 2.1])
    assert not np.isnan(out["ppm"]).any()
    assert any("skipped" in w for w in out["warnings"])


def test_process_warns_on_unit_mismatch_but_does_not_reject():
    # A raw 16-bit-ADC-like column under a generic header ("value") is still
    # accepted as ppm (renaming the accepted headers is out of scope) but flagged
    # — and never dropped, since a real huge leak can legitimately read high. Note
    # a 10/12-bit ADC's raw range (0-4095) overlaps the sensor's rated ppm ceiling
    # and genuinely can't be told apart by magnitude alone; this fixture uses a
    # wider raw range where the mismatch IS detectable. The check lives in
    # process_fieldtest (not parse_csv) so STORED tests re-warn on every read.
    out = fieldtest.parse_csv("value\n0\n20000\n40000\n65535\n")
    assert out["n"] == 4                           # nothing rejected
    res = fieldtest.process_fieldtest(out["ppm"])
    assert any("plausible" in w for w in res["warnings"])


def test_in_range_ppm_has_no_unit_warning():
    out = fieldtest.parse_csv("ppm\n1.9\n2.4\n3.0\n")
    assert out["warnings"] == []
    assert fieldtest.process_fieldtest(out["ppm"])["warnings"] == []


def test_parse_sorts_unsorted_timestamps():
    text = "time,ppm\n5,5.0\n1,1.0\n3,3.0\n"
    out = fieldtest.parse_csv(text)
    assert np.allclose(out["time"], [1, 3, 5])
    assert np.allclose(out["ppm"], [1.0, 3.0, 5.0])   # ppm permuted to match
    assert any("sorted by time" in w for w in out["warnings"])


def test_parse_flags_duplicate_timestamps():
    text = "time,ppm\n0,1.0\n0,1.1\n1,2.0\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 3                            # duplicates are kept, not dropped
    assert any("duplicate" in w for w in out["warnings"])


def test_process_interpolates_sparse_weather_gaps():
    # 9 of 10 humidity values present (>= _MIN_GOOD_FOR_INTERP=8): parse keeps the
    # gap as NaN (never invents data); process fills it for the subtraction step
    # and warns — a few dropped serial-print lines don't disable weather correction.
    rows = "\n".join(f"{i},2.0,{50 + i if i != 5 else ''}" for i in range(10))
    text = "time,ppm,humidity\n" + rows + "\n"
    out = fieldtest.parse_csv(text)
    assert out["humidity"] is not None and np.isnan(out["humidity"][5])
    res = fieldtest.process_fieldtest(out["ppm"], humidity=out["humidity"])
    assert res["humidity"].shape == (10,)
    assert not np.isnan(res["humidity"]).any()
    assert res["humidity"][5] == pytest.approx(55.0)   # linear fill between 54 (i=4) and 56 (i=6)
    assert any("interpolated" in w for w in res["warnings"])


def test_parse_warns_on_partial_time_column():
    # One blank timestamp used to silently drop the whole clock (file order assumed,
    # no sorting, no flag). The drop still happens — a gappy clock can't be trusted
    # — but it must be SAID.
    text = "time,ppm\n0,1.9\n,2.0\n2,2.1\n"
    out = fieldtest.parse_csv(text)
    assert out["time"] is None
    assert any("time" in w.lower() for w in out["warnings"])


def test_weather_fit_ignores_interpolated_gap_through_event():
    # The T/H logger drops out exactly during the plume event and the temperature
    # steps 20→26 °C across the gap (sensor heated). Interpolation fills the gap
    # with a ramp that rises WITH the plume bump — if the weather fit trains on
    # those invented samples it learns a fake a_temp and subtracts real methane.
    # The fit must use only REAL samples (where reading is flat at background).
    n = 120
    ppm = np.full(n, 1.9)
    ppm[40:70] += 5.0                                   # known truth: +5 ppm event
    temp = np.full(n, 20.0)
    temp[70:] = 26.0
    humid = np.full(n, 50.0)
    temp[40:70] = np.nan                                # gap spans the event
    humid[40:70] = np.nan
    res = fieldtest.process_fieldtest(ppm, temperature=temp, humidity=humid,
                                      noise_ppm=0.30)
    # Independent expectation: real samples are flat 1.9 at both T=20 and T=26, so
    # the true weather coefficient is 0 and the event must survive intact.
    event_excess = (res["raw"] - res["baseline"])[40:70].mean()
    assert event_excess == pytest.approx(5.0, rel=0.15)
    assert res["detection"].detected


def test_nasty_csv_end_to_end_alignment_and_warnings():
    # One deliberately horrible upload: shuffled timestamps, a duplicate stamp,
    # junk/NaN/Inf ppm cells, a T/H logger gap, and a negative reading. The
    # pipeline must (a) keep every surviving row aligned across columns after
    # sorting, (b) enumerate each problem in warnings, and (c) produce the same
    # cleaned series as the equivalent hand-sorted clean file.
    nasty = (
        "time,ppm,temperature,humidity\n"
        "4,2.4,20.4,50.4\n"
        "0,2.0,20.0,50.0\n"
        "1,junk,99,99\n"          # skipped row: its T/H must vanish with it
        "2,2.2,,\n"               # T/H gap at t=2
        "1,2.1,20.1,50.1\n"
        "3,2.3,20.3,50.3\n"
        "5,nan,1,1\n"             # skipped (NaN ppm)
        "6,inf,1,1\n"             # skipped (Inf ppm)
        "3,-0.5,20.3,50.3\n"      # duplicate stamp + negative reading, kept
    )
    out = fieldtest.parse_csv(nasty)
    assert out["n"] == 6
    assert np.allclose(out["time"], [0, 1, 2, 3, 3, 4])
    # Alignment survives the sort: ppm at t=0 is 2.0 and its T/H are the row's own.
    assert out["ppm"][0] == pytest.approx(2.0)
    assert out["temperature"][0] == pytest.approx(20.0)
    assert out["humidity"][5] == pytest.approx(50.4)
    assert np.isnan(out["temperature"][2]) and np.isnan(out["humidity"][2])
    w = " | ".join(out["warnings"])
    assert "skipped" in w and "sorted by time" in w and "duplicate" in w

    # The clean equivalent (pre-sorted, junk rows removed, gap left blank) must
    # produce byte-identical processing output — the mess itself carries no signal.
    clean = (
        "time,ppm,temperature,humidity\n"
        "0,2.0,20.0,50.0\n"
        "1,2.1,20.1,50.1\n"
        "2,2.2,,\n"
        "3,2.3,20.3,50.3\n"
        "3,-0.5,20.3,50.3\n"
        "4,2.4,20.4,50.4\n"
    )
    ref = fieldtest.parse_csv(clean)
    res_nasty = fieldtest.process_fieldtest(out["ppm"], temperature=out["temperature"],
                                            humidity=out["humidity"], time=out["time"])
    res_ref = fieldtest.process_fieldtest(ref["ppm"], temperature=ref["temperature"],
                                          humidity=ref["humidity"], time=ref["time"])
    assert np.allclose(res_nasty["excess"], res_ref["excess"], equal_nan=True)
    assert np.allclose(res_nasty["baseline"], res_ref["baseline"], equal_nan=True)


def test_process_fieldtest_invalid_sample_rate_falls_back():
    ppm = 1.9 + np.random.default_rng(0).normal(0, 0.1, size=50)
    res = fieldtest.process_fieldtest(ppm, sample_rate_hz=-1.0)
    assert np.allclose(res["time"], np.arange(50) / 1.0)
    assert any("plausible cadence" in w for w in res["warnings"])

    res2 = fieldtest.process_fieldtest(ppm, sample_rate_hz=1e6)
    assert np.allclose(res2["time"], np.arange(50) / 1.0)
    assert any("plausible cadence" in w for w in res2["warnings"])


def test_process_fieldtest_valid_sample_rate_has_no_warning():
    ppm = 1.9 + np.random.default_rng(0).normal(0, 0.1, size=50)
    res = fieldtest.process_fieldtest(ppm, sample_rate_hz=2.0)
    assert res["warnings"] == []


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


# ─── Fix 1: measured drift-inclusive floor must not let drift fabricate a detection ───
def test_measured_floor_rejects_drift_the_assumed_floor_fabricates():
    # A SOURCELESS record: realistic field noise (0.5 ppm) + slow drift, NO event. Truth,
    # independent by construction: detected=False. The MEASURED floor folds the noise+drift
    # into a data-driven threshold that clears the wander; the fixed assumed 0.30 floor does
    # not and fabricates a detection — the exact bug Fix 1 closes.
    rng = np.random.default_rng(11)
    n = 300
    t = np.arange(n, dtype=float)
    drift = 1.0 * np.sin(2 * np.pi * t / (2.3 * n))
    ppm = 1.9 + drift + rng.normal(0, 0.5, n)

    meas = fieldtest.process_fieldtest(ppm, time=t)                 # default → MEASURED floor
    old = fieldtest.process_fieldtest(ppm, time=t, noise_ppm=0.30)  # the assumed-0.30 bug

    assert meas["noise_floor_measured"] is True
    assert meas["noise_ppm"] > old["noise_ppm"]        # measured floor is honestly higher
    assert meas["detection"].detected is False         # ...so drift is not fabricated
    assert old["threshold"] == pytest.approx(0.90)
    assert old["detection"].detected is True           # the fixed 0.30 floor DOES fabricate


# ─── Fix 3: event-coincident humidity must not be mis-attributed as a weather term ───
def test_event_coincident_humidity_not_misattributed():
    # A real +4 ppm methane event; humidity is flat OUTSIDE the event and rises +12 %RH
    # ONLY during it. The TRUE humidity coefficient is ZERO by construction — the RH rise
    # merely coincides with the plume. The event-excluded weather fit must not learn a
    # phantom humidity coefficient and subtract real methane.
    from physics.processing import temp_humidity_correct
    n = 300
    t = np.arange(n, dtype=float)
    ev0, ev1 = 120, 200
    rng = np.random.default_rng(3)
    ppm = 1.9 + rng.normal(0, 0.05, n)
    ppm[ev0:ev1] += 4.0
    temp = np.full(n, 20.0)
    rh = np.full(n, 50.0)
    rh[ev0:ev1] += 12.0

    res = fieldtest.process_fieldtest(ppm, temperature=temp, humidity=rh, time=t)
    assert abs(res["weather_coeffs"].get("b_humid", 0.0)) < 0.05   # no phantom coefficient
    assert np.max(res["excess"]) > 0.9 * 4.0                       # the 4 ppm event survives

    # Independent contrast: a WHOLE-record fit (no event exclusion) DOES learn the phantom
    # (≈ 4 ppm / 12 %RH ≈ 0.33) — the exact bug the exclusion prevents.
    _, whole = temp_humidity_correct(ppm, temp, rh)               # ref_mask=None → whole record
    assert whole.get("b_humid", 0.0) > 0.15


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


def test_aggregate_inflates_sigma_for_autocorrelated_records():
    # F6: an AR(1)-correlated quiet record shrinks slower than sqrt(N), so its
    # sigma_ppm (the WLS weight / CRB input) must come out LARGER than the naive
    # i.i.d. assumption (n_avg=n_window) would give -- otherwise the inversion's
    # reported confidence is silently overconfident when a real source IS present.
    rng = np.random.default_rng(21)
    e = rng.normal(0, 0.30 * np.sqrt(1 - 0.9 ** 2), size=600)
    x = np.empty(600)
    x[0] = e[0]
    for i in range(1, 600):
        x[i] = 0.9 * x[i - 1] + e[i]
    ppm = 1.9 + x
    res = fieldtest.process_fieldtest(ppm, noise_ppm=0.30)
    pt = fieldtest.aggregate_for_inversion(res)

    assert pt.rho_hat > 0.6                       # recovers the strong correlation
    assert pt.n_eff is not None and pt.n_eff < pt.n_window

    from physics.sensor_sim import effective_noise_floor
    naive_sigma = effective_noise_floor(pt.random_ppm, pt.bias_ppm,
                                        n_avg=max(1, pt.n_window))
    assert pt.sigma_ppm > naive_sigma               # correctly inflated, not overconfident


def test_aggregate_white_noise_sigma_matches_naive_iid_assumption():
    # Control case: i.i.d. noise has rho_hat ~ 0, so n_eff ~ n_window and sigma_ppm
    # should be close to the old (pre-F6) naive iid calculation -- the fix must not
    # change behaviour on data that was already handled correctly.
    rng = np.random.default_rng(22)
    ppm = 1.9 + rng.normal(0, 0.30, size=600)
    res = fieldtest.process_fieldtest(ppm, noise_ppm=0.30)
    pt = fieldtest.aggregate_for_inversion(res)

    assert pt.rho_hat < 0.3
    from physics.sensor_sim import effective_noise_floor
    naive_sigma = effective_noise_floor(pt.random_ppm, pt.bias_ppm,
                                        n_avg=max(1, pt.n_window))
    assert pt.sigma_ppm == pytest.approx(naive_sigma, rel=0.25)


def test_aggregate_event_dominated_record_keeps_weight():
    # A short record where the detected event fills nearly everything leaves too
    # little quiet data to measure rho — the fallback must NOT measure rho on the
    # smooth plume bump itself (corrcoef of a bump → ~0.98 → n_eff floors at 1 and
    # the strongest detection gets down-weighted out of the WLS fit). With no
    # trustworthy quiet stretch, the honest answer is "no correction" (rho = 0).
    rng = np.random.default_rng(7)
    n = 40
    t = np.arange(n)
    ppm = 1.9 + 6.0 * np.exp(-0.5 * ((t - 20) / 8.0) ** 2) + rng.normal(0, 0.3, n)
    res = fieldtest.process_fieldtest(ppm, noise_ppm=0.30)
    assert res["detection"].detected
    pt = fieldtest.aggregate_for_inversion(res)
    assert pt.rho_hat == 0.0
    assert pt.n_eff == pytest.approx(pt.n_window)


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

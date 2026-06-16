"""
test_explain.py — lock in the deterministic caption paths of explain.py.

The OpenAI path needs a key + network, so it is not unit-tested here. What we DO pin
is the guarantee that matters: summarize_field extracts the right numbers, and
explain_field ALWAYS returns a sensible template caption when AI is off or unavailable
(a broken or absent API key must degrade cleanly, never crash the plot). No test makes
a real API call — the only AI path exercised has its key removed first.
"""

import numpy as np
import pytest

from ui.explain import (
    _fmt,
    ask_once,
    explain_field,
    scenario_facts,
    summarize_field,
    template_explanation,
)
from physics.plume import CH4_BACKGROUND, predict_ppm


def _demo_field():
    """A realistic ppm field from the forward model (demo.py's geometry, smaller grid)."""
    xs = np.linspace(10.0, 200.0, 60)
    ys = np.linspace(-60.0, 60.0, 50)
    XX, YY = np.meshgrid(xs, ys)
    rec = np.column_stack([XX.ravel(), YY.ravel()])
    ppm = predict_ppm((0.0, 0.0), Q=5.0, u=3.0, wind_dir_deg=270.0,
                      H=1.0, stability_class=4, receptors=rec).reshape(YY.shape)
    params = {"Q": 5.0, "u": 3.0, "H": 1.0, "stability_class": 4, "wind_dir_deg": 270.0}
    return ppm, xs, ys, params


def test_fmt_handles_none_and_numbers():
    assert _fmt(None) == "n/a"
    assert _fmt(1.2345, " ppm") == "1.23 ppm"
    assert _fmt(50.0, " m", nd=0) == "50 m"


def test_summarize_field_extracts_sane_numbers():
    ppm, xs, ys, params = _demo_field()
    f = summarize_field(ppm, xs, ys, params, noise_floor=0.30)
    assert f["background"] == CH4_BACKGROUND
    assert f["threshold"] == pytest.approx(3.0 * 0.30)
    # peak sits near the source (small x) on the centreline, at/above background
    assert f["peak_ppm"] >= CH4_BACKGROUND
    assert f["peak_x"] <= 50.0
    assert abs(f["peak_y"]) < 10.0
    # centreline ppm decays with downwind distance
    assert f["ppm_at_50"] > f["ppm_at_100"] > f["ppm_at_200"]


def test_summarize_field_none_when_field_is_pure_background():
    xs = np.linspace(10, 200, 20)
    ys = np.linspace(-50, 50, 20)
    ppm = np.full((ys.size, xs.size), CH4_BACKGROUND)
    f = summarize_field(ppm, xs, ys, {}, noise_floor=0.30)
    assert f["detect_reach_m"] is None
    assert f["half_width_at_50_m"] is None


def test_template_explanation_is_string_with_key_facts():
    ppm, xs, ys, params = _demo_field()
    text = template_explanation(summarize_field(ppm, xs, ys, params, 0.30))
    assert isinstance(text, str) and len(text) > 100
    assert "ppm" in text
    assert "steady-state" in text          # the honest caveat is always present


def test_template_explanation_frames_result_as_upper_bound():
    # The caption must not imply false precision: it states the leak size is an upper
    # bound (Q scales with the unknown wind) and that averaging beats only random noise.
    ppm, xs, ys, params = _demo_field()
    text = template_explanation(summarize_field(ppm, xs, ys, params, 0.30))
    assert "upper bound" in text
    assert "averaging" in text


def test_template_handles_missing_reach_and_params():
    # reach=None branch + params .get fallbacks + None values must not crash.
    facts = {
        "params": {}, "background": CH4_BACKGROUND, "noise_floor": 0.30,
        "threshold": 0.90, "peak_ppm": 2.0, "peak_x": 10.0, "peak_y": 0.0,
        "detect_reach_m": None, "ppm_at_50": None, "ppm_at_100": None,
        "ppm_at_200": None, "half_width_at_50_m": None,
    }
    text = template_explanation(facts)
    assert "never rises above" in text     # the no-detection branch
    assert "n/a" in text                   # None values render as n/a, not a crash


def test_explain_field_uses_template_when_ai_disabled():
    ppm, xs, ys, params = _demo_field()
    text, source = explain_field(ppm, xs, ys, params, 0.30, use_ai=False)
    assert source == "template"
    assert isinstance(text, str) and len(text) > 0


def test_explain_field_falls_back_to_template_without_api_key(monkeypatch):
    # use_ai=True but no key → ai_explanation returns None → template fallback (no API call).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ppm, xs, ys, params = _demo_field()
    text, source = explain_field(ppm, xs, ys, params, 0.30, use_ai=True)
    assert source == "template"


def test_ask_once_returns_error_string_without_key(monkeypatch):
    # ask_once must degrade to a clear, human-readable string (not None, not a crash,
    # and WITHOUT making a network call) when no key is configured.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = ask_once("what does the green line mean?")
    assert isinstance(out, str) and out
    assert "OPENAI_API_KEY" in out          # tells the user exactly what's missing


def test_ask_once_blank_question_is_guarded(monkeypatch):
    # An empty question never reaches the API — it returns usage help.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = ask_once("   ")
    assert isinstance(out, str) and "ask.py" in out


def test_scenario_facts_has_expected_keys():
    # Pure local compute (no API): builds the demo grid and summarises it.
    f = scenario_facts()
    for key in ("peak_ppm", "detect_reach_m", "threshold", "half_width_at_50_m"):
        assert key in f
    assert f["peak_ppm"] > CH4_BACKGROUND          # a real plume peak above background
    assert f["threshold"] == pytest.approx(3.0 * 0.30)

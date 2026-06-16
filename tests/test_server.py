"""
test_server.py — lock in the web layer: the shared compute_field helper and the
JSON API that the dashboard talks to.

Two guarantees matter here:
  1. compute_field is the single source of truth for the demo grid — it must
     reproduce the exact field the old inline construction produced, and feed
     scenario_facts unchanged.
  2. The API stays JSON-clean (no numpy leaking → 500) and degrades gracefully
     when the OpenAI key is absent (template caption, human-readable chat error).
No test makes a real network call — the AI paths are only exercised with the key removed.
"""
import json

import numpy as np
import pytest

pytest.importorskip("flask")  # web layer is optional; skip cleanly if Flask is absent

from ui.explain import compute_field, scenario_facts, DEFAULT_SCENARIO
from physics.plume import CH4_BACKGROUND, predict_ppm
from physics.sensor_sim import DETECT_K, SENSOR_NOISE_PPM


# ─── compute_field: the simplification must be numerically identical ───────────
def test_compute_field_grid_shapes():
    xs, ys, PPM, facts = compute_field()
    assert xs.shape == (250,) and ys.shape == (200,)
    assert PPM.shape == (200, 250)                     # (len(ys), len(xs))
    assert (xs[0], xs[-1]) == (10.0, 200.0)
    assert (ys[0], ys[-1]) == (-150.0, 150.0)


def test_compute_field_matches_old_inline_grid():
    # The refactor must not move a single number vs the pre-refactor construction.
    xs = np.linspace(10.0, 200.0, 250)
    ys = np.linspace(-150.0, 150.0, 200)
    XX, YY = np.meshgrid(xs, ys)
    ref = predict_ppm((0.0, 0.0), Q=5.0, u=3.0, wind_dir_deg=270.0, H=1.0,
                      stability_class=3,
                      receptors=np.column_stack([XX.ravel(), YY.ravel()])).reshape(YY.shape)
    _, _, PPM, _ = compute_field()
    assert np.allclose(PPM, ref)


def test_scenario_facts_is_compute_field_facts():
    # scenario_facts() is now a thin wrapper — identical facts, same signature.
    _, _, _, facts = compute_field()
    assert scenario_facts() == facts


# ─── API ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    from misc import server
    server._get_field.cache_clear()                    # isolate the field cache per test
    return server.app.test_client()


def test_api_field_shapes_and_constants(client):
    d = client.get("/api/field").get_json()
    assert len(d["xs"]) == 250 and len(d["ys"]) == 200
    assert len(d["ppm"]) == 200 and len(d["ppm"][0]) == 250
    assert d["detect_level"] == pytest.approx(CH4_BACKGROUND + DETECT_K * SENSOR_NOISE_PPM)  # 2.80
    assert d["background"] == CH4_BACKGROUND
    assert d["scenario"] == DEFAULT_SCENARIO
    for k in ("peak_ppm", "detect_reach_m", "threshold", "half_width_at_50_m"):
        assert k in d["facts"]


def test_api_field_is_pure_json(client):
    # tolist()/float() must leave no numpy types — otherwise jsonify 500s.
    resp = client.get("/api/field")
    assert resp.status_code == 200
    json.loads(resp.data)                              # raises if anything non-JSON leaked


def test_api_caption_falls_back_to_template_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = client.post("/api/caption", json={"use_ai": True}).get_json()
    assert d["source"] == "template"                   # degrades, never crashes
    assert isinstance(d["caption"], str) and len(d["caption"]) > 100


def test_api_caption_use_ai_false_is_clean_template(client):
    d = client.post("/api/caption", json={"use_ai": False}).get_json()
    assert d["source"] == "template"
    assert d["ai_error"] is None


def test_api_ask_returns_readable_error_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = client.post("/api/ask", json={"question": "what is the green line?"}).get_json()
    assert isinstance(d["answer"], str) and "OPENAI_API_KEY" in d["answer"]


def test_api_ask_blank_question_guarded(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = client.post("/api/ask", json={"question": "   "}).get_json()
    assert "ask.py" in d["answer"]                     # ask_once's blank-question guard


def test_api_feasibility_cells_and_verdict(client):
    d = client.get("/api/feasibility").get_json()
    assert len(d["cells"]) == 24                       # 6 Q × 4 distance × 1 stability
    assert {"Q", "distance", "stability", "ppm_total", "excess", "verdict"} <= d["cells"][0].keys()
    assert isinstance(d["verdict"], str) and "g/s" in d["verdict"]

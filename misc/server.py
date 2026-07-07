"""
server.py — Minimal localhost backend for the methane-plume dashboard.

Serves the static web/ dashboard and a tiny JSON API that calls the EXISTING
physics (explain.compute_field), caption (explain.explain_field), and Q&A
(explain.ask_once) functions — no physics is reimplemented here.

WHY THIS EXISTS
───────────────────────────────────────────────────────────────────────────────
The browser must not hold the OpenAI key. This backend keeps the key server-side
(loaded from .env by explain.py) and exposes only safe JSON endpoints, so the
dashboard can show an AI caption / answer questions without ever shipping a secret
to the client. It binds to 127.0.0.1 ONLY, so nothing on the network can reach the
key-bearing endpoints.

RUN
    python3 server.py            # → http://127.0.0.1:5050   (Ctrl+C to stop)
    python3 demo.py              # runs validation, saves the PNG, then serves this

Port 5050 (not 5000): macOS AirPlay Receiver squats on 5000 and answers with 403.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from misc import db
from ui import explain
from physics import fieldtest, weather
from ui.explain import (
    DEFAULT_SCENARIO,
    ask_once,
    compute_field,
    explain_field,
    interpret_fieldtest,
    summarize_fieldtest,
)
from physics.accuracy import accuracy_report
from physics.inversion import invert_field_tests_multi
from physics.plume import CH4_BACKGROUND, U_MIN
from physics.sensor_sim import DETECT_K, SENSOR_NOISE_PPM

# Detection-limit line on the plot: background + 3σ noise = 2.80 ppm. Same number
# the matplotlib demo and the feasibility verdict use.
DETECT_LEVEL = CH4_BACKGROUND + DETECT_K * SENSOR_NOISE_PPM

# The dashboard front-end lives in ui/web/ (this module is in misc/). Resolve it
# absolutely so the server works no matter what directory it is launched from.
_WEB_DIR = Path(__file__).resolve().parent.parent / "ui" / "web"

app = Flask(__name__, static_folder=str(_WEB_DIR), static_url_path="")

# ── One-time field cache ─────────────────────────────────────────────────────
# compute_field is ~2 ms, but caching means /api/field, /api/caption and /api/ask
# all describe ONE consistent field for the fixed DEFAULT_SCENARIO. maxsize=1: a
# single scenario, computed once for the life of the process.
@functools.lru_cache(maxsize=1)
def _get_field():
    """Cached (xs, ys, PPM, facts) for the fixed scenario."""
    return compute_field(DEFAULT_SCENARIO, noise_floor=SENSOR_NOISE_PPM)


@app.route("/")
def index():
    """Serve the dashboard shell."""
    return send_from_directory(_WEB_DIR, "index.html")


@app.route("/api/field")
def api_field():
    """The plume field + summary facts as JSON (numpy cast to plain lists/floats)."""
    xs, ys, PPM, facts = _get_field()
    return jsonify({
        "xs": xs.tolist(),                   # 250 downwind distances (m)
        "ys": ys.tolist(),                   # 200 crosswind distances (m)
        "ppm": PPM.tolist(),                 # 2-D total ppm, shape (200, 250)
        "facts": facts,                      # plain floats / None — already JSON-safe
        "detect_level": float(DETECT_LEVEL), # 2.80 ppm (green dashed line)
        "background": float(CH4_BACKGROUND), # 1.9 ppm
        "scenario": dict(DEFAULT_SCENARIO),
    })


@app.route("/api/caption", methods=["POST"])
def api_caption():
    """Plain-English caption for the graph (OpenAI if a key works, else template)."""
    body = request.get_json(silent=True) or {}
    use_ai = bool(body.get("use_ai", True))
    xs, ys, PPM, _ = _get_field()
    caption, source = explain_field(
        PPM, xs, ys, dict(DEFAULT_SCENARIO),
        noise_floor=SENSOR_NOISE_PPM, use_ai=use_ai,
    )
    return jsonify({
        "caption": caption,
        "source": source,                    # "openai" | "template"
        # Only surface an AI error when AI was actually attempted — LAST_AI_ERROR is a
        # module global that can hold a stale reason from an earlier request otherwise.
        "ai_error": explain.LAST_AI_ERROR if use_ai else None,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Answer one question about the model/graph, grounded in the current numbers."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    _, _, _, facts = _get_field()
    answer = ask_once(question, facts)       # always a string — errors included
    return jsonify({"answer": answer})


@app.route("/api/feasibility")
def api_feasibility():
    """The 'can this sensor see the leak?' sweep + plain-English verdict."""
    from physics.feasibility import sweep, verdict
    Q_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    dist_list = [25.0, 50.0, 100.0, 200.0]
    cells = sweep(Q_list=Q_list, distance_list=dist_list, stability_list=[4],
                  u=2.0, wind_dir_deg=270.0, H=1.0, z=1.0)
    rows = [{"Q": c.Q, "distance": c.distance, "stability": c.stability,
             "ppm_total": c.ppm_total, "excess": c.excess, "verdict": c.verdict}
            for c in cells]
    return jsonify({
        "cells": rows,
        "verdict": verdict(cells, fence_distance=50.0, stability=4),
    })


# ═════════════════════════════════════════════════════════════════════════════
#  FIELD TESTS — real measured readings: upload, store, process, interpret, compare
# ═════════════════════════════════════════════════════════════════════════════
# Storage is db.py (Railway Postgres when DATABASE_URL is set, else local SQLite).
# We store ONLY the raw readings and recompute the processed view (baseline,
# smoothing, detection) on read via fieldtest.py — so the cleaning can be retuned
# later without re-uploading. The OpenAI key stays server-side exactly as above.

# Create tables once per process per backend. Re-inits automatically if the backend
# changes (e.g. tests pointing LOCAL_DB_PATH at a fresh temp file).
_schema_ready: set = set()


def _ensure_schema() -> None:
    key = db.backend_label()
    if key not in _schema_ready:
        db.init_schema()
        _schema_ready.add(key)


def _form_float(form, name):
    v = (form.get(name) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _meta_from_form(form) -> dict:
    """Pull a field_tests metadata dict out of an upload's form fields."""
    fl = _form_float
    sc = fl(form, "stability_class")
    return {
        "name": (form.get("name") or "").strip() or None,
        "site": (form.get("site") or "").strip() or None,
        "test_date": (form.get("test_date") or "").strip() or None,
        "notes": (form.get("notes") or "").strip() or None,
        "wind_speed": fl(form, "wind_speed"),
        "wind_dir_deg": fl(form, "wind_dir_deg"),
        "stability_class": int(sc) if sc is not None else None,
        "sensor_distance_m": fl(form, "sensor_distance_m"),
        "noise_ppm": fl(form, "noise_ppm"),
        "sample_rate_hz": fl(form, "sample_rate_hz"),
    }


def _enrich_meta_with_weather(meta: dict) -> dict | None:
    """
    Fill missing wind/stability from the Open-Meteo archive, in place.

    Only attempts a lookup when the domain preconditions hold: the user did NOT
    already supply wind, the site resolves to known coordinates, and a test_date is
    present (a historical-archive query is undefined without a date to key it on).
    All three gates are domain logic; they also happen to keep offline/CI uploads
    (which omit test_date) network-free, but that is a side effect, not the reason —
    a test that exercises the lookup should patch the ``weather`` seam directly.
    Returns the WindEstimate on success (for the response) or None; any failure is
    silent (leaves the hand-entered/blank values) — the reason is in
    weather.LAST_WEATHER_ERROR.
    """
    if meta.get("wind_speed") is not None or meta.get("stability_class") is not None:
        return None
    if not meta.get("test_date") or weather.resolve_site(meta.get("site")) is None:
        return None
    est = weather.wind_for_site_date(meta["site"], meta["test_date"])
    if est is None:
        return None
    meta["wind_speed"] = round(est.u, 2)
    meta["wind_dir_deg"] = round(est.wind_dir_deg, 1)
    meta["stability_class"] = int(est.stability_class)
    return est


def _store_readings(ft_id: int, parsed: dict) -> None:
    """Persist parsed CSV arrays as raw reading rows.

    Temperature/humidity gaps arrive as NaN (parse_csv reports reality, it never
    invents values) and are stored as NULL — so a re-read rebuilds the same gappy
    column and process_fieldtest applies the one gap policy everywhere."""
    n = parsed["n"]
    time, temp, humid, ppm = (parsed["time"], parsed["temperature"],
                              parsed["humidity"], parsed["ppm"])

    def cell(col, i):
        if col is None or not np.isfinite(col[i]):
            return None
        return float(col[i])

    rows = [
        (i,
         cell(time, i),
         float(ppm[i]),
         cell(temp, i),
         cell(humid, i))
        for i in range(n)
    ]
    db.add_readings(ft_id, rows)


def _stored_arrays(ft: dict):
    """Load a stored test's readings as numpy arrays (None columns stay None).

    The clock must be complete to be trusted (same rule as parse_csv);
    temperature/humidity NULLs come back as NaN so process_fieldtest's gap
    policy treats a stored test exactly like a fresh upload."""
    readings = db.get_readings(ft["id"])
    if not readings:
        return None
    ppm = np.array([r["ppm_raw"] for r in readings], dtype=float)

    def full_col(key):
        vals = [r[key] for r in readings]
        return np.array(vals, dtype=float) if all(v is not None for v in vals) else None

    def gappy_col(key):
        vals = [r[key] for r in readings]
        if all(v is None for v in vals):
            return None
        return np.array([v if v is not None else np.nan for v in vals], dtype=float)

    return {"ppm": ppm, "time": full_col("t_seconds"),
            "temperature": gappy_col("temperature"), "humidity": gappy_col("humidity")}


def _process_stored(ft: dict):
    """Run the cleaning pipeline on a stored test using its saved params."""
    arrays = _stored_arrays(ft)
    if arrays is None:
        return None
    noise = ft.get("noise_ppm") or SENSOR_NOISE_PPM
    rate = ft.get("sample_rate_hz") or 1.0
    return fieldtest.process_fieldtest(
        arrays["ppm"], temperature=arrays["temperature"],
        humidity=arrays["humidity"], time=arrays["time"],
        noise_ppm=noise, sample_rate_hz=rate,
    )


def _empty_facts(ft: dict) -> dict:
    """Stub facts for a test with no readings (keeps the list endpoint 500-free)."""
    noise = float(ft.get("noise_ppm") or SENSOR_NOISE_PPM)
    return {
        "kind": "fieldtest", "name": ft.get("name"), "site": ft.get("site"),
        "test_date": ft.get("test_date"), "notes": ft.get("notes"),
        "n_samples": 0, "duration_s": 0.0, "background": float(CH4_BACKGROUND),
        "noise_ppm": noise, "threshold": float(DETECT_K * noise),
        "detect_k": float(DETECT_K), "raw_peak_ppm": None,
        "baseline_mean_ppm": None, "excess_peak_ppm": None, "detected": False,
        "confidence_sigmas": 0.0, "event_start_s": None, "event_end_s": None,
        "event_duration_s": None, "weather_corrected": False,
        "sensor_distance_m": ft.get("sensor_distance_m"),
        "wind_speed": ft.get("wind_speed"), "wind_dir_deg": ft.get("wind_dir_deg"),
        "stability_class": ft.get("stability_class"),
    }


def _facts_for(ft: dict) -> dict:
    result = _process_stored(ft)
    return summarize_fieldtest(result, ft) if result else _empty_facts(ft)


def _series_json(result: dict) -> dict:
    """Full plottable series for one test (raw, baseline, processed excess, event window)."""
    det = result["detection"]
    n = len(result["raw"])
    window = None
    if det.detected and det.start_idx >= 0:
        s = int(det.start_idx)
        e = int(min(det.end_idx, n) - 1)
        window = {"start_idx": s, "end_idx": e,
                  "start_s": float(result["time"][s]),
                  "end_s": float(result["time"][e])}
    return {
        "time": np.asarray(result["time"], dtype=float).tolist(),
        "raw": np.asarray(result["raw"], dtype=float).tolist(),
        "baseline": np.asarray(result["baseline"], dtype=float).tolist(),
        "excess": np.asarray(result["excess"], dtype=float).tolist(),
        "threshold": float(result["threshold"]),
        "weather_corrected": bool(result["weather_corrected"]),
        "windows": result["windows"],
        "detection": {"detected": bool(det.detected),
                      "confidence": float(det.confidence), "window": window},
    }


def _downsample_series(result: dict, max_points: int = 600) -> dict:
    """Lighter series (time/raw/excess) for the compare overlay."""
    t = np.asarray(result["time"], dtype=float)
    raw = np.asarray(result["raw"], dtype=float)
    ex = np.asarray(result["excess"], dtype=float)
    if t.size > max_points:
        idx = np.linspace(0, t.size - 1, max_points).round().astype(int)
        t, raw, ex = t[idx], raw[idx], ex[idx]
    return {"time": t.tolist(), "raw": raw.tolist(), "excess": ex.tolist(),
            "threshold": float(result["threshold"])}


def _spread(facts_list):
    """Across-test variance summary — answers 'how repeatable are my tests?'."""
    def stat(key):
        vals = [f[key] for f in facts_list if f.get(key) is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "min": float(min(vals)),
                "max": float(max(vals)), "spread": float(max(vals) - min(vals))}
    return {
        "n_tests": len(facts_list),
        "n_detected": sum(1 for f in facts_list if f.get("detected")),
        "excess_peak_ppm": stat("excess_peak_ppm"),
        "confidence_sigmas": stat("confidence_sigmas"),
        "raw_peak_ppm": stat("raw_peak_ppm"),
    }


@app.route("/api/fieldtests")
def api_fieldtests_list():
    """Every saved field test with its detection summary (newest first)."""
    _ensure_schema()
    tests = [{"meta": ft, "facts": _facts_for(ft)} for ft in db.list_field_tests()]
    return jsonify({"tests": tests, "backend": db.backend_label()})


@app.route("/api/fieldtests", methods=["POST"])
def api_fieldtests_create():
    """Upload a readings CSV (+ metadata) → store raw → return the processed view."""
    _ensure_schema()
    meta = _meta_from_form(request.form)
    if not meta["name"]:
        return jsonify({"error": "Please give the field test a name."}), 400
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "No CSV file was uploaded (field name 'file')."}), 400
    try:
        parsed = fieldtest.parse_csv(upload.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Auto-fill real wind + stability from Open-Meteo when possible (Problem 2):
    # collapses the Q-uncertainty swing the inversion otherwise inherits.
    wx = _enrich_meta_with_weather(meta)
    ft_id = db.create_field_test(meta)
    _store_readings(ft_id, parsed)
    ft = db.get_field_test(ft_id)
    result = _process_stored(ft)
    # Parse-time flags (skipped rows, sorted clock) exist only now — the raw CSV
    # isn't stored — so they ride the create response; process-level flags are
    # recomputed from storage on every read (see api_fieldtest_detail).
    return jsonify({"id": ft_id, "meta": ft, "facts": summarize_fieldtest(result, ft),
                    "series": _series_json(result), "columns": parsed["columns"],
                    "warnings": parsed["warnings"] + result["warnings"],
                    "accuracy": accuracy_report(result, meta=ft),
                    "weather": (wx.source if wx else None),
                    "weather_error": (None if wx else weather.LAST_WEATHER_ERROR)}), 201


@app.route("/api/fieldtests/sample", methods=["POST"])
def api_fieldtests_sample():
    """Generate + store a synthetic field test so the flow works before real data."""
    _ensure_schema()
    body = request.get_json(silent=True) or {}
    n = int(body.get("n", 600))
    event_ppm = float(body.get("event_ppm", 4.0))
    noise = float(body.get("noise_ppm", SENSOR_NOISE_PPM))
    sample = fieldtest.make_sample_readings(n=n, event_ppm=event_ppm, noise_ppm=noise)
    meta = {
        "name": (body.get("name") or "Synthetic sample test").strip(),
        "site": (body.get("site") or "Synthetic (sensor_sim)").strip(),
        "notes": f"Generated sample: {event_ppm:g} ppm event, {noise:g} ppm noise.",
        "noise_ppm": noise, "sample_rate_hz": 1.0, "sensor_distance_m": 50.0,
    }
    ft_id = db.create_field_test(meta)
    _store_readings(ft_id, sample)
    ft = db.get_field_test(ft_id)
    result = _process_stored(ft)
    return jsonify({"id": ft_id, "meta": ft, "facts": summarize_fieldtest(result, ft),
                    "series": _series_json(result),
                    "warnings": result["warnings"],
                    "accuracy": accuracy_report(result, meta=ft)}), 201


@app.route("/api/fieldtests/compare")
def api_fieldtests_compare():
    """Overlay series + an across-test spread for the selected ids (?ids=1,2,3)."""
    _ensure_schema()
    raw_ids = (request.args.get("ids") or "").split(",")
    ids = [int(x) for x in raw_ids if x.strip().isdigit()]
    tests, facts_list = [], []
    for ft_id in ids:
        ft = db.get_field_test(ft_id)
        if ft is None:
            continue
        result = _process_stored(ft)
        facts = summarize_fieldtest(result, ft) if result else _empty_facts(ft)
        facts_list.append(facts)
        tests.append({"id": ft_id, "meta": ft, "facts": facts,
                      "series": _downsample_series(result) if result else None})
    return jsonify({"tests": tests, "spread": _spread(facts_list)})


@app.route("/api/fieldtests/invert", methods=["POST"])
def api_fieldtests_invert():
    """
    Fuse 2+ selected field tests into one recovered source (x, y, Q).

    MVP framing: the DB stores one sensor_distance_m scalar, not a per-test sensor
    (x, y) — so this treats every selected test as one Snapshot recorded at the
    SAME fixed sensor (the map origin), under different wind episodes. That matches
    invert_multi's actual documented use case (triangulating a steady leak from
    several wind directions at one mounting point), not simultaneous multi-sensor
    triangulation, which would need new schema this single-Arduino-rig deployment
    doesn't need. Body: {"ids": [1, 2, 3]}.
    """
    _ensure_schema()
    body = request.get_json(silent=True) or {}
    ids = []
    for raw in (body.get("ids") or []):
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            pass
    if len(ids) < 2:
        return jsonify({"error": "select at least 2 field tests (same sensor, "
                                "different wind episodes) to fuse"}), 400

    snaps, distances = [], []
    for ft_id in ids:
        ft = db.get_field_test(ft_id)
        if ft is None:
            return jsonify({"error": f"field test {ft_id} not found"}), 404
        result = _process_stored(ft)
        if result is None:
            return jsonify({"error": f"field test {ft_id} has no readings"}), 400
        u, wind_dir = ft.get("wind_speed"), ft.get("wind_dir_deg")
        if u is None or wind_dir is None:
            return jsonify({"error": f"field test {ft_id} ('{ft.get('name')}') has "
                                    "no wind speed/direction set"}), 400
        if float(u) < U_MIN:
            return jsonify({"error": f"field test {ft_id} wind speed {u} m/s is "
                                    f"below the model minimum {U_MIN} m/s"}), 400
        snaps.append(([result], float(u), float(wind_dir), ft.get("stability_class")))
        distances.append((ft_id, ft.get("name"), ft.get("sensor_distance_m")))

    # Enforce the endpoint's own premise: one fixed sensor. Tests recorded at
    # different stored standoffs are different geometries — fusing them as
    # co-located would return a confidently-wrong fix.
    known = [(i, n, float(d)) for (i, n, d) in distances if d is not None]
    if known:
        d0 = known[0][2]
        if any(abs(d - d0) > 0.5 for (_, _, d) in known):
            listing = ", ".join(f"{n or i}: {d:g} m" for (i, n, d) in known)
            return jsonify({"error": "selected tests disagree on sensor_distance_m "
                                     f"({listing}) — fusion assumes the SAME fixed "
                                     "sensor across all wind episodes"}), 400

    try:
        est = invert_field_tests_multi(snaps, [[0.0, 0.0]])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "x": est.x, "y": est.y, "Q": est.Q, "stability_class": est.stability_class,
        "u": est.u, "wind_dir_deg": est.wind_dir_deg, "converged": est.converged,
        "ill_posed": est.ill_posed, "crb_std": est.crb_std,
        "n_snapshots": est.n_snapshots, "rmse_ppm": est.rmse_ppm, "note": est.note,
        "assumption": "Source (x, y) is reported relative to one fixed sensor (the "
                     "map origin), fusing the selected tests as different wind "
                     "episodes at that SAME sensor -- not simultaneous multi-sensor "
                     "triangulation.",
    })


@app.route("/api/fieldtests/<int:ft_id>")
def api_fieldtest_detail(ft_id):
    """One test's metadata + full processed series + facts + accuracy report.

    The accuracy report rides along (instead of a second round-trip to
    /accuracy) so viewing a test runs the cleaning pipeline ONCE, not twice."""
    _ensure_schema()
    ft = db.get_field_test(ft_id)
    if ft is None:
        return jsonify({"error": "field test not found"}), 404
    result = _process_stored(ft)
    if result is None:
        return jsonify({"meta": ft, "facts": _empty_facts(ft), "series": None,
                        "warnings": [], "accuracy": None})
    return jsonify({"meta": ft, "facts": summarize_fieldtest(result, ft),
                    "series": _series_json(result),
                    "warnings": result["warnings"],
                    "accuracy": accuracy_report(result, meta=ft)})


@app.route("/api/fieldtests/<int:ft_id>/accuracy")
def api_fieldtest_accuracy(ft_id):
    """The Accuracy Protocol for one test: measured noise floor (vs the assumed
    placeholder), detection limit + error bar, 0-100 grade, and a recommendation.
    Reuses accuracy.accuracy_report unchanged — no new physics here. (The UI reads
    the copy embedded in the detail/create responses; this stands alone for API
    users and scripts.)"""
    _ensure_schema()
    ft = db.get_field_test(ft_id)
    if ft is None:
        return jsonify({"error": "field test not found"}), 404
    result = _process_stored(ft)
    if result is None:
        return jsonify({"error": "no readings to assess"}), 400
    return jsonify(accuracy_report(result, meta=ft))


@app.route("/api/fieldtests/<int:ft_id>/interpret", methods=["POST"])
def api_fieldtest_interpret(ft_id):
    """Plain-English read of one test (OpenAI if a key works, else template)."""
    _ensure_schema()
    ft = db.get_field_test(ft_id)
    if ft is None:
        return jsonify({"error": "field test not found"}), 404
    use_ai = bool((request.get_json(silent=True) or {}).get("use_ai", True))
    text, source = interpret_fieldtest(_facts_for(ft), use_ai=use_ai)
    return jsonify({"text": text, "source": source,
                    "ai_error": explain.LAST_AI_ERROR if use_ai else None})


@app.route("/api/fieldtests/<int:ft_id>/ask", methods=["POST"])
def api_fieldtest_ask(ft_id):
    """Answer one question grounded in this test's numbers."""
    _ensure_schema()
    ft = db.get_field_test(ft_id)
    if ft is None:
        return jsonify({"error": "field test not found"}), 404
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    return jsonify({"answer": ask_once(question, _facts_for(ft))})


@app.route("/api/fieldtests/<int:ft_id>", methods=["DELETE"])
def api_fieldtest_delete(ft_id):
    """Delete a test and its readings."""
    _ensure_schema()
    if db.get_field_test(ft_id) is None:
        return jsonify({"error": "field test not found"}), 404
    db.delete_field_test(ft_id)
    return jsonify({"ok": True})


@app.route("/api/sample.csv")
def api_sample_csv():
    """Download a correctly-formatted sample CSV (so users see the accepted columns)."""
    sample = fieldtest.make_sample_readings(n=600, event_ppm=4.0)
    return (fieldtest.to_csv_text(sample), 200,
            {"Content-Type": "text/csv",
             "Content-Disposition": "attachment; filename=sample_field_test.csv"})


# Single source of truth for where the dashboard lives — demo.py reads these too.
# 127.0.0.1 ONLY (never 0.0.0.0): the key-bearing endpoints must stay off the network.
HOST, PORT = "127.0.0.1", 5050


def serve():
    """Run the dashboard server (blocks until Ctrl+C). use_reloader=False so the
    field/.env don't load twice."""
    try:
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except OSError as exc:
        # Almost always "address already in use" — a stale dashboard still running.
        raise SystemExit(
            f"\nCould not start the dashboard on {HOST}:{PORT} — {exc}\n"
            f"Another copy is probably already running; stop it (or free port {PORT}) "
            f"and try again."
        )


if __name__ == "__main__":
    serve()

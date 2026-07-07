"""
test_server_fieldtests.py — the field-test JSON API end to end.

Every test runs against a throwaway SQLite file (LOCAL_DB_PATH) with DATABASE_URL
removed, and the AI paths are only exercised with OPENAI_API_KEY removed — so the
whole file runs offline, no Postgres, no network. Mirrors tests/test_server.py.
"""
import io

import pytest

pytest.importorskip("flask")

from physics import fieldtest


@pytest.fixture(autouse=True)
def _offline_weather(monkeypatch):
    """Own our hermeticity instead of leaning on the upload's test_date gate: stub
    the Open-Meteo seam so no upload can reach the network, even one that supplies a
    resolvable site + test_date. Returning None exercises the offline-fallback path
    (hand-entered/blank wind is left untouched)."""
    from physics import weather
    monkeypatch.setattr(weather, "wind_for_site_date", lambda *a, **k: None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("LOCAL_DB_PATH", str(tmp_path / "ft.db"))
    from misc import server
    return server.app.test_client()


def _sample_csv_bytes(n=300, event_ppm=4.0, seed=0):
    sample = fieldtest.make_sample_readings(n=n, event_ppm=event_ppm, seed=seed)
    return fieldtest.to_csv_text(sample).encode("utf-8")


def _upload(client, name="Custer #1", **fields):
    data = {"name": name, "site": "Custer Road Transfer Station", "noise_ppm": "0.30"}
    data.update(fields)
    data["file"] = (io.BytesIO(_sample_csv_bytes()), "test.csv")
    return client.post("/api/fieldtests", data=data,
                       content_type="multipart/form-data")


# ─── upload ─────────────────────────────────────────────────────────────────────
def test_upload_stores_and_detects(client):
    resp = _upload(client)
    assert resp.status_code == 201
    d = resp.get_json()
    assert isinstance(d["id"], int)
    assert d["facts"]["detected"] is True
    assert d["facts"]["name"] == "Custer #1"
    assert len(d["series"]["time"]) == len(d["series"]["raw"]) > 0
    assert d["series"]["detection"]["window"] is not None      # shaded event window


def test_upload_requires_name(client):
    data = {"file": (io.BytesIO(_sample_csv_bytes()), "t.csv")}
    resp = client.post("/api/fieldtests", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_requires_file(client):
    resp = client.post("/api/fieldtests", data={"name": "x"},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_bad_csv_is_400(client):
    bad = io.BytesIO(b"time,temperature\n0,20\n1,21\n")    # no ppm column
    resp = client.post("/api/fieldtests",
                       data={"name": "bad", "file": (bad, "bad.csv")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# ─── list / detail / delete ─────────────────────────────────────────────────────
def test_list_includes_uploaded_test(client):
    ft_id = _upload(client).get_json()["id"]
    listed = client.get("/api/fieldtests").get_json()["tests"]
    assert any(t["meta"]["id"] == ft_id for t in listed)
    assert listed[0]["facts"]["kind"] == "fieldtest"


def test_detail_returns_series_and_404(client):
    ft_id = _upload(client).get_json()["id"]
    d = client.get(f"/api/fieldtests/{ft_id}").get_json()
    assert d["meta"]["id"] == ft_id
    assert "excess" in d["series"] and "threshold" in d["series"]
    assert client.get("/api/fieldtests/99999").status_code == 404


def test_delete_removes_test(client):
    ft_id = _upload(client).get_json()["id"]
    assert client.delete(f"/api/fieldtests/{ft_id}").get_json()["ok"] is True
    assert client.get(f"/api/fieldtests/{ft_id}").status_code == 404
    assert client.delete("/api/fieldtests/99999").status_code == 404


# ─── sample generator ────────────────────────────────────────────────────────────
def test_sample_endpoint_creates_detectable_test(client):
    resp = client.post("/api/fieldtests/sample", json={"event_ppm": 4.0})
    assert resp.status_code == 201
    assert resp.get_json()["facts"]["detected"] is True


def test_sample_csv_download(client):
    resp = client.get("/api/sample.csv")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/csv")
    assert b"ppm" in resp.data.splitlines()[0]


# ─── interpret / ask degrade gracefully without a key ────────────────────────────
def test_interpret_falls_back_to_template(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ft_id = _upload(client).get_json()["id"]
    d = client.post(f"/api/fieldtests/{ft_id}/interpret", json={"use_ai": True}).get_json()
    assert d["source"] == "template"
    assert isinstance(d["text"], str) and len(d["text"]) > 100


def test_ask_returns_readable_error_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ft_id = _upload(client).get_json()["id"]
    d = client.post(f"/api/fieldtests/{ft_id}/ask",
                    json={"question": "was methane detected?"}).get_json()
    assert isinstance(d["answer"], str) and "OPENAI_API_KEY" in d["answer"]


# ─── compare / variance ──────────────────────────────────────────────────────────
def test_compare_returns_spread(client):
    a = client.post("/api/fieldtests/sample", json={"event_ppm": 4.0}).get_json()["id"]
    b = client.post("/api/fieldtests/sample", json={"event_ppm": 3.0}).get_json()["id"]
    d = client.get(f"/api/fieldtests/compare?ids={a},{b}").get_json()
    assert d["spread"]["n_tests"] == 2
    assert d["spread"]["n_detected"] == 2
    assert len(d["tests"]) == 2
    assert d["spread"]["excess_peak_ppm"]["spread"] >= 0.0
    assert d["tests"][0]["series"] is not None


# ─── accuracy report ──────────────────────────────────────────────────────────────
def test_accuracy_endpoint_returns_grade_and_floor(client):
    ft_id = _upload(client).get_json()["id"]
    d = client.get(f"/api/fieldtests/{ft_id}/accuracy").get_json()
    assert "grade" in d and "score" in d
    assert d["assumed_floor_ppm"] == pytest.approx(0.30)
    assert "recommendation" in d and isinstance(d["recommendation"], str)


def test_accuracy_endpoint_404_for_missing_test(client):
    assert client.get("/api/fieldtests/99999/accuracy").status_code == 404


# ─── multi-snapshot inversion ─────────────────────────────────────────────────────
def _upload_with_wind(client, wind_dir_deg, name="snap"):
    return _upload(client, name=name, wind_speed="2.0",
                   wind_dir_deg=str(wind_dir_deg), stability_class="4").get_json()["id"]


def test_invert_endpoint_fuses_two_tests(client):
    a = _upload_with_wind(client, 250.0, name="a")
    b = _upload_with_wind(client, 290.0, name="b")
    d = client.post("/api/fieldtests/invert", json={"ids": [a, b]}).get_json()
    assert d["n_snapshots"] == 2
    for key in ("x", "y", "Q", "converged", "ill_posed", "crb_std", "note", "assumption"):
        assert key in d


def test_invert_endpoint_requires_at_least_two_ids(client):
    a = _upload_with_wind(client, 270.0)
    resp = client.post("/api/fieldtests/invert", json={"ids": [a]})
    assert resp.status_code == 400
    assert "at least 2" in resp.get_json()["error"]


def test_invert_endpoint_requires_wind_metadata(client):
    a = _upload_with_wind(client, 270.0)
    b = _upload(client, name="no-wind").get_json()["id"]   # no wind fields set
    resp = client.post("/api/fieldtests/invert", json={"ids": [a, b]})
    assert resp.status_code == 400
    assert "wind" in resp.get_json()["error"]


def test_invert_endpoint_rejects_calm_wind(client):
    a = _upload_with_wind(client, 270.0)
    b = _upload(client, name="calm", wind_speed="0.1", wind_dir_deg="270",
               stability_class="4").get_json()["id"]
    resp = client.post("/api/fieldtests/invert", json={"ids": [a, b]})
    assert resp.status_code == 400
    assert "below the model minimum" in resp.get_json()["error"]


def test_invert_endpoint_404_for_missing_test(client):
    a = _upload_with_wind(client, 270.0)
    resp = client.post("/api/fieldtests/invert", json={"ids": [a, 99999]})
    assert resp.status_code == 404


def test_invert_endpoint_rejects_mismatched_sensor_distances(client):
    # The endpoint's whole premise is "SAME fixed sensor, different winds" — tests
    # recorded at different stored standoffs must be refused, not silently fused
    # into a confidently-wrong source fix.
    a = _upload(client, name="near", wind_speed="2.0", wind_dir_deg="250",
                stability_class="4", sensor_distance_m="20").get_json()["id"]
    b = _upload(client, name="far", wind_speed="2.0", wind_dir_deg="290",
                stability_class="4", sensor_distance_m="80").get_json()["id"]
    resp = client.post("/api/fieldtests/invert", json={"ids": [a, b]})
    assert resp.status_code == 400
    assert "sensor_distance_m" in resp.get_json()["error"]


# ─── warnings must reach the user ────────────────────────────────────────────────
def _messy_csv_bytes():
    # Unsorted timestamps (parse-time flag) + implausibly large readings
    # (process-time flag, recomputable from storage on every read).
    return b"time,ppm\n5,20000\n1,30000\n3,40000\n" + \
           b"".join(f"{i + 6},1.9\n".encode() for i in range(60))


def test_upload_surfaces_parse_and_process_warnings(client):
    data = {"name": "messy", "file": (io.BytesIO(_messy_csv_bytes()), "m.csv")}
    resp = client.post("/api/fieldtests", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 201
    warnings = resp.get_json()["warnings"]
    assert any("sorted by time" in w for w in warnings)      # parse-level
    assert any("plausible" in w for w in warnings)           # process-level


def test_detail_recomputes_process_warnings(client):
    data = {"name": "messy2", "file": (io.BytesIO(_messy_csv_bytes()), "m.csv")}
    ft_id = client.post("/api/fieldtests", data=data,
                        content_type="multipart/form-data").get_json()["id"]
    d = client.get(f"/api/fieldtests/{ft_id}").get_json()
    # Parse-level flags are gone after upload (raw CSV isn't stored) but anything
    # recomputable from the stored readings must re-surface on every read.
    assert any("plausible" in w for w in d["warnings"])


# ─── the existing model endpoints must still work ────────────────────────────────
def test_model_field_endpoint_unaffected(client):
    assert client.get("/api/field").status_code == 200

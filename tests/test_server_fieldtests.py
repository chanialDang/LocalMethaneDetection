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


# ─── the existing model endpoints must still work ────────────────────────────────
def test_model_field_endpoint_unaffected(client):
    assert client.get("/api/field").status_code == 200

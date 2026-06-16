"""
test_db.py — lock in the storage layer on the SQLite fallback.

Every test points db.py at a throwaway SQLite file (LOCAL_DB_PATH) with
DATABASE_URL removed, so the suite runs offline with no Postgres and no network —
exactly how local dev and CI run. The Postgres path shares the same code; only
the id-column DDL, the parameter placeholder, and the new-id fetch differ, and
those are exercised structurally here on SQLite.
"""
import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """A db module bound to an empty, isolated SQLite file with tables created."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("LOCAL_DB_PATH", str(tmp_path / "test.db"))
    from misc import db
    db.init_schema()
    return db


def _meta(name="Custer fence #1", **over):
    base = dict(
        name=name, site="Custer Road Transfer Station", test_date="2026-06-13",
        notes="afternoon run", wind_speed=2.0, wind_dir_deg=270.0,
        stability_class=4, sensor_distance_m=50.0, noise_ppm=0.30, sample_rate_hz=1.0,
    )
    base.update(over)
    return base


def test_tests_use_sqlite_not_postgres(fresh_db):
    assert fresh_db.is_postgres() is False


def test_init_schema_is_idempotent(fresh_db):
    fresh_db.init_schema()                      # second call must not raise
    assert fresh_db.list_field_tests() == []


def test_create_and_get_field_test(fresh_db):
    ft_id = fresh_db.create_field_test(_meta())
    assert isinstance(ft_id, int)
    got = fresh_db.get_field_test(ft_id)
    assert got["name"] == "Custer fence #1"
    assert got["site"] == "Custer Road Transfer Station"
    assert got["stability_class"] == 4
    assert got["noise_ppm"] == pytest.approx(0.30)
    assert got["created_at"] is not None          # DEFAULT CURRENT_TIMESTAMP fired


def test_defaults_apply_when_keys_missing(fresh_db):
    ft_id = fresh_db.create_field_test({"name": "bare"})
    got = fresh_db.get_field_test(ft_id)
    assert got["noise_ppm"] == pytest.approx(0.30)   # column DEFAULT
    assert got["sample_rate_hz"] == pytest.approx(1.0)
    assert got["site"] is None


def test_ids_increase(fresh_db):
    a = fresh_db.create_field_test(_meta("a"))
    b = fresh_db.create_field_test(_meta("b"))
    assert b > a


def test_readings_roundtrip_ordered_by_idx(fresh_db):
    ft_id = fresh_db.create_field_test(_meta())
    # Insert deliberately out of order to prove ORDER BY idx on read.
    rows = [
        (2, 2.0, 2.4, 21.0, 55.0),
        (0, 0.0, 1.9, 20.0, 55.0),
        (1, 1.0, 2.1, 20.5, 55.0),
    ]
    fresh_db.add_readings(ft_id, rows)
    got = fresh_db.get_readings(ft_id)
    assert [r["idx"] for r in got] == [0, 1, 2]
    assert got[0]["ppm_raw"] == pytest.approx(1.9)
    assert got[2]["temperature"] == pytest.approx(21.0)


def test_readings_allow_null_weather(fresh_db):
    ft_id = fresh_db.create_field_test(_meta())
    fresh_db.add_readings(ft_id, [(0, 0.0, 1.95, None, None)])
    got = fresh_db.get_readings(ft_id)
    assert got[0]["temperature"] is None and got[0]["humidity"] is None


def test_add_readings_empty_is_noop(fresh_db):
    ft_id = fresh_db.create_field_test(_meta())
    fresh_db.add_readings(ft_id, [])
    assert fresh_db.get_readings(ft_id) == []


def test_list_is_newest_first(fresh_db):
    first = fresh_db.create_field_test(_meta("first"))
    second = fresh_db.create_field_test(_meta("second"))
    listed = fresh_db.list_field_tests()
    assert [t["id"] for t in listed] == [second, first]


def test_get_unknown_returns_none(fresh_db):
    assert fresh_db.get_field_test(99999) is None


def test_delete_removes_test_and_its_readings(fresh_db):
    ft_id = fresh_db.create_field_test(_meta())
    fresh_db.add_readings(ft_id, [(0, 0.0, 1.9, None, None), (1, 1.0, 2.0, None, None)])
    fresh_db.delete_field_test(ft_id)
    assert fresh_db.get_field_test(ft_id) is None
    assert fresh_db.get_readings(ft_id) == []        # readings gone too

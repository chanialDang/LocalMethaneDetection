"""
db.py — Persistence for real-world field-test readings.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS DOES (plain English)
───────────────────────────────────────────────────────────────────────────────
The forward model PREDICTS readings; this is where REAL ones get stored. Each
"field test" (one deployment of the Figaro sensor at a site) is a row in
``field_tests``; its raw time-series readings are rows in ``readings``. We store
ONLY the raw readings — the processed view (baseline removal, smoothing,
detection) is recomputed on read by fieldtest.py, so we can retune the cleaning
later without re-uploading anything.

TWO BACKENDS, ONE INTERFACE
───────────────────────────────────────────────────────────────────────────────
  • DATABASE_URL set (Railway Postgres)  → use psycopg (imported lazily, so it is
    only required where Postgres is actually used — never for local dev or tests).
  • DATABASE_URL unset                   → fall back to a local SQLite file
    (``fieldtests.db``), so the app and the whole test suite run offline with no
    services to start.

The dialect is re-checked on every connection (not cached at import) so tests can
point LOCAL_DB_PATH / DATABASE_URL at a temp database via monkeypatch.

WHY THE SQL IS HAND-ROLLED (no ORM)
───────────────────────────────────────────────────────────────────────────────
The project is deliberately lean (numpy + flask, vendored Plotly, no scipy in the
processing path). Two tables and a handful of queries do not justify pulling in
SQLAlchemy; we keep the same minimalist spirit and isolate the only two dialect
differences — the parameter placeholder (``?`` vs ``%s``) and how a new id comes
back (``lastrowid`` vs ``RETURNING id``) — inside this module. No SQL leaves here.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# .env — make DATABASE_URL available even if this module is used before explain.py
# ─────────────────────────────────────────────────────────────────────────────
def _load_dotenv(path: str | Path | None = None) -> None:
    """
    Minimal, dependency-free .env reader (mirrors explain._load_dotenv).

    Copies KEY=VALUE lines from a .env beside this module into os.environ, but
    only for names not already set, so a real shell variable always wins. Kept
    self-contained so this storage module pulls in no heavy imports (numpy/plume)
    just to find DATABASE_URL. A missing file is a silent no-op.
    """
    if path is None:
        # .env lives at the project root (one level up from misc/).
        path = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line[:7].lower() == "export ":
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Dialect detection — re-read os.environ each call so tests can monkeypatch it
# ─────────────────────────────────────────────────────────────────────────────
def _database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def is_postgres() -> bool:
    """True when DATABASE_URL points at a Postgres instance (Railway)."""
    return _database_url().lower().startswith(("postgres://", "postgresql://"))


def _sqlite_path() -> str:
    """Where the local SQLite file lives when no DATABASE_URL is set."""
    return os.environ.get(
        "LOCAL_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "fieldtests.db"),
    )


def backend_label() -> str:
    """Human-readable backend name for logs/health (e.g. 'Railway Postgres')."""
    return "Railway Postgres" if is_postgres() else f"local SQLite ({_sqlite_path()})"


def _placeholder() -> str:
    """Parameter marker for the active dialect (%s for psycopg, ? for sqlite)."""
    return "%s" if is_postgres() else "?"


def connect():
    """
    Open a fresh DB connection for the active dialect.

    One connection per call (opened and closed inside each helper). That keeps
    things thread-safe under Flask's threaded dev server — each request gets its
    own connection — and is plenty fast for this volume.
    """
    if is_postgres():
        import psycopg  # lazy: only needed when actually talking to Postgres
        url = _database_url()
        if url.startswith("postgres://"):
            # Some providers emit the legacy scheme; psycopg wants postgresql://.
            url = "postgresql://" + url[len("postgres://"):]
        return psycopg.connect(url)
    conn = sqlite3.connect(_sqlite_path())
    conn.execute("PRAGMA foreign_keys = ON")   # honour ON DELETE CASCADE
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
# Column lists are the single source of truth for row→dict mapping below.
_FIELD_TEST_COLS = [
    "id", "name", "site", "test_date", "notes",
    "wind_speed", "wind_dir_deg", "stability_class", "sensor_distance_m",
    "noise_ppm", "sample_rate_hz", "created_at",
]
_FIELD_TEST_INSERT_COLS = _FIELD_TEST_COLS[1:-1]   # everything except id + created_at
_READING_COLS = ["idx", "t_seconds", "ppm_raw", "temperature", "humidity"]


def _schema_statements() -> list[str]:
    """DDL for the active dialect. Only the id column type differs."""
    if is_postgres():
        ft_id = "id BIGSERIAL PRIMARY KEY"
        rd_id = "id BIGSERIAL PRIMARY KEY"
        real = "DOUBLE PRECISION"
        ts = "TIMESTAMP"
        ft_ref = "BIGINT"
    else:
        ft_id = "id INTEGER PRIMARY KEY AUTOINCREMENT"
        rd_id = "id INTEGER PRIMARY KEY AUTOINCREMENT"
        real = "REAL"
        ts = "TEXT"
        ft_ref = "INTEGER"
    return [
        f"""
        CREATE TABLE IF NOT EXISTS field_tests (
            {ft_id},
            name TEXT NOT NULL,
            site TEXT,
            test_date TEXT,
            notes TEXT,
            wind_speed {real},
            wind_dir_deg {real},
            stability_class INTEGER,
            sensor_distance_m {real},
            noise_ppm {real} DEFAULT 0.30,
            sample_rate_hz {real} DEFAULT 1.0,
            created_at {ts} DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS readings (
            {rd_id},
            field_test_id {ft_ref} NOT NULL
                REFERENCES field_tests(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            t_seconds {real},
            ppm_raw {real},
            temperature {real},
            humidity {real}
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_readings_ft ON readings(field_test_id)",
    ]


def init_schema() -> None:
    """Create the tables if they do not exist. Idempotent; safe to call at boot."""
    conn = connect()
    try:
        cur = conn.cursor()
        for stmt in _schema_statements():
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Row → dict helpers (cursors return plain tuples in both dialects)
# ─────────────────────────────────────────────────────────────────────────────
def _field_test_row(row) -> dict:
    d = dict(zip(_FIELD_TEST_COLS, row))
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])   # JSON-safe across both dialects
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────
def create_field_test(meta: dict) -> int:
    """
    Insert one field test from a metadata dict and return its new id.

    Recognised keys (all optional except ``name``): name, site, test_date, notes,
    wind_speed, wind_dir_deg, stability_class, sensor_distance_m, noise_ppm,
    sample_rate_hz. Unknown keys are ignored.

    Only columns the caller actually provides (non-None) are written, so the rest
    fall back to their DDL defaults — noise_ppm=0.30, sample_rate_hz=1.0, others
    NULL. (A column DEFAULT only fires when the column is omitted from the INSERT,
    not when an explicit NULL is supplied.)
    """
    cols = [c for c in _FIELD_TEST_INSERT_COLS if meta.get(c) is not None]
    if "name" not in cols:
        raise ValueError("a field test requires a non-empty 'name'")
    values = [meta[c] for c in cols]
    p = _placeholder()
    placeholders = ", ".join([p] * len(cols))
    sql = f"INSERT INTO field_tests ({', '.join(cols)}) VALUES ({placeholders})"
    conn = connect()
    try:
        cur = conn.cursor()
        if is_postgres():
            cur.execute(sql + " RETURNING id", values)
            new_id = cur.fetchone()[0]
        else:
            cur.execute(sql, values)
            new_id = cur.lastrowid
        conn.commit()
        return int(new_id)
    finally:
        conn.close()


def add_readings(field_test_id: int, rows) -> None:
    """
    Bulk-insert raw readings for a field test.

    ``rows`` is an iterable of (idx, t_seconds, ppm_raw, temperature, humidity);
    temperature/humidity may be None when the CSV lacked those columns.
    """
    p = _placeholder()
    sql = (f"INSERT INTO readings "
           f"(field_test_id, idx, t_seconds, ppm_raw, temperature, humidity) "
           f"VALUES ({p}, {p}, {p}, {p}, {p}, {p})")
    data = [(field_test_id, *tuple(r)) for r in rows]
    if not data:
        return
    conn = connect()
    try:
        cur = conn.cursor()
        cur.executemany(sql, data)
        conn.commit()
    finally:
        conn.close()


def list_field_tests() -> list[dict]:
    """All field tests, newest first (metadata only — no readings)."""
    sql = f"SELECT {', '.join(_FIELD_TEST_COLS)} FROM field_tests ORDER BY id DESC"
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return [_field_test_row(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_field_test(field_test_id: int) -> dict | None:
    """One field test's metadata, or None if the id is unknown."""
    p = _placeholder()
    sql = f"SELECT {', '.join(_FIELD_TEST_COLS)} FROM field_tests WHERE id = {p}"
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, (field_test_id,))
        row = cur.fetchone()
        return _field_test_row(row) if row else None
    finally:
        conn.close()


def get_readings(field_test_id: int) -> list[dict]:
    """All raw readings for a field test, ordered by sample index."""
    p = _placeholder()
    sql = (f"SELECT {', '.join(_READING_COLS)} FROM readings "
           f"WHERE field_test_id = {p} ORDER BY idx")
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, (field_test_id,))
        return [dict(zip(_READING_COLS, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def delete_field_test(field_test_id: int) -> None:
    """
    Delete a field test and its readings.

    Deletes readings explicitly first so removal works the same on both backends,
    independent of whether ON DELETE CASCADE is enforced.
    """
    p = _placeholder()
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM readings WHERE field_test_id = {p}", (field_test_id,))
        cur.execute(f"DELETE FROM field_tests WHERE id = {p}", (field_test_id,))
        conn.commit()
    finally:
        conn.close()

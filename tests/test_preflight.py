"""
test_preflight.py — the field self-check CLI (misc/preflight.py).

Preflight reuses the real parse→process→aggregate pipeline, so these tests pin the
VERDICT logic (GO / CHECK / STOP) and the exit codes a field script would gate on.
Expected verdicts come from hand-known inputs (clean vs. messy vs. unparseable).
"""
import numpy as np

from physics import fieldtest
from misc.preflight import format_preflight, main, run_preflight


def test_preflight_go_on_clean_sample():
    # A realistic clean file (fractional 2-6 ppm, seconds time, T/H present) → GO.
    text = fieldtest.to_csv_text(fieldtest.make_sample_readings(n=120, seed=3))
    report = run_preflight(text)
    assert report["verdict"] == "GO"
    assert report["warnings"] == []
    assert report["facts"]["n_readings"] == 120
    assert report["facts"]["columns"]["ppm"] == "ppm"


def test_preflight_check_on_messy_upload():
    # Headerless + millis time + ADC-looking values → parses, but must flag CHECK
    # and enumerate every issue for the user.
    report = run_preflight("0,512\n1000,1024\n2000,2048\n3000,900\n4000,700\n")
    assert report["verdict"] == "CHECK"
    joined = " ".join(report["warnings"]).lower()
    assert "no header" in joined and "millisecond" in joined and "adc" in joined


def test_preflight_stop_on_no_methane_column():
    report = run_preflight("time,temperature\n0,20\n1,21\n")
    assert report["verdict"] == "STOP"
    assert report["error"] is not None
    assert report["facts"] == {}


def test_format_preflight_renders_verdict_and_warnings():
    report = run_preflight("0,512\n1000,1024\n2000,2048\n3000,900\n4000,700\n")
    text = format_preflight(report)
    assert "CHECK" in text
    assert "DATA WARNINGS" in text
    # STOP path renders the reason, not a crash.
    stop = format_preflight(run_preflight("time,temperature\n0,20\n"))
    assert "STOP" in stop


def test_preflight_main_exit_codes(tmp_path):
    clean = tmp_path / "clean.csv"
    clean.write_text(fieldtest.to_csv_text(fieldtest.make_sample_readings(n=120, seed=3)))
    assert main([str(clean)]) == 0                       # GO

    messy = tmp_path / "messy.csv"
    messy.write_text("0,512\n1000,1024\n2000,2048\n3000,900\n4000,700\n")
    assert main([str(messy)]) == 1                       # CHECK

    broken = tmp_path / "broken.csv"
    broken.write_text("time,temperature\n0,20\n1,21\n")
    assert main([str(broken)]) == 2                      # STOP


def test_preflight_missing_file_returns_stop_code():
    assert main(["/nonexistent/path/nope.csv"]) == 2

"""
test_rehearse.py — the deployment dress-rehearsal harness (misc/rehearse.py).

Pins that the rehearsal actually recovers its known hidden source through the real
parse → process → multi-snapshot inversion path — so the tool a student runs to build
confidence can't silently rot. The truth is baked into the scenario and recovered blind.
"""
from misc.rehearse import SENSORS, TRUE_SOURCE, main, rehearse


def test_rehearse_recovers_hidden_source(tmp_path):
    r = rehearse(str(tmp_path), seed=0)
    assert r["n_files"] == 3 * len(SENSORS)     # 3 wind episodes × 5 sensors
    assert r["n_snapshots"] == 3
    assert r["converged"] is True
    assert r["true_source"] == TRUE_SOURCE
    assert r["error_m"] < 2.0                   # multi-wind fusion triangulates tightly
    # Files were actually written and are re-readable as CSVs.
    written = list(tmp_path.glob("*.csv"))
    assert len(written) == r["n_files"]


def test_rehearse_main_runs_and_reports(tmp_path, capsys):
    rc = main(["--out", str(tmp_path), "--seed", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dress rehearsal" in out.lower()
    assert "RECOVERED" in out and "TRUE source" in out

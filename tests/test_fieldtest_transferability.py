"""
test_fieldtest_transferability.py — REAL-WORLD CSV ingestion hardening.

Companion to test_fieldtest.py. These tests pin the behaviour needed for real
Custer/Melissa uploads from a student's Arduino/ESP32 logger, whose output deviates
from the clean committed sample (samples/custer_sample.csv) in predictable ways:
comment banners, non-comma delimiters, header spellings with units, headerless dumps,
millis()/epoch timestamps, and uncalibrated (ADC/ppb/%) values.

Policy under test (user-confirmed): the parser NEVER silently misinterprets. It
auto-corrects only when the correction is unambiguous (e.g. epoch-magnitude time),
and otherwise warns loudly and proceeds. Expected values come from an independent
route (hand-known inputs), never from the parser's own arithmetic.
"""
import numpy as np
import pytest

from physics import fieldtest


# ─── A3: leading comment / blank preamble ────────────────────────────────────────
def test_parse_skips_leading_comment_banner():
    # Many SD/datalogger sketches print a "# started ..." banner before the header.
    # DictReader would otherwise treat the '#' line as the header and fail to find ppm.
    text = ("# Started 2026-07-09 14:00:00\n"
            "# Figaro TGS2611 warmup complete\n"
            "time,ppm\n0,1.9\n1,2.4\n2,3.0\n")
    out = fieldtest.parse_csv(text)
    assert out["n"] == 3
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.0])
    assert out["columns"]["ppm"] == "ppm"


def test_parse_skips_leading_blank_lines():
    text = "\n\n   \ntime,ppm\n0,1.9\n1,2.0\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2
    assert np.allclose(out["ppm"], [1.9, 2.0])


def test_parse_slash_comment_banner():
    text = "// logger v3\ntime,ppm\n0,1.9\n1,2.2\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2


# ─── A4: delimiter detection + decimal comma ─────────────────────────────────────
def test_parse_semicolon_delimited():
    # European CSV export (Excel non-US locale) uses ';'.
    out = fieldtest.parse_csv("time;ppm\n0;1.9\n1;2.4\n2;3.0\n")
    assert out["n"] == 3
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.0])


def test_parse_tab_delimited():
    out = fieldtest.parse_csv("time\tppm\n0\t1.9\n1\t2.4\n")
    assert out["n"] == 2
    assert np.allclose(out["ppm"], [1.9, 2.4])


def test_parse_semicolon_with_decimal_comma():
    # ';' delimiter with decimal-comma numbers — the classic European spreadsheet.
    out = fieldtest.parse_csv("time;ppm\n0;1,9\n1;2,4\n2;3,0\n")
    assert out["n"] == 3
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.0])


def test_comma_delimited_still_default():
    # Regression guard: plain comma files must be unaffected by delimiter sniffing.
    out = fieldtest.parse_csv("time,ppm\n0,1.9\n1,2.4\n")
    assert out["n"] == 2
    assert np.allclose(out["ppm"], [1.9, 2.4])


# ─── A1: header matching (units suffix, preferred>generic, headerless) ────────────
def test_parse_header_with_units_suffix():
    # 'CH4 (ppm)' / 'Temperature (C)' / 'Humidity (%)' — the natural Arduino header.
    text = "time,CH4 (ppm),Temperature (C),Humidity (%)\n0,1.9,20,50\n1,2.4,20.1,49\n"
    out = fieldtest.parse_csv(text)
    assert np.allclose(out["ppm"], [1.9, 2.4])
    assert out["columns"]["ppm"] == "CH4 (ppm)"
    assert out["temperature"] is not None and out["humidity"] is not None


def test_parse_header_methane_ppm_underscore():
    out = fieldtest.parse_csv("t,methane_ppm\n0,2.0\n1,2.2\n")
    assert np.allclose(out["ppm"], [2.0, 2.2])
    assert out["columns"]["ppm"] == "methane_ppm"


def test_parse_prefers_named_ppm_over_generic_raw():
    # When both a raw/ADC column and a real ppm column exist, ppm must win —
    # otherwise the ADC counts silently become the "methane" signal (audit #3).
    out = fieldtest.parse_csv("time,raw,ppm\n0,512,1.9\n1,530,2.4\n")
    assert out["columns"]["ppm"] == "ppm"
    assert np.allclose(out["ppm"], [1.9, 2.4])   # NOT [512, 530]


def test_parse_headerless_two_columns():
    # A raw serial dump with no header (row 1 is data): assume [time, ppm].
    out = fieldtest.parse_csv("0,1.9\n1,2.4\n2,3.0\n")
    assert out["n"] == 3
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.0])
    assert out["time"] is not None and np.allclose(out["time"], [0, 1, 2])
    assert any("no header" in w.lower() for w in out["warnings"])


def test_parse_headerless_single_column():
    out = fieldtest.parse_csv("1.9\n2.1\n2.0\n")
    assert out["n"] == 3                      # the first value is data, not a header
    assert np.allclose(out["ppm"], [1.9, 2.1, 2.0])
    assert out["time"] is None


def test_parse_headerless_four_columns():
    out = fieldtest.parse_csv("0,1.9,20,50\n1,2.4,20.1,49\n")
    assert np.allclose(out["ppm"], [1.9, 2.4])
    assert out["temperature"] is not None and out["humidity"] is not None
    assert any("no header" in w.lower() for w in out["warnings"])


def test_real_header_still_beats_headerless_path():
    # Regression guard: a normal headed file must NOT trip the headerless fallback.
    out = fieldtest.parse_csv("time,ppm\n0,1.9\n1,2.4\n")
    assert not any("no header" in w.lower() for w in out["warnings"])
    assert out["columns"]["ppm"] == "ppm"


# ─── A2: time units (millis/epoch auto-scale, ISO/clock parse) ────────────────────
def test_parse_millis_time_scaled_to_seconds():
    # Arduino millis(): 0,1000,2000,... read as raw seconds would report an event
    # lasting 1000x too long and collapse the meander window. Must divide by 1000.
    out = fieldtest.parse_csv("time,ppm\n0,1.9\n1000,2.4\n2000,3.0\n3000,2.0\n")
    assert np.allclose(out["time"], [0, 1, 2, 3])
    assert any("millisecond" in w.lower() for w in out["warnings"])


def test_parse_epoch_seconds_rebased():
    out = fieldtest.parse_csv(
        "time,ppm\n1752069781,1.9\n1752069782,2.4\n1752069784,3.0\n")
    assert np.allclose(out["time"], [0, 1, 3])      # rebased, NOT divided
    assert any("epoch" in w.lower() for w in out["warnings"])


def test_parse_epoch_millis_rebased_and_scaled():
    out = fieldtest.parse_csv(
        "time,ppm\n1752069781000,1.9\n1752069782000,2.4\n1752069783000,3.0\n")
    assert np.allclose(out["time"], [0, 1, 2])      # rebase THEN /1000
    w = " ".join(out["warnings"]).lower()
    assert "epoch" in w and "millisecond" in w


def test_parse_iso8601_timestamps():
    out = fieldtest.parse_csv(
        "timestamp,ppm\n2026-07-09T14:03:01,1.9\n2026-07-09T14:03:02,2.4\n"
        "2026-07-09T14:03:04,3.0\n")
    assert np.allclose(out["time"], [0, 1, 3])
    assert any("clock" in w.lower() or "date" in w.lower() for w in out["warnings"])


def test_parse_clock_time_hhmmss():
    out = fieldtest.parse_csv("time,ppm\n14:03:01,1.9\n14:03:02,2.4\n14:03:05,3.0\n")
    assert np.allclose(out["time"], [0, 1, 4])


def test_normal_seconds_time_unchanged():
    # Regression guard: plain elapsed-seconds time must be untouched, no warning.
    out = fieldtest.parse_csv("time,ppm\n0,1.9\n1,2.4\n2,3.0\n")
    assert np.allclose(out["time"], [0, 1, 2])
    assert not any("time" in w.lower() or "millisecond" in w.lower()
                   for w in out["warnings"])


def test_millis_time_fixes_meander_window():
    # The real payoff: with millis auto-scaled, a 10 s meander window spans ~10
    # samples (at 1 Hz), not 1. Independent check via aggregate_for_inversion.
    rows = "".join(f"{i*1000},1.9\n" for i in range(120))   # 120 s at 1 Hz in millis
    out = fieldtest.parse_csv("time,ppm\n" + rows)
    res = fieldtest.process_fieldtest(out["ppm"], time=out["time"])
    pt = fieldtest.aggregate_for_inversion(res, window_s=10)
    assert pt.n_window >= 8      # ~10 samples, NOT 1 (would be 1 if time were ms)


# ─── A5: unit-sanity warnings (ADC / ppb / % volume) ─────────────────────────────
def test_process_flags_adc_integer_column():
    # Whole-number readings well above the ppm band → almost certainly raw ADC.
    ppm = np.array([512, 1024, 2048, 3000, 4095, 2048] * 10, dtype=float)
    res = fieldtest.process_fieldtest(ppm)
    assert any("adc" in w.lower() or "counts" in w.lower() for w in res["warnings"])


def test_process_flags_ppb_magnitude():
    # ppb (methane ~1900-3000 ppb = 1.9-3.0 ppm), non-integer so not read as ADC.
    ppm = np.array([1900.5, 2400.5, 3000.5, 2100.5, 2600.5] * 10, dtype=float)
    res = fieldtest.process_fieldtest(ppm)
    assert any("ppb" in w.lower() for w in res["warnings"])


def test_process_flags_sub_background_values():
    # % volume or an Rs/Ro ratio: everything sits below the 1.9 ppm background.
    ppm = np.array([0.10, 0.05, 0.20, 0.15, 0.08, 0.12] * 10, dtype=float)
    res = fieldtest.process_fieldtest(ppm)
    assert any("background" in w.lower() or "%" in w or "ratio" in w.lower()
               for w in res["warnings"])


def test_process_realistic_ppm_no_unit_warning():
    # Regression guard: fractional 2-6 ppm readings (like the real sensor) must NOT
    # trip any unit-sanity warning.
    rng = np.random.default_rng(0)
    ppm = 1.9 + np.abs(rng.normal(1.5, 0.8, size=200))    # ~2-5 ppm, fractional
    res = fieldtest.process_fieldtest(ppm)
    assert not any("adc" in w.lower() or "ppb" in w.lower() or "background" in w.lower()
                   or "plausible" in w.lower() for w in res["warnings"])


# ─── A6: limits (row truncation, BOM on str) ─────────────────────────────────────
def test_parse_warns_when_truncated(monkeypatch):
    # Silent truncation reads as "covered everything"; it must be SAID.
    monkeypatch.setattr(fieldtest, "MAX_ROWS", 3)
    text = "time,ppm\n" + "".join(f"{i},1.9\n" for i in range(10))
    out = fieldtest.parse_csv(text)
    assert out["n"] == 3
    assert any("ignored" in w.lower() and "read" in w.lower() for w in out["warnings"])


def test_parse_exactly_max_rows_no_truncation_warning(monkeypatch):
    monkeypatch.setattr(fieldtest, "MAX_ROWS", 5)
    text = "time,ppm\n" + "".join(f"{i},1.9\n" for i in range(5))
    out = fieldtest.parse_csv(text)
    assert out["n"] == 5
    assert not any("ignored" in w.lower() for w in out["warnings"])


def test_parse_strips_bom_on_decoded_str():
    # A pre-decoded str carrying a UTF-8 BOM must not leave an invisible char on the
    # first header (which would defeat 'ppm' matching when ppm is the first column).
    out = fieldtest.parse_csv("﻿ppm,time\n1.9,0\n2.4,1\n")
    assert out["columns"]["ppm"] == "ppm"
    assert np.allclose(out["ppm"], [1.9, 2.4])


# ─── Capstone: every quirk at once (a realistic European-Excel + millis upload) ───
def test_maximally_messy_upload_end_to_end():
    # Comment banner + semicolon delimiter + decimal-comma numbers + units-suffix
    # headers + Arduino millis() time — all in one file. Independent expectation:
    # the hand-known values [1.9, 2.4, 3.1, 2.7, 2.0] at 1 Hz must come through, with
    # the ms→s auto-scale applied, and the pipeline must still process it.
    csv = (
        "# Custer test 3 - 2026-07-09 14:00\n"
        "# Figaro TGS2611 warmup done\n"
        "time;CH4 (ppm);Temperature (C);Humidity (%)\n"
        "0;1,9;20,0;50,0\n"
        "1000;2,4;20,1;49,8\n"
        "2000;3,1;20,2;49,6\n"
        "3000;2,7;20,3;49,4\n"
        "4000;2,0;20,4;49,2\n"
    )
    out = fieldtest.parse_csv(csv)
    assert out["n"] == 5
    assert np.allclose(out["ppm"], [1.9, 2.4, 3.1, 2.7, 2.0])
    assert np.allclose(out["time"], [0, 1, 2, 3, 4])            # millis auto-scaled
    assert out["columns"]["ppm"] == "CH4 (ppm)"
    assert out["temperature"] is not None and out["humidity"] is not None
    assert any("millisecond" in w.lower() for w in out["warnings"])
    res = fieldtest.process_fieldtest(
        out["ppm"], temperature=out["temperature"],
        humidity=out["humidity"], time=out["time"])
    assert res["weather_corrected"] is True                     # T/H both varied → fit ran


# ═══ Round 2: adversarial-fuzz findings (crashes / silent misreads) ═══════════════
def test_nul_bytes_do_not_crash_parse_or_preflight():
    # A serial glitch injects a NUL byte; csv.reader raises "line contains NUL".
    # Must be stripped (not crash), and preflight must NEVER raise.
    from misc.preflight import run_preflight
    text = "time,ppm\n0,1.9\x00\n1,2.4\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2 and np.allclose(out["ppm"], [1.9, 2.4])
    assert run_preflight(text)["verdict"] in ("GO", "CHECK")
    # A NUL that merges two lines must also not crash.
    assert fieldtest.parse_csv("time,ppm\n12,1.9\n13,2.0\x0034,2.1\n")["n"] >= 2


def test_prose_banner_lines_skipped():
    # SD/serial loggers print non-'#' banners before the header row.
    text = "Logging started\nBattery 3.7V\ntime,ppm\n0,1.9\n1,2.4\n"
    out = fieldtest.parse_csv(text)
    assert out["n"] == 2 and np.allclose(out["ppm"], [1.9, 2.4])
    assert out["columns"]["ppm"] == "ppm"


def test_ragged_rows_are_warned_not_silent():
    # A row with MORE fields than the header is ambiguous — e.g. a decimal comma in a
    # COMMA file ("0,1,9" → time=0, ppm=1, extra '9'). CSV-correct reading is ppm=1,
    # but it must be flagged so the user catches the delimiter mistake.
    out = fieldtest.parse_csv("time,ppm\n0,1,9\n1,2,4\n")
    assert any("more column" in w.lower() for w in out["warnings"])


def test_iso_timezone_and_ampm_clock_parse():
    tz = fieldtest.parse_csv(
        "time,ppm\n2026-07-09T14:03:01+02:00,1.9\n2026-07-09T14:03:03+02:00,2.4\n")
    assert tz["time"] is not None and np.allclose(tz["time"], [0, 2])
    ap = fieldtest.parse_csv("time,ppm\n02:03:04 PM,1.9\n02:03:06 PM,2.4\n")
    assert ap["time"] is not None and np.allclose(ap["time"], [0, 2])


def test_invisible_unicode_in_header_stripped():
    # A zero-width space (U+200B) copy-pasted into a header must not defeat matching.
    out = fieldtest.parse_csv("time,p​pm\n0,1.9\n1,2.4\n")
    assert out["columns"]["ppm"] == "ppm"
    assert np.allclose(out["ppm"], [1.9, 2.4])

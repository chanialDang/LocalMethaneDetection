"""
fieldtest.py — Turn an uploaded CSV of real readings into a clean, plottable result.

═══════════════════════════════════════════════════════════════════════════════
WHERE THIS SITS (plain English)
───────────────────────────────────────────────────────────────────────────────
A field test is one deployment of the Figaro sensor: it produces a TIME SERIES of
ppm readings at a single location — not the smooth spatial field the forward model
predicts. This module is the bridge between a raw uploaded CSV and the per-test
graph the dashboard shows:

    CSV text ──parse_csv──▶ raw arrays ──process_fieldtest──▶ raw + baseline +
                                                              excess + detection

It does NOT reimplement any signal processing — it just feeds the existing,
tested tools in processing.py in the documented order
(temp/humidity correct → baseline subtract → smooth → detect) and packages the
result for plotting and for the chatbot to interpret.

DISPLAY FRAME NOTE
───────────────────────────────────────────────────────────────────────────────
When temperature/humidity columns are present we remove the weather-varying part
BEFORE estimating the baseline (it makes the plume stand out more cleanly). For
display, though, everything is reported back in the original RAW frame: the
returned ``baseline`` has the weather part added back in, so on the graph
``raw ≈ baseline + excess`` holds and the floor visibly hugs the raw trace.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from physics.accuracy import estimate_lag1_autocorrelation, estimate_noise_floor
from physics.processing import (
    detect_pattern,
    moving_average,
    subtract_baseline,
    temp_humidity_correct,
)
from physics.sensor_sim import (
    DETECT_K,
    SENSOR_NOISE_PPM,
    effective_noise_floor,
    make_plume_event,
    synthetic_timeseries,
)

# Upper bound on rows we ingest from one upload — guards against a runaway file.
MAX_ROWS = 200_000

# Plausibility band for a soft unit-mismatch warning — NOT the sensor's rated
# 500-12,500 ppm range (real readings sit at 2-40 ppm above background, far below
# "rated"); this is a wide sanity bound (25% over the rated ceiling) that only
# flags, never rejects, since a real large leak can legitimately read high.
# Checked in process_fieldtest (not parse_csv) so STORED tests re-warn on read.
_PPM_PLAUSIBLE_MIN = -1.0
_PPM_PLAUSIBLE_MAX = 15_000.0

# process_fieldtest: a partially-filled temperature/humidity column is only
# interpolated (instead of being dropped entirely) when at least this many real
# values remain — mirrors accuracy.estimate_noise_floor's minimum-sample threshold.
_MIN_GOOD_FOR_INTERP = 8

# process_fieldtest: a sample_rate_hz outside this is treated as bad input (Arduino
# loop rates are nowhere near this fast) and falls back to 1.0 Hz with a warning.
_MAX_SAMPLE_RATE_HZ = 1000.0

# Accepted header spellings (compared case-insensitively, stripped). The first
# matching column in the file wins.
_PPM_NAMES = {"ppm", "ch4", "ch4_ppm", "methane", "reading", "value", "conc",
              "concentration", "ppm_raw", "raw", "ch4_ppm_raw"}
# Generic/ambiguous synonyms — a column literally called 'raw'/'value'/'reading'
# could be ADC counts or voltage, not calibrated ppm. A PREFERRED name (ch4/methane/
# ppm/…) is always chosen over these when both are present; only when no preferred
# column exists do we fall back to a generic one (accepted, then flagged downstream).
_PPM_GENERIC = {"reading", "value", "raw"}
_PPM_PREFERRED = _PPM_NAMES - _PPM_GENERIC
_TIME_NAMES = {"time", "t", "seconds", "sec", "s", "timestamp", "elapsed",
               "time_s", "t_seconds", "ts"}
_TEMP_NAMES = {"temp", "temperature", "t_c", "tempc", "temp_c", "celsius", "tc"}
_HUMID_NAMES = {"humidity", "rh", "humid", "relative_humidity", "humidity_pct",
                "rh_pct", "humid_pct"}
# Raw sensor-voltage / ADC column spellings. A file with one of THESE but no ppm
# column is run through the sensor_frontend voltage→ppm converter (gated on the
# front-end being calibrated). Kept disjoint from _PPM_NAMES so 'raw'/'value'
# (generic ppm synonyms) are never mistaken for voltage, and vice-versa.
_VOLT_NAMES = {"v", "volt", "voltage", "vout", "v_out", "mv", "adc", "adc_raw",
               "counts"}
_ALL_COLUMN_NAMES = _PPM_NAMES | _TIME_NAMES | _TEMP_NAMES | _HUMID_NAMES | _VOLT_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# CSV PARSING
# ─────────────────────────────────────────────────────────────────────────────
# A decimal-comma numeric cell ("1,9" meaning 1.9) — only trusted when the file's
# delimiter is NOT a comma (European ';'/tab exports), so a real comma-delimited
# file's cells (already split) can never be misread as a decimal comma.
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d+$")


def _to_text(data) -> str:
    """Accept a file-like, bytes, or str and return decoded text (BOM-tolerant).

    Strips a UTF-8 BOM for BOTH bytes (via utf-8-sig) and already-decoded str
    (leading U+FEFF), so the first header — often the ppm column — is never left
    with an invisible BOM that defeats name matching.
    """
    if hasattr(data, "read"):
        data = data.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    # Drop characters that are never meaningful in a numeric readings CSV but do break
    # parsing: embedded NUL bytes (serial glitch — csv.reader rejects "line contains
    # NUL"), and zero-width / invisible unicode + stray BOMs (from a copy-pasted header
    # — they silently defeat column-name matching). Strip rather than crash/reject.
    for ch in ("\x00", "​", "‌", "‍", "⁠", "﻿"):
        data = data.replace(ch, "")
    return data


def _strip_preamble(text: str) -> str:
    """Drop leading blank lines, '#'/'//' comment lines, AND prose banner lines that
    precede the header.

    Dataloggers print a banner before the real header — either commented
    ('# Started 2026-…', '// logger v3') or plain prose ('Logging started',
    'Battery 3.7V') — or a blank lead-in; csv.DictReader would otherwise treat that
    first physical line as the header and fail to find a methane column. A leading
    line is KEPT (as the header / first data row) once it looks like tabular data: it
    contains a delimiter, OR is a single numeric value (header-less), OR carries a
    recognized column-name token. Everything before that — blanks, comments, and prose
    banners with no delimiter/number/known token — is skipped. Only LEADING junk is
    removed; a comment/banner AFTER the header stays for the row loop to skip as a
    no-reading row."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        if (any(d in s for d in (",", ";", "\t"))
                or _num(s) is not None
                or (_tokens(s) & _ALL_COLUMN_NAMES)):
            return "\n".join(lines[i:])
        # else: a prose banner line ('Logging started') — skip it and keep looking.
    return text   # all blank/comment/banner → let DictReader raise the friendly error


def _sniff_delimiter(text: str) -> str:
    """Pick the field delimiter from the header line by a simple frequency count.

    Deterministic and robust where csv.Sniffer is finicky (short files, quoted
    cells): comma unless ';' or tab clearly dominates the FIRST line. A student's
    Excel 'Save As CSV' on a non-US locale, or a serial capture, may use ';' or
    tab; everything else about parsing is unchanged."""
    header = text.splitlines()[0] if text else ""
    counts = {",": header.count(","), ";": header.count(";"), "\t": header.count("\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _num(value, decimal_comma: bool = False):
    """Parse one cell to float, or None if blank/non-numeric/non-finite (NaN/Inf).

    When ``decimal_comma`` is set (a non-comma-delimited file), a cell like "1,9"
    is read as 1.9. This is gated on the delimiter so a comma-delimited file — whose
    cells are already split and never contain a stray comma — is never affected.
    """
    if value is None:
        return None
    s = str(value).strip()
    try:
        v = float(s)
    except (ValueError, TypeError):
        if decimal_comma and _DECIMAL_COMMA_RE.match(s):
            try:
                v = float(s.replace(",", "."))
            except (ValueError, TypeError):
                return None
        else:
            return None
    return v if math.isfinite(v) else None


# Recognized date-time and clock-only spellings for a non-numeric time column.
_TS_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",          # RTC with timezone offset
    "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
)
_TS_CLOCK_FORMATS = ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M",
                     "%I:%M:%S %p", "%I:%M %p")             # 12-hour AM/PM


def _parse_timestamps(raws):
    """Convert an ISO-8601 / date / clock-time column to ELAPSED SECONDS.

    Returns a float ndarray (seconds from the first row), or None if the column
    isn't a recognizable, gap-free time format — any missing cell yields None, so a
    truly gappy clock is dropped rather than guessed. A clock-only ('HH:MM:SS')
    column is unwrapped across a single midnight rollover.
    """
    if not raws or any(r is None or str(r).strip() == "" for r in raws):
        return None
    strs = [str(r).strip().rstrip("Zz") for r in raws]

    def _first_match(formats):
        for fmt in formats:
            try:
                datetime.strptime(strs[0], fmt)
                return fmt
            except ValueError:
                continue
        return None

    if _first_match(_TS_DATETIME_FORMATS):
        kind, fmts = "datetime", _TS_DATETIME_FORMATS
    elif _first_match(_TS_CLOCK_FORMATS):
        kind, fmts = "clock", _TS_CLOCK_FORMATS
    else:
        return None

    parsed = []
    for s in strs:
        dt = None
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return None                        # column not uniformly this format
        parsed.append(dt)

    if kind == "clock":
        secs = np.asarray(
            [p.hour * 3600 + p.minute * 60 + p.second + p.microsecond / 1e6
             for p in parsed], dtype=float)
        if secs.size > 1:                      # unwrap one midnight rollover
            secs[1:] += np.cumsum(np.where(np.diff(secs) < -1.0, 86400.0, 0.0))
    else:
        t0 = parsed[0]
        secs = np.asarray([(p - t0).total_seconds() for p in parsed], dtype=float)
    return secs - secs[0]


def _normalize_time_units(time_arr, warnings: list):
    """Rebase epoch timestamps and auto-scale millisecond clocks to ELAPSED SECONDS.

    Auto-corrects only on strong evidence, always with a loud warning (policy: never
    silently misread). Two independent corrections:
      • Absolute (epoch) magnitude (>1e9): a field test never starts billions of
        seconds in, so this is unambiguously a wall-clock stamp → rebase to elapsed.
      • Implausibly large inter-sample step (median ≥ 50): Arduino millis() (or
        epoch-ms after rebasing) reads 1000× too slow → divide by 1000. A genuine
        very-slow cadence trips this too, but the warning tells the user to override.
    """
    t = np.asarray(time_arr, dtype=float)
    if t.size == 0:
        return t
    tmin = float(np.min(t))
    if tmin > 1e9:
        t = t - tmin
        warnings.append(
            "Time column looks like an absolute (epoch) timestamp; rebased to "
            "seconds elapsed from the first sample.")
    if t.size > 1:
        med_dt = float(np.median(np.abs(np.diff(t))))
        if med_dt >= 50.0:
            t = t / 1000.0
            warnings.append(
                f"Time advances ~{med_dt:g} units per sample — that looks like "
                "milliseconds, so it was divided by 1000 to get seconds. If your "
                "logger really samples this slowly, pass the sample rate explicitly.")
    return t


def _tokens(name) -> set:
    """Lowercased alphanumeric tokens of a header: 'CH4 (ppm)' → {'ch4', 'ppm'}."""
    return set(re.findall(r"[a-z0-9]+", (name or "").lower()))


def _pick(norm_to_orig: dict, candidates: set, preferred: set | None = None):
    """Return the original header matching `candidates`, else None.

    Match order (most specific first, so a real column always beats a fuzzy one):
      1. exact normalized-name match on a `preferred` name (methane: 'ppm'/'ch4'/…)
      2. exact normalized-name match on any candidate  (the original behaviour)
      3. token match on a `preferred` name             ('CH4 (ppm)' → token 'ch4')
      4. token match on any candidate                  ('Humidity (%)' → token 'humidity')

    Token matching splits on non-alphanumerics, so 'runtime' does NOT match 'time'
    (it is a single token) — avoiding the substring false positives a naive
    `in`-check would create.
    """
    items = list(norm_to_orig.items())            # (normalized_full, original), column order
    if preferred:
        for low, original in items:
            if low in preferred:
                return original
    for low, original in items:
        if low in candidates:
            return original
    if preferred:
        for low, original in items:
            if _tokens(original) & preferred:
                return original
    for low, original in items:
        if _tokens(original) & candidates:
            return original
    return None


def _col(vals):
    # A column is only usable if EVERY row has a number — used for `time`, where a
    # gap can't be sensibly interpolated against itself. See _col_gappy for the
    # temperature/humidity columns, whose gaps process_fieldtest may fill later.
    if any(v is None for v in vals):
        return None
    return np.asarray(vals, dtype=float)


def _col_gappy(vals):
    """Temperature/humidity column from float-or-None cells: NaN where missing,
    None when the column is entirely absent. parse_csv reports reality; the
    drop-or-interpolate POLICY lives in process_fieldtest (one home, so stored
    tests re-read from the DB get the identical treatment)."""
    if all(v is None for v in vals):
        return None
    return np.asarray([v if v is not None else np.nan for v in vals], dtype=float)


def parse_csv(data) -> dict:
    """
    Parse an uploaded readings CSV into aligned numpy arrays.

    Flexible about column names (see the *_NAMES sets): it needs a header row with
    at least a methane column; time, temperature, and humidity are optional. Rows
    without a usable numeric reading (blank, non-numeric, or NaN/Inf) are skipped.
    The time column is only usable if every row has one (a gappy clock is dropped,
    with a warning); temperature/humidity gaps are kept as NaN — parse reports
    reality, and ``process_fieldtest`` owns the drop-or-interpolate policy, so a
    test re-read from storage gets the identical treatment.

    Never raises on merely SUSPICIOUS data (unsorted timestamps, duplicates) —
    those are reported in `warnings` instead, since a meandering field upload may
    legitimately have an imperfect clock. (The out-of-range ppm check lives in
    ``process_fieldtest`` for the same one-home reason as the gap policy.)

    Returns
    -------
    dict with keys:
      ppm         : 1-D float array (required) — the raw reading incl. background.
      time        : 1-D float array or None (synthesized later if None), sorted.
      temperature : 1-D float array (NaN gaps) or None.
      humidity    : 1-D float array (NaN gaps) or None.
      n           : number of readings.
      columns     : which original headers were matched.
      warnings    : list of human-readable strings flagging suspicious input.

    Raises ValueError with a friendly message if there is no header or no readings.
    """
    text = _strip_preamble(_to_text(data))
    delimiter = _sniff_delimiter(text)
    decimal_comma = delimiter != ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError(
            "The CSV looks empty — it needs a header row and at least a "
            "methane column (e.g. 'ppm')."
        )

    norm = {(name or "").strip().lower(): name for name in reader.fieldnames}
    ppm_key = _pick(norm, _PPM_NAMES, preferred=_PPM_PREFERRED)
    volt_key = _pick(norm, _VOLT_NAMES)

    # Which column carries the reading, and whether it needs voltage→ppm conversion.
    # A real ppm column always wins; a raw sensor-voltage column is used only when
    # there is no ppm column, and only if the front-end is calibrated (else we STOP
    # loudly rather than emit a fabricated ppm — the gate).
    from_voltage = False
    reading_key = ppm_key
    if ppm_key is None and volt_key is not None:
        from physics import sensor_frontend as _sf
        ready, missing = _sf.calibration_status()
        if not ready:
            raise ValueError(
                f"Found a sensor-voltage column ({volt_key!r}) but no calibrated "
                "ppm column, and the voltage→ppm front-end is not calibrated yet "
                f"(missing: {', '.join(missing)}). Set those constants in "
                "physics/sensor_frontend.py (see docs/CALIBRATION.md), or upload a "
                "CSV that already has a ppm column."
            )
        reading_key = volt_key
        from_voltage = True
    elif ppm_key is None:
        # No recognizable methane header. If row 1 is entirely numeric there is
        # probably NO header at all (a raw serial dump) — fall back to positional
        # columns with a loud warning. Otherwise the header truly lacks methane.
        field_nums = [_num(fn, decimal_comma) for fn in reader.fieldnames]
        if reader.fieldnames and all(v is not None for v in field_nums):
            return _parse_headerless(text, delimiter, len(reader.fieldnames))
        raise ValueError(
            "No methane column found. Add a header like 'ppm' "
            "(accepted: ppm, ch4, methane, reading, value, concentration), a raw "
            "voltage column (v, voltage, vout, adc) once the front-end is "
            "calibrated, or upload a header-less file whose 1st column is time "
            "and 2nd is ppm."
        )
    time_key = _pick(norm, _TIME_NAMES)
    temp_key = _pick(norm, _TEMP_NAMES)
    humid_key = _pick(norm, _HUMID_NAMES)

    times, ppms, temps, humids, time_raws = [], [], [], [], []
    n_skipped = 0
    n_ragged = 0
    truncated = False
    for row in reader:
        if row.get(None):        # DictReader collects fields beyond the header here
            n_ragged += 1
        ppm = _num(row.get(reading_key), decimal_comma)
        if ppm is None:
            n_skipped += 1
            continue                          # skip rows without a usable reading
        ppms.append(ppm)
        times.append(_num(row.get(time_key), decimal_comma) if time_key else None)
        time_raws.append(row.get(time_key) if time_key else None)   # for date/clock strings
        temps.append(_num(row.get(temp_key), decimal_comma) if temp_key else None)
        humids.append(_num(row.get(humid_key), decimal_comma) if humid_key else None)
        if len(ppms) >= MAX_ROWS:
            truncated = next(reader, None) is not None   # were there more rows?
            break

    if not ppms:
        raise ValueError("No numeric methane readings were found in the file.")

    ppm_arr = np.asarray(ppms, dtype=float)
    warnings: list[str] = []

    if ppm_key is not None and volt_key is not None:
        warnings.append(
            f"Both a ppm column ({ppm_key!r}) and a raw voltage column "
            f"({volt_key!r}) were present; the calibrated ppm column was used and "
            "the voltage column ignored."
        )

    if n_skipped:
        warnings.append(
            f"{n_skipped} row(s) had no usable methane reading "
            "(blank, non-numeric, or NaN/Inf) and were skipped."
        )

    if truncated:
        warnings.append(
            f"File exceeded {MAX_ROWS:,} readings; only the first {MAX_ROWS:,} were "
            "read and the rest were ignored."
        )

    if n_ragged:
        warnings.append(
            f"{n_ragged} row(s) had more columns than the header; the extra value(s) "
            "were ignored — check your delimiter (a decimal comma in a comma-separated "
            "file does this)."
        )

    temp_arr = _col_gappy(temps)
    humid_arr = _col_gappy(humids)

    time_arr = _col(times)
    if time_arr is not None:
        time_arr = _normalize_time_units(time_arr, warnings)
    elif time_key is not None:
        # Numeric parse failed for some/all rows. Either genuine timestamp strings
        # (ISO-8601 / clock) we can convert to elapsed seconds, or a truly gappy
        # numeric clock we must drop (file order assumed).
        ts = _parse_timestamps(time_raws)
        if ts is not None and ts.size == len(ppm_arr):
            time_arr = ts
            warnings.append(
                "Time column was a date/clock format; converted to seconds "
                "elapsed from the first sample."
            )
        else:
            n_missing = sum(1 for v in times if v is None)
            warnings.append(
                f"The time column has {n_missing} unparseable value(s) (need "
                "numeric seconds or a recognizable date/clock format); it was "
                "ignored (file order assumed)."
            )
    if time_arr is not None and time_arr.size > 1:
        diffs = np.diff(time_arr)
        if np.any(diffs < 0):                 # cheap O(n) gate; sort only if needed
            order = np.argsort(time_arr, kind="stable")
            warnings.append(
                "Timestamps were not in increasing order; rows were sorted by time."
            )
            ppm_arr = ppm_arr[order]
            time_arr = time_arr[order]
            if temp_arr is not None:
                temp_arr = temp_arr[order]
            if humid_arr is not None:
                humid_arr = humid_arr[order]
            diffs = np.diff(time_arr)
        n_dupe = int(np.sum(diffs == 0))
        if n_dupe:
            warnings.append(f"{n_dupe} duplicate timestamp(s) found in the time column.")

    if from_voltage:
        # The 'ppm' arrays currently hold raw voltage; convert the whole column at
        # once (T/RH aligned and already sorted). Done AFTER sorting so the T/RH
        # correction lines up sample-for-sample with the reading.
        ppm_arr = np.asarray(
            _sf.voltage_to_ppm(ppm_arr, temp_arr, humid_arr), dtype=float
        )
        warnings.append(
            "Readings were a raw sensor-voltage column, converted to ppm by the "
            "front-end (physics/sensor_frontend.py). NOTE: the power-law A/m are "
            "DATASHEET-TYPICAL and unverified at the 2–40 ppm operating range — "
            "treat the ppm scale as provisional until you refit against span gas."
        )
        finite = np.isfinite(ppm_arr)
        n_bad_v = int((~finite).sum())
        if n_bad_v:
            ppm_arr = ppm_arr[finite]
            if time_arr is not None:
                time_arr = time_arr[finite]
            if temp_arr is not None:
                temp_arr = temp_arr[finite]
            if humid_arr is not None:
                humid_arr = humid_arr[finite]
            warnings.append(
                f"{n_bad_v} voltage sample(s) fell outside the valid divider range "
                "(0 < V_out < V_c) and were dropped after conversion."
            )
        if ppm_arr.size == 0:
            raise ValueError(
                "No valid readings after voltage→ppm conversion — every sample was "
                "outside the divider range (0 < V_out < V_c). Check V_c/wiring."
            )

    return {
        "ppm": ppm_arr,
        "time": time_arr,
        "temperature": temp_arr,
        "humidity": humid_arr,
        "n": len(ppm_arr),
        "columns": {"ppm": ppm_key, "time": time_key,
                    "temperature": temp_key, "humidity": humid_key,
                    "voltage": volt_key},
        "warnings": warnings,
    }


def _parse_headerless(text: str, delimiter: str, ncols: int) -> dict:
    """Re-parse a header-less numeric dump by assigning columns positionally.

    Row 1 parsed as all-numeric, so there is no header line; assume the logger
    wrote columns in the conventional order and prepend a synthetic header, then
    re-run ``parse_csv`` so EVERY row (including the former row 1) is read as data.
    A loud warning records the assumption — this is a best guess with the reasoning
    surfaced, not a silent reinterpretation. Column order assumed:
      1 col  → ppm
      2 cols → time, ppm
      3 cols → time, ppm, temperature
      4+ cols→ time, ppm, temperature, humidity  (extra columns ignored)
    """
    names = ["ppm"] if ncols == 1 else ["time", "ppm", "temperature", "humidity"][:ncols]
    out = parse_csv(delimiter.join(names) + "\n" + text)
    out["warnings"].insert(
        0,
        "No header row was detected (row 1 is numeric); assumed columns by "
        "position: " + ", ".join(names) + ". Verify this matches your logger's "
        "column order.",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def _odd(x) -> int:
    """Nearest odd integer ≥ 1 (the processing windows want odd, centred windows)."""
    x = int(x)
    if x < 1:
        return 1
    return x if x % 2 == 1 else x + 1


def _densify(col, name: str, warnings: list) -> tuple:
    """
    Apply the gap policy to one temperature/humidity column (NaN = missing).

    Dense already → unchanged. Fewer than ``_MIN_GOOD_FOR_INTERP`` real values →
    drop the column (too sparse to trust), with a warning. Otherwise fill the
    gaps by linear interpolation against sample index, with a warning — a few
    dropped serial-print lines don't disable weather correction for the record.

    Returns (dense_array_or_None, real_mask_or_None). The mask marks samples the
    logger actually recorded; the weather FIT must train only on those (an
    interpolated ramp through a plume-spanning gap correlates with the event and
    would teach the fit to subtract real methane — see the weather-fit note in
    ``process_fieldtest``).
    """
    if col is None:
        return None, None
    arr = np.asarray(col, dtype=float)
    real = np.isfinite(arr)
    n_good = int(real.sum())
    if n_good == arr.size:
        return arr, real
    if n_good < _MIN_GOOD_FOR_INTERP:
        warnings.append(
            f"The {name} column has only {n_good} usable value(s) — too sparse "
            "to trust, so it was dropped (no weather correction from it)."
        )
        return None, None
    idx = np.arange(arr.size, dtype=float)
    filled = arr.copy()
    filled[~real] = np.interp(idx[~real], idx[real], arr[real])
    warnings.append(
        f"{int(arr.size - n_good)} missing {name} value(s) were linearly "
        "interpolated (interpolated samples are excluded from the weather fit)."
    )
    return filled, real


def _unit_sanity_warnings(ppm, n: int) -> list:
    """Non-rejecting warnings when a column's magnitude doesn't look like ppm.

    The reading is NEVER dropped (a real large leak can legitimately read high) —
    these only flag a LIKELY unit mistake so a miscalibrated upload can't silently
    masquerade as ppm. Lives in process_fieldtest (not parse_csv) so STORED tests
    re-warn on every read (F14). Auto-correction is deliberately NOT done here: the
    ppb/ADC/% cases are ambiguous (a real leak can read high), so we warn and leave
    the value, matching the 'auto-fix only when unambiguous' policy.
    """
    w = []
    n_bad = int(np.sum((ppm < _PPM_PLAUSIBLE_MIN) | (ppm > _PPM_PLAUSIBLE_MAX)))
    if n_bad:
        w.append(
            f"{n_bad} of {n} reading(s) fall outside the plausible "
            f"{_PPM_PLAUSIBLE_MIN:g}-{_PPM_PLAUSIBLE_MAX:g} ppm range — "
            "check units (e.g. raw ADC counts mistaken for ppm)."
        )
    finite = ppm[np.isfinite(ppm)]
    if finite.size == 0:
        return w
    med = float(np.median(finite))
    integer_like = bool(np.allclose(finite, np.round(finite)))
    if integer_like and med > 50.0:
        w.append(
            "Readings are whole numbers well above the ppm band — they look like "
            "raw ADC/uncalibrated counts, not calibrated ppm. Apply your sensor's "
            "calibration curve before trusting the result."
        )
    elif med > 100.0:
        w.append(
            f"Readings (median {med:g}) are far above the expected ~2-40 ppm band — "
            "if this column is in ppb, divide by 1000 to get ppm."
        )
    if med < 0.5 and finite.size >= 5:
        w.append(
            f"Readings (median {med:g}) sit below the ~1.9 ppm methane background — "
            "check units (a % volume or an Rs/Ro ratio, not ppm?)."
        )
    return w


def process_fieldtest(
    ppm,
    temperature=None,
    humidity=None,
    time=None,
    noise_ppm: float | None = None,
    sample_rate_hz: float = 1.0,
    baseline_window: int | None = None,
    smooth_window: int | None = None,
    min_run: int = 5,
    k: float = DETECT_K,
) -> dict:
    """
    Run the standard cleaning pipeline on one field test's readings.

    Order (all from processing.py): optional temperature/humidity correction →
    rolling low-percentile baseline subtraction → moving-average smoothing →
    sustained-event detection against k·floor.

    Two passes (so detection is honest about DRIFT, not just fast noise): a
    provisional pass locates the event; the final pass (a) re-fits the weather
    correction EXCLUDING that event window, so humidity that rises during a plume
    can't be mis-attributed, and (b) measures the detection floor from the record's
    own quiet samples — `√(random² + bias²)` via `estimate_noise_floor` /
    `effective_noise_floor`, so non-averageable drift raises the bar instead of
    sailing past a fixed assumed floor. `noise_ppm=None` (default) measures the
    floor; passing a float overrides it verbatim (used by tests / legacy callers).

    Window defaults scale with record length: the baseline window is the whole
    record (capped at 601 samples) so a single plume bump sits well inside it, and
    the smoothing window is a small fraction of the record. Both can be overridden.

    Returns a dict of plottable arrays (time, raw, baseline, excess), the detection
    threshold, the Detection object, and the settings used. All in the RAW frame:
    ``raw ≈ baseline + excess`` (see the module docstring).
    """
    ppm = np.asarray(ppm, dtype=float)
    n = len(ppm)
    if n == 0:
        raise ValueError("No readings to process.")
    if not np.any(np.isfinite(ppm)):
        raise ValueError("All readings are non-finite (NaN/inf) — nothing to process.")

    warnings: list[str] = []
    if time is None:
        if sample_rate_hz and 0 < sample_rate_hz <= _MAX_SAMPLE_RATE_HZ:
            rate = sample_rate_hz
        else:
            rate = 1.0
            warnings.append(
                f"sample_rate_hz={sample_rate_hz!r} is not a plausible cadence; "
                "assumed 1.0 Hz."
            )
        time = np.arange(n) / rate
    else:
        time = np.asarray(time, dtype=float)

    warnings.extend(_unit_sanity_warnings(ppm, n))

    # Weather correction (only when both columns exist AND actually vary).
    # Gap policy: NaN cells are interpolated for the SUBTRACTION step, but the fit
    # trains only on samples the logger really recorded — an interpolated ramp
    # through a plume-spanning gap correlates with the event, and a fit trained on
    # it would learn a fake coefficient and subtract real methane.
    temperature, temp_real = _densify(temperature, "temperature", warnings)
    humidity, humid_real = _densify(humidity, "humidity", warnings)

    have_weather = (temperature is not None and humidity is not None
                    and (temperature.std() > 1e-9 or humidity.std() > 1e-9))
    base_mask = (temp_real & humid_real) if have_weather else None
    weather_usable = have_weather
    if have_weather and not np.all(base_mask) and int(base_mask.sum()) < _MIN_GOOD_FOR_INTERP:
        warnings.append(
            "Too few samples have BOTH temperature and humidity logged — "
            "weather correction skipped."
        )
        weather_usable = False

    # Window sizing.
    bw = _odd(min(601, n)) if baseline_window is None else _odd(baseline_window)
    sw = _odd(min(61, max(5, n // 50))) if smooth_window is None else _odd(smooth_window)

    def _clean(ref_mask):
        """Weather-correct (trained on ref_mask) → baseline → smooth. One stage of a pass."""
        corrected = ppm
        coeffs: dict = {}
        wcorr = False
        if weather_usable:
            corrected, coeffs = temp_humidity_correct(ppm, temperature, humidity,
                                                      ref_mask=ref_mask)
            wcorr = True
        excess_raw, baseline = subtract_baseline(
            corrected, window=bw, method="percentile", percentile=25.0)
        return corrected, coeffs, moving_average(excess_raw, window=sw), baseline, wcorr

    def _measured_floor(corrected, baseline, det):
        """Drift-inclusive 1σ floor from the record: random ⊕ non-averageable bias."""
        ne = estimate_noise_floor(corrected - baseline, det)
        return float(effective_noise_floor(ne.random_ppm, ne.bias_ppm, 1)), ne

    # ── Pass A — provisional: detect on the RAW signal (NO weather correction yet). ──
    # A methane event that coincides with a weather swing must stay VISIBLE here, so Pass B
    # can exclude it from the weather fit — otherwise the full-record fit silently subtracts
    # the event as "weather" and hides it from the detector (it can never be excluded). The
    # provisional floor is the RANDOM term only (successive-difference estimate, robust to
    # the event's slow shape) so the provisional pass stays sensitive; Pass B does the
    # honest drift-inclusive floor.
    excess_a_raw, base_a = subtract_baseline(
        ppm, window=bw, method="percentile", percentile=25.0)
    excess_a = moving_average(excess_a_raw, window=sw)
    if noise_ppm is not None:
        floor0 = float(noise_ppm)
    else:
        ne0 = estimate_noise_floor(ppm - base_a, None)
        floor0 = max(float(ne0.random_ppm), 1e-9)
    det_prov = detect_pattern(excess_a, noise_std=floor0, k=k, min_run=min_run)

    # ── Pass B — final: re-fit weather EXCLUDING the event (Fix 3), then measure the
    # drift-inclusive floor on the now-clean quiet samples and detect against it (Fix 1). ──
    ref_mask = base_mask
    if weather_usable and det_prov.detected:
        keep = np.ones(n, dtype=bool)
        lo = max(0, det_prov.start_idx - 5)
        hi = min(n, det_prov.end_idx + 5)
        keep[lo:hi] = False                              # drop the event window (guard=5)
        cand = base_mask & keep
        if int(cand.sum()) >= _MIN_GOOD_FOR_INTERP:
            ref_mask = cand
        else:
            warnings.append(
                "The detected event spans too much of the record to exclude it from the "
                "weather fit — used the full window (humidity may be partly mis-attributed)."
            )
    corrected, coeffs, excess, baseline, weather_corrected = _clean(ref_mask)

    if noise_ppm is not None:                            # explicit override honored verbatim
        floor, ne = float(noise_ppm), None
    else:
        floor, ne = _measured_floor(corrected, baseline, det_prov)
    detection = detect_pattern(excess, noise_std=floor, k=k, min_run=min_run)

    # Put the displayed baseline back in the raw frame so raw ≈ baseline + excess.
    weather_part = ppm - corrected                    # zeros when not corrected
    baseline_display = baseline + weather_part

    return {
        "time": time,
        "raw": ppm,
        "baseline": baseline_display,
        "excess": excess,
        "threshold": float(k * floor),
        "detection": detection,
        "weather_corrected": weather_corrected,
        "weather_coeffs": coeffs,
        "noise_ppm": float(floor),
        "noise_random_ppm": (float(ne.random_ppm) if ne is not None else None),
        "noise_bias_ppm": (float(ne.bias_ppm) if ne is not None else None),
        "noise_floor_measured": bool(ne is not None),
        "windows": {"baseline": bw, "smooth": sw, "min_run": int(min_run)},
        "warnings": warnings,
        "temperature": temperature,    # °C per sample, or None — for real T_K wiring
        "humidity": humidity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE FOR INVERSION  — turn one test's time series into one weighted datum
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InversionPoint:
    """
    One sensor's contribution to the Week-3 inversion: a model-comparable mean
    plus the σ that weights it. Built so a weighted least-squares fit can do
    ``residual = (mean_excess_ppm − predict_ppm(...)) / sigma_ppm`` directly.
    """
    mean_excess_ppm: float    # time-mean of the UNSMOOTHED excess over the window
    sigma_ppm: float          # measurement σ of that mean → weight = 1/σ²
    n_window: int             # samples averaged (the meander window length)
    window: tuple             # (start_idx, end_idx) used, end exclusive
    mean_raw_ppm: float       # time-mean of the raw reading (for the fit-bg path)
    baseline_ppm: float       # baseline level over the window (bias/background proxy)
    random_ppm: float         # averageable noise component (1/√N)
    bias_ppm: float           # non-averageable component (survives averaging)
    detected: bool            # did this sensor see a sustained event?
    mean_temperature_c: float | None = None  # window-mean air temp (°C), if logged
    rho_hat: float = 0.0      # measured lag-1 autocorrelation of the quiet record (F6)
    n_eff: float | None = None  # AR(1)-corrected effective N used for sigma_ppm


def _window_from_seconds(time, window_s):
    """Central index window spanning ``window_s`` seconds; whole record if None."""
    n = len(time)
    if window_s is None or n < 2:
        return 0, n
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        return 0, n
    half = int(round((window_s / dt) / 2))
    mid = n // 2
    return max(0, mid - half), min(n, mid + half + 1)


def aggregate_for_inversion(result: dict, window_s: float | None = None) -> InversionPoint:
    """
    Reduce ONE field-test result to one weighted datum for the inversion.

    Call this ONCE per sensor to build its InversionPoint *before* the
    ``scipy.optimize`` loop; the objective then reads the cached
    ``mean_excess_ppm``/``sigma_ppm``. Do NOT call it inside the optimizer — it
    rescans the whole record for the noise floor (``estimate_noise_floor``), which
    is a property of the sensor record, not of the trial (Q, source pos, stability).

    This is the "averaging done right" half of the reframe. The Gaussian plume is a
    *time-mean* field, so we hand the optimizer a **time-mean over a meander window**
    — NOT a noise-smoothed wiggle, and NOT the peak. Critically:

      • The mean is taken over the UNSMOOTHED excess (``raw − baseline``). Pre-smoothing
        with moving_average is deliberately skipped: a least-squares fit over the
        window already averages optimally, so smoothing first would only distort the
        noise model. (See the module + CLAUDE.md "two averaging roles" note.)
      • ``sigma_ppm`` is the σ OF THE MEAN, combining the averageable random noise
        (÷√N_eff over the window) with the non-averageable bias, via the shared
        ``sensor_sim.effective_noise_floor``. That is the correct weight for WLS.
        N_eff is the AR(1) effective sample size from the record's own measured
        lag-1 autocorrelation ρ̂ (``accuracy.estimate_lag1_autocorrelation``, F6):
        correlated background shrinks slower than √N, so using the raw window
        length N would under-estimate σ and over-state confidence in both the WLS
        weight and the inversion's CRB. NOTE the scope of this fix: it corrects
        ``sigma_ppm``/the CRB's confidence, NOT the separate ``inversion.converged``
        signal gate (which intentionally compares against a single-sample floor,
        not ``sigma_ppm``, for unrelated reasons — see its docstring) — a strongly
        autocorrelated sourceless record can still trip that gate; that fabrication
        -rate residual is not closed by this fix.

    Bias never averages away, so BOTH framings are always returned and the caller
    picks: subtract the baseline as a known zero (use ``mean_excess_ppm``), or fit
    background as a free parameter (use ``mean_raw_ppm`` with ``baseline_ppm`` as the
    start). They satisfy ``mean_raw_ppm ≈ baseline_ppm + mean_excess_ppm``.

    Window: a detected event's [start, end] when present; else the central
    ``window_s`` seconds; else the whole record (a legitimate "saw nothing"
    constraint — a near-zero mean that still bounds the source). Returns an
    InversionPoint.
    """
    time = np.asarray(result["time"], dtype=float)
    raw = np.asarray(result["raw"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    det = result["detection"]
    # Unsmoothed excess in the raw frame: raw − baseline cancels weather + floor and
    # leaves noise + any plume (the same quantity estimate_noise_floor wants).
    excess_unsmoothed = raw - baseline

    if det is not None and det.detected and det.start_idx >= 0:
        lo, hi = det.start_idx, det.end_idx        # end_idx is exclusive (F5)
    else:
        lo, hi = _window_from_seconds(time, window_s)
    lo, hi = int(lo), int(max(lo + 1, hi))
    n_window = hi - lo

    seg_excess = excess_unsmoothed[lo:hi]
    mean_excess = float(np.mean(seg_excess))
    mean_raw = float(np.mean(raw[lo:hi]))
    baseline_level = float(np.mean(baseline[lo:hi]))

    # Measure the two noise parts from the whole record's quiet samples, then turn
    # them into the σ of a mean over n_window samples (random shrinks, bias doesn't).
    ne = estimate_noise_floor(excess_unsmoothed, det)

    # F6: a window mean over autocorrelated samples shrinks slower than √N — measure
    # the record's own lag-1 ρ̂ (never assumed) and let the shared floor apply the
    # AR(1) effective-N so sigma_ppm isn't over-confident. (n_eff is recomputed here
    # only so the InversionPoint can REPORT the value the floor used.)
    rho_hat = estimate_lag1_autocorrelation(excess_unsmoothed, det)
    n_eff = max(1.0, n_window * (1.0 - rho_hat) / (1.0 + rho_hat))
    sigma_mean = float(effective_noise_floor(ne.random_ppm, ne.bias_ppm,
                                             n_avg=max(1, n_window), rho=rho_hat))

    # Real T_K wiring: window-mean of the CSV's own temperature, when logged, so the
    # inversion/CRB can use the measured air temperature instead of a 20°C default.
    mean_temperature_c = None
    temperature = result.get("temperature")
    if temperature is not None:
        temp_arr = np.asarray(temperature, dtype=float)
        if temp_arr.size == len(raw):
            seg_temp = temp_arr[lo:hi]
            if seg_temp.size and np.all(np.isfinite(seg_temp)):
                mean_temperature_c = float(np.mean(seg_temp))

    return InversionPoint(
        mean_excess_ppm=mean_excess,
        sigma_ppm=sigma_mean,
        n_window=n_window,
        window=(lo, hi),
        mean_raw_ppm=mean_raw,
        baseline_ppm=baseline_level,
        random_ppm=float(ne.random_ppm),
        bias_ppm=float(ne.bias_ppm),
        detected=bool(det.detected) if det is not None else False,
        mean_temperature_c=mean_temperature_c,
        rho_hat=rho_hat,
        n_eff=n_eff,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SAMPLE  — so the whole flow is usable before the real sensor exists
# ─────────────────────────────────────────────────────────────────────────────
def make_sample_readings(
    n: int = 600,
    event_ppm: float = 4.0,
    noise_ppm: float = SENSOR_NOISE_PPM,
    with_weather: bool = True,
    seed: int | None = 0,
) -> dict:
    """
    Build one realistic synthetic field test (reusing sensor_sim) for demos/tests.

    Returns the same array shape parse_csv produces (time/ppm/temperature/humidity),
    so it can flow through process_fieldtest and the storage layer exactly like a
    real upload. The 'ppm' here is the messy raw reading (background + plume + drift
    + weather + noise), so detection has something honest to recover.
    """
    true = make_plume_event(n, event_ppm=event_ppm)
    data = synthetic_timeseries(
        true, noise_ppm=noise_ppm, with_weather=with_weather, seed=seed
    )
    return {
        "time": data.time,
        "ppm": data.raw,
        "temperature": data.temperature if with_weather else None,
        "humidity": data.humidity if with_weather else None,
        "n": int(len(data.raw)),
    }


def to_csv_text(readings: dict) -> str:
    """
    Serialize parsed/sample readings back to CSV text (header + rows).

    Used to write the committed sample file and to let the dashboard offer a
    "download sample CSV" so users see the exact accepted format.
    """
    time = readings.get("time")
    ppm = readings["ppm"]
    temp = readings.get("temperature")
    humid = readings.get("humidity")
    n = len(ppm)

    header = ["time", "ppm"]
    if temp is not None:
        header.append("temperature")
    if humid is not None:
        header.append("humidity")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for i in range(n):
        row = [f"{(time[i] if time is not None else i):.3f}", f"{ppm[i]:.4f}"]
        if temp is not None:
            row.append(f"{temp[i]:.3f}")
        if humid is not None:
            row.append(f"{humid[i]:.3f}")
        writer.writerow(row)
    return buf.getvalue()

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

import numpy as np

from physics.processing import (
    detect_pattern,
    moving_average,
    subtract_baseline,
    temp_humidity_correct,
)
from physics.sensor_sim import (
    DETECT_K,
    SENSOR_NOISE_PPM,
    make_plume_event,
    synthetic_timeseries,
)

# Upper bound on rows we ingest from one upload — guards against a runaway file.
MAX_ROWS = 200_000

# Accepted header spellings (compared case-insensitively, stripped). The first
# matching column in the file wins.
_PPM_NAMES = {"ppm", "ch4", "ch4_ppm", "methane", "reading", "value", "conc",
              "concentration", "ppm_raw", "raw", "ch4_ppm_raw"}
_TIME_NAMES = {"time", "t", "seconds", "sec", "s", "timestamp", "elapsed",
               "time_s", "t_seconds", "ts"}
_TEMP_NAMES = {"temp", "temperature", "t_c", "tempc", "temp_c", "celsius", "tc"}
_HUMID_NAMES = {"humidity", "rh", "humid", "relative_humidity", "humidity_pct",
                "rh_pct", "humid_pct"}


# ─────────────────────────────────────────────────────────────────────────────
# CSV PARSING
# ─────────────────────────────────────────────────────────────────────────────
def _to_text(data) -> str:
    """Accept a file-like, bytes, or str and return decoded text (BOM-tolerant)."""
    if hasattr(data, "read"):
        data = data.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    return data


def _num(value):
    """Parse one cell to float, or None if blank/non-numeric."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _pick(norm_to_orig: dict, candidates: set):
    """Return the original header whose normalized form is in `candidates`, else None."""
    for low, original in norm_to_orig.items():
        if low in candidates:
            return original
    return None


def parse_csv(data) -> dict:
    """
    Parse an uploaded readings CSV into aligned numpy arrays.

    Flexible about column names (see the *_NAMES sets): it needs a header row with
    at least a methane column; time, temperature, and humidity are optional. Rows
    without a usable numeric reading are skipped. A column that is absent, or only
    partially numeric, comes back as None (so a half-filled weather column never
    corrupts the linear weather fit).

    Returns
    -------
    dict with keys:
      ppm         : 1-D float array (required) — the raw reading incl. background.
      time        : 1-D float array or None (synthesized later if None).
      temperature : 1-D float array or None.
      humidity    : 1-D float array or None.
      n           : number of readings.
      columns     : which original headers were matched.

    Raises ValueError with a friendly message if there is no header or no readings.
    """
    text = _to_text(data)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(
            "The CSV looks empty — it needs a header row and at least a "
            "methane column (e.g. 'ppm')."
        )

    norm = {(name or "").strip().lower(): name for name in reader.fieldnames}
    ppm_key = _pick(norm, _PPM_NAMES)
    if ppm_key is None:
        raise ValueError(
            "No methane column found. Add a header like 'ppm' "
            "(accepted: ppm, ch4, reading, value, concentration)."
        )
    time_key = _pick(norm, _TIME_NAMES)
    temp_key = _pick(norm, _TEMP_NAMES)
    humid_key = _pick(norm, _HUMID_NAMES)

    times, ppms, temps, humids = [], [], [], []
    for row in reader:
        ppm = _num(row.get(ppm_key))
        if ppm is None:
            continue                          # skip rows without a usable reading
        ppms.append(ppm)
        times.append(_num(row.get(time_key)) if time_key else None)
        temps.append(_num(row.get(temp_key)) if temp_key else None)
        humids.append(_num(row.get(humid_key)) if humid_key else None)
        if len(ppms) >= MAX_ROWS:
            break

    if not ppms:
        raise ValueError("No numeric methane readings were found in the file.")

    def _col(vals):
        # A column is only usable if EVERY row has a number — a partial column
        # would otherwise misalign or break the weather fit.
        if any(v is None for v in vals):
            return None
        return np.asarray(vals, dtype=float)

    return {
        "ppm": np.asarray(ppms, dtype=float),
        "time": _col(times),
        "temperature": _col(temps),
        "humidity": _col(humids),
        "n": len(ppms),
        "columns": {"ppm": ppm_key, "time": time_key,
                    "temperature": temp_key, "humidity": humid_key},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def _odd(x) -> int:
    """Nearest odd integer ≥ 1 (the processing windows want odd, centred windows)."""
    x = int(x)
    if x < 1:
        return 1
    return x if x % 2 == 1 else x + 1


def process_fieldtest(
    ppm,
    temperature=None,
    humidity=None,
    time=None,
    noise_ppm: float = SENSOR_NOISE_PPM,
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
    sustained-event detection against k·noise.

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

    if time is None:
        rate = sample_rate_hz if (sample_rate_hz and sample_rate_hz > 0) else 1.0
        time = np.arange(n) / rate
    else:
        time = np.asarray(time, dtype=float)

    # 1. Weather correction (only when both columns exist AND actually vary).
    corrected = ppm
    weather_corrected = False
    if temperature is not None and humidity is not None:
        temperature = np.asarray(temperature, dtype=float)
        humidity = np.asarray(humidity, dtype=float)
        if temperature.std() > 1e-9 or humidity.std() > 1e-9:
            corrected, _ = temp_humidity_correct(ppm, temperature, humidity)
            weather_corrected = True

    # Window sizing.
    bw = _odd(min(601, n)) if baseline_window is None else _odd(baseline_window)
    sw = _odd(min(61, max(5, n // 50))) if smooth_window is None else _odd(smooth_window)

    # 2. Baseline subtraction, then 3. smoothing of the excess.
    excess_raw, baseline = subtract_baseline(
        corrected, window=bw, method="percentile", percentile=25.0
    )
    excess = moving_average(excess_raw, window=sw)

    # 4. Detection of a sustained event above k·noise.
    detection = detect_pattern(excess, noise_std=noise_ppm, k=k, min_run=min_run)

    # Put the displayed baseline back in the raw frame so raw ≈ baseline + excess.
    weather_part = ppm - corrected                    # zeros when not corrected
    baseline_display = baseline + weather_part

    return {
        "time": time,
        "raw": ppm,
        "baseline": baseline_display,
        "excess": excess,
        "threshold": float(k * noise_ppm),
        "detection": detection,
        "weather_corrected": weather_corrected,
        "noise_ppm": float(noise_ppm),
        "windows": {"baseline": bw, "smooth": sw, "min_run": int(min_run)},
    }


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

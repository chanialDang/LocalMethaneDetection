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
from dataclasses import dataclass

import numpy as np

from physics.accuracy import estimate_noise_floor
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
        (÷√N over the window) with the non-averageable bias, via the shared
        ``sensor_sim.effective_noise_floor``. That is the correct weight for WLS.

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
    sigma_mean = float(effective_noise_floor(ne.random_ppm, ne.bias_ppm,
                                             n_avg=max(1, n_window)))

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

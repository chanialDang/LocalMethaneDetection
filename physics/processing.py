"""
processing.py — Pulling a tiny methane signal out of a messy sensor record.

═══════════════════════════════════════════════════════════════════════════════
THE PROBLEM (plain English)
───────────────────────────────────────────────────────────────────────────────
A raw MOX-sensor reading is the true plume signal buried under background,
slow drift, weather effects, and random noise (see sensor_sim.py). Our plume is
only ~2–5 ppm, while all that junk can be just as big. This module is the
toolbox that strips the junk away, step by step, so the small real signal can
show through.

The four tools, in the order you'd usually apply them:

  1. temp_humidity_correct  — remove the part of the reading that just tracks
                              temperature and humidity.
  2. subtract_baseline      — remove the slow wandering "zero" so what's left is
                              the EXCESS above the local floor (the plume).
  3. moving_average         — smooth away random noise so a real bump stands out.
  4. detect_pattern         — decide YES/NO: is there a sustained plume event,
                              or is this just noise?

Everything here is plain NumPy — no SciPy needed. Each function is pure (no
hidden state) so it is easy to test against the known-answer data from
sensor_sim.py.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. MOVING AVERAGE  — smooth away random noise
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth a signal by averaging each point with its neighbours.

    WHY: random noise is equally likely to be + or −, so averaging several
    readings makes the noise partly cancel while a real, steady signal survives.
    Averaging N points shrinks the noise by about √N.

    SCOPE (important): this is for DISPLAY and for the detect/explain pipeline — a
    nicer-looking trace and a stabler detection statistic. It is a *noise* smoother;
    it does NOT remove bias/drift (use subtract_baseline for that), and the Week-3
    inversion deliberately does NOT pre-smooth with it — a weighted least-squares
    fit averages optimally by itself, so smoothing first would only distort the
    noise model. The averaging the inversion DOES need (a time-mean over a
    meteorological "meander" window, to match the steady-state plume) lives in
    fieldtest.aggregate_for_inversion. See ACCURACY.md "two averaging roles".

    Edges are handled honestly: near the start/end, where a full window isn't
    available, we average over however many points DO exist (so the output is the
    same length as the input and is not biased toward zero at the ends).

    Parameters
    ----------
    signal : 1-D array of readings.
    window : number of samples to average over. window=1 returns the input
             unchanged. Even windows are bumped up by 1 so the window is centred.

    Returns
    -------
    Smoothed array, same length as `signal`.
    """
    signal = np.asarray(signal, dtype=float)
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        window += 1   # force odd so the window is symmetric about each point
    half = window // 2
    n = len(signal)

    # O(n) centred rolling mean via a cumulative sum: each point averages its
    # clipped [i-half, i+half] neighbourhood, dividing by how many points
    # actually contributed (so edges aren't biased toward zero). This never
    # builds a window-by-window matrix — memory stays O(n) — and the same idea
    # ports cleanly to an incremental ring-buffer sum on a microcontroller.
    csum = np.concatenate(([0.0], np.cumsum(signal)))
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half, n - 1)
    summed = csum[hi + 1] - csum[lo]
    counts = hi - lo + 1
    return summed / counts


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — rolling window statistic (used by baseline estimation)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_stat(signal: np.ndarray, window: int, stat: str, q: float = 25.0) -> np.ndarray:
    """
    Compute a rolling statistic (median / min / percentile) over a sliding window.

    Edges are padded by repeating the edge value so the output length matches the
    input. Used internally to estimate the slowly-varying baseline.

    stat : "median", "min", or "percentile" (with `q` the percentile 0–100).
    """
    signal = np.asarray(signal, dtype=float)
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)

    if stat == "median":
        return np.median(windows, axis=1)
    if stat == "min":
        return np.min(windows, axis=1)
    if stat == "percentile":
        return np.percentile(windows, q, axis=1)
    raise ValueError(f"unknown stat {stat!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BASELINE SUBTRACTION  — remove the slow wandering "zero"
# ─────────────────────────────────────────────────────────────────────────────

def subtract_baseline(
    signal: np.ndarray,
    window: int,
    method: str = "percentile",
    percentile: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the slowly-varying baseline and subtract it, leaving the EXCESS.

    WHY: the sensor's "zero" and the background drift up and down slowly over
    time. A plume is a relatively SHORT bump on top of that slow floor. If we
    estimate the floor in a wide window around each point and subtract it, what
    remains is the plume's contribution.

    HOW: within a wide window we take a LOW statistic (a low percentile, or the
    median/min) as the floor. A low percentile ignores short upward plume
    excursions (they're a minority of the window) while still tracking the slow
    drift — and it's far less jumpy than a pure minimum.

    Parameters
    ----------
    signal     : raw (ideally already temp/humidity-corrected) readings.
    window     : width of the baseline-estimation window (samples). Make this
                 several times longer than the longest plume event you expect.
    method     : "percentile" (default), "median", or "min".
    percentile : percentile to use when method="percentile" (0–100).

    Returns
    -------
    (excess, baseline) — both arrays the same length as `signal`.
    excess = signal − baseline  (the recovered plume contribution).
    """
    if method == "percentile":
        baseline = _rolling_stat(signal, window, "percentile", q=percentile)
    else:
        baseline = _rolling_stat(signal, window, method)
    excess = np.asarray(signal, dtype=float) - baseline
    return excess, baseline


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPERATURE / HUMIDITY CORRECTION  — remove weather-driven drift
# ─────────────────────────────────────────────────────────────────────────────

def temp_humidity_correct(
    signal: np.ndarray,
    temperature: np.ndarray,
    humidity: np.ndarray,
    ref_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Remove the part of the reading that simply tracks temperature and humidity.

    WHY: MOX sensors read differently when the air is hotter or more humid, even
    with the SAME amount of methane. If we know temperature and humidity at every
    sample, we can learn how strongly they push the reading and subtract that
    push out.

    HOW: fit a simple straight-line model on a "reference" stretch where we
    believe NO plume is present:

        reading ≈ a·temperature + b·humidity + c

    Then subtract that fitted weather contribution from the WHOLE record. We keep
    the constant `c` (it's just an offset the later baseline step handles), so the
    correction only removes the temperature/humidity-VARYING part.

    Parameters
    ----------
    signal       : readings to correct.
    temperature  : air temperature per sample (°C).
    humidity     : relative humidity per sample (%RH).
    ref_mask     : boolean array, True where we trust "no plume" (used to fit the
                   model). If None, the whole record is used (fine when plume
                   events are a small fraction of the record).

    Returns
    -------
    (corrected, coeffs) where coeffs = {"a_temp":…, "b_humid":…, "c_const":…}.
    `corrected` has the temperature/humidity-varying part removed.
    """
    signal = np.asarray(signal, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    humidity = np.asarray(humidity, dtype=float)

    if ref_mask is None:
        ref_mask = np.ones_like(signal, dtype=bool)

    # Drop a weather regressor that doesn't vary across the fit window: a (near-)
    # constant column is collinear with the intercept, so the lstsq is rank-deficient
    # and the min-norm split hands it a meaningless coefficient (F10). A non-varying
    # regressor adds nothing to the VARYING correction, so fit only the columns that
    # actually move and report a dropped one's coefficient as 0.
    t_ref, h_ref, s_ref = temperature[ref_mask], humidity[ref_mask], signal[ref_mask]

    def _varies(x):
        return float(np.ptp(x)) > 1e-9 * (abs(float(np.mean(x))) + 1.0)

    use_t, use_h = _varies(t_ref), _varies(h_ref)
    cols = ([t_ref] if use_t else []) + ([h_ref] if use_h else []) + [np.ones(s_ref.size)]
    sol = list(np.linalg.lstsq(np.column_stack(cols), s_ref, rcond=None)[0])
    a_temp = sol.pop(0) if use_t else 0.0
    b_humid = sol.pop(0) if use_h else 0.0
    c_const = sol.pop(0)

    # Subtract only the VARYING weather part (keep the constant offset c).
    weather_part = a_temp * temperature + b_humid * humidity
    corrected = signal - weather_part

    return corrected, {"a_temp": a_temp, "b_humid": b_humid, "c_const": c_const}


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATTERN DETECTION  — is there really a plume, or just noise?
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Result of looking for a plume event in a processed (excess) signal.

    detected   : True if a sustained above-threshold run was found.
    start_idx  : index where the detected event starts (-1 if none).
    end_idx    : index just past where it ends (-1 if none).
    confidence : how far above the noise the event sits, in "sigmas"
                 (mean excess in the event divided by the noise floor). 0 if none.
    threshold  : the ppm threshold used (k · noise_std).
    """
    detected: bool
    start_idx: int
    end_idx: int
    confidence: float
    threshold: float


def detect_pattern(
    excess: np.ndarray,
    noise_std: float,
    k: float = 3.0,
    min_run: int = 5,
) -> Detection:
    """
    Decide whether a processed signal contains a real plume event.

    WHY: after smoothing and baseline removal, noise alone will still wiggle a
    little. We only believe a plume if the signal climbs clearly above the noise
    (k times the noise level) AND stays there for several samples in a row — a
    single spike is probably noise; a sustained bump is probably real.

    Parameters
    ----------
    excess    : processed signal (plume contribution, ppm above baseline).
    noise_std : the noise floor (1σ), e.g. SENSOR_NOISE_PPM.
    k         : how many sigmas above noise counts as "signal" (default 3 ≈ a
                strict 99.7% confidence per sample).
    min_run   : how many consecutive above-threshold samples are required.

    Returns
    -------
    Detection (see the dataclass above).
    """
    excess = np.asarray(excess, dtype=float)
    threshold = k * noise_std
    above = excess > threshold

    # Find the longest run of consecutive True values in `above`.
    best_len = 0
    best_start = -1
    run_len = 0
    run_start = 0
    for i, flag in enumerate(above):
        if flag:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
        else:
            run_len = 0

    if best_len >= min_run:
        start_idx = best_start
        end_idx = best_start + best_len
        confidence = float(np.mean(excess[start_idx:end_idx]) / noise_std)
        return Detection(True, start_idx, end_idx, confidence, threshold)

    return Detection(False, -1, -1, 0.0, threshold)

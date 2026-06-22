"""
sensor_sim.py — Synthetic ("fake but realistic") methane-sensor data.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS (plain English)
───────────────────────────────────────────────────────────────────────────────
We don't have the real Figaro TGS 2611 sensor hooked up yet. But we still want
to build — and TEST — the software that will clean up its readings later
(baseline subtraction, averaging, temperature/humidity correction, plume
detection).

The trick: generate FAKE sensor readings where WE already know the right answer,
then check that our processing code recovers that known answer. This file is the
fake-data generator. `processing.py` is the cleaner. `tests/test_processing.py`
checks the cleaner against the truth.

WHAT A REAL MOX SENSOR READING LOOKS LIKE
───────────────────────────────────────────────────────────────────────────────
A real reading is NOT just "the methane concentration". It is a messy sum of:

    raw  =  true plume signal        (what we actually want)
          + atmospheric background    (1.9 ppm, always there)
          + slow baseline drift       (the sensor's "zero" wanders over hours)
          + temperature effect        (MOX sensors read hotter/cooler air wrong)
          + humidity effect           (water vapour shifts the reading too)
          + random noise              (electrical jitter, turbulence)

This module builds exactly that sum so we can practise pulling the true signal
back out.

⚠  IMPORTANT — THESE NUMBERS ARE ASSUMPTIONS, NOT MEASUREMENTS.
   The noise size, drift size, and temperature/humidity coefficients below are
   plausible placeholders chosen so the simulation behaves realistically. Once
   real sensor data exists, replace the constants in the "ASSUMED SENSOR
   CHARACTERISTICS" block with measured values — nothing else needs to change.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ASSUMED SENSOR CHARACTERISTICS  (placeholders — replace with measured values)
# ─────────────────────────────────────────────────────────────────────────────
#
# These are the single source of truth for "how messy is the sensor". Both the
# processing tests and the feasibility verdict (feasibility.py) import
# SENSOR_NOISE_PPM from here, so the whole project agrees on one noise floor.

SENSOR_NOISE_PPM = 0.30      # ppm — random reading-to-reading jitter (1σ).
                             # This IS the "noise floor": a plume smaller than a
                             # few times this is hard to distinguish from noise.
                             # KEY: this is the RANDOM part — averaging N samples
                             # shrinks it as 1/√N (see effective_noise_floor).

BIAS_FLOOR_PPM = 0.00        # ppm — the NON-random part of the error: slow drift,
                             # calibration offset, weather residual left after
                             # correction. Crucially, averaging does NOT remove it.
                             # Left at 0 by default (so nothing changes), but the
                             # field value is what really limits detection: a sensor
                             # with a 5 ppm calibration RMSE cannot be averaged down
                             # to 0.5 ppm if most of that 5 ppm is bias. ASSUMED —
                             # set from measured Figaro data once it exists.

DETECT_K = 3.0               # detection strictness: a signal must beat
                             # DETECT_K × SENSOR_NOISE_PPM to count as clearly
                             # "detectable". Single source of truth — imported by
                             # both feasibility.py and explain.py so the plot's
                             # detection line and its caption can never disagree.

BASELINE_DRIFT_PPM = 1.00    # ppm — amplitude of the slow wander of the sensor's
                             # zero point over the record (hours-scale).

TEMP_COEFF_PPM_PER_C = 0.05  # ppm per °C — how much a 1 °C change in air temp
                             # shifts the apparent reading.

HUMID_COEFF_PPM_PER_PCT = 0.02  # ppm per %RH — apparent shift per 1% change in
                                # relative humidity.

T0_REF_C = 20.0              # °C  — reference temperature (corrections measure
                            #        deviations from this).
H0_REF_PCT = 50.0           # %RH — reference humidity.

DEFAULT_SAMPLE_RATE_HZ = 1.0  # one reading per second by default.

# Background methane is duplicated here (rather than imported from plume.py) so
# this module stays standalone; the value is the same NOAA GML 2023 mean.
CH4_BACKGROUND_PPM = 1.9


def effective_noise_floor(
    random_ppm: float = SENSOR_NOISE_PPM,
    bias_ppm: float = BIAS_FLOOR_PPM,
    n_avg: int = 1,
) -> float:
    """
    Realistic detection floor (ppm, 1σ) after averaging ``n_avg`` samples.

    A sensor's error has two parts that behave OPPOSITELY under averaging:

      • RANDOM jitter (``random_ppm``) — independent each sample, so averaging
        N samples shrinks it as 1/√N (the classic "noise beats down" win).
      • BIAS / drift / calibration offset (``bias_ppm``) — essentially the SAME
        in every sample, so averaging does NOT reduce it at all.

    The combined 1σ floor adds them in quadrature::

        floor = √[ (random_ppm / √n_avg)² + bias_ppm² ]

    This is exactly why you cannot average your way to an arbitrarily small
    detection limit: past some N the bias term dominates and more averaging buys
    nothing. (It is also the correction to the common mistake of dividing a whole
    calibration RMSE by √N — only the random part earns that.) With the defaults
    (bias 0, n_avg 1) it returns ``random_ppm`` unchanged, so existing behaviour
    is preserved.

    Two distinct callers, one formula: (a) the *theoretical* detection-limit story
    in feasibility/accuracy (how low could the floor go if you averaged?), and
    (b) ``fieldtest.aggregate_for_inversion`` — the σ OF THE MEAN that weights one
    sensor's averaged reading in the Week-3 inversion. Both are the same quadrature;
    note neither is "pre-smooth then fit" — see ACCURACY.md "two averaging roles".
    """
    if n_avg < 1:
        raise ValueError(f"n_avg must be ≥ 1, got {n_avg}")
    if random_ppm < 0 or bias_ppm < 0:
        raise ValueError("noise terms must be non-negative")
    random_part = random_ppm / np.sqrt(n_avg)
    return float(np.hypot(random_part, bias_ppm))


@dataclass
class SensorData:
    """
    One synthetic sensor record. All arrays share the same length and indexing.

    Attributes
    ----------
    time         : seconds since start of record (s)
    raw          : the messy sensor reading (ppm-equivalent) — what a real
                   sensor would output, INCLUDING background, drift, weather,
                   and noise.
    temperature  : air temperature at each sample (°C)
    humidity     : relative humidity at each sample (%RH)
    true_ppm     : the GROUND TRUTH plume contribution (ppm ABOVE background).
                   This is what the processing code is trying to recover. In
                   real life you never get this column — here we keep it so the
                   tests can check their answers.
    noise_ppm    : the noise floor used to build this record (ppm, 1σ).
    """
    time: np.ndarray
    raw: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray
    true_ppm: np.ndarray
    noise_ppm: float


def make_weather(
    n: int,
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
    temp_amp_c: float = 5.0,
    humid_amp_pct: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build slowly-varying temperature and humidity tracks for a record of n points.

    Real weather drifts smoothly over a day. We model that as a gentle sine wave
    so that within any record there is *some* temperature/humidity variation for
    the correction step to work against. Humidity is given a half-cycle phase
    offset because in reality it tends to move opposite to temperature.

    Returns
    -------
    (temperature_C, humidity_pct) — two arrays of length n.
    """
    t = np.arange(n) / sample_rate
    span = max(float(t[-1]), 1.0)          # length of the record in seconds
    phase = 2.0 * np.pi * t / span         # one full slow cycle across the record

    temperature = T0_REF_C + temp_amp_c * np.sin(phase)
    humidity = H0_REF_PCT + humid_amp_pct * np.sin(phase + np.pi)
    return temperature, humidity


def make_baseline_drift(
    n: int,
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
    drift_ppm: float = BASELINE_DRIFT_PPM,
) -> np.ndarray:
    """
    Build a slow, smooth wander of the sensor's zero point.

    Modelled as a slow half-sine so the baseline starts low, rises, and falls —
    the kind of gentle drift a warming/cooling MOX sensor shows over a session.

    Returns an array of length n (ppm).
    """
    t = np.arange(n) / sample_rate
    span = max(float(t[-1]), 1.0)
    # Half a sine cycle => a single smooth hump across the record.
    return drift_ppm * np.sin(np.pi * t / span)


def make_plume_event(
    n: int,
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
    event_start_s: float | None = None,
    event_duration_s: float | None = None,
    event_ppm: float = 3.0,
) -> np.ndarray:
    """
    Build a GROUND-TRUTH plume signal: flat zero, then a plume "event", then zero.

    This represents the wind briefly carrying the plume over the sensor. The
    excess is `event_ppm` ABOVE background during the event and 0 otherwise.

    Parameters
    ----------
    n                : number of samples in the record.
    sample_rate      : samples per second.
    event_start_s    : when the plume arrives (s). Defaults to 40% into record.
    event_duration_s : how long it lingers (s). Defaults to 20% of the record.
    event_ppm        : plume height above background during the event (ppm).

    Returns
    -------
    true_ppm array of length n (ppm above background).
    """
    t = np.arange(n) / sample_rate
    total_s = max(float(t[-1]), 1.0)

    if event_start_s is None:
        event_start_s = 0.40 * total_s
    if event_duration_s is None:
        event_duration_s = 0.20 * total_s
    event_end_s = event_start_s + event_duration_s

    true_ppm = np.zeros(n, dtype=float)
    in_event = (t >= event_start_s) & (t < event_end_s)
    true_ppm[in_event] = event_ppm
    return true_ppm


def synthetic_timeseries(
    true_ppm: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
    noise_ppm: float = SENSOR_NOISE_PPM,
    drift_ppm: float = BASELINE_DRIFT_PPM,
    with_weather: bool = True,
    seed: int | None = 0,
) -> SensorData:
    """
    Turn a known true plume signal into a messy, realistic raw sensor record.

    raw = background + true_ppm + baseline_drift
          + temp_coeff·(T − T0) + humid_coeff·(H − H0) + gaussian_noise

    Parameters
    ----------
    true_ppm     : ground-truth plume contribution per sample (ppm above bkg).
                   Use make_plume_event(...) or a constant array.
    sample_rate  : samples per second.
    noise_ppm    : 1σ random noise to add (the noise floor).
    drift_ppm    : amplitude of the slow baseline wander.
    with_weather : if True, add temperature- and humidity-driven shifts.
    seed         : RNG seed for reproducible noise (tests rely on this).

    Returns
    -------
    SensorData with aligned time / raw / temperature / humidity / true_ppm arrays.
    """
    true_ppm = np.asarray(true_ppm, dtype=float)
    n = len(true_ppm)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate

    temperature, humidity = make_weather(n, sample_rate)
    drift = make_baseline_drift(n, sample_rate, drift_ppm)
    noise = rng.normal(0.0, noise_ppm, size=n)

    raw = CH4_BACKGROUND_PPM + true_ppm + drift + noise
    if with_weather:
        raw = raw + TEMP_COEFF_PPM_PER_C * (temperature - T0_REF_C)
        raw = raw + HUMID_COEFF_PPM_PER_PCT * (humidity - H0_REF_PCT)

    return SensorData(
        time=t,
        raw=raw,
        temperature=temperature,
        humidity=humidity,
        true_ppm=true_ppm,
        noise_ppm=noise_ppm,
    )

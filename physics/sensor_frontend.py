"""
sensor_frontend.py — the MISSING first stage: raw sensor voltage → ppm.

═══════════════════════════════════════════════════════════════════════════════
WHERE THIS SITS (plain English)
───────────────────────────────────────────────────────────────────────────────
Everything else in this project is "ppm in, source out": ``fieldtest.parse_csv``
reads a ppm column verbatim, and the plume model / inversion work in ppm. But a
real Figaro TGS 2611-E00 does NOT output ppm — it is a chemiresistor whose
resistance R_s changes with the methane it sees. What your ADC actually records is
a VOLTAGE across a load resistor. This module is the bridge nobody had built yet:

    ADC voltage ──voltage_to_ppm──▶ ppm  ──▶ (feeds parse_csv / the whole pipeline)

THE CHAIN (three steps, each pinned below)
───────────────────────────────────────────────────────────────────────────────
1. VOLTAGE DIVIDER → sensor resistance.
   Circuit:   V_c ─[ R_s (sensor) ]─ node(V_out) ─[ R_L (load) ]─ GND,
   the ADC measuring V_out across R_L. Solving the divider for the unknown R_s:

        R_s = R_L · (V_c − V_out) / V_out

   As methane rises, R_s falls (n-type MOX), so V_out rises. The "over-amplification"
   method this project is built on is exactly this: a LARGE R_L + a precision ADC so
   the tiny 2–40 ppm resistance change lands on many ADC codes instead of one.

2. POWER LAW → concentration. Figaro characterises the sensor as a straight line
   on log-log axes:

        R_s / R₀ = A · C^(−m)          (C = methane concentration in ppm)

   which we invert to recover C from the measured ratio:

        C = ( A / (R_s/R₀) )^(1/m)

3. TEMPERATURE / HUMIDITY correction. A MOX sensor's resistance also drifts with
   air T and RH. The datasheet gives this as a multiplicative correction on the
   R_s/R₀ ratio, so we divide it out BEFORE inverting the power law. Left as an
   identity (coefficients 0) until measured — see the constant block.

WHAT COMES OUT
───────────────────────────────────────────────────────────────────────────────
``voltage_to_ppm`` returns TOTAL ppm — the sensor responds to all the methane
present, background included — which is precisely what ``parse_csv``'s ppm column
already means ("the raw reading incl. background"). So this stage adds NO
background of its own; downstream baseline subtraction removes the ~1.9 ppm as it
always has.

═══════════════════════════════════════════════════════════════════════════════
THESE CONSTANTS ARE NOT MEASURED YET (read this before trusting a number)
───────────────────────────────────────────────────────────────────────────────
The circuit + bench constants (V_c, R_L, R₀) start as ``None`` — the converter
REFUSES to run until you set them, so it can never emit a fabricated ppm. The
power-law A/m ship as DATASHEET-TYPICAL placeholders so the chain is runnable, but
they are unverified at this project's 2–40 ppm operating range (the datasheet
curve is characterised at 500–12,500 ppm). Fill everything from a bench run — see
``docs/CALIBRATION.md``. As with ``sensor_sim.py``: replace the constants in the
block below with measured values and nothing else needs to change.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FRONT-END CONSTANTS  (placeholders — replace with measured values)
# ─────────────────────────────────────────────────────────────────────────────
#
# Single source of truth for "how the circuit turns methane into a voltage".
# The three circuit/bench values below are None on purpose: voltage_to_ppm raises
# until they are set, so an uncalibrated upload STOPs loudly instead of producing
# a made-up ppm. The power-law A/m are runnable datasheet-typical placeholders.

# ── Circuit constants — set once from the over-amplification front-end you build ──
SUPPLY_VOLTAGE_V    = None   # V — V_c, the supply across the R_s–R_L divider.
                             # HARDWARE. Read it off your circuit. Required.
LOAD_RESISTANCE_OHM = None   # Ω — R_L, the (large) over-amplification load resistor
                             # the ADC measures across. HARDWARE. Required.

# ── Bench-measured baseline — measure indoors in clean air AFTER burn-in ──
R0_OHM              = None   # Ω — R_s in clean reference air (~1.9 ppm background).
                             # MEASURE on the bench; it anchors the R_s/R₀ ratio.
                             # Required.

# ── Power-law calibration  R_s/R₀ = A · C^(−m)  ──
POWERLAW_A          = 1.0    # — DATASHEET-TYPICAL placeholder. In gas-free operation
                             # A is DERIVED from the background anchor (see BACKGROUND_PPM
                             # and anchor_A): apply_node overwrites this. Read the exact
                             # value off the sensitivity curve only if you skip anchoring.
POWERLAW_M          = 0.35   # — DATASHEET-TYPICAL placeholder (log-log slope).
                             # ⚠ UNVERIFIED at 2–40 ppm: the datasheet curve is
                             # 500–12,500 ppm, so this is an extrapolation until you
                             # refit A/m against certified span gas.

# ── Gas-free anchor: pin the clean-air background to a known ppm ──
BACKGROUND_PPM      = 1.9    # ppm — clean-air CH₄ background (NOAA GML). At clean air
                             # R_s = R₀ ⇒ ratio = 1 ⇒ C = A^(1/m); forcing that to equal
                             # BACKGROUND_PPM gives A = BACKGROUND_PPM^m. This is the one
                             # measured anchor a gas-free calibration has — see anchor_A.

# ── Temperature / humidity correction on the R_s/R₀ ratio (identity until measured) ──
TCORR_PER_C         = 0.0    # per °C — fractional shift of R_s/R₀ per °C away from
                             # T0_REF_C. ASSUMED 0 (no-op) until a T sweep measures it.
RHCORR_PER_PCT      = 0.0    # per %RH — fractional shift per %RH away from H0_REF_PCT.
                             # ASSUMED 0 (no-op) until an RH sweep measures it.
T0_REF_C            = 20.0   # °C  — reference temperature (matches sensor_sim.py).
H0_REF_PCT          = 50.0   # %RH — reference humidity   (matches sensor_sim.py).

# ── Kernel selection: which voltage→ppm model voltage_to_ppm dispatches on ──
KERNEL          = "powerlaw"   # "powerlaw" — gas-free, runs today (R₀ anchor + datasheet
                               #   slope; provisional scale).
                               # "mitchell" — Mitchell 2024 Eq. 16; ingests V_out directly
                               #   (does NOT use R₀) and REQUIRES gas-fit coefficients.
MITCHELL_COEFFS = None         # dict(C1,C2,C3,C4,C5,C7) fit to THIS sensor with calibration
                               # gas. None ⇒ the Mitchell kernel refuses (gate). The paper's
                               # coefficients are deliberately NOT a default — they give
                               # ≈ −165 ppm on our hardware; see calibration/mitchell/.

# Which constants MUST be set before any conversion can run. For the power-law kernel:
# the three circuit/bench values (a/m always have datasheet defaults, so they are not
# gated). For the Mitchell kernel: V_c + R_L for the divider-validity mask, plus the
# gas-fit MITCHELL_COEFFS — but NOT R₀ (Mitchell ingests V_out directly).
_REQUIRED_CONSTANTS = ("SUPPLY_VOLTAGE_V", "LOAD_RESISTANCE_OHM", "R0_OHM")
_REQUIRED_MITCHELL = ("SUPPLY_VOLTAGE_V", "LOAD_RESISTANCE_OHM", "MITCHELL_COEFFS")


def calibration_status() -> tuple[bool, list[str]]:
    """
    Report whether the front-end is ready to convert voltage → ppm.

    Returns (ready, missing): ``ready`` is True only when every required constant for
    the ACTIVE kernel has been set; ``missing`` lists the still-``None`` constant names
    (empty when ready). Read at call time from the live module globals, so a test or a
    deployment that sets the constants (or ``apply_node``) is reflected immediately. Used
    by ``fieldtest.parse_csv`` and ``misc.preflight`` to gate/label the voltage path.

    The required set depends on ``KERNEL``: the power-law kernel needs the circuit/bench
    trio; the Mitchell kernel needs V_c, R_L and the gas-fit ``MITCHELL_COEFFS`` (and
    refuses without them, so the paper's coefficients can never sneak in as a default).
    """
    g = globals()
    required = _REQUIRED_MITCHELL if g["KERNEL"] == "mitchell" else _REQUIRED_CONSTANTS
    missing = [name for name in required if g[name] is None]
    return (not missing, missing)


def anchor_A(background_ppm, m):
    """Derive the power-law prefactor A from the clean-air background anchor.

    At clean air R_s = R₀ ⇒ ratio = 1, and inverting R_s/R₀ = A·C^(−m) at ratio 1
    gives C = A^(1/m). Forcing that clean-air C to equal the known background pins
    ``A = background_ppm ** m`` — the single measured anchor a gas-free calibration
    has. Because A is DEFINED by (background, m), it is derived here rather than
    stored, so a later m refit can't silently leave A disagreeing with the anchor.
    """
    return float(background_ppm ** float(m))


def apply_node(cfg: dict) -> None:
    """Apply one node's measured constants onto the live module globals.

    ``cfg`` is a plain dict of measured truth owned by ``calibration/nodes/nodeN.py``
    (data, no logic). This is the ONE way node constants reach the runtime: it writes
    the same globals ``calibration_status``/``_resolve`` read, so the gate, the existing
    tests' monkeypatch pattern, and ``parse_csv``'s voltage path all keep working
    unchanged. ``POWERLAW_A`` is derived from the anchor (never taken from ``cfg``).

    Required keys: ``supply_voltage_v``, ``load_resistance_ohm``. Optional:
    ``r0_ohm``, ``powerlaw_m``, ``background_ppm``, ``tcorr_per_c``, ``rhcorr_per_pct``,
    ``kernel``, ``mitchell_coeffs``.
    """
    g = globals()
    g["SUPPLY_VOLTAGE_V"] = float(cfg["supply_voltage_v"])
    g["LOAD_RESISTANCE_OHM"] = float(cfg["load_resistance_ohm"])
    if cfg.get("r0_ohm") is not None:
        g["R0_OHM"] = float(cfg["r0_ohm"])
    g["POWERLAW_M"] = float(cfg.get("powerlaw_m", g["POWERLAW_M"]))
    g["BACKGROUND_PPM"] = float(cfg.get("background_ppm", g["BACKGROUND_PPM"]))
    g["POWERLAW_A"] = anchor_A(g["BACKGROUND_PPM"], g["POWERLAW_M"])   # derived, not stored
    g["TCORR_PER_C"] = float(cfg.get("tcorr_per_c", 0.0))
    g["RHCORR_PER_PCT"] = float(cfg.get("rhcorr_per_pct", 0.0))
    g["KERNEL"] = str(cfg.get("kernel", "powerlaw"))
    if cfg.get("mitchell_coeffs") is not None:
        g["MITCHELL_COEFFS"] = dict(cfg["mitchell_coeffs"])


_MITCHELL_KEYS = ("C1", "C2", "C3", "C4", "C5", "C7")


def _resolve_mitchell(v_c, coeffs):
    """Fill V_c and the Mitchell coefficients from the live globals, then validate.

    Mitchell Eq. 16 ingests V_out directly, so R₀/A/m are irrelevant here; only V_c
    (for the divider-validity mask) and the six gas-fit coefficients are needed. Raises
    ValueError — naming ``MITCHELL_COEFFS`` — when the kernel is selected without them,
    so the paper's coefficients can never be used by default (they give ≈ −165 ppm on
    our hardware).
    """
    v_c = SUPPLY_VOLTAGE_V if v_c is None else v_c
    coeffs = MITCHELL_COEFFS if coeffs is None else coeffs
    if v_c is None or coeffs is None:
        unset = [n for n, val in (("SUPPLY_VOLTAGE_V", v_c),
                                  ("MITCHELL_COEFFS", coeffs)) if val is None]
        raise ValueError(
            "Mitchell kernel selected but not calibrated — set "
            + ", ".join(unset)
            + ". MITCHELL_COEFFS must be fit to THIS sensor with calibration gas; the "
            "paper's coefficients are refused by design (they give ≈ −165 ppm here). "
            "See calibration/mitchell/ and docs/CALIBRATION.md."
        )
    missing = [k for k in _MITCHELL_KEYS if k not in coeffs]
    if missing:
        raise ValueError(f"MITCHELL_COEFFS missing keys: {missing}.")
    if not (v_c > 0):
        raise ValueError(f"SUPPLY_VOLTAGE_V must be positive, got {v_c!r}.")
    return v_c, coeffs


def _resolve(v_c, r_l, r0, a, m):
    """Fill any unset argument from the live module constant, then validate.

    Reads the CURRENT module globals (not def-time defaults) so a monkeypatched or
    deployment-set constant takes effect. Raises ValueError naming the offending
    constant(s) — an unset circuit/bench value, or a non-physical A/m — so the
    caller (and the upload endpoint) fails loud rather than emitting a bad ppm.
    """
    v_c = SUPPLY_VOLTAGE_V if v_c is None else v_c
    r_l = LOAD_RESISTANCE_OHM if r_l is None else r_l
    r0 = R0_OHM if r0 is None else r0
    a = POWERLAW_A if a is None else a
    m = POWERLAW_M if m is None else m

    unset = [name for name, val in
             (("SUPPLY_VOLTAGE_V", v_c), ("LOAD_RESISTANCE_OHM", r_l), ("R0_OHM", r0))
             if val is None]
    if unset:
        raise ValueError(
            "sensor front-end is not calibrated — set "
            + ", ".join(unset)
            + " in physics/sensor_frontend.py (or pass them explicitly) before "
            "converting voltage to ppm. See docs/CALIBRATION.md."
        )
    if not (v_c > 0 and r_l > 0 and r0 > 0):
        raise ValueError(
            f"circuit constants must be positive: V_c={v_c}, R_L={r_l}, R0={r0}."
        )
    if not (a > 0):
        raise ValueError(f"POWERLAW_A must be positive, got {a!r}.")
    if not (m > 0):
        raise ValueError(
            f"POWERLAW_M must be positive (it is a log-log slope), got {m!r}."
        )
    return v_c, r_l, r0, a, m


def _th_correction_factor(temperature=None, humidity=None,
                          tcorr=None, rhcorr=None,
                          t0=None, h0=None):
    """
    Multiplicative temperature/humidity correction applied to the R_s/R₀ ratio.

    Returns 1.0 (a true identity) whenever the coefficients are 0 or the column is
    absent — the default — so an uncalibrated front-end changes nothing here.
    NaN-safe: a missing per-sample T or RH contributes no correction for that
    sample (factor 1) rather than poisoning it to NaN, since T/RH gaps are common
    and the plume signal must survive them.
    """
    tcorr = TCORR_PER_C if tcorr is None else tcorr
    rhcorr = RHCORR_PER_PCT if rhcorr is None else rhcorr
    t0 = T0_REF_C if t0 is None else t0
    h0 = H0_REF_PCT if h0 is None else h0

    factor = 1.0
    if temperature is not None and tcorr:
        dT = np.asarray(temperature, dtype=float) - t0
        dT = np.where(np.isfinite(dT), dT, 0.0)     # NaN gap → no correction
        factor = factor * (1.0 + tcorr * dT)
    if humidity is not None and rhcorr:
        dH = np.asarray(humidity, dtype=float) - h0
        dH = np.where(np.isfinite(dH), dH, 0.0)
        factor = factor * (1.0 + rhcorr * dH)
    return factor


def _kernel_powerlaw(ratio_c, a, m):
    """Power-law kernel: invert R_s/R₀ = A·C^(−m) → C = (A / ratio)^(1/m).

    The one line that is kernel-SPECIFIC; everything around it (divider → ratio →
    T/RH factor → validity mask) is shared. Caller supplies the already-corrected
    ratio and gates on ratio > 0, so this is a pure arithmetic step.
    """
    return (a / ratio_c) ** (1.0 / m)


def _kernel_mitchell(v, temperature, humidity, coeffs):
    """Mitchell 2024 Eq. 16 kernel (dormant until gas-fit coefficients exist).

        M = C1 + C2·exp(C3·V − C4·ln(T+65) − C5·ln(H)) − C7·ln((T+65)·V)

    V in volts, T in °C, H in raw %RH, output linear ppm (Mitchell extraction §1/§3/§4).
    Ingests V_out DIRECTLY — no R_s/R₀, no multiplicative T/RH factor (its T/H terms are
    internal). Per-sample NaN where H ≤ 0 or T+65 ≤ 0 (log domain); the caller's
    0 < V < V_c mask already guards V. C7 carries the paper's negative sign (−0.0587), so
    the printed ``− C7·ln(…)`` becomes a positive contribution.
    """
    C1, C2, C3, C4, C5, C7 = (float(coeffs[k]) for k in _MITCHELL_KEYS)
    v = np.asarray(v, dtype=float)
    T = np.asarray(temperature, dtype=float)
    H = np.asarray(humidity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(v) & np.isfinite(T) & np.isfinite(H) & (H > 0.0) & (T + 65.0 > 0.0)
        expo = C3 * v - C4 * np.log(T + 65.0) - C5 * np.log(H)
        m = C1 + C2 * np.exp(expo) - C7 * np.log((T + 65.0) * v)
        return np.where(ok, m, np.nan)


def voltage_to_ppm(v_out, temperature=None, humidity=None, *,
                   v_c=None, r_l=None, r0=None, a=None, m=None,
                   tcorr=None, rhcorr=None, kernel=None, coeffs=None):
    """
    Convert a raw ADC voltage (or array of them) to methane ppm.

    Dispatches on the active ``KERNEL`` (or the ``kernel`` override):

    • "powerlaw" (default): divider → R_s → R_s/R₀ → T/RH correction → invert the
      power law (see the module header). Gas-free; provisional ppm scale.
    • "mitchell": Mitchell Eq. 16, which ingests V_out directly (no R₀, no
      multiplicative T/RH factor) — REQUIRES gas-fit ``coeffs`` and T + RH.

    Returns TOTAL ppm including background — do NOT add background after.

    Parameters
    ----------
    v_out        : measured node voltage across R_L (V). Scalar or array.
    temperature  : air temperature (°C), scalar/array/None. REQUIRED for "mitchell".
    humidity     : relative humidity (%RH), scalar/array/None. REQUIRED for "mitchell".
    v_c, r_l, r0, a, m : override the power-law module constants (tests pass these).
    kernel, coeffs : override the active kernel / Mitchell coefficients.

    Returns
    -------
    float if v_out is scalar, else an ndarray of ppm. A sample outside the valid
    divider region (v_out ≤ 0 or ≥ V_c), or that yields a non-physical ratio / log
    domain, or that is non-finite, comes back as NaN — which ``parse_csv`` already
    skips — rather than raising, so one bad ADC code can't sink a whole record.

    Raises
    ------
    ValueError if the constants for the active kernel are unset (front-end not
    calibrated) or non-physical — a whole-record configuration error, distinct from a
    single bad sample. The Mitchell kernel also raises if T or RH is absent.
    """
    kernel = KERNEL if kernel is None else kernel
    v = np.asarray(v_out, dtype=float)
    scalar = v.ndim == 0

    if kernel == "mitchell":
        v_c, coeffs = _resolve_mitchell(v_c, coeffs)
        if temperature is None or humidity is None:
            raise ValueError(
                "Mitchell kernel requires temperature and humidity (its T/RH terms "
                "are internal to the model) — none were supplied."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            valid_v = np.isfinite(v) & (v > 0.0) & (v < v_c)
            ppm = np.where(valid_v, _kernel_mitchell(v, temperature, humidity, coeffs),
                           np.nan)
        return float(ppm) if scalar else ppm

    # Power-law kernel (default).
    v_c, r_l, r0, a, m = _resolve(v_c, r_l, r0, a, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Valid divider region only: 0 < V_out < V_c. Outside it the divider
        # inversion is meaningless (open/short/rail), so mark the sample NaN.
        valid_v = np.isfinite(v) & (v > 0.0) & (v < v_c)
        rs = np.where(valid_v, r_l * (v_c - v) / v, np.nan)

        ratio = rs / r0
        ratio_c = ratio / _th_correction_factor(temperature, humidity, tcorr, rhcorr)

        # Invert R_s/R₀ = A·C^(−m)  →  C = (A / ratio)^(1/m). Needs ratio > 0.
        good = np.isfinite(ratio_c) & (ratio_c > 0.0)
        ppm = np.where(good, _kernel_powerlaw(ratio_c, a, m), np.nan)

    return float(ppm) if scalar else ppm


def ppm_to_voltage(c_ppm, temperature=None, humidity=None, *,
                   v_c=None, r_l=None, r0=None, a=None, m=None,
                   tcorr=None, rhcorr=None):
    """
    Inverse of ``voltage_to_ppm``: methane ppm → the ADC voltage that would produce it.

    The exact algebraic inverse (forward power law → ratio → R_s → divider solved
    for V_out), so ``voltage_to_ppm(ppm_to_voltage(C)) == C`` to floating point.
    Exists for two reasons: it makes the round-trip test independent of the forward
    arithmetic (the honest check per CLAUDE.md's practice), and it lets a harness
    synthesise realistic voltage CSVs from a known plume.

    Returns a float for scalar input, else an ndarray. C ≤ 0 → NaN (no voltage
    corresponds to negative concentration).
    """
    v_c, r_l, r0, a, m = _resolve(v_c, r_l, r0, a, m)

    c = np.asarray(c_ppm, dtype=float)
    scalar = c.ndim == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        good = np.isfinite(c) & (c > 0.0)
        ratio_c = np.where(good, a * c ** (-m), np.nan)          # A·C^(−m)
        ratio = ratio_c * _th_correction_factor(temperature, humidity, tcorr, rhcorr)
        rs = ratio * r0
        # Divider solved for V_out: V_out = R_L · V_c / (R_s + R_L).
        v = np.where(np.isfinite(rs), r_l * v_c / (rs + r_l), np.nan)

    return float(v) if scalar else v


def ppm_per_volt(v_baseline, temperature=None, humidity=None, *, dv=1e-4, **cal):
    """Local sensitivity dppm/dV at a baseline voltage — the calibration SLOPE.

    A central finite difference of the REAL ``voltage_to_ppm``, so it is kernel-
    agnostic (works for power-law and Mitchell with no extra math) and can be checked
    in tests against an independent analytic derivative. This slope is what converts an
    electrical noise spec (e.g. a 1.65 mV baseline jitter) into a ppm floor:
    ``random_ppm ≈ |ppm_per_volt(V_baseline)| · noise_V``.

    ⚠ That converts the SHORT-TERM electrical jitter only. It is a LOWER bound on the
    real floor, which is dominated by non-averageable drift and T/RH sensitivity — do
    not report it as "the" detection floor (see misc/calibrate.py).

    ``**cal`` forwards any override (v_c, r_l, r0, a, m, tcorr, rhcorr, kernel, coeffs)
    to ``voltage_to_ppm``.
    """
    hi = voltage_to_ppm(v_baseline + dv, temperature, humidity, **cal)
    lo = voltage_to_ppm(v_baseline - dv, temperature, humidity, **cal)
    return float((hi - lo) / (2.0 * dv))

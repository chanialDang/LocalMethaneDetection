"""
plume.py — Gaussian plume dispersion model for methane point-source detection.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES (plain English)
───────────────────────────────────────────────────────────────────────────────
Imagine you're standing near a landfill and someone pokes a hole in a trash
bag. Methane leaks out and the wind carries it downwind like a slowly widening
ribbon. This module answers the question:

    "If I know WHERE the leak is, HOW MUCH gas it emits, and WHICH WAY the
     wind is blowing — what concentration will my sensor read?"

That question is the "forward model". Later work inverts it: given sensor
readings, find the leak.

The physics model is called the Gaussian plume because the concentration
profile — if you slice across the plume at any downwind distance — looks like
a bell curve (Gaussian distribution) in both the horizontal and vertical
directions. It's been the standard workhorse for atmospheric dispersion since
the 1960s and is still the basis of EPA regulatory models.

═══════════════════════════════════════════════════════════════════════════════
UNITS AND CONSTANTS — every one stated with its source
───────────────────────────────────────────────────────────────────────────────
  Quantity           Value        Unit        Source
  ─────────────────────────────────────────────────────────────────────────
  R  (gas const)     8.314        J/(mol·K)   NIST CODATA 2018
  M_CH4              16.04        g/mol       IUPAC atomic weights 2021
  T  default         293.15       K           20 °C standard temperature
  P  default         101325       Pa          1 atm standard pressure
  CH4_BACKGROUND     1.9          ppm         NOAA GML 2023 global mean
  σ  coefficients    table        x in m,     otm33a_dispersion_sigma.csv
                     lookup +     σ in m      (Briggs 1973 formulas tabulated
                     interp                    at 1 m steps — see loader note)
  u_min guard        0.5          m/s         below this the model diverges
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import math
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DISPERSION COEFFICIENT TABLE LOADER
# ─────────────────────────────────────────────────────────────────────────────
#
# Instead of evaluating the Briggs (1973) formulas on the fly, this module now
# reads pre-computed σ_y / σ_z values from `otm33a_dispersion_sigma.csv` and
# interpolates between tabulated distances. The CSV is laid out as:
#
#     distance_m, sigma_y_class1 … sigma_y_class6, sigma_z_class1 … sigma_z_class6
#
# covering integer distances 1–200 m for the six Pasquill stability classes
# (1 = A = very unstable … 6 = F = very stable).
#
# PROVENANCE — read this before citing the table anywhere:
#   The σ values in the shipped CSV are the Briggs (1973) open-country formulas
#   tabulated at 1-metre steps; they are NOT independent measurements. Switching
#   from the closed-form formulas to this table changes the *interface* (numeric
#   classes, lookup + interpolation) but not the physics. If/when a genuinely
#   different source table is supplied, drop it in at the same path and the rest
#   of the pipeline is unchanged.

_SIGMA_TABLE = None   # cached dict: distance(float) → {'sigma_y':[6], 'sigma_z':[6]}
_N_CLASSES = 6        # Pasquill A–F mapped to integers 1–6


def _load_sigma_table() -> dict:
    """
    Load and cache the dispersion-coefficient lookup table from CSV.

    Returns a dict keyed by downwind distance (float, metres). Each value is
    {'sigma_y': [σ for class 1..6], 'sigma_z': [σ for class 1..6]}.

    Raises
    ------
    FileNotFoundError  if the CSV is missing next to this module.
    """
    global _SIGMA_TABLE
    if _SIGMA_TABLE is not None:
        return _SIGMA_TABLE

    csv_path = Path(__file__).parent / "otm33a_dispersion_sigma.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dispersion-coefficient table not found at {csv_path}. "
            "sigma_y/sigma_z now read from this CSV instead of computing Briggs "
            "formulas inline."
        )

    data = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            distance = float(row["distance_m"])
            data[distance] = {
                "sigma_y": [float(row[f"sigma_y_class{i}"]) for i in range(1, _N_CLASSES + 1)],
                "sigma_z": [float(row[f"sigma_z_class{i}"]) for i in range(1, _N_CLASSES + 1)],
            }

    _SIGMA_TABLE = data
    return data


def _lookup_sigma(x: float, stability_class: int, key: str) -> float:
    """
    Shared table lookup + linear interpolation for σ_y and σ_z.

    Parameters
    ----------
    x               : downwind distance (m). Must lie within the table range.
    stability_class : 1–6 (1 = A = most unstable, 6 = F = most stable).
    key             : 'sigma_y' or 'sigma_z'.

    Returns
    -------
    Interpolated σ in metres.

    Raises
    ------
    ValueError  if the class is out of 1–6 or x is outside the tabulated range.
    """
    if not (isinstance(stability_class, (int, np.integer)) and 1 <= stability_class <= _N_CLASSES):
        raise ValueError(
            f"stability_class must be an integer 1–{_N_CLASSES} "
            f"(1=A … {_N_CLASSES}=F), got {stability_class!r}."
        )

    data = _load_sigma_table()
    distances = sorted(data.keys())
    d_min, d_max = distances[0], distances[-1]

    # Guard the table edges. A tiny epsilon absorbs floating-point overshoot at
    # the boundary (e.g. a coordinate rotation yielding 200.0000000000003 m),
    # then we clamp into range so the interpolation below stays well-defined.
    eps = 1e-6
    if x < d_min - eps or x > d_max + eps:
        raise ValueError(
            f"Downwind distance {x:.4f} m is outside the tabulated range "
            f"[{d_min:g}, {d_max:g}] m. The model is only defined where the "
            f"σ table has data."
        )
    x = min(max(x, d_min), d_max)

    class_idx = stability_class - 1

    # Find the bracketing tabulated distances.
    x_lower = max(d for d in distances if d <= x)
    x_upper = min(d for d in distances if d >= x)
    if x_lower == x_upper:
        return data[x_lower][key][class_idx]

    s_lower = data[x_lower][key][class_idx]
    s_upper = data[x_upper][key][class_idx]
    frac = (x - x_lower) / (x_upper - x_lower)
    return s_lower + frac * (s_upper - s_lower)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

R_GAS         = 8.314   # J/(mol·K)  — NIST CODATA 2018
                         # The universal gas constant; links temperature,
                         # pressure, and the number of molecules in a gas.

M_CH4         = 16.04   # g/mol  — IUPAC 2021
                         # Molecular weight of methane: 1 carbon (12) + 4
                         # hydrogen (4×1.01) = 16.04 g per mole.

CH4_BACKGROUND = 1.9    # ppm  — NOAA Global Monitoring Laboratory 2023
                         # The ambient methane concentration everywhere on
                         # Earth right now, before any local source adds to it.
                         # Our sensors see this as their "zero" baseline.

U_MIN         = 0.5     # m/s — minimum valid wind speed
                         # The Gaussian plume formula has wind speed in the
                         # denominator (C ∝ 1/u). As wind → 0, concentration
                         # → infinity, which is physically wrong (gas just
                         # piles up and diffuses in all directions instead).
                         # Below 0.5 m/s we refuse to calculate.


# ─────────────────────────────────────────────────────────────────────────────
# DISPERSION COEFFICIENTS  σ_y and σ_z
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW THE PLUME SPREADS — plain English
# ─────────────────────────────────────
# As a puff of gas travels downwind it mixes with surrounding air due to
# turbulence. The further it travels, the more spread out it becomes.
#
# We capture this spreading with two numbers:
#   σ_y  = how wide the plume is sideways (crosswind), in metres
#   σ_z  = how tall the plume is vertically, in metres
#
# Both grow with downwind distance x. Their exact growth rate depends on how
# turbulent the atmosphere is — described by the Pasquill stability class.
# Classes are now integers 1–6 (was 'A'–'F') to match the lookup-table columns
# and the downstream Week-3 inversion:
#
#   1 = A = Very unstable     (sunny, light wind — lots of turbulent mixing)
#   2 = B = Unstable
#   3 = C = Slightly unstable
#   4 = D = Neutral           (overcast, or moderate wind — intermediate)
#   5 = E = Slightly stable
#   6 = F = Very stable       (clear night, calm — very little mixing)
#
# σ_y and σ_z are read from `otm33a_dispersion_sigma.csv` and linearly
# interpolated between tabulated distances (see _lookup_sigma above). The
# shipped table is the Briggs (1973) open-country formulas tabulated at 1 m
# steps, so the numbers are identical to the old closed-form code up to small
# interpolation error — the change is one of interface and data source, not
# physics. The table covers 1–200 m; outside that range the model is undefined
# and a ValueError is raised. As before, sub-100 m and rough-terrain results
# are order-of-magnitude estimates only.

def sigma_y(x: float, stability_class: int) -> float:
    """
    Crosswind (horizontal) dispersion coefficient σ_y at downwind distance x.

    Looked up from `otm33a_dispersion_sigma.csv` with linear interpolation
    between tabulated 1 m distances. The shipped table is the Briggs (1973)
    open-country σ_y values; see the module loader note on provenance.

    Parameters
    ----------
    x               : downwind distance from source (m); must lie in 1–200 m.
    stability_class : 1–6  (1 = A = most unstable, 6 = F = most stable).

    Returns
    -------
    σ_y in metres
    """
    return _lookup_sigma(x, stability_class, "sigma_y")


def sigma_z(x: float, stability_class: int) -> float:
    """
    Vertical dispersion coefficient σ_z at downwind distance x.

    Looked up from `otm33a_dispersion_sigma.csv` with linear interpolation
    between tabulated 1 m distances. The shipped table is the Briggs (1973)
    open-country σ_z values; see the module loader note on provenance.

    Parameters
    ----------
    x               : downwind distance (m); must lie in 1–200 m.
    stability_class : 1–6  (1 = A = most unstable, 6 = F = most stable).

    Returns
    -------
    σ_z in metres
    """
    return _lookup_sigma(x, stability_class, "sigma_z")


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE ROTATION
# ─────────────────────────────────────────────────────────────────────────────
#
# THE PROBLEM — plain English
# ────────────────────────────
# The Gaussian plume formula only knows about two directions: "downwind" (x)
# and "crosswind" (y). But our map has East and North axes, and the wind can
# blow any direction.
#
# Solution: rotate the map so that the "downwind" axis lines up with +x.
# Every receptor position gets translated into this wind-aligned frame before
# we run the plume equation.
#
# WIND DIRECTION CONVENTION
# ─────────────────────────
# Meteorologists describe wind by WHERE IT COMES FROM, not where it goes.
# "Wind from the west" (φ = 270°) means air is blowing eastward.
# This is counter-intuitive but universal in weather data.
#
# DERIVATION (pinned here to prevent sign errors)
# ─────────────────────────────────────────────────
# Map axes: x = East, y = North.
# φ = meteorological from-direction (clockwise from North).
#
# Wind travels TOWARD (φ − 180°), giving the downwind unit vector:
#     û_down  = (−sin φ,  −cos φ)   [in East, North components]
#
# The crosswind unit vector is 90° counterclockwise from û_down:
#     v̂_cross = ( cos φ,  −sin φ)
#
# Receptor displacement from source:
#     dx = x_receptor − x_source
#     dy = y_receptor − y_source
#
# Project onto wind-frame axes:
#     x_wind = dx·(−sin φ) + dy·(−cos φ)    positive = downwind
#     y_wind = dx·( cos φ) + dy·(−sin φ)    crosswind (enters as y² so sign ok)
#
# Quick check: φ = 270° → sin φ = −1, cos φ = 0
#     x_wind = dx·(+1) + dy·0 = dx
#     Receptor east of source (dx > 0) → x_wind > 0 → downwind ✓
#     Receptor west of source (dx < 0) → x_wind < 0 → upwind  ✓

def rotate_to_wind_frame(
    x_r: float,
    y_r: float,
    src_x: float,
    src_y: float,
    wind_dir_deg: float,
) -> tuple[float, float]:
    """
    Convert a receptor map position into the source-centred wind frame.

    Map convention: x = East, y = North.
    wind_dir_deg (φ) = meteorological FROM-direction, clockwise from North.
      e.g. φ = 270° → wind comes FROM the west → plume travels eastward.

    See the module-level derivation above for the full math.

    Parameters
    ----------
    x_r, y_r      : receptor position in map frame (m)
    src_x, src_y  : source position in map frame (m)
    wind_dir_deg  : meteorological wind-from direction (° clockwise from N)

    Returns
    -------
    (x_wind, y_wind) in metres.
    x_wind > 0  →  receptor is downwind of source  →  plume formula applies.
    x_wind ≤ 0  →  receptor is upwind              →  returns background only.
    """
    phi = math.radians(wind_dir_deg)   # convert degrees → radians for trig

    # Receptor's displacement from the source in map coordinates
    dx = x_r - src_x
    dy = y_r - src_y

    # Rotate into wind-aligned frame using the dot products derived above
    x_wind = dx * (-math.sin(phi)) + dy * (-math.cos(phi))  # downwind component
    y_wind = dx * ( math.cos(phi)) + dy * (-math.sin(phi))  # crosswind component

    return x_wind, y_wind


# ─────────────────────────────────────────────────────────────────────────────
# THE GAUSSIAN PLUME EQUATION
# ─────────────────────────────────────────────────────────────────────────────
#
# THE CORE PHYSICS — plain English
# ─────────────────────────────────
# Picture a fire hose spraying smoke continuously. Downwind, the smoke forms
# a cone that gets wider and taller as it travels. At any cross-section of
# that cone, the concentration of smoke peaks at the centreline and falls off
# symmetrically in a bell-curve pattern — that's the Gaussian shape.
#
# The full formula (Pasquill 1961, Gifford 1961) for a continuous point source
# with a perfectly reflective ground (no gas is absorbed into the soil):
#
#   C = ────────────────── · exp(−y²/2σ_y²) · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
#         2π · σ_y · σ_z · u
#
# Breaking this down term by term:
#
#   Q / (2π σ_y σ_z u)          Mass conservation: all the emitted gas has to
#                                 go somewhere. This ensures the total mass
#                                 flowing through any cross-section equals Q.
#
#   exp(−y²/2σ_y²)               The crosswind bell curve: concentration falls
#                                 off as you move sideways from the plume axis.
#
#   exp(−(z−H)²/2σ_z²)           The "real" source contribution: gas emitted at
#                                 height H spreads vertically; at ground level
#                                 z=0 this is exp(−H²/2σ_z²).
#
#   + exp(−(z+H)²/2σ_z²)         The "image" source: the ground reflects the
#                                 plume like a mirror. We pretend there's an
#                                 identical source at depth −H underground.
#                                 When H=0, both terms equal 1, so the 2 in
#                                 the denominator cancels → C = Q/(π σ_y σ_z u).

def concentration_gm3(
    x: float,
    y: float,
    Q: float,
    u: float,
    H: float,
    sy: float,
    sz: float,
    z: float = 0.0,
) -> float:
    """
    Steady-state Gaussian plume concentration at a receptor (g/m³).

    Uses the Pasquill (1961) / Gifford (1961) formula with perfect ground
    reflection (the ground acts like a mirror — no gas is absorbed).

    See the module-level explanation above for the physics intuition.

    Parameters
    ----------
    x, y  : wind-frame coordinates (m); x must be positive (downwind).
    Q     : source emission rate (g/s) — how much gas leaks per second.
    u     : mean wind speed (m/s). Must be ≥ 0.5 m/s.
    H     : effective release height (m). 0–2 m for near-ground sources.
    sy    : σ_y from sigma_y(x, class) — plume's crosswind width (m).
    sz    : σ_z from sigma_z(x, class) — plume's vertical height (m).
    z     : sensor height above ground (m); default 0 (ground level).
            Use 1.0–1.5 m for fence-mounted sensors — important because
            the vertical concentration gradient is steep near H≈0 sources.

    Returns
    -------
    float  mass concentration in g/m³.
            Returns 0.0 for x ≤ 0 (upwind — no plume there).

    Raises
    ------
    ValueError  if u < U_MIN (0.5 m/s). The model diverges at calm winds.
    """
    if x <= 0.0:
        # Nothing to calculate upwind — the plume hasn't arrived yet.
        return 0.0

    if u < U_MIN:
        raise ValueError(
            f"Wind speed u={u} m/s is below the model minimum ({U_MIN} m/s). "
            "Concentration diverges as u→0; use a puff model for calm conditions."
        )

    # ── Crosswind (horizontal) bell-curve factor ──
    # This is 1.0 at y=0 (centreline) and decreases to ~0.6 at y=σ_y,
    # ~0.14 at y=2σ_y, reaching ~0 by y=3σ_y.
    term_y = math.exp(-0.5 * (y / sy) ** 2)

    # ── Vertical bell-curve factor (real source + ground reflection image) ──
    # exp(−(z−H)²/2σ_z²)  →  real source contribution
    # exp(−(z+H)²/2σ_z²)  →  reflected image source contribution
    # When H=0: both equal exp(0) = 1 → term_z = 2
    term_z = (
        math.exp(-0.5 * ((z - H) / sz) ** 2) +
        math.exp(-0.5 * ((z + H) / sz) ** 2)
    )

    # ── Assemble the full Gaussian plume formula ──
    # The 2π in the denominator + term_z=2 at H=0 combine to give π,
    # matching the simpler "ground-level with reflection" formula you often
    # see written as  C = Q / (π σ_y σ_z u).
    return Q / (2.0 * math.pi * sy * sz * u) * term_y * term_z


# ─────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION: g/m³  →  ppm
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY WE NEED THIS — plain English
# ──────────────────────────────────
# The Gaussian formula gives concentration as mass per volume (g/m³ of air).
# Sensors and atmospheric scientists prefer "parts per million" (ppm) — the
# fraction of air molecules that are methane, times a million.
#
# DERIVATION using the ideal gas law (PV = nRT)
# ──────────────────────────────────────────────
# Step 1: At temperature T and pressure P, the total number of air moles per
#         cubic metre is:  n_air/V = P / (R·T)   [mol/m³]
#
# Step 2: The number of methane moles per cubic metre is:
#         n_CH4/V = C_gm3 / M_CH4   [mol/m³]
#         (divide mass concentration by molecular weight to get molar)
#
# Step 3: ppm = (methane moles) / (air moles) × 10⁶
#             = [C_gm3/M_CH4] / [P/(R·T)] × 10⁶
#             = C_gm3 · R · T / (M_CH4 · P) × 10⁶
#
# IMPORTANT: This function is a pure unit converter. Background methane
# (1.9 ppm) is added in predict_ppm(), not here, so this function stays
# reusable for other gases or calculation pipelines.

def gm3_to_ppm_methane(C_gm3: float, T_K: float = 293.15, P_Pa: float = 101325.0) -> float:
    """
    Convert CH4 mass concentration (g/m³) → volume mixing ratio (ppm).

    Assumes ideal gas: PV = nRT. Does NOT add atmospheric background —
    that is done in predict_ppm() to keep this function a pure converter.

    Constants used:
        R     = 8.314   J/(mol·K)  — NIST CODATA 2018
        M_CH4 = 16.04   g/mol      — IUPAC 2021

    Parameters
    ----------
    C_gm3 : CH4 mass concentration (g/m³) — output from concentration_gm3()
    T_K   : air temperature (K); default 293.15 K = 20 °C
    P_Pa  : air pressure (Pa); default 101325 Pa = 1 atm

    Returns
    -------
    float  CH4 plume contribution in ppm (volume/volume, dry-air basis).
           Does NOT include the 1.9 ppm background — add that separately.
    """
    # This is the derivation above in one line:
    #   [C_gm3 · R · T] gives energy units that cancel with [M_CH4 · P]
    #   The 1e6 converts the mole fraction to "parts per million"
    return C_gm3 * (R_GAS * T_K) / (M_CH4 * P_Pa) * 1.0e6


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_ppm(
    src_pos: tuple,
    Q: float,
    u: float,
    wind_dir_deg: float,
    H: float,
    stability_class: int,
    receptors: np.ndarray,
    T_K: float = 293.15,
    P_Pa: float = 101325.0,
    z: float = 0.0,
) -> np.ndarray:
    """
    Predict total CH4 (ppm) at every receptor, including 1.9 ppm background.

    This is the function you'll call most often. It orchestrates the full
    calculation pipeline for an array of sensor positions:

        For each receptor:
          1. Rotate map coordinates → wind frame (rotate_to_wind_frame)
          2. If upwind → return background (1.9 ppm), skip steps 3–5
          3. Look up σ_y, σ_z at that downwind distance
          4. Evaluate Gaussian plume → g/m³  (concentration_gm3)
          5. Convert → ppm  (gm3_to_ppm_methane)
          6. Add CH4_BACKGROUND (1.9 ppm, NOAA GML 2023)

    Parameters
    ----------
    src_pos        : (src_x, src_y) source location in map frame (m), East/North
    Q              : emission rate (g/s)
    u              : mean wind speed (m/s) — must be ≥ 0.5 m/s
    wind_dir_deg   : meteorological FROM-direction (° clockwise from North)
    H              : effective release height (m)
    stability_class: Pasquill–Gifford class as integer 1–6 (1=A … 6=F)
    receptors      : array of shape (N, 2) with [x_east, y_north] in metres
    T_K            : air temperature (K); default 293.15 K
    P_Pa           : air pressure (Pa); default 101325 Pa
    z              : sensor height above ground (m); default 0.0

    Returns
    -------
    np.ndarray  shape (N,) — total CH4 in ppm (plume contribution + background).
    Upwind receptors return exactly CH4_BACKGROUND = 1.9 ppm.
    """
    if not (isinstance(stability_class, (int, np.integer)) and 1 <= stability_class <= 6):
        raise ValueError(
            f"stability_class must be an integer 1–6 (1=A … 6=F), "
            f"got {stability_class!r}."
        )

    src_x, src_y = src_pos
    receptors = np.atleast_2d(np.asarray(receptors, dtype=float))
    n = len(receptors)

    # Start every receptor at background concentration — upwind ones stay here
    result = np.full(n, CH4_BACKGROUND, dtype=float)

    for i in range(n):
        rx, ry = receptors[i, 0], receptors[i, 1]

        # Step 1: rotate from map frame to wind-aligned frame
        xw, yw = rotate_to_wind_frame(rx, ry, src_x, src_y, wind_dir_deg)

        # Step 2: skip upwind receptors — the plume doesn't reach them
        if xw <= 0.0:
            continue  # result[i] stays at CH4_BACKGROUND

        # Step 3: how wide/tall is the plume at this downwind distance?
        sy = sigma_y(xw, stability_class)
        sz = sigma_z(xw, stability_class)

        # Step 4: raw mass concentration from the Gaussian formula
        C = concentration_gm3(xw, yw, Q, u, H, sy, sz, z)

        # Steps 5–6: convert units and add background
        result[i] = gm3_to_ppm_methane(C, T_K, P_Pa) + CH4_BACKGROUND

    return result

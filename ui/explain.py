"""
explain.py — Turn a plume result into a plain-English explanation.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS DOES (plain English)
───────────────────────────────────────────────────────────────────────────────
A contour plot is pretty, but if you're new to this it doesn't tell you what to
THINK about it. This module reads the numbers behind the plot and writes a short,
human paragraph: where the plume is strongest, how far away it's still
detectable, how fast it fades, and the bottom-line verdict.

Two ways the paragraph gets written:

  • template_explanation(...)  — built straight from the numbers. Always works,
                                 needs no internet, no API key, no extra package.
                                 This is the guaranteed fallback.

  • ai_explanation(...)        — if an OpenAI API key is available, send just the
                                 summary NUMBERS to OpenAI and let it write a
                                 richer paragraph. If anything goes wrong (no key,
                                 no network, no package), we silently fall back to
                                 the template — the program never breaks.

The key is read from the OPENAI_API_KEY environment variable, which is also
auto-loaded from a local .env file (see _load_dotenv below). It is never stored
in the code. To enable the AI version, put ONE line in .env (no quotes, no
spaces, no `export`):
    OPENAI_API_KEY=sk-...
or set it in your shell:  export OPENAI_API_KEY=sk-...
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from physics.plume import CH4_BACKGROUND, predict_ppm
# Detection strictness shared with feasibility.py (single source of truth) so the
# caption's "detectable out to X m" matches the plot's detection-limit line.
# SENSOR_NOISE_PPM is the assumed noise floor used to build scenario facts.
from physics.sensor_sim import DETECT_K, SENSOR_NOISE_PPM


# ─────────────────────────────────────────────────────────────────────────────
# .env loader — let the OpenAI key live in one local file instead of the shell
# ─────────────────────────────────────────────────────────────────────────────
def _load_dotenv(path: str | Path | None = None) -> None:
    """
    Minimal, dependency-free .env reader so OPENAI_API_KEY can live in one file.

    Copies KEY=VALUE lines from a .env beside this module into os.environ, but
    only for names not already set — so a real shell variable always wins. It is
    deliberately forgiving: a leading ``export``, spaces around ``=``, surrounding
    quotes, blank lines and #-comments are all tolerated, because those are the
    usual ways a key gets pasted in wrong. A missing/empty file is a silent no-op
    (the AI path then just falls back to the template, exactly as before).
    """
    if path is None:
        # .env lives at the project root (one level up from ui/).
        path = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line[:7].lower() == "export ":          # tolerate `export KEY=...`
            line = line[7:].strip()
        key, sep, value = line.partition("=")       # split on the FIRST = only
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")  # drop spaces, then quotes
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env at import time so OPENAI_API_KEY is in place before ai_explanation runs.
_load_dotenv()

# OpenAI model used for the optional richer explanation. One place to change it.
OPENAI_MODEL = "gpt-4o-mini"

# Holds the reason the most recent AI call fell back to the template (e.g. no key,
# no internet, API error). The plot caption stays graceful/silent, but demo.py reads
# this so it can tell you WHY it used the template instead of guessing.
LAST_AI_ERROR: str | None = None

# The canonical plot scenario — single source of truth shared by demo.py's contour
# plot and ask.py's "what does the graph show" context, so questions are answered
# against the exact numbers on the saved figure.
DEFAULT_SCENARIO = {
    "Q": 5.0,               # g/s   — representative landfill hotspot
    "u": 3.0,               # m/s   — moderate wind
    "H": 1.0,               # m     — near-ground release
    "stability_class": 3,   # Class C (slightly unstable, common daytime)
    "wind_dir_deg": 270.0,  # FROM west → plume runs east
}

# System prompt: gives the model real methane domain grounding so the caption is
# specific to CH4 detection rather than generic "gas plume" boilerplate. Single
# source of truth — edit here to retune the AI voice. Every fact below is standard
# literature-level knowledge; the model is explicitly told NOT to invent numbers
# beyond the per-plot values it is handed.
METHANE_SYSTEM_PROMPT = (
    "You are an atmospheric-chemistry assistant who specializes in methane (CH4) "
    "and its measurement near ground-level sources. Use this domain knowledge only "
    "where it genuinely helps explain the specific plot:\n"
    "- CH4 is a colorless, odorless gas, molar mass 16.04 g/mol. It is lighter than "
    "air (~29 g/mol), so a release is positively buoyant and tends to rise and "
    "dilute as it drifts downwind.\n"
    "- The global background is about 1.9 ppm (NOAA). At a landfill or transfer-"
    "station fenceline the signal of interest is usually only a few ppm above that "
    "background — this is trace detection, far below the lower explosive limit "
    "(~5% by volume = 50,000 ppm), so it is a measurement problem, not a safety "
    "alarm.\n"
    "- These sources are weak, intermittent, and near the ground, so wind direction "
    "and atmospheric stability (how turbulent the air is) strongly control whether "
    "the plume ever reaches a fixed sensor.\n"
    "- The sensor here is a Figaro TGS 2611-E00 metal-oxide (MOX) sensor, "
    "characterized by the maker at 500-12,500 ppm; using it down at 2-40 ppm above "
    "background is the core challenge, because baseline drift and temperature/"
    "humidity cross-sensitivity are comparable in size to the signal.\n"
    "- CH4 is also a potent greenhouse gas (far stronger than CO2 over a 20-year "
    "horizon) with an atmospheric lifetime of roughly a decade, which is why "
    "fenceline monitoring matters.\n"
    "Write for a sharp high-school researcher: plain English, define any technical "
    "term in a few words, and stay strictly consistent with the numbers you are "
    "given. Never invent specific values; if a number is null or missing, say it is "
    "not available rather than guessing."
)


def summarize_field(ppm_grid, xs, ys, params: dict, noise_floor: float) -> dict:
    """
    Boil a 2-D ppm field down to the handful of numbers worth explaining.

    Parameters
    ----------
    ppm_grid    : 2-D array, shape (len(ys), len(xs)) — total ppm incl. background.
    xs          : 1-D downwind distances (m), length = ppm_grid.shape[1].
    ys          : 1-D crosswind distances (m), length = ppm_grid.shape[0].
    params      : dict of the run settings (Q, u, H, stability, wind_dir_deg).
    noise_floor : sensor noise (1σ, ppm).

    Returns
    -------
    dict of plain facts (peak, detection reach, decay, width, verdict inputs).
    """
    ppm_grid = np.asarray(ppm_grid, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    excess = ppm_grid - CH4_BACKGROUND
    threshold = DETECT_K * noise_floor

    # Peak (hottest point in the field).
    peak_flat = int(np.argmax(ppm_grid))
    peak_row, peak_col = np.unravel_index(peak_flat, ppm_grid.shape)
    peak_ppm = float(ppm_grid[peak_row, peak_col])

    # Centreline = the crosswind row closest to y = 0.
    cl_idx = int(np.argmin(np.abs(ys)))
    cl_excess = excess[cl_idx, :]

    # Farthest downwind distance where the centreline still beats the threshold.
    above = np.where(cl_excess > threshold)[0]
    detect_reach = float(xs[above[-1]]) if above.size else None

    # Centreline ppm at a few reference distances (nearest grid column).
    def cl_ppm_at(target_x):
        if target_x < xs.min() or target_x > xs.max():
            return None
        return float(np.interp(target_x, xs, ppm_grid[cl_idx, :]))

    ppm_at = {d: cl_ppm_at(d) for d in (50.0, 100.0, 200.0)}

    # Plume half-width at the nearest column to 50 m: how far sideways until the
    # excess falls to half its centreline value.
    ref_col = int(np.argmin(np.abs(xs - 50.0)))
    col_excess = excess[:, ref_col]
    cl_val = excess[cl_idx, ref_col]
    half_width = None
    if cl_val > 0:
        half_level = 0.5 * cl_val
        # distances (abs y) where excess is at least half the centreline value
        ys_above_half = np.abs(ys[col_excess >= half_level])
        if ys_above_half.size:
            half_width = float(ys_above_half.max())

    return {
        "params": dict(params),
        "noise_floor": float(noise_floor),
        "threshold": float(threshold),
        "background": float(CH4_BACKGROUND),
        "peak_ppm": peak_ppm,
        "peak_x": float(xs[peak_col]),
        "peak_y": float(ys[peak_row]),
        "detect_reach_m": detect_reach,
        "ppm_at_50": ppm_at[50.0],
        "ppm_at_100": ppm_at[100.0],
        "ppm_at_200": ppm_at[200.0],
        "half_width_at_50_m": half_width,
    }


def _fmt(value, unit="", nd=2):
    """Format a possibly-None number for display."""
    if value is None:
        return "n/a"
    return f"{value:.{nd}f}{unit}"


def _fmt_duration(seconds):
    """Human-friendly duration: seconds → '45 s' / '3.2 min' / '1.5 h'."""
    if seconds is None:
        return "an unknown length"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


def template_explanation(facts: dict) -> str:
    """
    Write the plain-English paragraph straight from the summary numbers.

    Always available — no API, no network. This is the fallback that guarantees
    there is ALWAYS a sensible caption.
    """
    p = facts["params"]
    bkg = facts["background"]
    reach = facts["detect_reach_m"]

    lines = []
    lines.append(
        f"This map shows a methane plume from a {p.get('Q', '?')} g/s source in "
        f"Class {p.get('stability_class', p.get('stability', '?'))} air with wind "
        f"{p.get('u', '?')} m/s from {p.get('wind_dir_deg', '?')}°. Background is "
        f"{bkg:g} ppm; colours show methane above that."
    )
    lines.append(
        f"The signal peaks at about {_fmt(facts['peak_ppm'], ' ppm')} near the "
        f"source and fades downwind as the plume spreads and dilutes: roughly "
        f"{_fmt(facts['ppm_at_50'], ' ppm')} at 50 m, "
        f"{_fmt(facts['ppm_at_100'], ' ppm')} at 100 m, and "
        f"{_fmt(facts['ppm_at_200'], ' ppm')} at 200 m on the centreline."
    )

    if reach is not None:
        lines.append(
            f"Taking the sensor's noise into account, the plume stays clearly "
            f"detectable (above the {_fmt(facts['threshold'], ' ppm')} threshold = "
            f"{DETECT_K:g}× the {_fmt(facts['noise_floor'], ' ppm')} noise floor) "
            f"out to about {reach:.0f} m downwind. The plume is roughly "
            f"{_fmt(facts['half_width_at_50_m'], ' m', nd=0)} wide (half-max) at 50 m."
        )
    else:
        lines.append(
            f"Taking the sensor's noise into account, the plume never rises above "
            f"the {_fmt(facts['threshold'], ' ppm')} detection threshold "
            f"({DETECT_K:g}× the {_fmt(facts['noise_floor'], ' ppm')} noise floor) "
            f"on this grid — this source would be hard to detect here."
        )

    lines.append(
        "Remember: this is a steady-state, open-country estimate. Real readings "
        "scatter more, and sub-100 m values are order-of-magnitude only. Because the "
        "recovered leak size scales directly with the assumed wind speed and "
        "stability class, treat any single emission figure as an upper bound, not a "
        "precise value — and note that averaging beats down only random noise (as "
        "√N), never a fixed calibration bias."
    )
    return " ".join(lines)


def ai_explanation(facts: dict, model: str = OPENAI_MODEL) -> str | None:
    """
    Ask OpenAI to write a richer, methane-specific paragraph from the summary
    numbers.

    A methane-domain system prompt (METHANE_SYSTEM_PROMPT) grounds the model so
    the caption reflects real CH4 behaviour and the Figaro sensor's limits rather
    than generic gas-plume boilerplate. Only the small set of summary numbers is
    sent — never the raw grid.

    Returns the text on success, or None on ANY problem (missing key, missing
    `openai` package, network/API error) so the caller can fall back cleanly.
    On failure it records the reason in the module-level LAST_AI_ERROR so the
    caller can report WHY it fell back (the plot itself never breaks).
    """
    global LAST_AI_ERROR
    LAST_AI_ERROR = None
    if not os.environ.get("OPENAI_API_KEY"):
        LAST_AI_ERROR = "no OPENAI_API_KEY set"
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        user_prompt = (
            "Explain this methane plume dispersion plot in 3-4 short, plain-English "
            "sentences. Cover, in order: (1) the peak concentration and roughly "
            "where it is, (2) how the signal fades with downwind distance, (3) how "
            "far downwind the plume stays detectable above the sensor's noise floor, "
            "and (4) one honest caveat about what this steady-state, open-country "
            "model cannot capture. Tie every statement to these numbers (JSON):\n\n"
            f"{json.dumps(facts, indent=2, default=str)}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": METHANE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=320,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        # Any failure → record why, then let the caller use the template instead.
        LAST_AI_ERROR = f"{type(exc).__name__}: {exc}"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CHAT SESSION
# ─────────────────────────────────────────────────────────────────────────────

# System prompt that encodes the full project knowledge so the chat can answer
# questions about the physics, tests, optimizer guard rails, and known caveats.
# Written once here so it stays in sync with the rest of the codebase.
_CHAT_SYSTEM_PROMPT = """\
You are an expert assistant embedded inside a methane-detection research project.
The user is a high-school researcher. Answer in plain English, define any jargon
the first time you use it, and tie every answer to the specific code/numbers below.
Never invent values; if something is unknown say so.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: detect and locate methane (CH4) leaks at fenceline distances (10–200 m) using
a low-cost Figaro TGS 2611-E00 metal-oxide (MOX) sensor. The sensor is rated by the
manufacturer at 500–12,500 ppm, but we are using it at 2–40 ppm above the 1.9 ppm
global background — that's the research contribution: making a cheap off-label sensor
work for trace atmospheric detection.

Real-data sequence:
  1. Custer Road Transfer Station, Allen TX — first deployment, moderate signal.
  2. Melissa TX landfill — follow-on test, expected harder / lower-ppm case.
Everything must work for both sites; design choices are tuned for the harder Melissa case.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHYSICS: THE GAUSSIAN PLUME FORWARD MODEL (plume.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The core equation (Pasquill 1961, Gifford 1961):

  C = Q / (2π σ_y σ_z u)
    × exp(−y²/2σ_y²)
    × [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]

  Q  = emission rate (g/s)         σ_y = crosswind plume width (m)
  u  = wind speed (m/s)            σ_z = vertical plume height (m)
  H  = release height (m)          y   = crosswind offset (m)
  z  = sensor height (m)           The second bracket is the ground-reflection term.

At H=z=y=0 this collapses to C = Q / (π σ_y σ_z u) — the standard ground-level form.
Background CH4 = 1.9 ppm (NOAA GML 2023) is added at the very end in predict_ppm(),
NOT inside the unit-conversion function gm3_to_ppm_methane().

Dispersion coefficients σ_y, σ_z — Briggs (1973) open-country formulas:
  Tabulated at 1 m steps from 1–200 m in briggs_dispersion_sigma.csv.
  Six Pasquill stability classes, integers 1–6:
    1=A very unstable (sunny, light wind), 2=B, 3=C slight unstable,
    4=D neutral (overcast/moderate wind), 5=E slight stable, 6=F very stable (clear night).
  Class A: plume spreads wide and dilutes fast. Class F: stays narrow and concentrated.

Wind direction: meteorological FROM-direction (clockwise from North).
  φ=270° = wind FROM west → plume travels east along +x.
  Rotation formula: x_wind = dx·(−sinφ) + dy·(−cosφ). Upwind receptors → background only.

Hard guard: u < 0.5 m/s raises ValueError (C → ∞ as u → 0; use a puff model instead).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMIZER GUARD RAILS (the clamp_to_table flag — Week-3 inversion)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
predict_ppm(... clamp_to_table=False) — default / strict mode
  Raises ValueError if any downwind distance falls outside the 1–200 m σ table.
  Used in demo.py, tests, the contour plot. Hard errors surface bad geometry.

predict_ppm(... clamp_to_table=True) — OPTIMIZER-SAFE MODE (use this in Week-3 inversion)
  Never raises on geometry. scipy.optimize probes millions of candidate source positions;
  a single out-of-table evaluation would crash the whole run in strict mode.
  Policy when out of range:
    < 1 m downwind  → clamped to the 1 m floor (sub-1 m is degenerate and
                       sub-100 m is order-of-magnitude anyway).
    > 200 m downwind → returns background only (source that far is undetectable;
                        a pure-background reading is the honest answer).
  Inside 1–200 m the two modes are numerically identical (verified by test).

Three bounds the OPTIMIZER still owns (not auto-fixed by the model):
  1. u ≥ 0.5 m/s   — calm-wind guard still raises; bound wind speed in the optimizer.
  2. stability_class is a discrete integer 1–6 — fix from weather data or loop;
     never let a continuous optimizer vary it (it will try non-integer values).
  3. Q ≥ 0         — negative emission is unphysical; add a lower bound of 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENSOR SIMULATION & SIGNAL PROCESSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sensor_sim.py generates synthetic readings since the real Figaro isn't deployed yet:
  raw = background(1.9) + true_plume + slow drift + temp_effect + humid_effect + noise

ASSUMED constants (⚠ placeholders — replace when real sensor is characterized):
  SENSOR_NOISE_PPM  = 0.30 ppm (1σ Gaussian)
  BASELINE_DRIFT    = 1.00 ppm (slow sensor zero wander over hours)
  TEMP_COEFF        = 0.05 ppm/°C
  HUMID_COEFF       = 0.02 ppm/%RH

Processing pipeline (processing.py), applied in order:
  1. temp_humidity_correct() — linear fit on reference window, subtract weather-varying part.
  2. subtract_baseline()     — rolling low-percentile baseline removal.
  3. moving_average()        — centred rolling mean (O(n) cumsum, not O(n·w) convolution).
  4. detect_pattern()        — flags event where excess > k·noise_std for ≥ min_run samples.

Detection threshold = DETECT_K × SENSOR_NOISE_PPM = 3.0 × 0.30 = 0.90 ppm.
This is the SAME threshold used in the feasibility sweep, the contour plot's green
detection-limit line, and the processing tests — one number, consistent everywhere.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEASIBILITY VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sweep across Q, distance, stability class. At 50 m fenceline, neutral Class-D, u=2 m/s:
  Minimum detectable leak ≈ 0.05 g/s (anything weaker stays below 0.90 ppm threshold).
Caveat: this is the CENTRELINE, DOWNWIND geometry — the most favourable possible.
Off-centreline or off-axis sensors see smaller signals and have higher detection floors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE TEST SUITE (41 tests, all green)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tests/test_plume.py — physics core:
  • σ lookup reproduces Briggs formula at exact table row (100 m, Class D)
  • σ interpolates correctly between rows; monotonic in distance and class
  • Out-of-range distance and invalid class both raise ValueError
  • C = Q/(π σ_y σ_z u) at y=z=H=0 (the standard ground-level form)
  • Upwind receptor returns exactly 1.9 ppm background
  • Calm wind raises when a receptor is downwind; returns background cleanly when all upwind
  • Vectorized predict_ppm matches the scalar pipeline point-by-point (the hot-path guard)
  • clamp_to_table=False (strict) raises below 1 m and above 200 m (6 tests)
  • clamp_to_table=True never raises; far-field→background; near-field clamps to 1 m floor;
    clamp mode == strict mode everywhere inside valid range

tests/test_processing.py (Tests 1–5):
  • Baseline subtraction recovers a known excess
  • Moving average reduces noise by √N
  • Temp/humidity correction removes weather component
  • Pattern detection flags event above threshold
  • Full pipeline end-to-end recovers synthetic plume event

tests/test_feasibility.py:
  • Pins the 0.05 g/s fenceline detection verdict (regression guard)

tests/test_explain.py:
  • Template explanation runs without API key
  • AI-absent fallback is clean (no crash, returns template text)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN DISCREPANCIES AND HONEST CAVEATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Steady-state only. The model gives a time-mean picture; real turbulence produces
   intermittent puffs — individual sensor readings fluctuate far more than the smooth
   contour plot suggests. Pasquill-Gifford σ values represent 10-min to hourly averages.

2. Sub-100 m extrapolation. Briggs formulas were validated at ≥100 m over flat open
   terrain. Everything left of the orange dashed line in the contour plot (10–100 m)
   is extrapolated below the valid range — treat as order-of-magnitude only.

3. Open-country σ underestimates dilution at rough sites. A transfer-station fenceline
   has structures, equipment, and a perimeter berm that add turbulent mixing above
   open-field levels. The real plume dilutes FASTER than the model predicts, making
   detection harder in practice than the contour plot shows.

4. Calm-wind undefined. u < 0.5 m/s is when a weak source is most detectable (less
   dilution) — yet the model is explicitly undefined there. Field data near-calm must
   be handled separately (puff model or flagged for manual review).

5. Assumed sensor characteristics. SENSOR_NOISE_PPM (0.30 ppm), drift (1.00 ppm),
   TEMP_COEFF (0.05 ppm/°C), HUMID_COEFF (0.02 ppm/%RH) are all placeholders.
   They were chosen to be plausible, not measured. The whole detection floor —
   including the green line on the plot and the 0.05 g/s fenceline verdict — shifts
   if the real Figaro noise floor is different. THIS IS THE BIGGEST UNKNOWN.

6. Linear weather correction. The temp/humidity correction fits a flat linear model
   (a·T + b·H + c). Real MOX sensors may have nonlinear cross-sensitivity (a T·H
   interaction term, or saturation effects). This is a candidate improvement once
   real sensor data exists.

7. Ground treated as a perfect mirror. The model assumes no CH4 is absorbed by the
   soil (the second term in the vertical bracket). In reality some fraction is absorbed,
   which would reduce ground-level concentration slightly.

8. Single source assumed. The inversion (Week 3) will try to fit one source position
   and emission rate Q. If there are multiple independent leak points, the fit will
   find a phantom "centroid" source rather than the real individual sources.
"""


def compute_field(scenario: dict | None = None,
                  noise_floor: float = SENSOR_NOISE_PPM):
    """
    Build the canonical demo field once and return everything callers need.

    Single source of truth for the 250×200 receptor grid that demo.py's contour
    plot, scenario_facts(), and the web dashboard (server.py) all render — so the
    grid definition lives in exactly one place instead of being duplicated.

    Parameters
    ----------
    scenario    : dict like DEFAULT_SCENARIO (Q, u, H, stability_class, wind_dir_deg).
                  Defaults to DEFAULT_SCENARIO.
    noise_floor : sensor noise (1σ, ppm); defaults to the assumed SENSOR_NOISE_PPM.

    Returns
    -------
    (xs, ys, PPM, facts)
        xs    : 1-D downwind distances (m), shape (250,)   — linspace(10, 200, 250)
        ys    : 1-D crosswind distances (m), shape (200,)  — linspace(-150, 150, 200)
        PPM   : 2-D total ppm incl. background, shape (200, 250) == (len(ys), len(xs))
        facts : dict from summarize_field(PPM, xs, ys, scenario, noise_floor)
    """
    s = dict(DEFAULT_SCENARIO if scenario is None else scenario)

    xs = np.linspace(10.0, 200.0, 250)     # downwind axis (m) — σ-table range
    ys = np.linspace(-150.0, 150.0, 200)   # crosswind axis (m)
    XX, YY = np.meshgrid(xs, ys)
    receptors = np.column_stack([XX.ravel(), YY.ravel()])
    PPM = predict_ppm(
        src_pos=(0.0, 0.0), Q=s["Q"], u=s["u"], wind_dir_deg=s["wind_dir_deg"],
        H=s["H"], stability_class=s["stability_class"], receptors=receptors, z=0.0,
    ).reshape(YY.shape)                     # (NY, NX) == (len(ys), len(xs))

    facts = summarize_field(PPM, xs, ys, s, noise_floor)
    return xs, ys, PPM, facts


def scenario_facts(scenario: dict | None = None,
                   noise_floor: float = SENSOR_NOISE_PPM) -> dict:
    """
    Summary facts for the standard scenario — thin wrapper over compute_field.

    Lets ask.py ground its answers in the exact numbers on the saved figure
    (peak ppm, detection reach, half-width, …) without rebuilding the grid here.
    """
    _, _, _, facts = compute_field(scenario, noise_floor)
    return facts


def ask_once(question: str, facts: dict | None = None,
             model: str = OPENAI_MODEL) -> str:
    """
    Answer ONE question about the plume model with a single OpenAI call, then return.

    This is the non-interactive replacement for the old chat loop: it needs no
    terminal input (the question comes in as an argument), so it works in any
    context that has internet + an API key. It uses the full-project _CHAT_SYSTEM_PROMPT
    so the answer is grounded in this project's physics, tests, and guard rails, and
    optionally includes `facts` (the current graph's numbers) for plot-specific
    questions.

    Unlike ai_explanation (which silently falls back to a template so the plot never
    breaks), this returns a clear, human-readable ERROR STRING on failure — because
    when you ask a question you want to know WHY it couldn't answer.

    Parameters
    ----------
    question : the user's plain-English question.
    facts    : optional dict from scenario_facts()/summarize_field() for grounding.
    model    : OpenAI model name (defaults to OPENAI_MODEL).

    Returns
    -------
    str — the answer, or a plain-English explanation of what went wrong.
    """
    question = (question or "").strip()
    if not question:
        return "No question given. Try:  python3 ask.py \"what does the green line mean?\""

    if not os.environ.get("OPENAI_API_KEY"):
        return ("No OPENAI_API_KEY found. Add a line  OPENAI_API_KEY=sk-...  to the "
                ".env file next to this project, then re-run.")

    try:
        from openai import OpenAI
    except ImportError:
        return "The 'openai' package isn't installed. Run:  pip install openai"

    # Ground the answer in the current graph's numbers when we have them.
    context = ""
    if facts:
        context = ("\n\nHere are the numbers from the plume graph currently being "
                   "discussed (JSON) — tie your answer to these where relevant:\n"
                   f"{json.dumps(facts, indent=2, default=str)}")

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": question + context},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else "(The model returned an empty response — try again.)"
    except Exception as exc:
        return (f"Couldn't reach OpenAI ({type(exc).__name__}: {exc}).\n"
                "Most likely no internet on this machine, or the API key is invalid/"
                "out of quota. Try again from a network-connected terminal.")


def explain_field(
    ppm_grid,
    xs,
    ys,
    params: dict,
    noise_floor: float,
    use_ai: bool = True,
) -> tuple[str, str]:
    """
    One-call explanation: summarise the field, then write the paragraph.

    Tries OpenAI first (when use_ai and a key is present); otherwise — or on any
    failure — uses the always-available template.

    Returns
    -------
    (text, source) where source is "openai" or "template".
    """
    facts = summarize_field(ppm_grid, xs, ys, params, noise_floor)
    if use_ai:
        ai_text = ai_explanation(facts)
        if ai_text:
            return ai_text, "openai"
    return template_explanation(facts), "template"


# ═════════════════════════════════════════════════════════════════════════════
# FIELD-TEST INTERPRETATION  — the same template/AI split, for REAL readings
# ═════════════════════════════════════════════════════════════════════════════
# A field test is a measured ppm-vs-time record (see fieldtest.py), not a predicted
# spatial field. These three functions mirror summarize_field / template_explanation
# / explain_field exactly, so the dashboard's "interpret this test" button and the
# chatbot behave identically to the model side — including the guaranteed template
# fallback when no OpenAI key is available.

def summarize_fieldtest(result: dict, meta: dict | None = None) -> dict:
    """
    Boil a processed field-test result (from fieldtest.process_fieldtest) plus its
    metadata down to the handful of numbers worth explaining / charting / asking about.

    Returns a flat dict of plain floats / bools / None — JSON-safe and shaped like
    summarize_field's output so ask_once() can ground answers in it unchanged.
    """
    meta = dict(meta or {})
    time = np.asarray(result["time"], dtype=float)
    raw = np.asarray(result["raw"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    excess = np.asarray(result["excess"], dtype=float)
    det = result["detection"]
    n = int(len(raw))
    noise = float(result["noise_ppm"])
    duration = float(time[-1] - time[0]) if n > 1 else 0.0

    event_start_s = event_end_s = event_duration_s = None
    if det.detected and det.start_idx >= 0:
        s = int(det.start_idx)
        e = int(min(det.end_idx, n) - 1)
        event_start_s = float(time[s])
        event_end_s = float(time[e])
        event_duration_s = float(event_end_s - event_start_s)

    return {
        "kind": "fieldtest",
        "name": meta.get("name"),
        "site": meta.get("site"),
        "test_date": meta.get("test_date"),
        "notes": meta.get("notes"),
        "n_samples": n,
        "duration_s": duration,
        "background": float(CH4_BACKGROUND),
        "noise_ppm": noise,
        "threshold": float(result["threshold"]),
        "detect_k": float(DETECT_K),
        "raw_peak_ppm": float(np.max(raw)) if n else None,
        "baseline_mean_ppm": float(np.mean(baseline)) if n else None,
        "excess_peak_ppm": float(np.max(excess)) if n else None,
        "detected": bool(det.detected),
        "confidence_sigmas": float(det.confidence),
        "event_start_s": event_start_s,
        "event_end_s": event_end_s,
        "event_duration_s": event_duration_s,
        "weather_corrected": bool(result.get("weather_corrected", False)),
        "sensor_distance_m": meta.get("sensor_distance_m"),
        "wind_speed": meta.get("wind_speed"),
        "wind_dir_deg": meta.get("wind_dir_deg"),
        "stability_class": meta.get("stability_class"),
    }


def template_fieldtest_explanation(facts: dict) -> str:
    """Plain-English read of a field test, straight from the numbers. Always available."""
    name = facts.get("name") or "this field test"
    site = facts.get("site")
    where = f" at {site}" if site else ""
    n = facts.get("n_samples") or 0
    noise = facts.get("noise_ppm")
    thr = facts.get("threshold")

    lines = [f"Field test “{name}”{where} ran "
             f"{_fmt_duration(facts.get('duration_s'))} ({n} samples)."]

    if facts.get("detected"):
        lines.append(
            f"A methane event WAS detected: the cleaned signal peaked at "
            f"{_fmt(facts.get('excess_peak_ppm'), ' ppm')} above the local baseline — about "
            f"{_fmt(facts.get('confidence_sigmas'), '×', nd=1)} the {_fmt(noise, ' ppm')} "
            f"noise floor (threshold {_fmt(thr, ' ppm')}) — and stayed above it from "
            f"{_fmt(facts.get('event_start_s'), ' s', nd=0)} to "
            f"{_fmt(facts.get('event_end_s'), ' s', nd=0)} "
            f"({_fmt_duration(facts.get('event_duration_s'))})."
        )
    else:
        lines.append(
            f"No sustained methane event cleared the {_fmt(thr, ' ppm')} detection threshold "
            f"({_fmt(facts.get('detect_k'), '×', nd=0)} the {_fmt(noise, ' ppm')} noise "
            f"floor); the cleaned signal peaked at only "
            f"{_fmt(facts.get('excess_peak_ppm'), ' ppm')} above baseline. The plume may not "
            f"have reached the sensor, or the leak was too weak/far, or it was lost in noise."
        )

    weather = ("Temperature/humidity drift was modelled and removed before detection. "
               if facts.get("weather_corrected")
               else "No temperature/humidity columns were supplied, so weather drift was "
                    "left uncorrected. ")
    lines.append(
        f"The raw reading sat on a {_fmt(facts.get('baseline_mean_ppm'), ' ppm')} baseline "
        f"and peaked at {_fmt(facts.get('raw_peak_ppm'), ' ppm')}. {weather}"
    )
    lines.append(
        "Reminder: the noise floor is an assumed placeholder until the real Figaro is "
        "characterised, so this detection margin will shift once it is measured."
    )
    return " ".join(lines)


def ai_fieldtest_explanation(facts: dict, model: str = OPENAI_MODEL) -> str | None:
    """
    Ask OpenAI to interpret a real field test from its summary numbers.

    Grounded in the full-project _CHAT_SYSTEM_PROMPT so the read reflects this
    project's physics, sensor, and caveats. Returns text on success, or None on ANY
    failure (recording the reason in LAST_AI_ERROR) so interpret_fieldtest can fall
    back to the template cleanly.
    """
    global LAST_AI_ERROR
    LAST_AI_ERROR = None
    if not os.environ.get("OPENAI_API_KEY"):
        LAST_AI_ERROR = "no OPENAI_API_KEY set"
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        user_prompt = (
            "Interpret this REAL methane field-test record for a sharp high-school "
            "researcher in 3-5 short, plain-English sentences. Cover, in order: (1) "
            "whether a sustained methane event was detected and how confident it is in "
            "sigmas, (2) when it happened and how long it lasted, (3) how the excess "
            "compares to the sensor's noise floor and threshold, and (4) one honest "
            "caveat (the assumed noise floor, the weather correction, or the single-"
            "sensor/steady-state limits). Use ONLY these numbers (JSON); if a value is "
            "null, say it is not available rather than guessing:\n\n"
            f"{json.dumps(facts, indent=2, default=str)}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=360,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        LAST_AI_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def interpret_fieldtest(facts: dict, use_ai: bool = True) -> tuple[str, str]:
    """
    One-call interpretation of a field test: AI when a key works, else the template.

    Returns (text, source) where source is "openai" or "template" — mirrors
    explain_field so the API/JS can treat model captions and field-test reads alike.
    """
    if use_ai:
        ai_text = ai_fieldtest_explanation(facts)
        if ai_text:
            return ai_text, "openai"
    return template_fieldtest_explanation(facts), "template"

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

The key is read from the OPENAI_API_KEY environment variable. It is never stored
in the code. To enable the AI version:  export OPENAI_API_KEY=sk-...
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os

import numpy as np

from plume import CH4_BACKGROUND
# Detection strictness shared with feasibility.py (single source of truth) so the
# caption's "detectable out to X m" matches the plot's detection-limit line.
from sensor_sim import DETECT_K

# OpenAI model used for the optional richer explanation. One place to change it.
OPENAI_MODEL = "gpt-4o-mini"


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
        "scatter more, and sub-100 m values are order-of-magnitude only."
    )
    return " ".join(lines)


def ai_explanation(facts: dict, model: str = OPENAI_MODEL) -> str | None:
    """
    Ask OpenAI to write a richer paragraph from the summary numbers.

    Returns the text on success, or None on ANY problem (missing key, missing
    `openai` package, network/API error) so the caller can fall back cleanly.
    Only the small set of summary numbers is sent — never raw grids.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "You are helping a high-school researcher understand a methane plume "
            "dispersion plot. Write 3-4 short, plain-English sentences explaining "
            "what these numbers mean for detecting the leak. Avoid jargon; if you "
            "use a technical term, explain it. Here are the facts (JSON):\n\n"
            f"{facts}\n\n"
            "Mention the peak, how it fades with distance, how far it stays "
            "detectable above the sensor noise, and one honest caveat."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        # Any failure → let the caller use the template instead.
        return None


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

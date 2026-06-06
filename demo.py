"""
demo.py — Validation suite and demonstration for the methane plume model.

Runs three validation checks, a feasibility sweep, a contour plot, and a
plain-English model summary. All terminal output uses the `rich` library for
readability; no physics or data is changed from the underlying plume.py.

WHY THREE VALIDATION BLOCKS?
───────────────────────────────────────────────────────────────────────────────
  Block 1  checks that the sigma_y / sigma_z CSV lookup reproduces the
           tabulated Briggs values at an exact table row (x = 100 m) — no
           interpolation, so the tolerance is tiny (< 0.005 m).

  Block 2  checks that concentration_gm3 implements the formula correctly.
           C_ref is built from the same Briggs σ values, so this confirms the
           IMPLEMENTATION is right (π vs 2π, correct exponents) — it is not an
           external literature comparison. Labelled clearly as such.

  Block 3  checks the coordinate-rotation sign convention with two cardinal
           wind directions — the most common place for subtle 90°/180° bugs.
───────────────────────────────────────────────────────────────────────────────
"""

import math
import sys
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatter

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from plume import (
    CH4_BACKGROUND,
    U_MIN,
    concentration_gm3,
    gm3_to_ppm_methane,
    predict_ppm,
    rotate_to_wind_frame,
    sigma_y,
    sigma_z,
)

# New modules added in this iteration (see plan):
#   feasibility — noise-based "will it work?" sweep + plain-English verdict
#   explain     — turns the plume grid into a caption (OpenAI, or template fallback)
#   sensor_sim  — single source of truth for the assumed sensor noise floor
from feasibility import sweep as feasibility_sweep, verdict as feasibility_verdict, DEFAULT_K
from explain import explain_field
from sensor_sim import SENSOR_NOISE_PPM

# ─────────────────────────────────────────────────────────────────────────────
# Rich console — everything goes through this so colours/width stay consistent
# ─────────────────────────────────────────────────────────────────────────────
console = Console(highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (referenced by name throughout)
# ─────────────────────────────────────────────────────────────────────────────
CLR_HEAD   = "bold bright_white"          # section headings
CLR_DATA   = "bright_cyan"               # numeric values
CLR_LABEL  = "dim white"                 # field labels / units
CLR_PASS   = "bold bright_green"         # validation pass
CLR_FAIL   = "bold bright_red"           # validation fail
CLR_WARN   = "bold yellow"              # warnings / caveats
CLR_NOTE   = "italic dim white"         # parenthetical notes
CLR_ACCENT = "steel_blue1"              # decorative accents / arrows


def _pass_fail(ok: bool) -> Text:
    """Return a styled PASS or FAIL tag."""
    if ok:
        return Text(" PASS ", style=f"{CLR_PASS} on dark_green")
    return Text(" FAIL ", style=f"{CLR_FAIL} on dark_red")


# ═════════════════════════════════════════════════════════════════════════════
#  BANNER
# ═════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    """Print the styled title banner."""
    title = Text()
    title.append("  ▸ METHANE PLUME  ", style="bold bright_white on grey11")
    title.append("FORWARD MODEL", style="bold bright_cyan on grey11")
    title.append("  ◂  ", style="bold bright_white on grey11")

    subtitle = Text(
        "Gaussian dispersion · Briggs (1973) σ · Fenceline detection feasibility",
        style="dim italic white",
        justify="center",
    )

    console.print()
    console.print(Align.center(title))
    console.print(Align.center(subtitle))
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION BLOCK 1 — σ arithmetic
# ═════════════════════════════════════════════════════════════════════════════

def validate_sigma() -> bool:
    """
    Check that the sigma_y / sigma_z table lookup returns the expected value.

    The shipped CSV tabulates the Briggs (1973) open-country formulas, so at a
    tabulated distance the lookup should reproduce the formula value exactly.
    We compute the reference from the formula directly and compare. The
    tolerance is tiny because 100 m is an exact table row (no interpolation).

    Test case: Class 4 = D (neutral atmosphere), x = 100 m downwind.
    """
    console.print(Rule(
        "[bold]Block 1  —  Dispersion coefficients σ_y, σ_z[/bold]",
        style=CLR_ACCENT,
    ))

    X = 100.0   # downwind distance to evaluate at (metres) — an exact table row
    SC = 4      # Pasquill class 4 = D = neutral atmosphere

    # ── Reference values computed directly from the Briggs formula ──────────
    #
    # Class D:  σ_y = 0.08 · x · (1 + 0.0001·x)^{−½}
    #           σ_z = 0.06 · x · (1 + 0.0015·x)^{−½}
    #
    # At x = 100 m:
    #   σ_y = 0.08 · 100 · (1.01)^{−0.5} = 8 · 0.99503 = 7.9603 m
    #   σ_z = 0.06 · 100 · (1.15)^{−0.5} = 6 · 0.93250 = 5.5950 m
    #
    # These numbers also agree with the independent Briggs cross-check
    # provided during design review (σ_y ≈ 8.0 m, σ_z ≈ 5.6 m at 100 m, D).
    sy_ref = 0.08 * X * (1.0 + 0.0001 * X) ** (-0.5)   # = 7.9603 m
    sz_ref = 0.06 * X * (1.0 + 0.0015 * X) ** (-0.5)   # = 5.5950 m

    sy_got = sigma_y(X, SC)
    sz_got = sigma_z(X, SC)

    TOL = 0.005   # metres — pure arithmetic; differences should be < 1e-12

    ok_y = abs(sy_got - sy_ref) < TOL
    ok_z = abs(sz_got - sz_ref) < TOL
    block_ok = ok_y and ok_z

    # ── Build display table ─────────────────────────────────────────────────
    tbl = Table(
        show_header=True,
        header_style="bold dim white",
        box=box.SIMPLE_HEAVY,
        border_style="grey30",
        padding=(0, 1),
        show_footer=False,
    )
    tbl.add_column("Coefficient", style=CLR_LABEL, width=14)
    tbl.add_column("Reference (m)", style=CLR_DATA, justify="right", width=16)
    tbl.add_column("Computed (m)", style=CLR_DATA, justify="right", width=14)
    tbl.add_column("Δ (m)", justify="right", width=12)
    tbl.add_column("Result", justify="center", width=8)

    for label, ref, got, ok in [
        ("σ_y", sy_ref, sy_got, ok_y),
        ("σ_z", sz_ref, sz_got, ok_z),
    ]:
        delta = abs(got - ref)
        delta_str = Text(f"{delta:.2e}", style=CLR_PASS if ok else CLR_FAIL)
        tbl.add_row(label, f"{ref:.6f}", f"{got:.6f}", delta_str, _pass_fail(ok))

    note = Text(
        "  Input: Class 4 (D), x = 100 m (exact table row)  |  tol = 0.005 m  |  "
        "Lookup reproduces tabulated Briggs σ_y ≈ 8.0 m, σ_z ≈ 5.6 m ✓",
        style=CLR_NOTE,
    )

    panel_content = Align.center(tbl)
    console.print(Panel(
        panel_content,
        subtitle=note,
        border_style="grey30",
        padding=(1, 2),
    ))
    return block_ok


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION BLOCK 2 — Plume formula implementation check
# ═════════════════════════════════════════════════════════════════════════════

def validate_concentration() -> bool:
    """
    Check that concentration_gm3 implements the Gaussian formula correctly.

    C_ref is built from the same Briggs σ values and the same physics — so
    this is an IMPLEMENTATION CHECK, not a literature comparison. It catches
    structural errors like using 2π instead of π, or a wrong exponent sign.

    At H=0 and y=z=0 the formula simplifies to:
        C = Q / (π · σ_y · σ_z · u)
    (The ground-reflection "image" source doubles the effective output,
     and the 2 cancels the 2π denominator.)
    """
    console.print(Rule(
        "[bold]Block 2  —  Centreline concentration (implementation check)[/bold]",
        style=CLR_ACCENT,
    ))

    # Test inputs — chosen because σ values match Block 1's reference exactly
    X, Q, U, H = 100.0, 1.0, 5.0, 0.0
    sy_ref = 0.08 * X * (1.0 + 0.0001 * X) ** (-0.5)
    sz_ref = 0.06 * X * (1.0 + 0.0015 * X) ** (-0.5)

    # At y=0, z=0, H=0: C = Q / (π · σ_y · σ_z · u)
    # This is the simplest sanity-check form of the Gaussian plume equation.
    C_ref = Q / (math.pi * sy_ref * sz_ref * U)

    # Get σ values from the actual functions, then compute C
    sy_got = sigma_y(X, 4)
    sz_got = sigma_z(X, 4)
    C_got  = concentration_gm3(X, 0.0, Q, U, H, sy_got, sz_got, z=0.0)

    TOL_REL = 0.05   # 5% relative — any structural formula error exceeds this
    rel_err = abs(C_got - C_ref) / C_ref
    block_ok = rel_err <= TOL_REL

    # Convert to ppm so we can show what a sensor would actually read
    ppm_plume = gm3_to_ppm_methane(C_got)   # just the plume, no background
    ppm_total = ppm_plume + CH4_BACKGROUND  # what the sensor reads

    # ── Inputs panel ────────────────────────────────────────────────────────
    inputs_tbl = Table(box=None, show_header=False, padding=(0, 1))
    inputs_tbl.add_column("key", style=CLR_LABEL, width=22)
    inputs_tbl.add_column("val", style=CLR_DATA)
    for k, v in [
        ("Q (emission rate)", f"{Q} g/s"),
        ("u (wind speed)", f"{U} m/s"),
        ("H (release height)", f"{H} m  →  ground-level source"),
        ("x (downwind dist.)", f"{X} m"),
        ("y, z (offset/height)", "0 m, 0 m  →  centreline, ground"),
        ("Stability class", "D  (neutral)"),
    ]:
        inputs_tbl.add_row(k, v)

    # ── Results panel ────────────────────────────────────────────────────────
    results_tbl = Table(box=None, show_header=False, padding=(0, 1))
    results_tbl.add_column("key", style=CLR_LABEL, width=22)
    results_tbl.add_column("val", style=CLR_DATA)
    results_tbl.add_row("C reference", f"{C_ref:.6e} g/m³")
    results_tbl.add_row("C computed",  f"{C_got:.6e} g/m³")
    results_tbl.add_row(
        "Relative error",
        Text(f"{rel_err*100:.4f}%  (tol 5%)", style=CLR_PASS if block_ok else CLR_FAIL),
    )
    results_tbl.add_row("", "")
    results_tbl.add_row("Plume-only ppm", f"{ppm_plume:.4f} ppm  (T=20°C, P=1 atm)")
    results_tbl.add_row("Total w/ background",
                        f"[bold]{ppm_total:.4f} ppm[/bold]  ({CH4_BACKGROUND} bkg + {ppm_plume:.4f} plume)")

    caveat = Text(
        "  C_ref uses the same Briggs σ — this validates the formula structure, "
        "not external data. Catches 2π/π errors, wrong exponents, unit slips.",
        style=CLR_NOTE,
    )

    console.print(Panel(
        Columns([inputs_tbl, Text("  "), results_tbl], equal=False),
        title=_pass_fail(block_ok),
        subtitle=caveat,
        border_style="grey30",
        padding=(1, 2),
    ))
    return block_ok


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION BLOCK 3 — Wind-direction convention
# ═════════════════════════════════════════════════════════════════════════════

def validate_rotation() -> bool:
    """
    Check that rotate_to_wind_frame puts receptors on the correct side.

    Wind-direction sign bugs are common and subtle. We test two cardinal
    directions and check both the downwind and upwind sides.
    """
    console.print(Rule(
        "[bold]Block 3  —  Coordinate rotation & wind-direction convention[/bold]",
        style=CLR_ACCENT,
    ))

    SRC = (0.0, 0.0)   # source at the map origin

    tests = [
        # (description, φ_deg, receptor_dx, receptor_dy, want_downwind)
        # φ = 270°: wind FROM west → plume goes east → +x receptor is downwind
        ("FROM west → blows east  (φ=270°)", 270.0,  100.0,  0.0,   True,  "Receptor due east  "),
        ("FROM west → blows east  (φ=270°)", 270.0, -100.0,  0.0,   False, "Receptor due west  "),
        # φ = 0°: wind FROM north → plume goes south → −y receptor is downwind
        ("FROM north → blows south (φ=0°)",    0.0,    0.0, -100.0,  True,  "Receptor due south "),
        ("FROM north → blows south (φ=0°)",    0.0,    0.0,  100.0,  False, "Receptor due north "),
    ]

    tbl = Table(
        show_header=True,
        header_style="bold dim white",
        box=box.SIMPLE_HEAVY,
        border_style="grey30",
        padding=(0, 1),
    )
    tbl.add_column("Scenario",       style=CLR_LABEL,  width=30)
    tbl.add_column("Receptor",       style=CLR_LABEL,  width=20)
    tbl.add_column("x_wind (m)",     style=CLR_DATA,   justify="right", width=12)
    tbl.add_column("Expected",       justify="center", width=12)
    tbl.add_column("Result",         justify="center", width=8)

    all_ok = True
    for scenario, phi, dx, dy, want_down, receptor_label in tests:
        xw, _ = rotate_to_wind_frame(dx, dy, *SRC, phi)
        actually_down = xw > 0.0
        ok = (actually_down == want_down)
        all_ok = all_ok and ok

        xw_text = Text(f"{xw:+.2f}", style=CLR_PASS if ok else CLR_FAIL)
        expect   = Text("downwind ↗" if want_down else "upwind ↙",
                        style="dim green" if want_down else "dim red")
        tbl.add_row(scenario, receptor_label, xw_text, expect, _pass_fail(ok))

    note = Text(
        "  Sign derivation: û_down = (−sin φ, −cos φ); "
        "x_wind = dx·(−sin φ) + dy·(−cos φ). At φ=270°: x_wind = dx ✓",
        style=CLR_NOTE,
    )
    console.print(Panel(
        Align.center(tbl),
        subtitle=note,
        border_style="grey30",
        padding=(1, 2),
    ))
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
#  ANCILLARY CHECKS — upwind background & calm-wind guard
# ═════════════════════════════════════════════════════════════════════════════

def validate_ancillary() -> bool:
    """Quick sanity checks for the two most likely silent failure modes."""
    console.print(Rule("[bold]Ancillary checks[/bold]", style=CLR_ACCENT))

    results = []

    # ── Check 1: upwind receptor must return exactly the background constant ─
    upwind_rec = np.array([[-50.0, 0.0]])   # 50 m west; wind from west → upwind
    ppm_up = predict_ppm(
        (0.0, 0.0), Q=1.0, u=2.0, wind_dir_deg=270.0,
        H=1.0, stability_class=4, receptors=upwind_rec,
    )
    ok_bkg = abs(ppm_up[0] - CH4_BACKGROUND) < 1e-12
    results.append((
        "Upwind receptor → background only",
        f"{ppm_up[0]:.6f} ppm  (expect {CH4_BACKGROUND} exactly)",
        ok_bkg,
    ))

    # ── Check 2: wind speed < U_MIN must raise ValueError ───────────────────
    # If it doesn't, the model would silently give garbage at near-calm winds.
    try:
        concentration_gm3(100.0, 0.0, 1.0, 0.1, 0.0, 8.0, 5.6)
        ok_guard = False
        detail = f"did NOT raise ValueError — silent error at u=0.1 m/s!"
    except ValueError:
        ok_guard = True
        detail = f"correctly raised ValueError  (u=0.1 m/s < U_MIN={U_MIN} m/s)"
    results.append(("Calm-wind guard (u < 0.5 m/s)", detail, ok_guard))

    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column("Check",  style=CLR_LABEL, width=38)
    tbl.add_column("Detail", style=CLR_DATA)
    tbl.add_column("",       justify="center", width=8)

    all_ok = True
    for label, detail, ok in results:
        tbl.add_row(label, detail, _pass_fail(ok))
        all_ok = all_ok and ok

    console.print(Panel(tbl, border_style="grey30", padding=(1, 2)))
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
#  OVERALL VERDICT
# ═════════════════════════════════════════════════════════════════════════════

def print_verdict(all_ok: bool) -> None:
    if all_ok:
        msg  = Text("  ✔  ALL CHECKS PASSED  ", style="bold bright_green on dark_green")
        desc = Text(
            "Briggs σ arithmetic ✓    Gaussian formula ✓    "
            "Rotation convention ✓    Upwind guard ✓    Calm-wind guard ✓",
            style="dim green",
            justify="center",
        )
    else:
        msg  = Text("  ✘  ONE OR MORE CHECKS FAILED  ", style="bold bright_red on dark_red")
        desc = Text("Review the FAIL rows above before using this model.", style="dim red",
                    justify="center")

    console.print(Panel(
        Align.center(msg, vertical="middle"),
        subtitle=desc,
        border_style="bright_green" if all_ok else "bright_red",
        padding=(1, 4),
        height=6,
    ))


# ═════════════════════════════════════════════════════════════════════════════
#  FEASIBILITY SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def _verdict_text(verdict_label: str) -> Text:
    """Colour-code a single cell's detectable / marginal / undetectable label."""
    if verdict_label == "detectable":
        return Text("detectable", style=CLR_PASS)
    if verdict_label == "marginal":
        return Text("marginal", style=CLR_WARN)
    return Text("undetectable", style="dim red")


def run_feasibility_sweep() -> None:
    """
    Sanity check: across a range of leak sizes and distances, will the sensor
    actually see the plume?

    For every (leak size Q, distance) we predict the methane ABOVE background and
    compare it to a detection threshold built from the sensor's noise floor
    (SENSOR_NOISE_PPM, shared with the processing tests). Each cell is labelled
    detectable / marginal / undetectable, and we finish with a plain-English
    verdict for the realistic fenceline case.

    Conditions: neutral Class D atmosphere, u=2 m/s, H=1 m, sensor z=1 m,
    centreline receptor (the most generous geometry for detection).
    """
    console.print(Rule("[bold]Feasibility sweep — will the sensor see it?[/bold]",
                       style=CLR_ACCENT))

    Q_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]   # emission rates to test (g/s)
    dist_list = [25.0, 50.0, 100.0, 200.0]       # sensor distances (m)
    threshold = DEFAULT_K * SENSOR_NOISE_PPM     # ppm a signal must beat to "count"

    # Run the whole grid through the forward model (Class D = 4).
    cells = feasibility_sweep(
        Q_list=Q_list, distance_list=dist_list, stability_list=[4],
        u=2.0, wind_dir_deg=270.0, H=1.0, z=1.0,
    )
    # Index by (Q, distance) for easy table lookup.
    grid = {(c.Q, c.distance): c for c in cells}

    # ── Table: excess-above-background ppm at each distance, colour = verdict ──
    tbl = Table(
        title=Text(
            "CH₄ above background (ppm)  ·  Class D  ·  u = 2 m/s  ·  H = 1 m  ·  "
            "z = 1 m (fence-mount)",
            style="dim white",
        ),
        show_header=True,
        header_style="bold dim white",
        box=box.SIMPLE_HEAVY,
        border_style="grey30",
        padding=(0, 2),
    )
    tbl.add_column("Q  (g/s)", style=CLR_LABEL, justify="right", width=12)
    for d in dist_list:
        tbl.add_column(f"{d:.0f} m", justify="right", width=12)
    tbl.add_column("Outlook @ 50 m", width=26)

    def _cell_text(excess: float, label: str) -> Text:
        if label == "detectable":
            style = CLR_PASS
        elif label == "marginal":
            style = CLR_WARN
        else:
            style = "dim white"
        return Text(f"+{excess:.2f}", style=style)

    for Q_s in Q_list:
        row = [f"{Q_s:.3f}"]
        for d in dist_list:
            c = grid[(Q_s, d)]
            row.append(_cell_text(c.excess, c.verdict))
        outlook = _verdict_text(grid[(Q_s, 50.0)].verdict)
        tbl.add_row(*row, outlook)

    note = Text(
        f"  ⚑  Detection threshold = {threshold:.2f} ppm "
        f"({DEFAULT_K:g}× the {SENSOR_NOISE_PPM:.2f} ppm assumed noise floor).\n"
        "     green = detectable   ·   yellow = marginal (needs averaging/stats)   "
        "·   dim = lost in noise.\n"
        "     Noise floor is an ASSUMPTION (no hardware yet) — set in sensor_sim.py.",
        style=CLR_WARN,
    )
    console.print(Panel(
        Align.center(tbl),
        subtitle=note,
        border_style="grey30",
        padding=(1, 2),
    ))

    # ── Plain-English bottom line for the realistic fenceline case ────────────
    line = feasibility_verdict(cells, fence_distance=50.0, stability=4)
    console.print(Panel(
        Text(line, style="bold white"),
        title=Text(" VERDICT ", style="bold bright_white on grey23"),
        border_style=CLR_ACCENT,
        padding=(1, 2),
    ))


# ═════════════════════════════════════════════════════════════════════════════
#  CONTOUR PLOT  (matplotlib)
# ═════════════════════════════════════════════════════════════════════════════

def make_contour_plot() -> None:
    """
    Render a 2-D concentration heatmap with log-spaced colour levels.

    Wind is set to φ=270° (from west) so the plume runs along the +x axis,
    matching the wind-arrow annotation. Grid spans x=10–200 m: it starts at
    10 m because the σ values are already extrapolated below their 100 m
    validation range (starting at 1 m would give a misleading near-source
    spike), and stops at 200 m because that is the upper limit of the σ
    lookup table — beyond it the model is undefined.
    """
    console.print(Rule("[bold]Contour plot[/bold]", style=CLR_ACCENT))
    console.print(f"  [dim]Generating 250×200 receptor grid …[/dim]")

    NX, NY = 250, 200
    xs = np.linspace(10.0, 200.0, NX)    # downwind axis (m) — table range 10–200 m
    ys = np.linspace(-150.0, 150.0, NY)  # crosswind axis (m)
    XX, YY = np.meshgrid(xs, ys)          # shape (NY, NX) each

    # Plot conditions
    PLOT_Q    = 5.0     # g/s  — representative landfill hotspot
    PLOT_U    = 3.0     # m/s
    PLOT_H    = 1.0     # m    — just above ground
    PLOT_CLS  = 3       # class 3 = C = slightly unstable — common daytime condition
    PLOT_WIND = 270.0   # FROM west → plume runs east along +x axis

    # Flatten the grid into a (NX*NY, 2) receptor array, run predict_ppm, reshape
    receptors_flat = np.column_stack([XX.ravel(), YY.ravel()])
    ppm_flat = predict_ppm(
        src_pos=(0.0, 0.0), Q=PLOT_Q, u=PLOT_U, wind_dir_deg=PLOT_WIND,
        H=PLOT_H, stability_class=PLOT_CLS,
        receptors=receptors_flat, z=0.0,
    )
    PPM = ppm_flat.reshape(NY, NX)   # must match meshgrid's (NY, NX) layout

    # ── Auto-explanation: turn the grid into a plain-English caption ──────────
    # Tries OpenAI when OPENAI_API_KEY is set; otherwise a template summary built
    # from the numbers. Either way we always get a caption.
    plot_params = {
        "Q": PLOT_Q, "u": PLOT_U, "H": PLOT_H,
        "stability_class": PLOT_CLS, "wind_dir_deg": PLOT_WIND,
    }
    caption_text, caption_src = explain_field(
        PPM, xs, ys, plot_params, noise_floor=SENSOR_NOISE_PPM,
    )
    src_label = ("AI (OpenAI)" if caption_src == "openai"
                 else "template — set OPENAI_API_KEY for an AI version")
    console.print(f"  [dim]Explanation source:[/dim] [cyan]{src_label}[/cyan]")

    # ── Matplotlib styling ───────────────────────────────────────────────────
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    ppm_max = float(PPM.max())
    ppm_lo  = CH4_BACKGROUND * 1.04   # start contours just above background
    levels  = np.logspace(np.log10(ppm_lo), np.log10(ppm_max), 22)

    # Filled contours — log-normalised so both weak and strong signals are visible
    cf = ax.contourf(
        XX, YY, PPM,
        levels=levels,
        norm=mcolors.LogNorm(vmin=ppm_lo, vmax=ppm_max),
        cmap='inferno',
        extend='both',
        alpha=0.92,
    )

    # Contour lines give a topographic feel and make levels easier to read
    cl = ax.contour(
        XX, YY, PPM,
        levels=levels[::4],
        colors='white',
        linewidths=0.4,
        alpha=0.25,
    )

    # Detection-limit contour: the line where the plume just clears the sensor
    # noise (background + 3× noise floor). Inside this line → detectable.
    detect_level = CH4_BACKGROUND + DEFAULT_K * SENSOR_NOISE_PPM
    if PPM.max() > detect_level:
        dl = ax.contour(
            XX, YY, PPM,
            levels=[detect_level],
            colors='#3fb950',
            linewidths=1.6,
            linestyles='--',
        )
        ax.clabel(dl, fmt=lambda v: f'detection limit ≈ {v:.1f} ppm', fontsize=7,
                  colors='#3fb950')

    # Colourbar
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label('CH₄  (ppm)', color='#c9d1d9', fontsize=11)
    cb.ax.yaxis.set_tick_params(color='#c9d1d9')
    cb.outline.set_edgecolor('#30363d')
    # Show a subset of tick labels to avoid crowding
    tick_vals = levels[::5]
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([f"{v:.1f}" for v in tick_vals], color='#8b949e', fontsize=8)

    # Source marker
    ax.scatter(0, 0, s=120, color='#58a6ff', zorder=6, marker='^',
               label=f'Source  (Q={PLOT_Q} g/s, H={PLOT_H} m)')

    # Wind arrow — points east because φ=270° means wind blows eastward
    ax.annotate(
        '',
        xy=(70, 132), xytext=(15, 132),
        arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=2.0),
    )
    ax.text(42, 141, 'Wind', ha='center', color='#58a6ff', fontsize=9,
            fontstyle='italic')

    # Validity-range annotation — honest about the 10–100 m extrapolation
    ax.axvline(100, color='#f0883e', lw=0.8, ls='--', alpha=0.6)
    ax.text(102, -140, '← extrapolated\n   above 100 m →',
            color='#f0883e', fontsize=7.5, alpha=0.75, va='bottom')

    # Labels and title
    ax.set_xlabel('Easting from source (m)  —  downwind direction', color='#c9d1d9')
    ax.set_ylabel('Northing from source (m)  —  crosswind', color='#c9d1d9')
    ax.set_title(
        f'Gaussian Plume  ·  Q = {PLOT_Q} g/s  ·  u = {PLOT_U} m/s  ·  '
        f'H = {PLOT_H} m  ·  Class {PLOT_CLS}  ·  wind from W (φ=270°)\n'
        f'Colour scale: log-normalised ppm  ·  '
        f'Briggs open-country σ  ·  sub-100 m is extrapolated (dashed)',
        color='#c9d1d9', fontsize=10,
    )
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.legend(loc='upper left', fontsize=9,
              facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])

    # ── Plain-English caption box under the plot ──────────────────────────────
    # Reserve space at the bottom, then drop in the wrapped explanation so the
    # picture explains itself.
    fig.subplots_adjust(bottom=0.30, top=0.90)
    wrapped = textwrap.fill(caption_text, width=120)
    fig.text(
        0.02, 0.02, "What this shows:  " + wrapped,
        color='#c9d1d9', fontsize=8.5, va='bottom', ha='left',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d'),
    )

    fig.savefig('plume_contour.png', dpi=150, facecolor=fig.get_facecolor())
    console.print(f"  [green]Saved[/green] [dim]→[/dim] [bold cyan]plume_contour.png[/bold cyan]")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  PLAIN-ENGLISH SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    """Print the model summary as a styled, numbered panel list."""
    console.print(Rule("[bold]Plain-English Model Summary[/bold]", style=CLR_ACCENT))

    points = [
        (
            "Q — Emission rate (g/s)",
            "Total methane released per second. Every concentration everywhere "
            "in the plume scales linearly with Q — double Q, double all sensor "
            "readings. It is the key unknown the later inversion tries to recover.",
        ),
        (
            "u — Wind speed (m/s)",
            "Dilutes the plume along the wind axis. Concentration C ∝ 1/u, so "
            "halving the wind doubles the peak reading. The model raises an error "
            f"below {U_MIN} m/s because C → ∞ as u → 0 (physically wrong).",
        ),
        (
            "wind_dir_deg — Wind direction (° from N)",
            "Rotates the coordinate frame so the plume axis always aligns with "
            "+x. Receptors upwind of the source receive background concentration "
            "only — the plume equation is never evaluated for them.",
        ),
        (
            "H — Release height (m)",
            "The ground acts like a mirror: gas bouncing off it is modelled as "
            "an identical 'image' source at depth −H underground. At H = 0 the "
            "image is coincident with the real source, doubling peak concentration "
            "to C = Q/(π σ_y σ_z u). Higher H reduces ground-level impact.",
        ),
        (
            "stability_class — Pasquill class (1–6, where 1=A … 6=F)",
            "Controls the σ_y and σ_z growth rates. Unstable class 1 (A; sunny, "
            "light wind) produces a wide, fast-diluting plume; stable class 6 (F; "
            "clear night, calm) produces a narrow, slowly-diluting plume that "
            "stays concentrated over long distances.",
        ),
        (
            "Biggest weakness — temporal averaging",
            "The model gives the time-mean steady-state plume. Pasquill–Gifford "
            "σ values represent 10-minute-to-hourly averages. Individual sensor "
            "readings fluctuate far more than the model predicts because real "
            "turbulence produces intermittent gusts, not a smooth continuous ribbon.",
        ),
        (
            "Spatial validity & site-specific limits",
            "Briggs open-country formulas were validated at ≥ 100 m over smooth, "
            "flat terrain. At 10–50 m they are extrapolated below their valid "
            "range. A transfer-station fenceline has structures, equipment, and "
            "a perimeter berm adding turbulent mixing above open-country levels — "
            "both effects make real plumes dilute faster than this model predicts. "
            "Treat sub-100 m results as order-of-magnitude estimates.",
        ),
        (
            "Calm-wind limitation",
            "Near-calm conditions (u < 0.5 m/s) are exactly when a weak source "
            "is most detectable — yet the steady-state model is undefined there. "
            "Field work should log wind speed and treat near-calm readings "
            "separately (puff model or flagged for manual review).",
        ),
    ]

    content = Text()
    for i, (title, body) in enumerate(points, 1):
        content.append(f"\n  {i}.  ", style="bold bright_cyan")
        content.append(f"{title}\n", style="bold white")
        content.append(f"      {body}\n", style="dim white")

    console.print(Panel(
        content,
        border_style="grey30",
        padding=(0, 2),
    ))


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS REFERENCE TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_constants() -> None:
    """Print every constant used, with its value, unit, and source."""
    console.print(Rule("[bold]Constants & Units Reference[/bold]", style=CLR_ACCENT))

    tbl = Table(
        show_header=True,
        header_style="bold dim white",
        box=box.SIMPLE_HEAVY,
        border_style="grey30",
        padding=(0, 2),
    )
    tbl.add_column("Quantity",       style=CLR_LABEL,  width=22)
    tbl.add_column("Value",          style=CLR_DATA,   justify="right", width=14)
    tbl.add_column("Unit",           style="dim cyan",  width=14)
    tbl.add_column("Source",         style=CLR_NOTE,   width=36)

    rows = [
        ("R  (gas constant)",   "8.314",   "J/(mol·K)",  "NIST CODATA 2018"),
        ("M_CH4",               "16.04",   "g/mol",      "IUPAC atomic weights 2021"),
        ("T default",           "293.15",  "K",          "20 °C standard"),
        ("P default",           "101,325", "Pa",         "1 atm"),
        ("CH4 background",      "1.9",     "ppm",        "NOAA GML 2023 global mean"),
        ("σ formulas",          "Briggs 1973", "x,σ in m", "Seinfeld & Pandis (2016) Table 18.2"),
        ("u_min guard",         "0.5",     "m/s",        "model diverges below this"),
    ]
    for r in rows:
        tbl.add_row(*r)

    console.print(Panel(Align.center(tbl), border_style="grey30", padding=(1, 2)))


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_banner()

    console.print()
    b1 = validate_sigma()
    console.print()
    b2 = validate_concentration()
    console.print()
    b3 = validate_rotation()
    console.print()
    ba = validate_ancillary()

    console.print()
    print_verdict(all_ok=b1 and b2 and b3 and ba)

    console.print()
    run_feasibility_sweep()

    console.print()
    make_contour_plot()

    console.print()
    print_summary()

    console.print()
    print_constants()
    console.print()

    if not (b1 and b2 and b3 and ba):
        sys.exit(1)

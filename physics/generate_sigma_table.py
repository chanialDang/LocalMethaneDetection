"""
generate_sigma_table.py — Build the dispersion-coefficient lookup table.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS SCRIPT EXISTS (provenance, in plain English)
───────────────────────────────────────────────────────────────────────────────
`plume.py` reads σ_y / σ_z from a CSV instead of evaluating formulas on every
call. That CSV must come from SOMEWHERE traceable — a table you cannot reproduce
is a table you cannot trust. This script IS that traceable source: it computes
the table directly from the published Briggs (1973) open-country dispersion
formulas, so anyone can regenerate and audit the numbers.

    Run:  python3 generate_sigma_table.py
    Out:  briggs_dispersion_sigma.csv   (1–200 m, 1 m steps, classes A–F)

THE FORMULAS — Briggs (1973), open-country / rural form
───────────────────────────────────────────────────────────────────────────────
x is downwind distance in metres; σ_y, σ_z are in metres. These are the
standard Briggs (1973) open-country coefficients, as compiled in references
such as Hanna, Briggs & Hosker (1982), "Handbook on Atmospheric Diffusion"
(DOE/TIC-11223). The exact expressions used here are pinned in code below and
checked against hand-computed anchors in _self_check(); a specific table/page
locator is deliberately not cited because it has not been independently verified:

  Class   σ_y                                σ_z
  A   0.22 x (1 + 1e-4 x)^-1/2          0.20 x
  B   0.16 x (1 + 1e-4 x)^-1/2          0.12 x
  C   0.11 x (1 + 1e-4 x)^-1/2          0.08 x (1 + 2e-4 x)^-1/2
  D   0.08 x (1 + 1e-4 x)^-1/2          0.06 x (1 + 1.5e-3 x)^-1/2
  E   0.06 x (1 + 1e-4 x)^-1/2          0.03 x (1 + 3e-4 x)^-1
  F   0.04 x (1 + 1e-4 x)^-1/2          0.016 x (1 + 3e-4 x)^-1

Stability classes map to integer columns 1–6 (1 = A = most unstable, 6 = F =
most stable) to match the lookup columns in plume.py and the downstream
Week-3 inversion.

VALIDITY CAVEAT (also enforced in plume.py / shown in demo.py)
───────────────────────────────────────────────────────────────────────────────
The Briggs open-country forms were fit to data at downwind distances of roughly
100 m – 10 km over flat, uniform terrain. Below 100 m, and at a built-up
transfer-station fenceline, the table values are extrapolations and should be
treated as order-of-magnitude only.
"""

import csv
from pathlib import Path

# Write the table next to this module (inside physics/) so plume.py finds it
# regardless of the working directory the generator is run from.
OUTPUT_CSV = str(Path(__file__).resolve().parent / "briggs_dispersion_sigma.csv")
DISTANCES_M = range(1, 201)          # 1 m … 200 m inclusive, 1 m steps
DECIMALS = 4                         # enough for demo.py's 0.005 m tolerance
CLASS_LETTERS = ["A", "B", "C", "D", "E", "F"]   # column order = class 1 … 6

# σ_y prefactor a in  σ_y = a · x · (1 + 1e-4 x)^-1/2  (same denominator for all)
_SIGMA_Y_A = {"A": 0.22, "B": 0.16, "C": 0.11, "D": 0.08, "E": 0.06, "F": 0.04}


def sigma_y_briggs(x: float, letter: str) -> float:
    """Briggs (1973) open-country crosswind σ_y (m) at distance x (m)."""
    return _SIGMA_Y_A[letter] * x * (1.0 + 1.0e-4 * x) ** -0.5


def sigma_z_briggs(x: float, letter: str) -> float:
    """Briggs (1973) open-country vertical σ_z (m) at distance x (m)."""
    if letter == "A":
        return 0.20 * x
    if letter == "B":
        return 0.12 * x
    if letter == "C":
        return 0.08 * x * (1.0 + 2.0e-4 * x) ** -0.5
    if letter == "D":
        return 0.06 * x * (1.0 + 1.5e-3 * x) ** -0.5
    if letter == "E":
        return 0.03 * x * (1.0 + 3.0e-4 * x) ** -1.0
    if letter == "F":
        return 0.016 * x * (1.0 + 3.0e-4 * x) ** -1.0
    raise ValueError(f"Unknown stability class {letter!r}")


def build_rows() -> list[dict]:
    """Compute every (distance, class) σ value as a list of CSV row dicts.

    Keys are inserted in CSV column order — all σ_y classes, then all σ_z
    classes — so write_csv can derive the header straight from the row keys and
    the schema lives in exactly one place. Fixed-decimal strings (e.g. "5.5950")
    keep the table aligned and stable under diff; trailing zeros carry no false
    precision.
    """
    rows = []
    for x in DISTANCES_M:
        row = {"distance_m": x}
        for i, letter in enumerate(CLASS_LETTERS, start=1):
            row[f"sigma_y_class{i}"] = f"{sigma_y_briggs(x, letter):.{DECIMALS}f}"
        for i, letter in enumerate(CLASS_LETTERS, start=1):
            row[f"sigma_z_class{i}"] = f"{sigma_z_briggs(x, letter):.{DECIMALS}f}"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str = OUTPUT_CSV) -> None:
    """Write rows to CSV. The column schema and order come from the row keys
    (set in build_rows), so plume.py's expected header is defined in one place."""
    if not rows:
        raise ValueError("build_rows() produced no rows — nothing to write.")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _self_check() -> None:
    """
    Guard against silent regressions: verify a few anchor values that are easy
    to compute by hand. Class D (4) at 100 m is the row demo.py validates.
    """
    anchors = [
        # (x, letter, sigma_y, sigma_z)  — hand-checked Briggs values
        (100.0, "D", 7.9603, 5.5950),
        (100.0, "A", 21.8908, 20.0000),
        (200.0, "F", 7.9212, 3.0189),
    ]
    for x, letter, want_y, want_z in anchors:
        got_y = round(sigma_y_briggs(x, letter), DECIMALS)
        got_z = round(sigma_z_briggs(x, letter), DECIMALS)
        assert abs(got_y - want_y) < 1e-3, f"σ_y({x},{letter})={got_y} != {want_y}"
        assert abs(got_z - want_z) < 1e-3, f"σ_z({x},{letter})={got_z} != {want_z}"


if __name__ == "__main__":
    _self_check()
    rows = build_rows()
    write_csv(rows)
    print(f"Wrote {OUTPUT_CSV}: {len(rows)} distances × {len(CLASS_LETTERS)} classes "
          f"(A–F), Briggs (1973) open-country formulas.")
    print(f"  Anchor  σ_y(100 m, D) = {rows[99]['sigma_y_class4']} m   "
          f"σ_z(100 m, D) = {rows[99]['sigma_z_class4']} m   (demo.py Block-1 row)")

"""
fit_powerlaw.py — recover the power-law slope m from the datasheet sensitivity curve.

The Figaro characteristic is a straight line on log-log axes: Rs/R0 = A_ds · ppm^(−m).
So log(Rs/R0) = log(A_ds) − m·log(ppm), and a linear fit gives the slope −m. Only m is
used by the calibration: the prefactor A is set separately from the clean-air R0 anchor
(A = background_ppm^m), NOT from this table's A_ds. See calibration/README.md.

Usage:
    python3 calibration/datasheet/fit_powerlaw.py [tgs2611_sensitivity.csv]

Prints the fitted m (and R² of the log-log fit). Reads-only; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np


def load_curve(path):
    """Read (ppm, rs_over_r0) rows, skipping '#' comment lines and the header."""
    ppm, ratio = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("ppm"):
            continue
        a, b = line.split(",")[:2]
        ppm.append(float(a))
        ratio.append(float(b))
    return np.asarray(ppm), np.asarray(ratio)


def fit_m(ppm, ratio):
    """Least-squares slope of log(Rs/R0) vs log(ppm); returns (m, r2)."""
    x, y = np.log(ppm), np.log(ratio)
    slope, intercept = np.polyfit(x, y, 1)
    m = -float(slope)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return m, r2


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = argv[0] if argv else Path(__file__).with_name("tgs2611_sensitivity.csv")
    ppm, ratio = load_curve(path)
    m, r2 = fit_m(ppm, ratio)
    print(f"datasheet points : {len(ppm)}  (ppm {ppm.min():g}–{ppm.max():g})")
    print(f"fitted slope m   : {m:.4f}   (log-log R² = {r2:.4f})")
    print("set powerlaw_m in the node cfg to this value (A is derived from the R0 anchor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

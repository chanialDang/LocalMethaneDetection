# Mitchell Eq. 16 — dormant-kernel spec

Implementation contract for `physics/sensor_frontend._kernel_mitchell`. Full extraction
(symbols, fitting method, caveats) in `../docs/mitchell2024_eq16_extraction.md`.

## The equation
```
M = C1 + C2·exp(C3·V − C4·ln(T+65) − C5·ln(H)) − C7·ln((T+65)·V)
```
- **Inputs:** `V` = V_out in volts, `T` = °C, `H` = raw %RH. **Ingests V_out directly — no
  R_s, no R_s/R₀, does NOT use R₀.** No separate multiplicative T/RH factor (its T/H terms
  are internal log terms).
- **Output:** `M` = linear ppm (total, incl. background), valid 0–200 ppm.
- **Coefficients:** six — `C1,C2,C3,C4,C5,C7`. The printed `C7` is Table-5 row `C6`
  (= −0.0587); it carries a negative sign, so the printed `− C7·ln(…)` is a positive
  contribution. Guard `H > 0` and `T + 65 > 0` (log domain) → NaN otherwise.

## Why it is DORMANT (gated off until gas)
1. Coefficients are hardware-specific and REQUIRE known-concentration calibration gas to
   fit. There is no gas-free / datasheet-only route to them.
2. The paper's own coefficients give ≈ −165 ppm on our hardware (see `paper_coeffs.py`) —
   never usable as a default. `MITCHELL_COEFFS = None` makes the kernel refuse.
3. Even perfectly fit, Eq. 16's RMSE ≈ 5.1 ppm — the same order as our 2–40 ppm signal.

## To activate (when calibration gas is available)
1. Collect paired `(V_out, T, H) → known ppm` points across T/RH with span gas.
2. Fit `C1..C7` (nonlinear least squares) to THIS sensor.
3. Put the result in a node cfg as `mitchell_coeffs=dict(C1=…, …, C7=…)` and set
   `kernel="mitchell"`. `apply_node` wires it in; the gate then passes.

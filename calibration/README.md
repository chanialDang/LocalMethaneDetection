# calibration/ — voltage → ppm, the honest pipeline

This folder is the **single home for the calibration math + measured node constants +
reference docs**, so that when real field CSVs arrive they follow one clear path to a
trustworthy ppm and an honest detection verdict.

## The pipeline
```
raw ADC voltage
   │  (divider:  Rs = R_L·(V_c − V_out)/V_out)
   ▼
[ CALIBRATION KERNEL ]  ── voltage_to_ppm ──▶  ppm  (total, incl. ~1.9 ppm background)
   │   powerlaw (gas-free, now)  |  mitchell (dormant, needs gas)
   ▼
[ HONEST NOISE FLOOR ]  ── random ⊕ drift/bias ⊕ model ──▶  LOD/LOQ + GO/CHECK/STOP
   ▼
[ INVERSION ]  ── ppm + wind ──▶  source (x, y, Q)     (downstream, separate stage)
```
Run it on real data:
- `python3 -m misc.calibrate calibration/nodes/node1.py "<baseline.csv>"` — derive a
  node's constants + the make-or-break ppm noise floor + verdict.
- `python3 -m misc.preflight "<field.csv>"` — is a field upload trustworthy (GO/CHECK/STOP).

## Reconciliation rule (why the engine isn't in here)
`calibration/` is the **source of measured truth**; `physics/sensor_frontend.py` is the
**engine that consumes it**. The engine owns the kernel math, the gate, and the module
globals that everything downstream (and every existing test) depends on. Data flows one way:

```
calibration/nodes/nodeN.py  ──sensor_frontend.apply_node(cfg)──▶  sensor_frontend globals
```

Never hardcode a node's numbers into the shared engine (that would fork it per node); put
them in `nodes/nodeN.py` and let `apply_node` land them on the globals.

## What's here
- `nodes/node1.py` — Node 1 measured constants (data only).
- `datasheet/tgs2611_sensitivity.csv` + `fit_powerlaw.py` — recover the log-log slope `m`.
  ⚠ the CSV is a **placeholder** (generated from m=0.35) until digitized from the real curve.
- `mitchell/eq16_spec.md` + `paper_coeffs.py` — the dormant Mitchell kernel spec; the paper
  coefficients are **reference/tests only, never a default**.
- `docs/` — the hardware handoff and the Mitchell Eq. 16 extraction.

## Provisional-scale caveats (read before trusting an absolute ppm)
- **Gas-free anchor.** `A = background_ppm ** m` pins the clean-air baseline to 1.9 ppm using
  the measured clean-air R₀. The datasheet's own A/m are defined against the datasheet's R₀
  (Rs/R₀ @ 5000 ppm), so re-referencing to clean-air R₀ makes the absolute ppm **provisional**
  — harmless for *locating* a leak (the plume is linear in Q) but not a certified concentration.
- **Extrapolated slope.** `m` is a 500–12,500 ppm datasheet slope stretched to 2–40 ppm.
- **The electrical floor is a lower bound.** 1.65 mV jitter → ~0.018 ppm is the short-term
  *random* floor only; the real floor is dominated by non-averageable drift and T/RH
  sensitivity (unmeasured). `misc/calibrate.py` reports the **total honest floor**, not 0.018.
- **Mitchell needs gas** and even then RMSE ≈ 5.1 ppm (~the size of the low-end signal).

## Note: stale dispersion-table reference
The handoff (`docs/HANDOFF_to_coding_agent.md` §7) cites `otm33a_dispersion_sigma.csv`. That
file does **not** exist; the repo's inversion actually uses `physics/briggs_dispersion_sigma.csv`
(Briggs 1973). Inversion-side only, unrelated to calibration — flagged here for whoever wires
the source-localization step.

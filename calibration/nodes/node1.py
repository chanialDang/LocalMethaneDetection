"""
node1.py — measured constants for Node 1 (the first physical sensor).

DATA ONLY (no logic). Handed over from the hardware/bench side on 2026-07-12 — see
``calibration/docs/`` (the handoff) and ``node1_clean_baseline.csv``. Feed this dict to
``physics.sensor_frontend.apply_node(NODE1)`` to configure the runtime front-end; the
loader derives ``POWERLAW_A`` from the anchor (never stored here).

Provenance (all measured in clean indoor air after burn-in, ~1.9 ppm ambient):
  • V_c = 5.0 V, R_L = 10 kΩ         — the divider circuit (read off the board).
  • R₀ ≈ 79,960 Ω (median 79,938)    — clean-air baseline resistance.
  • baseline V = 0.5558 V, jitter ≈ 1.65 mV (detrended std) — the electrical noise spec.
  • T ≈ 24.8 °C, RH ≈ 48.8 %, P ≈ 994.3 hPa during the baseline.

⚠ Provisional: POWERLAW_M = 0.35 is a DATASHEET-TYPICAL slope (500–12,500 ppm) extrapolated
to 2–40 ppm; TCORR/RHCORR are honest no-ops until a bench T/RH sweep measures them; the
electrical 1.65 mV jitter is a LOWER bound on the real floor (drift + T/RH dominate). See
``calibration/README.md``.
"""

NODE1 = dict(
    # ── circuit / bench (gate the power-law kernel) ──
    supply_voltage_v=5.0,
    load_resistance_ohm=10000.0,
    r0_ohm=79960.0,

    # ── power-law calibration (gas-free) ──
    background_ppm=1.9,        # anchor: clean air reads this ppm  →  A = 1.9 ** m (derived)
    powerlaw_m=0.35,           # datasheet-typical log-log slope (provisional; refit w/ gas)

    # ── temperature / humidity correction (no-op until a bench sweep) ──
    tcorr_per_c=0.0,
    rhcorr_per_pct=0.0,

    # ── active kernel ──
    kernel="powerlaw",         # "mitchell" would require gas-fit coefficients (dormant)

    # ── measured floor inputs (consumed by misc/calibrate.py, not by the kernel) ──
    baseline_voltage_v=0.5558,
    electrical_noise_mv=1.65,
    baseline_temp_c=24.8,
    baseline_rh_pct=48.8,
)

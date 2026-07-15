"""
calibrate.py — turn one node's clean-air baseline into constants + an HONEST ppm floor.

Run this once per node, on its clean-air baseline CSV, to (1) apply the node's measured
constants, (2) sanity-check that the gas-free anchor makes clean air read ~1.9 ppm, and
(3) compute the make-or-break number: the detection floor IN PPM.

    python3 -m misc.calibrate calibration/nodes/node1.py "Claude Chat -> Code/node1_clean_baseline.csv"

Sibling to ``misc.preflight`` (which asks "is this FIELD upload trustworthy?"); this asks
"what are this node's constants and how small a signal can it honestly see?". It writes
nothing — constants are applied to in-memory globals only. Exit 0 = GO, 1 = CHECK, 2 = STOP.

⚠ THE HONESTY POINT. The 1.65 mV electrical jitter converts to a tiny ~0.018 ppm — that is
the SHORT-TERM RANDOM floor only, and it is NOT the detection limit. The real floor is set by
non-averageable baseline drift and T/RH sensitivity (measured from the record here) plus the
datasheet-extrapolation model error. This tool reports the TOTAL honest floor, and the verdict
keys off that — so the flattering 0.018 can never masquerade as the limit.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from physics import sensor_frontend as sf
from physics.fieldtest import parse_csv, process_fieldtest
from physics.accuracy import estimate_noise_floor, honest_detection_floor, MODEL_UNCERTAINTY_PPM
from physics.sensor_sim import DETECT_K


def load_node_cfg(path) -> tuple[str, dict]:
    """Import a node cfg module from a file path and return (name, cfg dict).

    A node file (e.g. calibration/nodes/node1.py) exposes a plain dict with the
    calibration keys. Finds the first module-level dict carrying 'supply_voltage_v'.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, val in vars(mod).items():
        if isinstance(val, dict) and "supply_voltage_v" in val:
            return name, val
    raise ValueError(f"{path} defines no node-constants dict (needs 'supply_voltage_v').")


def calibrate(cfg: dict, data, sample_rate_hz: float = 0.5) -> dict:
    """Apply cfg, convert the baseline, and compute the honest ppm floor. Never raises on
    bad DATA — returns a dict with a verdict (STOP/CHECK/GO), mirroring run_preflight."""
    report: dict = {"verdict": "STOP", "error": None, "facts": {}, "warnings": []}
    sf.apply_node(cfg)

    ready, missing = sf.calibration_status()
    if not ready:
        report["error"] = f"front-end not calibrated — missing {missing}"
        return report

    try:
        parsed = parse_csv(data)
        res = process_fieldtest(
            parsed["ppm"], temperature=parsed["temperature"],
            humidity=parsed["humidity"], time=parsed["time"],
            sample_rate_hz=sample_rate_hz,
        )
    except Exception as e:                       # bad data → STOP, never crash
        report["error"] = f"could not parse/process the baseline ({type(e).__name__}: {e})"
        return report

    ppm = np.asarray(res["raw"], dtype=float)
    baseline = np.asarray(res["baseline"], dtype=float)
    median_ppm = float(np.median(ppm))

    # (1) Anchor sanity: clean air should read ~background. Fail loud if it doesn't.
    bg = float(sf.BACKGROUND_PPM)
    anchor_ok = abs(median_ppm - bg) <= 0.5      # 0.5 ppm slack over a clean record

    # (2) The floor, three ways.
    #  random_electrical: the 1.65 mV spec pushed through the calibration slope (lower bound).
    slope = abs(sf.ppm_per_volt(cfg["baseline_voltage_v"]))
    random_electrical = slope * (cfg["electrical_noise_mv"] / 1000.0)
    #  random_record + bias_record: measured from the CONVERTED baseline itself.
    ne = estimate_noise_floor(ppm - baseline, res["detection"])
    #  total honest floor: measured random ⊕ measured drift/bias ⊕ datasheet-model error.
    hf = honest_detection_floor(ne.random_ppm, ne.bias_ppm)
    total_floor, lod, loq = hf["total_floor_ppm"], hf["lod_ppm"], hf["loq_ppm"]

    provisional = sf.KERNEL == "powerlaw"        # datasheet-slope power-law → ppm is provisional

    report["facts"] = {
        "kernel": sf.KERNEL,
        "constants": {
            "supply_voltage_v": sf.SUPPLY_VOLTAGE_V,
            "load_resistance_ohm": sf.LOAD_RESISTANCE_OHM,
            "r0_ohm": sf.R0_OHM,
            "powerlaw_m": sf.POWERLAW_M,
            "powerlaw_A_derived": sf.POWERLAW_A,
            "background_ppm": bg,
        },
        "n_readings": int(parsed["n"]),
        "median_ppm": median_ppm,
        "anchor_ok": anchor_ok,
        "ppm_per_volt": slope,
        "random_electrical_ppm": random_electrical,
        "measured_random_ppm": ne.random_ppm,
        "measured_bias_ppm": ne.bias_ppm,
        "model_uncertainty_ppm": MODEL_UNCERTAINTY_PPM,
        "total_honest_floor_ppm": total_floor,
        "lod_ppm": lod,
        "loq_ppm": loq,
        "provisional_scale": provisional,
    }
    report["warnings"] = list(parsed["warnings"]) + list(res["warnings"])

    if not anchor_ok:
        report["verdict"] = "STOP"
        report["error"] = (f"anchor sanity FAILED: clean air reads {median_ppm:g} ppm, "
                           f"expected ~{bg:g}. Check R0/constants before trusting any ppm.")
    elif provisional or report["warnings"]:
        report["verdict"] = "CHECK"
    else:
        report["verdict"] = "GO"
    return report


def format_report(name: str, report: dict) -> str:
    v = report["verdict"]
    banner = {
        "GO": "✅ GO    — calibrated; clean air anchors and the floor is trustworthy.",
        "CHECK": "⚠️  CHECK — calibrated, but the ppm scale is provisional (read below).",
        "STOP": "⛔ STOP  — calibration could not be trusted.",
    }[v]
    lines = ["", "═" * 70, f"  CALIBRATE [{name}]: {banner}", "═" * 70]
    if v == "STOP":
        lines += ["", f"  Reason: {report['error']}", ""]
        if not report["facts"]:
            return "\n".join(lines)

    f = report["facts"]
    c = f["constants"]
    lines += [
        "",
        "  CONSTANTS APPLIED",
        f"    kernel      : {f['kernel']}",
        f"    V_c / R_L   : {c['supply_voltage_v']:g} V / {c['load_resistance_ohm']:g} Ω",
        f"    R0          : {c['r0_ohm']:g} Ω",
        f"    m (slope)   : {c['powerlaw_m']:g}   A (derived from {c['background_ppm']:g} ppm anchor) : {c['powerlaw_A_derived']:.4f}",
        "",
        "  ANCHOR SANITY",
        f"    {f['n_readings']} readings, median {f['median_ppm']:.3f} ppm "
        f"(want ~{c['background_ppm']:g})  →  {'PASS' if f['anchor_ok'] else 'FAIL'}",
        "",
        "  DETECTION FLOOR (ppm)",
        f"    calibration slope        : {f['ppm_per_volt']:.2f} ppm/V",
        f"    random electrical (1.65mV): {f['random_electrical_ppm']:.4f} ppm   ← lower bound only",
        f"    measured random / drift  : {f['measured_random_ppm']:.3f} / {f['measured_bias_ppm']:.3f} ppm",
        f"    datasheet-model error    : {f['model_uncertainty_ppm']:.2f} ppm",
        f"    ►  TOTAL HONEST FLOOR    : {f['total_honest_floor_ppm']:.3f} ppm",
        f"       LOD ({DETECT_K:g}σ) / LOQ (10σ)  : {f['lod_ppm']:.2f} / {f['loq_ppm']:.2f} ppm",
    ]
    if f.get("provisional_scale"):
        lines += ["",
                  "  ⚠ PROVISIONAL: power-law kernel uses a datasheet-typical slope extrapolated",
                  "    to 2–40 ppm and a clean-air-R0 anchor. Absolute ppm is not certified until",
                  "    a gas fit (Mitchell) — fine for LOCATING a leak, not for a certified value."]
    if report["warnings"]:
        lines += ["", "  DATA WARNINGS", *[f"    • {w}" for w in report["warnings"]]]
    lines += ["", "═" * 70, ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m misc.calibrate",
        description="Calibrate a node from its clean-air baseline; report the honest ppm floor.",
    )
    parser.add_argument("node", help="path to a node cfg (e.g. calibration/nodes/node1.py)")
    parser.add_argument("csv", help="path to that node's clean-air baseline CSV")
    parser.add_argument("--sample-rate", type=float, default=0.5, dest="sample_rate_hz",
                        help="Hz to assume if the CSV has no time column (default 0.5)")
    args = parser.parse_args(argv)

    try:
        name, cfg = load_node_cfg(args.node)
    except Exception as e:
        print(f"⛔ STOP  — cannot load node cfg {args.node!r}: {e}", file=sys.stderr)
        return 2
    try:
        with open(args.csv, "rb") as fh:
            data = fh.read()
    except OSError as e:
        print(f"⛔ STOP  — cannot open {args.csv!r}: {e}", file=sys.stderr)
        return 2

    report = calibrate(cfg, data, sample_rate_hz=args.sample_rate_hz)
    print(format_report(name, report))
    return {"GO": 0, "CHECK": 1, "STOP": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())

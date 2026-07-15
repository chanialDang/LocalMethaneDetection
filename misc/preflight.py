"""
preflight.py — a field self-check for a real methane CSV before you trust it.

Run this on your FIRST real Custer/Melissa upload, right after you pull the SD card,
so you catch a format problem WHILE you can still re-log — not after you've left the
site:

    python3 -m misc.preflight path/to/readings.csv
    python3 -m misc.preflight path/to/readings.csv --sample-rate 2   # if no time column

It reuses the EXACT parse → process → aggregate pipeline the dashboard uses
(``physics.fieldtest``), so whatever it reports here is what the web app will see.
It prints: which header columns were matched, the time units/cadence it inferred,
a ppm-magnitude sanity check, EVERY data warning, the detection outcome, and a
one-line verdict:

    GO     — parsed cleanly, values look like ppm, ready to upload/invert.
    CHECK  — parsed, but there are warnings to read (a unit was auto-corrected, a
             column was guessed, values look off). Not necessarily wrong — verify.
    STOP   — could not parse at all (no methane column, empty file, …). Fix and re-run.

Nothing here changes your file. Exit code is 0 for GO, 1 for CHECK, 2 for STOP, so
it can gate a script.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from physics.fieldtest import (
    aggregate_for_inversion,
    parse_csv,
    process_fieldtest,
)
from physics.accuracy import estimate_noise_floor, honest_detection_floor


def run_preflight(data, sample_rate_hz: float = 1.0) -> dict:
    """
    Run the full ingestion pipeline on `data` (CSV text/bytes/file-like) and return
    a JSON-safe diagnostics dict. Never raises on bad DATA — a parse failure is
    reported as ``verdict="STOP"`` with the friendly message, so the CLI and any
    caller degrade gracefully.
    """
    report: dict = {"verdict": "STOP", "error": None, "warnings": [], "facts": {}}

    # Whole body guarded so run_preflight NEVER raises — the CLI and callers rely on a
    # dict with a verdict, not an exception, no matter how broken the input is.
    try:
        parsed = parse_csv(data)
    except ValueError as e:
        report["error"] = str(e)
        return report
    except Exception as e:                       # defense-in-depth (e.g. a csv.Error)
        report["error"] = f"could not parse the file ({type(e).__name__}: {e})"
        return report

    try:
        res = process_fieldtest(
            parsed["ppm"],
            temperature=parsed["temperature"],
            humidity=parsed["humidity"],
            time=parsed["time"],
            sample_rate_hz=sample_rate_hz,
        )
        warnings = list(parsed["warnings"]) + list(res["warnings"])

        time = np.asarray(res["time"], dtype=float)
        ppm = np.asarray(res["raw"], dtype=float)
        excess = np.asarray(res["excess"], dtype=float)
        det = res["detection"]
        cadence = float(np.median(np.diff(time))) if time.size > 1 else None

        pt = aggregate_for_inversion(res)

        # Honest ppm floor measured from THIS record (random ⊕ drift ⊕ datasheet-model
        # error). Surfaced so a quiet record can't be mistaken for a sensitive one — the
        # detection limit is dominated by non-averageable drift + model error, not jitter.
        ne = estimate_noise_floor(ppm - np.asarray(res["baseline"], dtype=float), det)
        hf = honest_detection_floor(ne.random_ppm, ne.bias_ppm)
        from_voltage = bool(parsed["columns"].get("voltage")) and parsed["columns"]["ppm"] is None

        facts = {
            "n_readings": int(parsed["n"]),
            "columns": parsed["columns"],
            "time_present": parsed["time"] is not None,
            "cadence_s": cadence,
            "duration_s": float(time[-1] - time[0]) if time.size > 1 else 0.0,
            "ppm_min": float(np.min(ppm)),
            "ppm_median": float(np.median(ppm)),
            "ppm_max": float(np.max(ppm)),
            "baseline_mean_ppm": float(np.mean(res["baseline"])),
            "excess_peak_ppm": float(np.max(excess)) if excess.size else 0.0,
            "threshold_ppm": float(res["threshold"]),
            "weather_corrected": bool(res["weather_corrected"]),
            "detected": bool(det.detected),
            "event_start_s": float(time[det.start_idx]) if det.detected else None,
            "event_end_s": float(time[min(det.end_idx, time.size - 1)]) if det.detected else None,
            "confidence_sigmas": float(det.confidence) if det.detected else 0.0,
            "inversion_mean_excess_ppm": float(pt.mean_excess_ppm),
            "inversion_sigma_ppm": float(pt.sigma_ppm),
            "inversion_n_window": int(pt.n_window),
            "floor_random_ppm": hf["random_ppm"],
            "floor_bias_ppm": hf["bias_ppm"],
            "floor_model_ppm": hf["model_ppm"],
            "floor_total_ppm": hf["total_floor_ppm"],
            "lod_ppm": hf["lod_ppm"],
            "loq_ppm": hf["loq_ppm"],
            "ppm_provisional": from_voltage,
        }

        report["facts"] = facts
        report["warnings"] = warnings
        report["verdict"] = "CHECK" if warnings else "GO"
    except Exception as e:
        report["error"] = f"parsed, but processing failed ({type(e).__name__}: {e})"
        report["verdict"] = "STOP"
    return report


def format_preflight(report: dict) -> str:
    """Render a run_preflight() dict as a human-readable terminal report."""
    verdict = report["verdict"]
    banner = {
        "GO": "✅ GO    — parsed cleanly and values look like ppm.",
        "CHECK": "⚠️  CHECK — parsed, but read the warnings below before trusting it.",
        "STOP": "⛔ STOP  — could not parse this file.",
    }[verdict]
    lines = ["", "═" * 68, f"  CSV PREFLIGHT: {banner}", "═" * 68]

    if verdict == "STOP":
        lines += ["", f"  Reason: {report['error']}", ""]
        return "\n".join(lines)

    f = report["facts"]
    cols = f["columns"]

    def _fmt(v, unit=""):
        return "—" if v is None else f"{v:g}{unit}"

    lines += [
        "",
        "  COLUMNS DETECTED",
        f"    methane     : {cols['ppm']!r}"
        + ("  (converted from voltage — see warnings)"
           if cols.get("voltage") and cols["ppm"] is None else ""),
        f"    voltage     : {cols.get('voltage')!r}",
        f"    time        : {cols['time']!r}"
        + ("" if f["time_present"] else "  (none — cadence assumed)"),
        f"    temperature : {cols['temperature']!r}",
        f"    humidity    : {cols['humidity']!r}",
        "",
        "  SIGNAL",
        f"    readings    : {f['n_readings']}",
        f"    cadence     : {_fmt(f['cadence_s'], ' s/sample')}"
        f"   duration : {_fmt(f['duration_s'], ' s')}",
        f"    ppm min/med/max : {f['ppm_min']:g} / {f['ppm_median']:g} / {f['ppm_max']:g}",
        f"    baseline    : {f['baseline_mean_ppm']:g} ppm"
        f"   excess peak : {f['excess_peak_ppm']:g} ppm  (threshold {f['threshold_ppm']:g})",
        f"    weather correction applied : {f['weather_corrected']}",
        "",
        "  DETECTION",
        (f"    DETECTED event  {_fmt(f['event_start_s'])}–{_fmt(f['event_end_s'])} s,"
         f"  {f['confidence_sigmas']:.1f}σ above noise"
         if f["detected"] else
         "    no sustained event (a quiet record is still a valid null constraint)"),
        "",
        "  INVERSION-READINESS (aggregate_for_inversion)",
        f"    mean excess : {f['inversion_mean_excess_ppm']:g} ± "
        f"{f['inversion_sigma_ppm']:g} ppm over {f['inversion_n_window']} samples",
        "",
        "  HONEST DETECTION FLOOR (ppm)",
        f"    random / drift / model : {f['floor_random_ppm']:.3f} / "
        f"{f['floor_bias_ppm']:.3f} / {f['floor_model_ppm']:.2f} ppm",
        f"    ►  total floor : {f['floor_total_ppm']:.3f} ppm"
        f"    LOD : {f['lod_ppm']:.2f} ppm   LOQ : {f['loq_ppm']:.2f} ppm"
        + ("   (ppm scale PROVISIONAL — from voltage)" if f['ppm_provisional'] else ""),
    ]

    if report["warnings"]:
        lines += ["", "  DATA WARNINGS (read these!)"]
        lines += [f"    • {w}" for w in report["warnings"]]
    else:
        lines += ["", "  DATA WARNINGS: none 🎉"]

    lines += ["", "═" * 68, ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m misc.preflight",
        description="Field self-check for a real methane CSV before you trust it.",
    )
    parser.add_argument("csv", help="path to the readings CSV")
    parser.add_argument(
        "--sample-rate", type=float, default=1.0, dest="sample_rate_hz",
        help="Hz to assume if the file has no time column (default 1.0)",
    )
    parser.add_argument(
        "--node", default=None,
        help="path to a node cfg (e.g. calibration/nodes/node1.py) to apply first — "
             "required if the CSV is raw voltage (no ppm column)",
    )
    args = parser.parse_args(argv)

    if args.node is not None:
        from physics import sensor_frontend as sf
        from misc.calibrate import load_node_cfg
        try:
            _name, cfg = load_node_cfg(args.node)
            sf.apply_node(cfg)
        except Exception as e:
            print(f"⛔ STOP  — cannot apply node cfg {args.node!r}: {e}", file=sys.stderr)
            return 2

    try:
        with open(args.csv, "rb") as fh:
            data = fh.read()
    except OSError as e:
        print(f"⛔ STOP  — cannot open {args.csv!r}: {e}", file=sys.stderr)
        return 2

    report = run_preflight(data, sample_rate_hz=args.sample_rate_hz)
    print(format_preflight(report))
    return {"GO": 0, "CHECK": 1, "STOP": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())

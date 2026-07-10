"""
rehearse.py — a full deployment DRESS REHEARSAL before you have real data.

Generates a realistic multi-sensor, multi-wind scenario with a KNOWN hidden source,
writes it as CSV files (the exact format the app reads), then runs the real field
workflow — parse → process → multi-snapshot inversion — and shows how close the
recovered source lands to the truth. Run it to rehearse the whole pipeline end to end
and build confidence before Custer/Melissa:

    python3 -m misc.rehearse
    python3 -m misc.rehearse --out ./my_rehearsal    # keep the generated CSVs to inspect

Nothing here is used in production — it's a teaching / dry-run harness that composes the
real modules (physics.plume, physics.fieldtest, physics.inversion). The scenario is
deterministic given --seed, so a rehearsal is reproducible.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from physics.fieldtest import parse_csv, process_fieldtest, to_csv_text
from physics.inversion import invert_field_tests_multi
from physics.plume import (
    CH4_BACKGROUND,
    RELEASE_HEIGHT_M,
    SENSOR_HEIGHT_M,
    predict_ppm,
)

# A fixed, realistic fence-line scenario. The source is hidden behind the fence; the
# five sensors sit on it; three wind episodes let the fusion triangulate (a single
# wind on this line is under-determined — see CLAUDE.md F7 / DEPLOYMENT.md §3).
TRUE_SOURCE = (12.0, -4.0)
TRUE_Q = 3.0
STABILITY = 4
SENSORS = np.array([[50, -20], [50, -10], [50, 0], [50, 10], [50, 20]], dtype=float)
WINDS = [(3.0, 270.0), (3.0, 240.0), (3.0, 300.0)]   # (u m/s, from-direction °)


def _sensor_series(total_ppm: float, n: int, ev0: int, ev1: int, rng) -> dict:
    """One sensor's realistic time series: quiet 1.9 ppm background with a sustained
    plume event at the forward-model level, plus gentle temperature/humidity drift."""
    ppm = CH4_BACKGROUND + rng.normal(0, 0.05, n)
    ppm[ev0:ev1] = total_ppm + rng.normal(0, 0.05, ev1 - ev0)
    t = np.arange(n, dtype=float)
    return {"time": t, "ppm": ppm, "temperature": 20.0 + 0.005 * t,
            "humidity": 50.0 - 0.005 * t}


def generate(out_dir: str, seed: int = 0, n: int = 200, ev0: int = 80,
             ev1: int = 150) -> list:
    """Write one CSV per (wind episode × sensor). Returns a manifest of file paths."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for wi, (u, wd) in enumerate(WINDS):
        total = predict_ppm(TRUE_SOURCE, TRUE_Q, u, wd, RELEASE_HEIGHT_M, STABILITY,
                            SENSORS, z=SENSOR_HEIGHT_M)   # total ppm each sensor reads
        for si, ppm_i in enumerate(total):
            path = os.path.join(out_dir, f"wind{wi + 1}_sensor{si}.csv")
            with open(path, "w") as f:
                f.write(to_csv_text(_sensor_series(float(ppm_i), n, ev0, ev1, rng)))
            manifest.append(path)
    return manifest


def rehearse(out_dir: str, seed: int = 0) -> dict:
    """Generate the scenario, run the real parse→process→fuse pipeline, return a report."""
    manifest = generate(out_dir, seed=seed)
    snapshots, detections = [], 0
    for wi, (u, wd) in enumerate(WINDS):
        results = []
        for si in range(len(SENSORS)):
            path = os.path.join(out_dir, f"wind{wi + 1}_sensor{si}.csv")
            with open(path, "rb") as f:
                parsed = parse_csv(f.read())
            res = process_fieldtest(parsed["ppm"], temperature=parsed["temperature"],
                                    humidity=parsed["humidity"], time=parsed["time"])
            detections += int(res["detection"].detected)
            results.append(res)
        snapshots.append((results, u, wd))

    est = invert_field_tests_multi(snapshots, SENSORS, stability_class=STABILITY)
    return {
        "out_dir": out_dir,
        "true_source": TRUE_SOURCE, "true_Q": TRUE_Q,
        "recovered": (est.x, est.y, est.Q),
        "error_m": float(np.hypot(est.x - TRUE_SOURCE[0], est.y - TRUE_SOURCE[1])),
        "converged": bool(est.converged), "n_snapshots": int(est.n_snapshots),
        "detections": detections, "n_files": len(manifest),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m misc.rehearse",
        description="Dress-rehearse the full field pipeline with a known hidden source.")
    ap.add_argument("--out", default=None,
                    help="directory for the generated CSVs (default: a temp dir)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = args.out or tempfile.mkdtemp(prefix="ch4_rehearsal_")
    r = rehearse(out, seed=args.seed)
    (tx, ty), (ex, ey, eq) = r["true_source"], r["recovered"]
    print(
        "\n  ── CH₄ pipeline dress rehearsal ─────────────────────────────────\n"
        f"  Generated {r['n_files']} sensor CSVs under: {r['out_dir']}\n"
        f"  Wind episodes fused : {r['n_snapshots']}     sensor detections: {r['detections']}\n"
        f"  TRUE source   : x={tx:+.1f}  y={ty:+.1f}  Q={r['true_Q']:.2f} g/s\n"
        f"  RECOVERED     : x={ex:+.1f}  y={ey:+.1f}  Q={eq:.2f} g/s\n"
        f"  Position error: {r['error_m']:.2f} m      converged: {r['converged']}\n"
        "  ──────────────────────────────────────────────────────────────────\n"
        f"  Inspect a file:  python3 -m misc.preflight "
        f"{os.path.join(r['out_dir'], 'wind1_sensor2.csv')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

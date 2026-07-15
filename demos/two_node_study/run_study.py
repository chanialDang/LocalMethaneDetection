"""
run_study.py — does a SECOND node credibly help locate a leak?

Builds faked-but-realistic CSVs for a KNOWN hidden source seen by two fenceline nodes under
three wind episodes (turbulent scatter + drift + sensor noise baked in, so the readings have
the "variance" real data has). Then runs the forward→inverse pipeline for three scenarios and
compares them, so the benefit of the 2nd dataset is measured against ground truth, not asserted:

    (A) 1 node  × 3 winds     — one sensor, even with winds, is geometrically starved
    (B) 2 nodes × 1 wind      — two sensors but a single wind is under-determined (F7)
    (C) 2 nodes × 3 winds     — the combination the pipeline actually needs

Run:  python3 -m demos.two_node_study.run_study   (writes CSVs under demos/two_node_study/data/)
"""
import os
import numpy as np

from physics.plume import (predict_ppm, CH4_BACKGROUND, RELEASE_HEIGHT_M,
                           SENSOR_HEIGHT_M)
from physics.fieldtest import parse_csv, process_fieldtest
from physics.inversion import invert_field_tests_multi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ── ground truth (hidden from the inversion; used only to score recovery) ──
TRUE_SOURCE = (10.0, 0.0)      # 10 m in from the fence, on the centreline
TRUE_Q = 6.0                   # g/s — a clearly-detectable low-ppm leak (~7–8 ppm at the fence)
STAB = 4                       # Pasquill D
NODES = {"A": (55.0, -8.0), "B": (55.0, 8.0)}       # two sensors on a 55 m fenceline
# The straight (270°) wind sends the plume down the centreline (both nodes ~equal → can't tell
# left from right); each veered wind swings it onto ONE node — that asymmetry is what lets the
# fusion locate. A single wind (esp. 270°) is symmetric ⇒ under-determined.
WINDS = [("wind1", 3.0, 270.0), ("wind2", 3.0, 250.0), ("wind3", 3.0, 290.0)]
N, EV0, EV1 = 360, 120, 300     # samples; plume passes over [120, 300) — long enough to average
TURB = 0.18                     # plume meander: ±18% scatter around the mean (realistic)


def generate(rng):
    """Write one CSV per (wind × node) with realistic variance. Returns a path manifest."""
    os.makedirs(DATA, exist_ok=True)
    manifest = {}
    for (wname, u, wd) in WINDS:
        mean_ppm = predict_ppm(TRUE_SOURCE, TRUE_Q, u, wd, RELEASE_HEIGHT_M, STAB,
                               np.array(list(NODES.values())), z=SENSOR_HEIGHT_M)
        for node, m in zip(NODES, mean_ppm):
            t = np.arange(N, dtype=float)
            ppm = CH4_BACKGROUND + 0.25 * np.sin(2 * np.pi * t / (3.0 * N))  # slow drift
            ppm = ppm + rng.normal(0, 0.35, N)                               # sensor noise
            excess = np.zeros(N)
            excess[EV0:EV1] = m - CH4_BACKGROUND
            # TURBULENCE: the plume meanders, so the event scatters ±TURB around its mean.
            excess[EV0:EV1] *= np.clip(1.0 + rng.normal(0, TURB, EV1 - EV0), 0.0, None)
            ppm = np.clip(ppm + excess, 0.05, None)
            temp = 20.0 + 0.01 * t
            rh = 50.0 + 8.0 * np.sin(2 * np.pi * t / N)
            path = os.path.join(DATA, f"{wname}_node{node}.csv")
            rows = ["time_s,ch4_ppm,temp_C,humidity_pct"]
            rows += [f"{t[i]:.0f},{ppm[i]:.4f},{temp[i]:.2f},{rh[i]:.2f}" for i in range(N)]
            with open(path, "w") as f:
                f.write("\n".join(rows) + "\n")
            manifest[(wname, node)] = path
    return manifest


def _process(path):
    p = parse_csv(open(path, "rb").read())
    return process_fieldtest(p["ppm"], temperature=p["temperature"],
                             humidity=p["humidity"], time=p["time"])


def invert(node_ids, wind_eps, manifest):
    sensors = np.array([NODES[nd] for nd in node_ids])
    snaps = [([_process(manifest[(w, nd)]) for nd in node_ids], u, wd)
             for (w, u, wd) in wind_eps]
    return invert_field_tests_multi(snaps, sensors, stability_class=STAB)


def report(label, est):
    err = float(np.hypot(est.x - TRUE_SOURCE[0], est.y - TRUE_SOURCE[1]))
    crb = est.crb_std
    crb_xy = (f"±({crb['x']:.1f},{crb['y']:.1f}) m" if crb and crb["x"] < 1e4 else
              ("±HUGE (unconstrained)" if crb else "—"))
    print(f"  {label:26s}  converged={str(est.converged):5s}  ill_posed={str(est.ill_posed):5s}  "
          f"pos_err={err:6.1f} m   Q={est.Q:4.2f}   CRB(x,y)={crb_xy}")
    return err


def main():
    rng = np.random.default_rng(7)
    manifest = generate(rng)
    print(f"\n  Wrote {len(manifest)} CSVs to {os.path.relpath(DATA)}/")
    print(f"  HIDDEN truth: source=({TRUE_SOURCE[0]:.0f}, {TRUE_SOURCE[1]:.0f}) m,  Q={TRUE_Q} g/s")

    # 1) Does each node SEE a plume? (detection per node, wind1)
    print("\n  STEP 1 — each node's own graph (wind1): is there a plume bump?")
    for nd in NODES:
        r = _process(manifest[("wind1", nd)])
        d = r["detection"]
        print(f"    node {nd}: detected={str(d.detected):5s}  confidence={d.confidence:4.1f}σ  "
              f"excess_peak={np.max(r['excess']):.2f} ppm   (floor {r['noise_ppm']:.2f} ppm)")

    # 2) Can we LOCATE the source, and does the 2nd node/more winds help?
    print("\n  STEP 2 — locating the source: 1 vs 2 nodes, 1 vs 3 winds")
    print("  " + "-" * 92)
    report("(A) 1 node  × 3 winds", invert(["A"], WINDS, manifest))
    report("(B) 2 nodes × 1 wind", invert(["A", "B"], WINDS[:1], manifest))
    est_c = invert(["A", "B"], WINDS, manifest)
    report("(C) 2 nodes × 3 winds", est_c)
    print("  " + "-" * 92)
    print(f"  Recovered by (C): source=({est_c.x:.1f}, {est_c.y:.1f}) m,  Q={est_c.Q:.2f} g/s")
    print("  Read it as: each node SEES the plume (Step 1), but only (C) can LOCATE it —")
    print("  (A)/(B) are under-determined and honestly flagged, not confident wrong answers.\n")


if __name__ == "__main__":
    main()

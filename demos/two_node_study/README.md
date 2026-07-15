# Two-node study — is a second dataset credibly worth it?

**Synthetic demo.** Faked-but-realistic CSVs for a KNOWN hidden leak, seen by two fenceline
nodes under three wind episodes, pushed through the real forward→inverse pipeline. Because the
source is known, recovery is *scored against truth* — the benefit is measured, not asserted.

Run: `python3 -m demos.two_node_study.run_study` (writes CSVs to `data/`).

## Setup
- Hidden source at `(10, 0) m`, `Q = 6 g/s` (a low-ppm leak: ~7–8 ppm at the fence).
- Two nodes on a 55 m fence: `A = (55, −8)`, `B = (55, +8)`.
- Three winds: 270° (straight down the centreline) and ±20° veers.
- Each record carries realistic **variance**: slow drift + sensor noise + ±18% plume meander.

## What it shows

**Step 1 — each node sees a plume.** On the straight wind both nodes detect the bump at ~21σ
(~8 ppm above the ~1.9 ppm background). So *detection* — "is there methane?" — works with one node.

**Step 2 — locating the source needs the second node AND multiple winds:**

| scenario | converged | ill_posed | pos error | CRB (x,y) | verdict |
|---|---|---|---|---|---|
| (A) 1 node × 3 winds | True | **True** | ~100 m | ±HUGE | degenerate — honestly flagged |
| (B) 2 nodes × 1 wind | True | **True** | ~100 m | ±HUGE | symmetric single wind — under-determined |
| (C) 2 nodes × 3 winds | True | False | **~0.2 m** | **±1 m** | located, error bar matches error |

(Exact numbers are deterministic given the seed; re-run to reproduce.)

## The takeaways
1. **One node is enough to DETECT, not to LOCATE.** Locating a leak is 3 unknowns (x, y, Q);
   one sensor — even across winds — is geometrically starved.
2. **Two nodes on a single wind can still be under-determined.** The 270° wind sends the plume
   straight down the centreline, so both nodes read ~equally and can't tell left from right —
   the pipeline flags this (`ill_posed=True`, ±HUGE CRB) instead of guessing.
3. **Two nodes × a few winds is the credible combination.** The veered winds swing the plume
   onto one node at a time; that asymmetry pins the source to sub-metre, and the reported error
   bar (±1 m) honestly matches the actual error (0.2 m).
4. **Read `converged` together with `ill_posed` and the CRB.** `converged=True` means "signal
   present and a fit ran"; `ill_posed=True` / a huge CRB means "but the geometry can't trust the
   position." (A) and (B) are the honest "I can't locate this" — not confident wrong answers.

**So: the second dataset is credibly worth it** — but the payoff is unlocked by capturing the
leak under a few *different winds*, not by adding sensors alone.

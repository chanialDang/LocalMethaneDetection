
# What This Code Does — Plain-English Explanation

## The big picture

You're trying to detect a methane leak near the Custer Road Transfer Station
using a cheap MOX sensor. Before you touch any hardware, you need to answer
one question: **does the physics even allow detection?**

This code builds a **forward model** — a calculator that answers:

> "If there's a leak at position X emitting Q grams per second, with the wind
> blowing at speed u from direction φ, what concentration (in ppm) will my
> sensor read?"

Once the forward model works, Week 3 flips it: given sensor readings, find
the leak. That inversion is only meaningful if the forward direction is correct.

---

## The two files

### `plume.py` — the physics engine

Contains five functions. Here's what each one does in plain English:

---

#### 1. `sigma_y` and `sigma_z` — how fat is the plume?

When gas leaks from a source, it forms a cone-shaped cloud downwind. The
further it travels, the more it spreads out. `σ_y` measures how wide it is
sideways (crosswind), and `σ_z` measures how tall it is vertically.

Both are calculated using **Briggs (1973)** formulas — closed-form equations
calibrated against open-field tracer experiments. The spread rate depends on
atmospheric stability:

| Class | Condition | What it means |
|-------|-----------|---------------|
| A | Very unstable | Hot sunny day, light wind — plume fans out fast |
| B–C | Unstable | Partly cloudy, moderate wind |
| D | Neutral | Overcast or moderate wind — the most common case |
| E–F | Stable | Clear night, calm air — plume stays narrow and concentrated |

**Accuracy note:** These formulas were tested at distances ≥ 100 m over flat,
smooth terrain. At 10–50 m (your operating range), and at a site with structures
and equipment, treat them as order-of-magnitude estimates.

---

#### 2. `rotate_to_wind_frame` — aligning coordinates with the wind

The plume equation works in "wind coordinates" — one axis pointing downwind,
one perpendicular (crosswind). But your map uses East/North axes, and the wind
blows in some arbitrary direction.

This function takes a receptor position in map coordinates and rotates it into
wind coordinates. Upwind receptors (negative downwind coordinate) get
background concentration only — the plume hasn't reached them.

The wind direction `φ` follows the **meteorological convention**: it's the
direction the wind comes FROM, clockwise from North. φ = 270° means wind
blows from the west (eastward).

---

#### 3. `concentration_gm3` — the core Gaussian plume formula

This is the actual physics equation. For a continuous point source:

```
C = Q / (2π σ_y σ_z u)
  · exp(−y²/2σ_y²)        ← crosswind bell curve
  · [exp(−(z−H)²/2σ_z²)   ← real source
   + exp(−(z+H)²/2σ_z²)]  ← ground reflection (mirror image)
```

Breaking it down:

- **`Q / (2π σ_y σ_z u)`** — mass conservation. All the gas has to go
  somewhere; this ensures mass flux through any cross-section equals Q.
- **`exp(−y²/2σ_y²)`** — the crosswind bell curve. Concentration is highest
  on the plume centreline (y=0) and falls off like a Gaussian going sideways.
- **Ground reflection term** — the ground acts like a perfect mirror for the
  gas. We pretend there's an identical source at depth -H underground. This
  doubles concentration at ground level when H=0.

Returns concentration in **g/m³**. Raises `ValueError` if wind speed < 0.5 m/s
(the formula diverges as u→0 — physically, still air doesn't form a steady
plume, so the model is simply undefined).

---

#### 4. `gm3_to_ppm_methane` — unit conversion

Your sensor reports in ppm (parts per million by volume). The physics formula
returns g/m³. The conversion uses the ideal gas law:

```
ppm = C_gm3 · (R · T) / (M_CH4 · P) · 10⁶
```

This is a **pure converter** — it doesn't add the 1.9 ppm atmospheric background.
Background is added in `predict_ppm` to keep this function reusable.

---

#### 5. `predict_ppm` — the function you actually call

Orchestrates the full pipeline for an array of sensor positions:

1. Rotate each receptor into wind coordinates.
2. If upwind → return 1.9 ppm (background) and skip.
3. Look up σ_y, σ_z at that downwind distance.
4. Evaluate the Gaussian plume formula → g/m³.
5. Convert → ppm.
6. Add 1.9 ppm background.

Returns a NumPy array of total CH₄ readings, one per sensor.

---

### `demo.py` — validation and feasibility

Runs four validation blocks and two outputs:

**Validation Block 1 — σ arithmetic**
Checks that `sigma_y(100, 'D')` and `sigma_z(100, 'D')` reproduce the exact
Briggs formula values. At x=100 m, Class D:
- σ_y = 7.96 m, σ_z = 5.60 m.

**Validation Block 2 — Concentration formula**
Checks the Gaussian formula implementation at a known operating point
(Q=1 g/s, u=5 m/s, H=0, centreline, Class D, x=100 m). Pass criterion: ±5%
of the reference value computed directly from the same Briggs formulas.
This is an **implementation check**, not a comparison against external
measurements — both sides use the same physics.

**Validation Block 3 — Coordinate rotation**
Tests four cardinal-direction cases (wind from west, receptor east/west; wind
from north, receptor south/north). Catches sign-flip bugs that are easy to
introduce and hard to notice.

**Ancillary checks**
- Upwind receptor must return exactly 1.9 ppm.
- `u=0.1 m/s` must raise `ValueError`.

**Feasibility sweep** — the most important scientific output:

Prints predicted concentrations at 50 m and 100 m for emission rates spanning
Q = 0.01–5.0 g/s (transfer station range). Example output for Class D, u=2 m/s:

```
Q (g/s) |  50 m (ppm) | 100 m (ppm)
--------|-------------|------------
  0.01  |  ~1.9–2.0   |   ~1.9      ← borderline/below sensor floor
  0.1   |  ~2–3       |   ~2.0      ← marginal, needs stats
  5.0   |  ~10–50     |   ~5–15     ← clearly detectable
```

This directly answers whether detection is physically feasible before you
buy or build anything.

**Contour plot** — a 2D heatmap of the plume (saved as `plume_contour.png`)
for Q=5 g/s, Class C, u=3 m/s. Colour scale is log-normalised so both the
dilute fringe and the hot centreline are visible. A dashed line marks 100 m
to flag where the Briggs formulas leave their validated range.

---

## Key caveats you must keep in mind

1. **Steady-state ≠ instantaneous.** Your sensor takes a reading in a few
   seconds. The model gives the 10-minute average. Real readings fluctuate far
   more — expect large scatter around the model prediction.

2. **Open-country σ at a rough site.** The transfer station has buildings,
   equipment, and a berm. All of these increase turbulence and make the real
   plume dilute faster than the model predicts. Peak concentrations in the
   field will be lower than what `predict_ppm` returns.

3. **Sub-100 m is extrapolated.** Your fenceline sensors are likely at 10–50 m.
   Briggs never tested this close. Use model outputs as order-of-magnitude.

4. **Calm wind is the hardest case.** u < 0.5 m/s is exactly when a weak
   source is most detectable (less dilution) — and exactly when the model is
   undefined. You'll need to flag those field measurements separately.

5. **The sensor is 2–3 orders of magnitude outside its characterised range.**
   The TGS 2611-E00 is rated for 500–12,500 ppm. Your plumes are 2–40 ppm.
   Signal extraction requires statistical baseline subtraction, not simple
   threshold crossing.

---

## How this fits the 8-week project

```
Week 2 (now) ── Forward model ──► given (source, wind) → predicted ppm
                                        │
                                        ▼
Week 3 ────── Inversion ──────────► given (sensor readings) → find source
                                        │
                                        ▼
Week 4 ────── Feasibility sweep ──► is detection possible before field work?
```

The feasibility sweep in `demo.py` is already Week 4's core output. If it
shows Q=0.01 g/s is undetectable at 100 m under neutral conditions, that
tells you exactly what emission rate or sensor placement you need before
collecting a single data point.

## Collecting real data (the Field Tests page)

The forward model *predicts* readings; the **Field Tests** page (`/fieldtests.html`)
takes in *real* ones. The important thing to understand: the model draws a
**map** (concentration everywhere at once), but a real sensor only gives you a
**graph over time** at one spot — methane shows up as a bump rising out of the
1.9 ppm background as the wind carries the plume past. So real data gets its own
**ppm-vs-time graph, one per field test**, and the map stays the model/reference.

What happens when you upload a CSV (columns: at least `ppm`; optionally `time`,
`temperature`, `humidity`):

1. The same cleaning pipeline from `processing.py` runs — remove weather drift,
   subtract the slow baseline, smooth, then decide whether a real event is there
   (the excess must stay above 3× the noise floor for several samples).
2. You get a graph (raw + baseline + cleaned excess + threshold + the detected
   window), metric cards, and a plain-English **interpretation** (the same
   AI-or-template path as the model caption — it never breaks without a key).
3. The raw readings are **stored** (Railway Postgres if configured, otherwise a
   local file), so you can reopen tests later and **compare runs** to see how
   repeatable they are — directly exposing variance/deviation across field tests.

No sensor yet? Hit **Generate sample test** (or download the sample CSV) to walk
the whole flow with realistic synthetic data first. See `CLAUDE.md` →
"Field-test data collection" for the API and the one-line Railway `DATABASE_URL`
setup.

# LocalMethaneDetection — CLAUDE.md

## What this project is

A **Gaussian plume forward model** for methane (CH₄) detection near the
Custer Road Transfer Station in Allen, TX. Built by a high school student
as Week 2 of an 8-week research project:

- **Week 2 (this code)** — Forward model: given a source, predict sensor readings.
- **Week 3** — Inversion: given sensor readings, find the source (`scipy.optimize`).
- **Week 4** — Feasibility sweep: determine whether detection is physically possible.

The sensor is a **Figaro TGS 2611-E00 MOX sensor**, characterised by the
manufacturer at 500–12,500 ppm. Using it at 2–40 ppm above a 1.9 ppm background
is the research contribution.

---

## File layout

```
LocalMethaneDetection/
├── plume.py          # Core physics module — import this in downstream work
├── demo.py           # Validation suite, feasibility sweep, contour plot
├── plume_contour.png # Output from demo.py's contour plot
└── requirements.txt  # numpy, matplotlib, rich
```

---

## Key physics and formulas

### Gaussian plume equation (Pasquill 1961, Gifford 1961)

```
C = Q / (2π σ_y σ_z u)
  · exp(−y²/2σ_y²)
  · [exp(−(z−H)²/2σ_z²) + exp(−(z+H)²/2σ_z²)]
```

- The second bracket is the **ground reflection** term — models the ground
  as a perfect mirror (no gas absorbed by soil).
- At H=0, y=0, z=0 this simplifies to `C = Q / (π σ_y σ_z u)`.

### Dispersion coefficients — Briggs (1973) open-country formulas

`σ_y` and `σ_z` (metres) depend on downwind distance `x` (metres) and the
Pasquill stability class (A–F, unstable to stable). See `plume.py:100–206` for
the exact expressions. **Key validity note:** these formulas were validated at
≥100 m over flat open terrain. Sub-100 m results and rough-terrain sites
(structures, equipment) are order-of-magnitude estimates only.

### Wind-frame coordinate rotation

`wind_dir_deg` is the **meteorological from-direction** (clockwise from North).
270° = wind blows FROM the west, TOWARD the east. The downwind unit vector is:

```
û_down = (−sin φ, −cos φ)   [East, North components]
x_wind = dx·(−sin φ) + dy·(−cos φ)    # positive = downwind
y_wind = dx·(cos φ)  + dy·(−sin φ)    # crosswind
```

---

## Constants (cite these if you change them)

| Symbol | Value | Unit | Source |
|--------|-------|------|--------|
| `R_GAS` | 8.314 | J/(mol·K) | NIST CODATA 2018 |
| `M_CH4` | 16.04 | g/mol | IUPAC 2021 |
| `CH4_BACKGROUND` | 1.9 | ppm | NOAA GML 2023 |
| `U_MIN` | 0.5 | m/s | Model undefined below this |

---

## Public API (`plume.py`)

| Function | Returns | Purpose |
|----------|---------|---------|
| `sigma_y(x, stability_class)` | float (m) | Crosswind dispersion coefficient |
| `sigma_z(x, stability_class)` | float (m) | Vertical dispersion coefficient |
| `rotate_to_wind_frame(x_r, y_r, src_x, src_y, wind_dir_deg)` | (float, float) | Map → wind frame |
| `concentration_gm3(x, y, Q, u, H, sy, sz, z=0)` | float (g/m³) | Raw Gaussian plume concentration |
| `gm3_to_ppm_methane(C_gm3, T_K, P_Pa)` | float (ppm) | Unit conversion only — no background added |
| `predict_ppm(src_pos, Q, u, wind_dir_deg, H, stability_class, receptors, ...)` | ndarray (ppm) | Full pipeline including background |

**`predict_ppm` is the function you'll call in downstream work.** It handles
coordinate rotation, upwind masking, σ lookup, unit conversion, and background
addition in one call.

---

## Running the demo

```bash
pip install numpy matplotlib rich
python demo.py
```

`demo.py` runs four validation blocks, a feasibility sweep table, and a
contour plot (saved as `plume_contour.png`). Exit code 1 if any validation fails.

---

## Constraints and caveats

1. **Steady-state only.** The model gives a time-mean picture. Individual
   sensor readings fluctuate far more due to real turbulence.
2. **u ≥ 0.5 m/s required.** `concentration_gm3` raises `ValueError` below
   this. Calm-wind readings must be handled separately.
3. **Open-country σ underestimates dilution** at a rough site (structures,
   berm, equipment increase mixing beyond open-field levels).
4. **Sub-100 m predictions are order-of-magnitude only.**
5. `gm3_to_ppm_methane` is a pure unit converter — it does NOT add background.
   Background is added in `predict_ppm` only.

---

## Project context

- Operating range: 10–200 m; release height H ≈ 0–2 m.
- Site: Custer Road Transfer Station (not a landfill — methane is weak and
  intermittent).
- Target signal: 2–40 ppm above 1.9 ppm background.
- Sensor: Figaro TGS 2611-E00 MOX (rated 500–12,500 ppm — low-ppm use is the
  research contribution).
- This forward model feeds directly into a `scipy.optimize` inversion in Week 3.

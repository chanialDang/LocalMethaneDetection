# Deployment runbook — taking this from code to a real result

A plain-English checklist for running the methane experiment at Custer (then Melissa) so
the data actually means something. It folds in the 2026-07-09 literature validation (see
`CLAUDE.md` 📚 LIT notes) and the hardened pipeline. Work top to bottom.

---

## 0. The one thing that decides whether any verdict is trustworthy

Every "detectable? / how big is the leak?" number depends on the **sensor's real noise
floor and drift**, which are still placeholders in the code (`SENSOR_NOISE_PPM = 0.30`,
`BIAS_FLOOR_PPM = 0.00`). Peer-reviewed characterization of *your exact sensor* (Figaro
TGS 2611-E00; Shah et al. 2023, Furuta et al. 2024) says the realistic resolution is
**~1–2 ppm, not 0.30**, and that slow **baseline drift is the dominant error and does not
average away**. Until you measure these on the bench, treat every detectability verdict as
an optimistic best case — especially at low-ppm Melissa.

**→ Do a zero-air / clean-air bench run FIRST** (§1). Everything downstream re-computes
automatically once the measured constants are set.

---

## 1. Before you go — bench (a few hours, indoors)

1. **Warm up** the Figaro per its datasheet, then log it in clean air (no leak) for at
   least a couple of hours.
2. From that quiet record, read off:
   - **random noise σ** (short-term jitter) → `SENSOR_NOISE_PPM` in `physics/sensor_sim.py`
   - **drift over hours** (the slow wander that averaging can't remove) → `BIAS_FLOOR_PPM`
   - **temperature / humidity sensitivity** if you vary T/H → `TEMP_COEFF_PPM_PER_C`,
     `HUMID_COEFF_PPM_PER_PCT`
   (`physics/accuracy.estimate_noise_floor` can estimate the random/bias split from a quiet
   record, but a real zero-air bench run is the gold standard — an in-record proxy can't see
   a constant calibration offset.)
3. **Measure the geometry** you'll deploy and set it in `physics/plume.py`:
   `RELEASE_HEIGHT_M` (H, how high the leak is) and `SENSOR_HEIGHT_M` (z, fence-mount height).
4. Re-run the tests (`MPLBACKEND=Agg python3 -m pytest tests/ -q`) — still green — then
   re-read the accuracy grade; it now reflects your real sensor.

## 2. Logging setup — the CSV

The parser is now robust to real logger output, but you'll have the least surprises if you
log **`time,ppm`** (or add `temperature,humidity`) with **time in seconds**. It will also
handle: `;`/tab delimiters, decimal commas, `CH4 (ppm)`-style headers, header-less dumps,
`millis()`/epoch/ISO/clock time, comment banners, and BOM/CRLF. It **warns** (never silently
misreads) on anything ambiguous.

**Sanity-check your format before the field** with the preflight tool:

```
python3 -m misc.preflight your_test_file.csv
```

Fix anything until it prints **GO** (or a **CHECK** whose warnings you understand). `STOP`
means it couldn't parse — fix the file and re-run.

**Rehearse the whole workflow once** before the field, so nothing about the
parse→process→invert flow is new on data day:

```
python3 -m misc.rehearse
```

It invents a known source, writes realistic multi-sensor CSVs, runs the full pipeline, and
prints how close the recovered source lands — a dry run of exactly what you'll do at the site.

## 3. On site — data collection

- **Sensor placement (matters for the inversion).** A single wind on a close fenceline is
  *under-determined* — the math can't pin the source (Riddick et al. 2022 saw exactly this
  <100 m). Two independent fixes, either works:
  - place **≥3 sensors that all sit inside the plume**, OR
  - capture the **same leak under ≥2 different wind directions** (wait for the wind to veer)
    and fuse them (§4). Wind variability is what makes the inverse solvable.
- **Sensors closer than ~40 m are the least reliable** (the plume-spread formulas are worst
  there). ~50 m near the plume centerline is the sweet spot.
- **Log the wind** during each episode (on-site anemometer, or `physics/weather.py` pulls
  Open-Meteo archive wind + a Pasquill stability class for your coordinates/time). Note the
  start/stop time of each distinct wind episode.
- Calm wind (< 0.5 m/s) is the most detectable but the model is undefined there — record it,
  but don't quantify a leak size from it.

## 4. After collection — process → locate the source

For each sensor's CSV:

```python
from physics.fieldtest import parse_csv, process_fieldtest
parsed = parse_csv(open("sensorA.csv","rb").read())
resA   = process_fieldtest(parsed["ppm"], temperature=parsed["temperature"],
                           humidity=parsed["humidity"], time=parsed["time"])
```

Then invert. **Single wind** (only if ≥3 sensors are in the plume):

```python
from physics.inversion import invert_field_tests
est = invert_field_tests([resA, resB, resC], sensor_positions, u=..., wind_dir_deg=...)
```

**Multiple winds (recommended — triangulates the source):**

```python
from physics.inversion import invert_field_tests_multi
est = invert_field_tests_multi([
    ([resA1, resB1, resC1], u1, dir1),   # wind episode 1
    ([resA2, resB2, resC2], u2, dir2),   # wind episode 2 (same sensors, later)
], sensor_positions)
```

`est` gives the source `x, y, Q`, a `converged` flag, `crb_std` (the best-possible 1σ), and
an `ill_posed` flag. Pin `u`/`wind_dir_deg` from `weather.py` per episode, not a guess.

## 5. Reading the result honestly

- **`converged is False`** → no sensor saw a real signal; the position is unconstrained and
  only an upper bound on Q is meaningful. Not a failure — a valid "saw nothing" result.
- **`ill_posed is True`** → the geometry is degenerate (e.g. all sensors on one line, or one
  wind); add a wind episode or a sensor in the plume.
- **The recovered Q carries a wide error bar by nature.** Near-field Gaussian-plume inversion
  is documented to *over-estimate* emissions (Mbua et al. 2025: 1.6–3.9×; Riddick et al. 2022:
  40–60% uncertainty), and open-country dispersion curves over-predict ppm at a built fenceline.
  Report Q as an order-of-magnitude with its `crb_std`, not a precise figure.
- Sub-100 m plume results are **order-of-magnitude only** — this is honest, not a bug.

## 6. Still open — needs your decision (don't let me change these blind)

These change the *scientific behavior* and you should weigh in before they ship:

- **Nonlinear temperature/humidity correction.** Literature for this sensor uses
  log(water-vapor) + log(time) terms; the code fits a linear `a·T + b·H`. Humidity actually
  *dominates* the raw signal at ~2 ppm, so this may matter at Melissa. (Enhancement, needs a
  model-form choice + validation against your bench data.)
- **F7 single-snapshot honesty flag.** A single-wind under-determined fit currently reports
  `converged=True` with an optimistic CRB. The cure in place is *operational* (fuse winds);
  an alternative is to make the flag itself honest (widen CRB / set `converged=False`). Both
  are defensible — pick one.
- **Site roughness `z0`.** Default is open-country (0.1 m); a built fenceline is ~0.3–1.0 m,
  which shifts the wind-height adjustment and the inferred stability class. Set a
  site-specific value once you know the terrain.

---

*Pipeline status (2026-07-09): forward + inversion math independently verified clean; CSV
ingestion hardened against ~100 adversarial cases; full test suite green. The gates above are
measurement- and judgment-bound, not code bugs.*

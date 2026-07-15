# Sensor calibration runbook — turning volts into ppm

The pipeline is **ppm-in, source-out**: everything downstream assumes your CSV column is
already methane in ppm. A real Figaro TGS 2611-E00 doesn't output ppm — it's a
chemiresistor, and your ADC records a **voltage**. `physics/sensor_frontend.py` is the
stage that bridges the two. This runbook fills its constants from a bench session so the
converter can run.

Do this **before** the deployment runbook (`docs/DEPLOYMENT.md` §1) — the noise-floor
measurement in step 5 here is the same "one thing that decides whether any verdict is
trustworthy" that §0 of that doc warns about, so the two bench sessions are really one.

Everything you set lives in one place — the constant block at the top of
`physics/sensor_frontend.py` (and the noise floor in `physics/sensor_sim.py`). Replace the
placeholders with measured values; nothing else needs to change.

---

## The chain you're calibrating

```
ADC voltage ─► R_s = R_L·(V_c − V_out)/V_out ─► R_s/R₀ = A·C^(−m) ─► (÷ T/RH factor) ─► ppm
```

| Constant | Where | What it is | How you get it |
|----------|-------|-----------|----------------|
| `SUPPLY_VOLTAGE_V` (V_c) | `sensor_frontend.py` | supply across the divider | read off your circuit |
| `LOAD_RESISTANCE_OHM` (R_L) | `sensor_frontend.py` | the over-amplification load resistor | read off your circuit |
| `R0_OHM` (R₀) | `sensor_frontend.py` | R_s in clean air | **measure on the bench** (step 3) |
| `POWERLAW_A`, `POWERLAW_M` | `sensor_frontend.py` | log-log sensitivity curve | datasheet now, span gas later |
| `TCORR_PER_C`, `RHCORR_PER_PCT` | `sensor_frontend.py` | T/RH drift of the ratio | optional T/RH sweep (step 6) |
| `SENSOR_NOISE_PPM`, `BIAS_FLOOR_PPM` | `sensor_sim.py` | random jitter + non-averageable drift | quiet log (step 5) |

Until `SUPPLY_VOLTAGE_V`, `LOAD_RESISTANCE_OHM`, and `R0_OHM` are set, the converter
**refuses to run** (a loud STOP), so it can never emit a fabricated ppm.

---

## Where to run it: indoors is fine — with one rule

The real constraint is **not** indoor vs outdoor, it's *known gas vs ambient air* and
*clean air vs contaminated air*. For the parts of this runbook that need a stable baseline
(burn-in, R₀, the noise floor), a room is actually **better** than outdoors — steadier
temperature and humidity than a yard where weather swings all day.

> **The one rule:** put the sensor in a room with **no gas stove, gas furnace, or gas water
> heater**, and don't cook, spray aerosols, or use cleaning products near it during a run.
> A MOX sensor cross-reacts to alcohol, cooking VOCs, and hydrogen, and gas appliances leak
> methane — either will quietly corrupt R₀ and the drift baseline. A bedroom or an
> electric-only garage is ideal.

**Safety:** at the 2–40 ppm this project works in, methane is ~1000× below its flammable
limit (~50,000 ppm), so low-ppm work is not a fire hazard. The caution above is only about
keeping *interfering* gases away from the sensor, not danger.

---

## The bench session, in order

### 1. Build the front-end, record the circuit values
Wire the divider `V_c —[R_s sensor]— node(V_out) —[R_L load]— GND` with a **large R_L** and
a **precision ADC** (the over-amplification method — a big load resistor puts the tiny
low-ppm resistance change onto many ADC codes). Record the supply and load values and set:
```python
SUPPLY_VOLTAGE_V    = 5.0       # your measured V_c
LOAD_RESISTANCE_OHM = 47_000.0  # your R_L
```

### 2. Burn in
Power the sensor and leave it running per the datasheet's preheat guidance — MOX baselines
keep creeping for days after first power-up. Location doesn't matter here; just keep it
powered and away from gas appliances.

### 3. Measure R₀ in clean air
Once burned in and sitting in clean room air (≈1.9 ppm background), read the steady sensor
resistance and set:
```python
R0_OHM = 8_200.0   # your measured clean-air R_s
```
> **Caveat worth knowing:** the datasheet A/m (step 4) are defined against the *datasheet's*
> R₀ reference, which may not be your clean-air R₀. Using your own R₀ with datasheet A/m
> introduces a scale offset — harmless for locating the leak, but it's one more reason the
> ppm scale is provisional until the span-gas refit (step 4's note).

### 4. Set the power-law A and m (datasheet now)
Read the sensitivity line off the TGS 2611-E00 datasheet's `R_s/R₀ vs. concentration` chart
and set:
```python
POWERLAW_A = 1.0    # placeholder — read the real value
POWERLAW_M = 0.35   # placeholder — the log-log slope
```
> ⚠ **These are unverified at your operating range.** The datasheet curve is characterised at
> **500–12,500 ppm**; you're running at **2–40 ppm**, so this is an extrapolation. Every ppm
> the front-end produces is therefore *provisional* — good enough to locate a leak and
> exercise the pipeline, not yet a defensible absolute concentration. When you can get
> **certified low-ppm span gas**, expose the sensor to a few known concentrations, fit A and
> m to *those* points, and overwrite the two constants — nothing else changes.

### 5. Measure the noise floor and drift (the trustworthiness step)
From the same burn-in log (a few quiet hours), read off and set **in `sensor_sim.py`**:
```python
SENSOR_NOISE_PPM = 0.9    # random short-term jitter (1σ), in ppm
BIAS_FLOOR_PPM   = 0.6    # slow drift over hours that averaging can't remove
```
`physics/accuracy.estimate_noise_floor` will split a quiet record into its random and bias
parts for you (feed it the converted-ppm series). Literature on this exact sensor (Shah 2023,
Furuta 2024) puts the realistic resolution near **1–2 ppm** and says **drift, not random
noise, is the dominant floor** — so expect `BIAS_FLOOR_PPM` to matter more than
`SENSOR_NOISE_PPM`, especially at low-ppm Melissa. This is the number that turns every
"detectable?" verdict from an optimistic guess into a real one.

### 6. (Optional) Temperature / humidity sweep
If you can vary room T and RH while logging clean air, fit the fractional shift of R_s/R₀ per
°C and per %RH and set `TCORR_PER_C` / `RHCORR_PER_PCT`. Left at 0 they're a no-op, which is
the honest default until measured. (Humidity is known to dominate this sensor's raw signal at
low ppm, so this is worth doing before Melissa.)

---

## Verify

Log a short clean-air capture as a `voltage` CSV and preflight it:
```
python3 -m misc.preflight your_voltage_test.csv
```
- With the three circuit/bench constants **unset**, you'll get **STOP** naming exactly what's
  missing — that's the gate working.
- With them **set**, it converts the voltage column to ppm and reports **GO**/**CHECK** (a
  CHECK just flags that the datasheet A/m make the ppm scale provisional).

Then run the suite to confirm nothing regressed:
```
MPLBACKEND=Agg python3 -m pytest tests/ -q
```

A CSV that already carries a real `ppm` column is unaffected by any of this — the front-end
only engages when the file has a raw `voltage`/`adc` column and no ppm column.

---

*Once these constants are measured, hand off to `docs/DEPLOYMENT.md` for the field run. The
A/m span-gas refit and the T/RH sweep are the only pieces this runbook leaves open — both are
one-constant edits with no other code change.*

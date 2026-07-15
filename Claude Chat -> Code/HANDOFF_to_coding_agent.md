# Handoff: Node 1 Baseline Data + Calibration Task

**For:** the coding agent working on the methane-sensor project
**Date:** July 12, 2026
**From:** hardware/bench side (node 1 is built and logging)

---

## 1. What this project is (1 paragraph)
Low-cost methane source-localization + emission-rate estimation using a small network of Figaro **TGS 2611-E00** MOX sensors + Gaussian-plume math. High-school research project. The forward model and both inversions (PSG single-sensor + multi-sensor least-squares) are already built and ~95% validated. The open critical-path item is **CALIBRATION**: converting the physical sensor's raw voltage into methane ppm. Everything downstream (noise floor in ppm, whether we get "clean detection" vs "upper-bound only", and the inversions) is blocked on this.

## 2. Where hardware stands
- **Node 1** (ESP32-DevKitC-32E + ADS1115 16-bit ADC + BME280 + microSD) is built and logging.
- Sensor is in a **voltage divider**: `VC = 5.0 V`, load resistor `R_L = 10,000 Ω`. The ADS reads the divider junction (`A0`).
- Sensor resistance is computed as: **`Rs = R_L * (VC - V) / V`**.
- Known issue: breadboard connections are loose (being fixed by soldering to protoboard). This corrupted part of the run — see filtering below. This does NOT affect the clean data handed over here.

## 3. THE DATA (attached files)
Two files accompany this handoff:

### `node1_clean_baseline.csv`  ← USE THIS
The **filtered, clean clean-air baseline.** 2,716 rows, ~91 minutes, all in clean indoor air (~1.9 ppm ambient methane background). Columns:
`time_ms, t_s, voltage_V, Rs_ohm, temp_C, humidity_pct, pressure_hPa`

### `run3_raw.csv` (the full raw run, for reference only)
The full 6.6-hour raw log (11,777 rows). **~30% of it is garbage** and must NOT be used as-is. Included only so the filtering is reproducible/auditable.

## 4. FILTERING LOGIC (already applied to the clean file — documented so you can reproduce or extend it)
The raw run had two failure modes that must be stripped out of ANY future run:

1. **ADS/voltage dropout** → `voltage_V` reads ~0 or rails. Filter: keep only `0.05 < V < 4.9` (and for a clean-air baseline specifically, tighten to `0.50 < V < 0.60` since baseline sits at ~0.556 V).
2. **BME280 dropout** → produces sentinel garbage `temp_C = 180.13` and `humidity_pct = 100.00` (and negative pressure). Filter: drop rows with `temp_C > 100`, `temp_C < -40`, or `humidity_pct >= 100`.

Additionally: the first ~1 min of any run is ADS boot noise (V≈0) and the BME on this run died at **minute 276**, so the clean window used was **min 45.2–136.0** (after warm-up, before the BME failure). Reproducible filter:
```python
seg = df[(t_min>=45.2) & (t_min<=136.0)]
mask = (seg.voltage_V>0.50) & (seg.voltage_V<0.60) & (seg.temp_C<100) & (seg.temp_C>-40) & (seg.humidity_pct<100)
clean = seg[mask]
```

## 5. MEASURED RESULTS from the clean baseline
| Quantity | Value |
|---|---|
| **R₀ (clean-air baseline resistance)** | **≈ 79,960 Ω (80 kΩ)**, median 79,938 Ω |
| Baseline voltage | 0.5558 V (std 2.03 mV) |
| **Noise floor (short-term jitter, detrended std)** | **≈ 1.65 mV** |
| Noise floor (point-to-point median) | ≈ 0.40 mV |
| Temp during baseline | 24.82 °C (range 24.59–24.92) |
| Humidity during baseline | 48.83 %RH (range 47.87–50.40) |
| Pressure | 994.3 hPa |
| Duration | 90.7 min |

**⚠️ CRITICAL INTERPRETATION NOTE:** the noise floor above is in **millivolts (electrical)**, NOT in ppm. Whether ~1.65 mV corresponds to "a few tenths of a ppm" (clean detection — good) or ">1 ppm" (upper-bound only) depends **entirely on the calibration slope** (dV/dppm, i.e. how much the sensor voltage moves per ppm of methane at low concentration). **We cannot state the ppm noise floor until calibration is done.** This is the whole point of the task below.

## 6. THE TASK: calibration (voltage/Rs → ppm)
Method already chosen for this project (do not switch approaches without discussion):
- Use **Mitchell, Cox & Lewis 2024 (Sensors 24, 1066) Eq. 16 FORM**, with every coefficient **re-fit to THIS physical sensor** (do not use the paper's coefficients directly).
- **Anchor to the measured R₀ above (~80 kΩ).** The datasheet sensitivity curve is expressed as the ratio **Rs/R₀ vs ppm**, so R₀ is the reference point.
- **Temperature & humidity correction is mandatory and is the single biggest error source** — use the BME280 `temp_C` and `humidity_pct` columns. Do not treat this as a minor term.
- The datasheet curve (rated 500–12,500 ppm) will be extrapolated DOWN to the 2–40 ppm range of interest — flag/track the extrapolation error.
- No calibration gas is available; calibration = datasheet curve shape + measured R₀ + temp/humidity correction.

### What to build
1. A calibration function `ppm = f(Rs, temp_C, humidity_pct)` in the Mitchell Eq. 16 form, coefficients fit to this sensor, anchored to R₀ = 80 kΩ.
2. Apply it to the clean baseline as a sanity check — it should return ≈ 1.9 ppm (ambient background) for the clean-air data. If it doesn't, the fit/anchor is off.
3. **Then compute the noise floor IN PPM** by pushing the ~1.65 mV voltage jitter through the calibration slope. THIS is the number the whole project hinges on.

### Keep calibration and inversion SEPARATE
Calibration (Rs + T + RH → ppm) is a distinct stage that runs FIRST and per-node. The inversion (ppm + wind → source location + emission rate) runs AFTER and combines nodes. Do not merge them. Pipeline:
`voltage → Rs → [CALIBRATION → ppm] → [INVERSION → source]`

## 7. Reference assets already in the project
- `otm33a_dispersion_sigma.csv` (EPA OTM 33A dispersion table) — for the inversion, not calibration.
- Mitchell 2024 (calibration Eq. 16), Foster-Wittig/Thoma/Albertson 2015 (PSG inversion), Van den Bossche 2017 (low-end linear fit; benchmark ±1.7 ppm variable error on this same sensor — useful to compare our residual calibration error against).

## 8. Open questions for the agent to keep in mind
- How large is the residual calibration error once fit? (Compare to Van den Bossche's ±1.7 ppm.)
- Does extrapolating the datasheet curve from 500 ppm down to 2–40 ppm hold up with acceptable error?
- What ppm does the ~1.65 mV noise floor translate to — the make-or-break number?

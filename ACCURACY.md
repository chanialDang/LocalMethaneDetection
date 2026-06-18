# ACCURACY.md — How accurate is it, and how accurate could it ever be?

A detection only means something if it comes with a number for how sure we are —
especially here, where we run a sensor rated for 500–12,500 ppm down at 2–40 ppm
above a 1.9 ppm background, and the harder Melissa site will be fainter still.
This document defines the project's **accuracy framework** (`physics/accuracy.py`),
in plain English alongside the math.

The framework is one **ordered protocol**. Each step is a function you can call;
together they produce a single self-assessing report.

```
1. CHARACTERIZE  → measure the sensor's real noise from the data
2. VALIDATE      → score recovery against a known truth (synthetic)
4. BOUND         → the best leak we can detect now (empirical) + the best a
                   source fix could ever be (theoretical, Cramér-Rao)
3+5. QUANTIFY+GRADE → fold it into one 0-100 score, a grade, and a next action
```

---

## The accuracy model: two kinds of error

A sensor's error is **not one number**. Two parts behave oppositely when you
average many readings:

| Part | What it is | Under averaging of N samples |
|------|------------|------------------------------|
| **Random** (`random_ppm`) | electrical/turbulent jitter, different each sample | shrinks as **1/√N** |
| **Bias** (`bias_ppm`) | slow drift, calibration offset, leftover weather | **does not shrink** |

Combined 1σ floor after averaging N samples (added in quadrature):

```
floor(N) = √[ (random_ppm / √N)² + bias_ppm² ]
```

This is `sensor_sim.effective_noise_floor`. The key consequence: **you cannot
average your way to zero.** Past about `N ≈ (random/bias)²` the bias term wins and
more averaging buys almost nothing — the framework reports that crossover.

---

## 1. CHARACTERIZE — measure the real noise (`estimate_noise_floor`)

Instead of trusting the assumed `SENSOR_NOISE_PPM = 0.30` placeholder, we measure
the noise **from the record itself**, on its **quiet** samples (everything outside
the detected event; the whole record if nothing was detected).

- **Random** is read from the *successive differences* of the quiet signal. Slow
  drift cancels when you subtract neighbouring samples, so what remains is the
  sample-to-sample jitter; a robust (median-absolute-deviation) spread ÷ √2 turns
  it back into a 1σ noise. Robust = a few spikes don't inflate it.
- **Bias** is read from the *heavily smoothed* quiet signal — the slow wander that
  a single averaging window can't remove.

> ⚠ **Honesty caveat.** The random term is genuinely measurable from any record.
> The in-record bias is only a **proxy** for residual drift — a *constant*
> calibration offset is indistinguishable from the baseline within one record and
> can only be pinned with a **known reference (a zero-air bench run)**. Protocol
> step 1 in the lab is: sit the real Figaro in clean air and measure both terms
> directly, then set the constants in `sensor_sim.py` from data.

---

## 2. VALIDATE — score recovery vs known truth (`recovery_metrics`)

On synthetic data we *know* the true plume (`sensor_sim` keeps the ground truth),
so we can grade how well the cleaning pipeline recovered it:

- **RMSE** and **mean bias** of recovered − true (ppm),
- **% recovery** = recovered peak ÷ true peak,
- **R²** and **correlation** against the truth,
- **detection outcome** — true/false positive/negative,
- **event timing error** — how far off the detected start/end are (s).

For *real* uploads there is no truth, so these fields are reported as `null` and
the grade leans on the self-measured floor instead. This cleanly separates
**algorithm validation** (synthetic, has truth) from **runtime self-assessment**
(real, no truth).

---

## 4a. BOUND, empirical — the smallest leak we can claim now (`detection_limit`)

From a noise floor we derive the standard analytical-chemistry limits:

```
LOD (limit of detection)      = k · floor      (k = DETECT_K = 3)
LOQ (limit of quantification)  = 10 · floor
```

Then we convert the ppm limit into a **minimum detectable leak Q (g/s)**. Because
the Gaussian plume is *linear in Q*, this is a single back-solve (reusing
`feasibility._excess_at` / `implied_Q_range`) — no optimizer. Crucially we report
it **with an error bar**: the same fenceline signal implies very different Q across
plausible wind (0.5–5 m/s) and stability (classes B–F), summarized as a
`swing_factor` (q_max / q_min). The framework also traces the **averaging curve** —
how the limit drops with N until the bias floor stops it.

---

## 4b. BOUND, theoretical — the best a source fix could ever be (`crb_source_bound`)

The **Cramér-Rao bound (CRB)** is a hard floor from estimation theory: given
Gaussian measurement noise of 1σ `sigma_ppm` independent at each receptor, **no
unbiased estimator** of the source parameters can do better than the inverse
**Fisher information**:

```
J = ∂(predicted ppm at receptors)/∂(x, y, Q)     ← numerical Jacobian of predict_ppm
F = Jᵀ J / σ²                                     ← Fisher information
Cov_best ≥ F⁻¹    →   best-possible 1σ = √diag(F⁻¹)
```

This needs no inversion to exist yet — it stands alone. When the **Week-3**
optimizer lands, compare its achieved scatter to this bound: near the CRB means it
is as good as the physics allows; far above means there is room to improve. More
(or closer) receptors and lower noise tighten the bound; the test suite pins both.

Caveats (documented in the function): assumes Gaussian iid noise, a single source,
and local linearity — it is a best **case**, optimistic if any of those break.

---

## 3 + 5. QUANTIFY + GRADE — one report (`accuracy_report`)

`accuracy_report(result, meta, true_ppm=None)` runs the whole protocol on one
processed field test and returns a single JSON-safe dict: measured-vs-assumed
floor, SNR, detection confidence, the detection limit + error bar, optional
recovery metrics, and a **0–100 score + letter grade + one recommendation**.

**Grading rubric (a documented heuristic, not a physical law):**

| Lever | Points | Rewards |
|-------|--------|---------|
| **Confidence** | 0–50 | detection margin (sigmas if detected, else SNR); full marks at 2× the threshold |
| **Stability** | 0–30 | a low *bias fraction* — error that averaging can actually beat |
| **Recovery** | 0–20 | low RMSE vs the known truth (synthetic only; rescaled away for real data) |

Letters: 90 = A, 80 = B, 70 = C, 60 = D, else F. The **recommendation** attacks the
weakest lever — *recalibrate* (bias-limited), *average more / move closer* (signal
near noise), or *signal-rich* (comfortable margin).

---

## Worked example

Running the framework on the built-in synthetic sample (4 ppm event, 0.30 ppm
injected noise, sensor at the 50 m fence, Class D, u = 2 m/s):

```python
from physics.fieldtest import make_sample_readings, process_fieldtest
from physics import accuracy

s   = make_sample_readings(n=600, event_ppm=4.0, noise_ppm=0.30)
res = process_fieldtest(s["ppm"], temperature=s["temperature"],
                        humidity=s["humidity"], time=s["time"], noise_ppm=0.30)
rep = accuracy.accuracy_report(
    res, meta={"sensor_distance_m": 50.0, "wind_speed": 2.0, "stability_class": 4})
```

| Field | Value | Reading |
|-------|-------|---------|
| `measured_random_ppm` | **0.29** | recovers the injected 0.30 ✓ |
| `measured_bias_ppm` | 0.26 | residual drift the whole-record baseline can't remove |
| `measured_floor_ppm` | 0.39 | random ⊕ bias — higher than the 0.30 placeholder (honest) |
| `confidence_sigmas` | 14.3 | the 4 ppm event sits far above the floor |
| `detection_limit.min_detectable_Q` | **0.063 g/s** | best leak detectable here now |
| `detection_limit.swing_factor` | **≈163×** | wind/stability error bar on that Q |
| `score` / `grade` | 82.4 / **B** | docked from A by the bias fraction (~47%) |
| `recommendation` | *"Signal-rich…"* | comfortable margin at this geometry |

The **B (not A)** is the framework doing its job: the synthetic data carries real
drift, so the sensor is not purely random-limited, and the score reflects that.

---

## What sharpens with real data

Everything above is honest *given the assumed sensor constants*. The single
biggest unknown is still the real Figaro's noise and bias. Once the lab zero-air
run exists:

1. set `SENSOR_NOISE_PPM` and `BIAS_FLOOR_PPM` in `sensor_sim.py` from measurement;
2. every limit, error bar, and grade re-computes automatically — no other change;
3. the in-record bias proxy can be validated against the measured bias.

Design target: this must still give a trustworthy floor at the **Melissa** site,
where the signal is expected to be only a little above the 1.9 ppm background.

---

## Running it

```bash
python3 -m pytest tests/test_accuracy.py -v     # pins the whole protocol
```

See `physics/accuracy.py` for the API and `EXPLANATION.md` for the plain-English
project overview.

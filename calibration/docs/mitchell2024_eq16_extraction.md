# Extraction: Eq. 16 calibration model — Mitchell, Cox & Lewis (2024)

**Paper:** Mitchell H.L., Cox S.J., Lewis H.G. (2024) "Calibration of a Low-Cost
Methane Sensor Using Machine Learning." *Sensors* **24**(4), 1066.
DOI: 10.3390/s24041066. Open access (CC BY 4.0). Read from the PMC full text
(PMC10892608) + Tables 1–5.

> **Reader beware — three framing mismatches with the requester's project**
> 1. **Input is voltage, not resistance.** Every model equation ingests
>    `V_out` (volts, across the load resistor). There is **no** `Rs`, `Rs/R₀`,
>    or resistance-ratio term anywhere in the fitted models. The measured
>    clean-air `R₀ ≈ 80 kΩ` is **not used**.
> 2. **This is a data-driven fit that REQUIRES known-concentration calibration
>    gas.** The paper's novelty is avoiding an expensive *reference analyzer*,
>    but it still calibrates against 200 ppm methane-in-air + calibration-air
>    mixtures. There is **no** datasheet-curve-only / gas-free fitting route.
> 3. **The model's valid range is 0–200 ppm**, so a 2–40 ppm target is *inside*
>    the paper's range (not an extrapolation relative to this paper), but the
>    reported RMSE ≈ 5.1 ppm means absolute error is comparable to the signal at
>    the low end.

---

## 1. Eq. 16 in full (verbatim, from Table 1)

```
M = C1 + C2 * exp( C3*Vout - C4*ln(T + 65) - C5*ln(H) ) - C7*ln( (T + 65) * Vout )
```

LaTeX:
```
M = C_1 + C_2 \exp\!\big(C_3 V_{out} - C_4 \ln(T+65) - C_5 \ln(H)\big)
      - C_7 \ln\!\big((T+65)\,V_{out}\big)
```

Structure: an offset `C1`, plus an exponential whose exponent is linear in
`Vout` with logarithmic temperature and humidity terms **inside** the `exp`,
minus a temperature×voltage interaction log term **outside** the `exp`.

### Fitted coefficients (Table 5; fit to experimental 0–200 ppm data, one sensor)

| Symbol in equation | Table 5 row | Estimate | Std. error | t | p |
|---|---|---|---|---|---|
| C1 | C1 | **−1370** | 116.08 | −11.80 | 4.2e−32 |
| C2 | C2 | **4081.4** | 7.335 | 556.43 | 0 |
| C3 | C3 | **0.1109** | 0.0088 | 12.59 | 2.8e−36 |
| C4 | C4 | **0.2152** | 0.0170 | 12.64 | 1.4e−36 |
| C5 | C5 | **0.0808** | 0.0065 | 12.50 | 8.3e−36 |
| **C7** (interaction) | **C6** | **−0.0587** | 0.0047 | −12.55 | 4.6e−36 |

> **AMBIGUITY (must handle in code):** Table 1 writes the interaction
> coefficient as **`C7`** (it skips `C6`, which is reserved for a
> `(T+α)(H+β)` term used only in Eqs 13/14). But Table 5 lists only six rows,
> labelling the sixth **`C6` = −0.0587**. So the **`C7` in the printed
> equation == the `C6` row in Table 5 == −0.0587.** There are six fitted
> coefficients, not eight.

Fully numeric form:
```
M = -1370 + 4081.4*exp(0.1109*Vout - 0.2152*ln(T+65) - 0.0808*ln(H))
          - (-0.0587)*ln((T+65)*Vout)
  = -1370 + 4081.4*exp(0.1109*Vout - 0.2152*ln(T+65) - 0.0808*ln(H))
          + 0.0587*ln((T+65)*Vout)
```
(The interaction coefficient is negative and the term is subtracted, so it
contributes a *positive* `+0.0587·ln((T+65)·Vout)`.)

---

## 2. Dependency equations

**Eq. (1) — potential-divider / how `Vout` is obtained (verbatim):**
```
Vout = (VC * RL) / (RL + RS)
```
- `VC` = supply voltage = 5 V; `RL` = fixed load resistor = 10.0 kΩ ± 1 %
  (datasheet standard test conditions); `RS` = measured sensor resistance.
- This is algebraically identical to the requester's `Rs = RL*(Vc − Vout)/Vout`
  (solve Eq. 1 for RS → `RS = RL*(VC − Vout)/Vout`). So the hardware convention
  matches — **but the model wants `Vout`, so you feed `Vout` directly and never
  convert to `Rs` for the model.**

**Eq. (2) — the exponential starting point (verbatim):**
```
M = C1 * exp(C2 * Vout)
```
(Here `C1`, `C2` are *local* symbols for this form only — temperature/humidity-
dependent "constants". Do not confuse with the `C1..C8` of Eq. 16.)

**Eq. (3) — the general proposed model form (verbatim, as printed):**
```
M = C1 + C2*exp( C3*Vout - C4*ln(T+α) - C5*ln(H+β)
      - C6*ln((T+α)(H+β)) - C7*ln((T+α)Vout) - C8*ln((H+β)Vout) )
```
> **AMBIGUITY:** As typeset, Eq. (3) reads as if *all* the `C6/C7/C8`
> interaction log-terms sit **inside** the `exp`. But the actual tested
> equations in Table 1 (Eqs 13–20, incl. Eq. 16) place `C4·ln(T+α)` and
> `C5·ln(H)` **inside** the `exp` and the `C6/C7/C8` interaction terms
> **outside** it (subtracted from the whole exponential). **Trust the Table 1
> form for Eq. 16, not the Eq. 3 typesetting.**

**Definition of `R0` in the paper (context only — NOT used in Eq. 16):** the
datasheet response ratio `RS/R0`, where `R0` = sensor resistance in **5000 ppm
methane at 20 °C, 65 %RH**. Used solely to read the Figaro datasheet curves
(Figs 3–5), which were then converted to `Vout` via Eq. 1. This is a completely
different `R0` from a clean-air baseline.

---

## 3. Every symbol (Eq. 16 + dependencies)

| Symbol | Meaning | Role | Units |
|---|---|---|---|
| `M` | methane concentration (predicted) | **output** | ppm (linear, NOT log, NOT ratio) |
| `Vout` | sensor output voltage across `RL` | **input** | V |
| `T` | air temperature | **input** | °C |
| `H` | relative humidity | **input** | % (raw percent, e.g. 48.8) |
| `C1` | additive offset | coefficient | ppm |
| `C2` | exponential pre-factor | coefficient | ppm |
| `C3` | voltage sensitivity | coefficient | V⁻¹ |
| `C4` | temperature log-coefficient (in exponent) | coefficient | dimensionless |
| `C5` | humidity log-coefficient (in exponent) | coefficient | dimensionless |
| `C7` (=Table 5 `C6`) | temperature×voltage interaction coefficient | coefficient | ppm |
| `65` | temperature offset α (see item 5) | fixed structural constant | °C |
| `RS`,`RL`,`VC` | Eq.1 divider terms | — | Ω, Ω, V |

- **Input is `Vout`** (not `Rs`, not `Rs/R0`).
- **Output is linear ppm** (not `log10(ppm)`, not a ratio). Interpreted as
  *total* methane in the sampled air over 0–200 ppm — the paper does not model
  "concentration above background"; background appears as the model's estimate
  near 0 ppm.

---

## 4. How T and RH enter

- **Temperature:** inside the `exp`, as `− C4·ln(T + 65)` (logarithmic, with a
  +65 offset on T in °C). Also appears in the **interaction** term outside the
  `exp`: `− C7·ln((T + 65)·Vout)`.
- **Humidity:** inside the `exp`, as `− C5·ln(H)` (logarithmic, **no offset** —
  raw %RH). Eq. 16 has **no standalone humidity interaction** term (that would
  be the `C8·ln(H·Vout)` term, which defines Eq. 18, not Eq. 16).
- **Cross term:** the only interaction in Eq. 16 is **temperature × voltage**,
  `ln((T+65)·Vout)`. No `T·RH` term (that is the `C6` term of Eqs 13/14). No
  `RH·Vout` term. No plain polynomial terms.
- Forms are **logarithmic** in T and H (the paper found log fits beat linear for
  the T- and H-vs-`Vout` datasheet curves).

---

## 5. Reference conditions T0 / RH0

- **Not stated as physical reference conditions.** `α` and `β` are described only
  as "offsets to be applied to the temperature and humidity terms." For the
  shortlisted models (Eqs 5,6,11–20, incl. **Eq. 16**): **α = 65** (used as
  `T+65`, T in °C) and **β = 0** (H used raw, `ln(H)`).
- Other equation families use different offsets: Eqs 3,4,21,22 use `T+273.15`
  (i.e. Kelvin, α = 273.15); Eq. 21 uses `H+50`. So **α is not universal.**
- The value **65** is never derived or justified in the text. (It numerically
  coincides with the datasheet's 65 %RH reference, but the paper does not link
  them — treat "65" as an empirical offset, not a reference temperature.)
- **There is no explicit T0/RH0 baseline that corrections are measured relative
  to.** Flag as **ambiguous / unstated.**

---

## 6. Universal vs. sensor-specific coefficients

- **All of C1–C7 are fitted, none are universal.** They were obtained by
  nonlinear regression on the experimental calibration data of **a single
  sensor** (one of four; the four "showed very similar responses," so they
  trained on one). Table 5 values are specific to that unit + that circuit +
  that dataset.
- **α = 65, β = 0** are fixed *structural* choices (not optimized by the fitter),
  carried across the shortlisted models.
- The coefficients are tied to the **NGM2611-E13 module's** `Vout` (its own
  onboard `RL`/circuit). A bare TGS 2611-E00 with a user-supplied 10 kΩ load may
  produce a different `Vout` scale even at identical gas/T/RH → **coefficients
  must be re-fit per sensor/hardware.** No coefficient can be lifted verbatim
  with confidence.

---

## 7. Fitting procedure

- **Method:** nonlinear least-squares regression via MATLAB **`fitnlm`**.
- **Settings:** initial estimates = 0 for *all* coefficients; termination
  tolerance = 1×10⁻⁸; iteration limit = 200.
- **Two stages:** (a) first fit on *training data derived from the Figaro
  datasheet curves* (Figs 3–5 → `Vout`, Fig. 7), used only to shortlist model
  forms; (b) **re-fit on experimental calibration data**, 0–200 ppm, collected
  in a vacuum chamber using 200 ppm methane-in-air calibration gas and
  calibration air, at varied T and RH. Table 5 coefficients are from stage (b).
- **What the fit requires you to have:** paired samples of
  `(Vout, T, H) → known M`. The known `M` comes from **calibration gas of known
  concentration** (200 ppm mix + ~0 ppm calibration air, with intermediate
  levels via chamber fill/decay).
  **→ The method assumes calibration gas is available. It does NOT provide a way
  to fit from the datasheet curve + a measured clean-air R₀ alone.** A requester
  with no calibration gas cannot reproduce the coefficient fit as published.

---

## 8. Validated concentration range

- **Model trained & evaluated over 0–200 ppm methane.** (Abstract: "0–200 ppm
  methane, 5–30 °C, 40–80 %RH". Methods text states calibration conditions
  "span 5–35 °C and 40–85 %RH" — a minor internal inconsistency on the T/RH
  bounds; flag it.)
- **Decay-validation experiments:** ~0–~225 ppm, 8–30 °C (expt 2) / 19–25 °C
  (expt 1).
- Manufacturer datasheet characterization: 300–10,000 ppm (the paper's whole
  point is extending *below* that).
- **Implication for a 2–40 ppm target:** within the paper's 0–200 ppm envelope
  (not an extrapolation vs. this paper), but the low end is where a ~5 ppm
  absolute error dominates.

---

## 9. Reported accuracy

- **Eq. 16 on experimental 0–200 ppm data (Table 4): RMSE = 5.09 ppm,
  R² = 0.997.** (Abstract rounds to "5.1 ppm, R² 0.997"; this is the paper's
  selected/headline model.)
- Eq. 16 on the datasheet-derived training data (Table 3): RMSE = 224 ppm,
  R² = 0.993 (units span 0–10,000 ppm there — not comparable to the 0–200 ppm
  figure).
- **Complexity of Eq. 16: reported as 13 in Tables 3–4 but "12" in the
  Discussion** — minor inconsistency; flag it.
- Best models overall (Eqs 16/18/20/21) clustered at RMSE 4.5–5.1 ppm,
  R² 0.997–0.998. Eq. 16 is presented as the best performance/complexity
  compromise and the one whose coefficients are tabulated.

---

## 10. Stated caveats / limitations

- **Hydrogen cross-sensitivity:** the TGS 2611-E00 charcoal filter suppresses
  ethanol/iso-butane but the sensor remains H₂-sensitive → unreliable for low
  methane where H₂ is present. (Less of an issue outdoors.)
- **Warm-up:** ~4 h to thermal equilibrium after power-on; pre-equilibrium data
  do not follow the normal linear `Vout`–temperature relation and were excluded
  from calibration. Relevant to intermittently powered / low-power deployments.
- **Overfitting:** explicitly demonstrated — Eq. 22 (same family, higher
  complexity, `H+0` offset) achieved the best training RMSE (4.79 ppm, R² 0.998)
  yet failed the decay validation badly (temperature-correlated sawtooth,
  gross overestimation). Higher complexity ≠ better field behaviour.
- **T–RH coupling in the calibration rig:** RH is "roughly inversely
  proportional to air temperature in a sealed volume," so the T- and H-related
  terms partly stand in for each other. **This coupling may not hold at a
  fenceline where T and RH vary independently** — a genuine transfer risk for
  the requester's application (not stated in these words, but implied by the
  paper's own reasoning; flag).
- **Extrapolation beyond 200 ppm:** a linear fit "cannot be assumed for wider
  ranges"; an exponential holds 300–10,000 ppm. Nothing validated above ~225 ppm
  here.
- **Drift:** the paper motivates ML calibration partly to offset long-term drift
  but does **not** quantify drift for Eq. 16 or include a drift/time term
  (indeed it argues *against* time-based predictors for transferable
  calibration). No drift figure is reported. Flag as **silent**.

---

## Practical guidance for the re-implementation

1. **Feed `Vout` (volts) to the model, not `Rs/R0`.** You already measure
   `Vout`; skip the resistance conversion for the model input. Keep your `Rs`
   calc only if you want it for diagnostics.
2. **`R₀ ≈ 80 kΩ` is not an input to Eq. 16.** Don't wire it in.
3. **Do not reuse Table 5 coefficients as-is on your hardware.** They are fit to
   one NGM2611-E13 module's `Vout`. Your bare TGS 2611-E00 + your 10 kΩ / 5 V
   divider will have a different `Vout` scale; the exponential's `C2`, `C3` and
   the offset `C1` especially will be wrong. You need known-concentration points
   to re-fit — which the paper's method assumes and you currently lack.
4. **Watch units:** `T` in °C (offset +65 applied inside the model), `H` in raw
   percent, output `M` in linear ppm.
5. **Interaction-coefficient label:** implement the printed `C7` term using the
   Table 5 `C6` value (−0.0587).
6. **At 2–40 ppm, a ±5 ppm model error is the same order as your signal above a
   1.9 ppm background.** Budget for it; humidity handling is, as you expect, a
   dominant term (the `C5·ln(H)` term plus the T–RH coupling caveat).

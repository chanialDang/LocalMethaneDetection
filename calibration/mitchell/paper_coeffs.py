"""
paper_coeffs.py — Mitchell 2024 Eq. 16 coefficients, FOR REFERENCE / TESTS ONLY.

Table 5 of Mitchell, Cox & Lewis (2024) Sensors 24(4):1066. These are fit to the paper's
own NGM2611-E13 module + its circuit — NOT to our bare TGS 2611-E00 + 5 V / 10 kΩ divider.
Plugged into Eq. 16 at our Node 1 baseline (V=0.5558, T=24.8, H=48.8) they give ≈ −165 ppm,
which is physically impossible — the concrete proof that Mitchell coefficients CANNOT be
lifted across hardware and MUST be re-fit with calibration gas.

⚠ ``physics/sensor_frontend.py`` deliberately NEVER imports this. It exists only so a test
can demonstrate the −165 ppm failure and prove the Mitchell gate refuses it. The label
quirk (paper prints C7 for the interaction term but Table 5 rows it as C6 = −0.0587) is
resolved here: the key is ``C7`` with the −0.0587 value. See ``eq16_spec.md``.
"""

PAPER_COEFFS_REFERENCE = dict(
    C1=-1370.0,
    C2=4081.4,
    C3=0.1109,
    C4=0.2152,
    C5=0.0808,
    C7=-0.0587,   # Table-5 "C6" row == the printed-equation C7 interaction coefficient
)

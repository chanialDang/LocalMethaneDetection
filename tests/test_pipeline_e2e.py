"""
test_pipeline_e2e.py — the full deployment path, end to end.

Exercises exactly what a real Custer/Melissa run does: several sensors' CSV TEXT →
parse_csv → process_fieldtest → invert_field_tests → a recovered source. Every other
test pins one stage; this one pins that they COMPOSE — a regression anywhere in the
bridge (column mapping, windowing, aggregation, the WLS handoff) shows up here.

Independent truth: the source (x, y, Q) is chosen first, sensor readings are generated
forward from predict_ppm, serialized to CSV, then recovered blind through the pipeline.
"""
import numpy as np
import pytest

from physics.fieldtest import parse_csv, process_fieldtest, to_csv_text
from physics.inversion import invert_field_tests
from physics.plume import (
    CH4_BACKGROUND,
    RELEASE_HEIGHT_M,
    SENSOR_HEIGHT_M,
    predict_ppm,
)


def _sensor_csv(ppm_true_total, n, ev0, ev1, rng):
    """One sensor's CSV text: quiet background with a sustained plume event whose
    level is the forward-model total ppm at that sensor."""
    series = CH4_BACKGROUND + rng.normal(0, 0.05, n)
    series[ev0:ev1] = ppm_true_total + rng.normal(0, 0.05, ev1 - ev0)
    return to_csv_text({"time": np.arange(n, dtype=float), "ppm": series,
                        "temperature": None, "humidity": None})


def test_csv_text_to_recovered_source_end_to_end():
    rng = np.random.default_rng(0)
    true_src = (12.0, -4.0)
    Q_true, u, wind, sc = 3.0, 3.0, 270.0, 4
    sensors = np.array([[40, -12], [45, 0], [45, 12], [70, -6], [95, 4]], dtype=float)

    true_total = predict_ppm(true_src, Q_true, u, wind, RELEASE_HEIGHT_M, sc, sensors,
                             z=SENSOR_HEIGHT_M)   # total ppm each sensor should read

    n, ev0, ev1 = 200, 80, 150
    results = []
    for ppm_i in true_total:
        parsed = parse_csv(_sensor_csv(float(ppm_i), n, ev0, ev1, rng))
        results.append(process_fieldtest(parsed["ppm"], time=parsed["time"],
                                         noise_ppm=0.30))

    est = invert_field_tests(results, sensors, u=u, wind_dir_deg=wind, stability_class=sc)

    # The pipeline composed and returned a usable, converged fix in the right place.
    assert est.converged is True
    assert est.Q > 0.0
    assert est.x == pytest.approx(true_src[0], abs=15.0)
    assert est.y == pytest.approx(true_src[1], abs=15.0)
    # At least the near sensors detected their event (the CSV→detection bridge works).
    assert sum(r["detection"].detected for r in results) >= 2


def test_e2e_quiet_sensors_are_a_null_constraint_not_a_crash():
    # A run where NO sensor sees a real plume (all background) must not crash the
    # bridge — it aggregates to near-zero excess and reports converged=False.
    rng = np.random.default_rng(1)
    sensors = np.array([[45, -10], [45, 0], [45, 10]], dtype=float)
    results = []
    for _ in sensors:
        parsed = parse_csv(_sensor_csv(CH4_BACKGROUND, 150, 60, 90, rng))
        results.append(process_fieldtest(parsed["ppm"], time=parsed["time"],
                                         noise_ppm=0.30))
    est = invert_field_tests(results, sensors, u=3.0, wind_dir_deg=270.0,
                             stability_class=4)
    assert est.converged is False        # honest: no signal → position unconstrained
    assert np.isfinite(est.Q)

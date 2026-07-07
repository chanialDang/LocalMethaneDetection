"""
test_weather.py — Open-Meteo wind/stability layer (Problem 2).

All hermetic: the only networked function (fetch_wind_archive) is monkeypatched or
exercised through its failure path, and everything else runs on the committed
fixture tests/fixtures/openmeteo_archive.json. No test touches the network.
"""
import json
import pathlib

import numpy as np
import pytest

from physics import weather

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "openmeteo_archive.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def archive(payload):
    return weather.parse_archive(payload)


# ── parsing ──────────────────────────────────────────────────────────────────
def test_parse_archive_shapes(archive):
    assert len(archive) == 10
    assert archive.u10.shape == (10,)
    assert archive.direction.shape == (10,)
    assert archive.u10[0] == pytest.approx(1.8)
    assert archive.shortwave[6] == pytest.approx(860.0)


def test_parse_archive_missing_hourly_raises():
    with pytest.raises(ValueError):
        weather.parse_archive({"latitude": 1.0})


def test_parse_archive_nulls_become_nan():
    payload = {"hourly": {"time": ["t0", "t1"],
                          "wind_speed_10m": [2.0, None],
                          "wind_direction_10m": [270, 270],
                          "cloud_cover": [0, 0],
                          "shortwave_radiation": [0, 0]}}
    a = weather.parse_archive(payload)
    assert np.isnan(a.u10[1])
    assert float(np.nanmedian(a.u10)) == pytest.approx(2.0)


# ── Pasquill class ───────────────────────────────────────────────────────────
def test_pasquill_day_strong_sun_light_wind_is_unstable():
    # strong insolation + calm → class A (1)
    assert weather.pasquill_class(1.0, 900.0, 0.0) == 1


def test_pasquill_day_strong_sun_strong_wind_is_more_neutral():
    # same strong sun but brisk wind → less unstable (≥ C)
    assert weather.pasquill_class(7.0, 900.0, 0.0) >= 3


def test_pasquill_night_calm_clear_is_most_stable():
    # no sun, calm, clear sky → class F (6)
    assert weather.pasquill_class(1.0, 0.0, 0.0) == 6


def test_pasquill_night_calm_overcast_less_stable_than_clear():
    overcast = weather.pasquill_class(1.0, 0.0, 90.0)
    clear = weather.pasquill_class(1.0, 0.0, 0.0)
    assert overcast < clear        # cloud blankets → less stable at night


def test_pasquill_returns_valid_class_range():
    for u in (0.5, 2.5, 4.0, 7.0):
        for rad in (0.0, 400.0, 800.0):
            c = weather.pasquill_class(u, rad, 30.0)
            assert 1 <= c <= 6


# ── F8: EPA-454/R-99-005 Table 6-7 "Key to the SRDT Method", p.6-15 ─────────────
# Expected values below are taken directly from the cited EPA table, independent
# of the code under test (NOT a re-pin of whatever this code happened to output).
def test_pasquill_strong_vs_moderate_insolation_boundary_925_diverges_at_high_wind():
    # EPA: >=925 W/m^2 is "strong", 675-925 is "moderate". At wind >=6 m/s, EPA
    # Table 6-7 maps strong->C and moderate->D -- the two bands genuinely diverge
    # here, so this is the boundary where a wrong cutoff is actually observable
    # (lower wind bands map strong and moderate to the same letter).
    u = 6.5
    strong = weather.pasquill_class(u, 950.0, 0.0)     # clearly >= 925
    moderate = weather.pasquill_class(u, 900.0, 0.0)   # 675 <= 900 < 925
    assert strong == 3      # C
    assert moderate == 4    # D
    # Old (pre-fix) 700/350 cutoffs would have called 900 W/m^2 "strong" too,
    # collapsing this distinction -- the divergence IS the regression guard.
    assert strong != moderate


def test_pasquill_near_neutral_below_175_is_always_class_d():
    # EPA's 4th daytime band (<175 W/m^2) is near-neutral -> class D regardless
    # of wind; this band was entirely absent before the fix.
    for u in (0.5, 3.0, 7.0):
        assert weather.pasquill_class(u, 100.0, 0.0) == 4   # D


def test_pasquill_slight_insolation_band_175_to_675():
    # EPA's "slight" daytime band (175-675 W/m^2) at a wind row where slight
    # differs from moderate (2-3 m/s: moderate->B, slight->C per Table 6-7).
    moderate = weather.pasquill_class(2.5, 700.0, 0.0)   # 675 <= 700 < 925: moderate
    slight = weather.pasquill_class(2.5, 300.0, 0.0)     # 175 <= 300 < 675: slight
    assert moderate == 2    # B
    assert slight == 3      # C


def test_pasquill_exact_boundary_values_are_inclusive():
    # EPA Table 6-7 bands are >=-inclusive at the lower edge: exactly 925 is
    # "strong", exactly 675 is "moderate", exactly 175 is "slight" (NOT the D
    # band). Exercised at wind rows where adjacent bands map to different
    # letters, so an off-by-one comparison is observable.
    assert weather.pasquill_class(6.5, 925.0, 0.0) == 3   # strong → C at ≥6 m/s
    assert weather.pasquill_class(2.5, 675.0, 0.0) == 2   # moderate → B at 2-3 m/s
    assert weather.pasquill_class(2.5, 175.0, 0.0) == 3   # slight → C, not D
    assert weather.pasquill_class(2.5, 174.9, 0.0) == 4   # just below → D


# ── height adjustment ────────────────────────────────────────────────────────
def test_height_adjust_lowers_wind_toward_ground():
    u10 = 5.0
    u2 = weather.adjust_wind_to_height(u10, z_target=2.0)
    assert u2 < u10
    # log-law ratio ln(2/0.1)/ln(10/0.1)
    ratio = np.log(2.0 / 0.1) / np.log(10.0 / 0.1)
    assert u2 == pytest.approx(u10 * ratio)


def test_height_adjust_identity_at_reference():
    assert weather.adjust_wind_to_height(4.0, z_target=10.0) == pytest.approx(4.0)


# ── summary ──────────────────────────────────────────────────────────────────
def test_summarize_archive_fixture(archive):
    est = weather.summarize_archive(archive, z_target=2.0)
    assert est.n_hours == 10
    assert 1 <= est.stability_class <= 6
    # 10 m median wind ≈ 3.35 m/s → height-adjusted to ~2.18 m/s, so well below.
    assert est.u < float(np.nanmedian(archive.u10))
    assert 200.0 < est.wind_dir_deg < 280.0


def test_summarize_empty_raises():
    empty = weather.ArchiveWind(times=[], u10=np.array([]), direction=np.array([]),
                                shortwave=np.array([]), cloud=np.array([]))
    with pytest.raises(ValueError):
        weather.summarize_archive(empty)


def test_summarize_archive_without_direction_does_not_crash():
    # F9: a degraded archive missing the direction column (parse_archive → None) must
    # still summarize (default direction 0°), NOT raise on ~np.isnan(None). The
    # offline-safe contract is "never crash on a degraded archive".
    no_dir = weather.ArchiveWind(
        times=["2026-06-10T12:00", "2026-06-10T13:00"],
        u10=np.array([3.0, 4.0]), direction=None,
        shortwave=np.array([800.0, 850.0]), cloud=np.array([10.0, 20.0]),
    )
    est = weather.summarize_archive(no_dir)
    assert est.wind_dir_deg == 0.0
    assert 1 <= est.stability_class <= 6
    assert est.n_hours == 2


# ── site resolution ──────────────────────────────────────────────────────────
def test_resolve_site_by_name():
    assert weather.resolve_site("Custer Road Transfer Station") == weather.SITES["custer"]
    assert weather.resolve_site("Melissa landfill") == weather.SITES["melissa"]


def test_resolve_site_tuple_passthrough():
    assert weather.resolve_site((33.0, -96.0)) == (33.0, -96.0)


def test_resolve_site_unknown_is_none():
    assert weather.resolve_site("Atlantis") is None
    assert weather.resolve_site(None) is None


# ── networked entry points (offline behaviour) ───────────────────────────────
def test_fetch_offline_returns_none_and_sets_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("no network in tests")
    monkeypatch.setattr(weather, "urlopen", boom)
    out = weather.fetch_wind_archive(33.1, -96.6, "2026-06-10", "2026-06-10")
    assert out is None
    assert "OSError" in (weather.LAST_WEATHER_ERROR or "")


def test_wind_for_site_date_unknown_site_is_none():
    out = weather.wind_for_site_date("Nowhere", "2026-06-10")
    assert out is None
    assert "unknown site" in (weather.LAST_WEATHER_ERROR or "")


def test_wind_for_site_date_uses_fetch(monkeypatch, archive):
    # Patch the network call to return our fixture archive; rest is real code.
    monkeypatch.setattr(weather, "fetch_wind_archive",
                        lambda *a, **k: archive)
    est = weather.wind_for_site_date("Custer Road Transfer Station", "2026-06-10")
    assert isinstance(est, weather.WindEstimate)
    assert est.n_hours == 10


def test_wind_for_site_date_no_date_is_none():
    assert weather.wind_for_site_date("Custer", "") is None
    assert "no date" in (weather.LAST_WEATHER_ERROR or "")

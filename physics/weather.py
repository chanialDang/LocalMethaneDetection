"""
weather.py — Real wind + stability from Open-Meteo, for grounding the inversion.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS EXISTS (plain English)
───────────────────────────────────────────────────────────────────────────────
The plume reads C ∝ Q / (σ_y σ_z u). Wind speed ``u`` and the stability class
(which sets σ_y, σ_z) are the two knobs we used to just GUESS. That guess is the
single worst source of error in recovering Q: across plausible wind (0.5–5 m/s)
and stability (classes B–F) the SAME fenceline signal implies Q over a ~160×
range (see feasibility.implied_Q_range). Pin u and the class with real data and
that swing collapses.

This module fetches the wind that was ACTUALLY blowing at a site, at the time a
reading was logged, from the Open-Meteo **historical archive** (free, no API key),
and derives a Pasquill stability class from the sun/cloud conditions. It is the
"Problem 2" half of the averaging + wind work — it makes the Week-3 inversion's
geometry honest instead of assumed.

DESIGN NOTES
───────────────────────────────────────────────────────────────────────────────
• Network is isolated to ONE function (``fetch_wind_archive``). Everything that
  interprets the data (``parse_archive``, ``summarize_archive``, ``pasquill_class``,
  ``adjust_wind_to_height``) is pure, so the tests run on a committed fixture with
  no network and CI stays hermetic.
• Offline-safe: any fetch failure returns None and records the reason in the
  module-level ``LAST_WEATHER_ERROR`` — mirroring explain.LAST_AI_ERROR — so callers
  fall back to their hardcoded defaults and the app never breaks.
• The 10 m archive wind is height-adjusted toward the near-ground release height
  with a log-law profile; using 10 m wind raw would overstate u and so understate Q.

⚠ VALIDATION CAVEAT (do not skip)
───────────────────────────────────────────────────────────────────────────────
The Pasquill insolation cutoffs below (radiation/cloud/wind → class) are the
*standard* Pasquill–Turner scheme, but the exact numeric thresholds vary by
reference. They are coded here as named constants and must be VERIFIED against a
cited source (e.g. Turner 1964 / the EPA dispersion workbook) before any result is
trusted. Treat the auto-class as a good first guess, not an authority — the
feasibility swing is still reported across neighbouring classes for this reason.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

# Set by fetch_wind_archive on failure; read by callers to explain a fallback.
LAST_WEATHER_ERROR: str | None = None

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The hourly variables we request. shortwave_radiation doubles as a day/night flag
# (>0 ⇒ daylight), which is all the Pasquill scheme needs.
_HOURLY_VARS = "wind_speed_10m,wind_direction_10m,cloud_cover,shortwave_radiation"

# Known deployment sites → (lat, lon). Approximate — VERIFY exact coordinates
# before a real run. Custer Road Transfer Station, Allen TX; Melissa landfill, TX.
SITES = {
    "custer": (33.10, -96.63),
    "melissa": (33.29, -96.57),
}

# ── Pasquill–Turner cutoffs (SEE VALIDATION CAVEAT — verify before trusting) ──
# Daytime incoming-solar bands (W/m²): strong / moderate / slight insolation.
_INSOL_STRONG = 700.0
_INSOL_MODERATE = 350.0
_INSOL_SLIGHT = 1.0          # >0 but weak; below this we treat it as night
# Night cloud-cover split (%): "overcast/cloudy" vs "mostly clear".
_CLOUD_OVERCAST = 50.0
# Roughness length (m) for the log-law height adjustment. ~0.1 m = open country;
# a structure-cluttered transfer-station fenceline is rougher (use 0.3–1.0 there).
DEFAULT_Z0 = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — no network, directly unit-tested
# ─────────────────────────────────────────────────────────────────────────────
def adjust_wind_to_height(u_ref: float, z_target: float = 2.0,
                          z_ref: float = 10.0, z0: float = DEFAULT_Z0) -> float:
    """
    Scale a wind speed from a reference height to a target height (log-law).

    The neutral logarithmic wind profile is u(z) = u* / κ · ln(z / z0), so the
    ratio between two heights is just ln(z_target/z0) / ln(z_ref/z0). Open-Meteo
    reports wind at 10 m; the methane release is near the ground (~2 m), where the
    wind is slower — using 10 m wind raw would inflate u and deflate the recovered Q.

    z_target is floored just above z0 so the logarithm stays positive.
    """
    z_target = max(z_target, z0 * math.e)   # keep ln(z_target/z0) ≥ 1
    return float(u_ref * math.log(z_target / z0) / math.log(z_ref / z0))


def pasquill_class(wind_speed: float, shortwave_radiation: float,
                   cloud_cover: float) -> int:
    """
    Map surface wind + sun/cloud to a Pasquill stability class as int 1–6 (A–F).

    Daytime (shortwave_radiation > slight threshold) keys off insolation strength;
    night keys off cloud cover. Stronger sun + lighter wind → more unstable (lower
    class); clear calm night → more stable (higher class). Half-classes (e.g. "A–B")
    are resolved to the more UNSTABLE integer — the more dispersive, larger-implied-Q
    side — which is the conservative choice when reporting Q as an upper bound.

    SEE the module VALIDATION CAVEAT: the cutoff constants must be checked against a
    cited reference. Returns 1=A (most unstable) … 6=F (most stable).
    """
    u = float(wind_speed)
    is_day = float(shortwave_radiation) > _INSOL_SLIGHT

    if is_day:
        rad = float(shortwave_radiation)
        if rad >= _INSOL_STRONG:
            insol = "strong"
        elif rad >= _INSOL_MODERATE:
            insol = "moderate"
        else:
            insol = "slight"
        # rows by wind band; values are the (unstable-rounded) class A–F.
        if u < 2.0:
            cls = {"strong": "A", "moderate": "A", "slight": "B"}[insol]
        elif u < 3.0:
            cls = {"strong": "A", "moderate": "B", "slight": "C"}[insol]
        elif u < 5.0:
            cls = {"strong": "B", "moderate": "B", "slight": "C"}[insol]
        elif u < 6.0:
            cls = {"strong": "C", "moderate": "C", "slight": "D"}[insol]
        else:
            cls = {"strong": "C", "moderate": "D", "slight": "D"}[insol]
    else:
        overcast = float(cloud_cover) >= _CLOUD_OVERCAST
        if u < 3.0:
            cls = "E" if overcast else "F"
        elif u < 5.0:
            cls = "D" if overcast else "E"
        else:
            cls = "D"

    return "ABCDEF".index(cls) + 1


def _circular_mean_deg(degrees) -> float:
    """Mean of wind directions (degrees), done on the circle so 350° & 10° → 0°."""
    rad = np.radians(np.asarray(degrees, dtype=float))
    ang = math.atan2(float(np.mean(np.sin(rad))), float(np.mean(np.cos(rad))))
    return float(np.degrees(ang) % 360.0)


# ─────────────────────────────────────────────────────────────────────────────
# Archive parsing + summary — pure, fixture-tested
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ArchiveWind:
    """Hourly wind/sun arrays from one Open-Meteo archive response."""
    times: list                # ISO hour strings, e.g. "2026-06-10T14:00"
    u10: np.ndarray            # wind speed at 10 m (m/s)
    direction: np.ndarray      # meteorological from-direction (° from N)
    shortwave: np.ndarray      # incoming solar (W/m²) — also the day/night flag
    cloud: np.ndarray          # cloud cover (%)

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class WindEstimate:
    """A single representative wind+stability for a record (median over the hours)."""
    u: float                   # wind speed (m/s) at the target height
    wind_dir_deg: float        # meteorological from-direction (° from N)
    stability_class: int       # Pasquill 1–6 (A–F)
    n_hours: int               # hours that fed the estimate
    source: str                # human-readable provenance


def parse_archive(payload: dict) -> ArchiveWind:
    """
    Turn an Open-Meteo archive JSON payload into aligned numpy arrays.

    Pure (no network) so it is unit-tested against a committed fixture. Raises
    ValueError if the expected ``hourly`` block is missing.
    """
    hourly = (payload or {}).get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response had no 'hourly' block.")

    def arr(key):
        vals = hourly.get(key)
        if vals is None:
            return None
        # Open-Meteo uses null for gaps; coerce to NaN so nanmedian skips them.
        return np.array([np.nan if v is None else v for v in vals], dtype=float)

    return ArchiveWind(
        times=list(hourly["time"]),
        u10=arr("wind_speed_10m"),
        direction=arr("wind_direction_10m"),
        shortwave=arr("shortwave_radiation"),
        cloud=arr("cloud_cover"),
    )


def summarize_archive(archive: ArchiveWind, z_target: float = 2.0,
                      z0: float = DEFAULT_Z0) -> WindEstimate:
    """
    Reduce an hourly archive to one representative (u, direction, stability).

    Uses robust medians (nan-aware) over the hours so a single bad hour can't skew
    the result, height-adjusts the median wind to ``z_target``, and derives the
    stability class from the median sun/cloud/wind. This is what fills a field
    test's wind metadata and seeds the inversion's geometry.
    """
    if len(archive) == 0:
        raise ValueError("Empty archive — nothing to summarize.")

    u10 = float(np.nanmedian(archive.u10))
    u = adjust_wind_to_height(u10, z_target=z_target, z0=z0)
    direction = _circular_mean_deg(
        archive.direction[~np.isnan(archive.direction)]
    )
    rad = float(np.nanmedian(archive.shortwave))
    cloud = float(np.nanmedian(archive.cloud)) if archive.cloud is not None else 0.0
    cls = pasquill_class(u10, rad, cloud)   # class from the 10 m wind (as tabulated)

    return WindEstimate(
        u=u, wind_dir_deg=direction, stability_class=cls,
        n_hours=len(archive),
        source=f"Open-Meteo archive, {len(archive)} h, u(10m)={u10:.1f}→u({z_target:g}m)={u:.1f} m/s",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Site resolution + the one networked entry point
# ─────────────────────────────────────────────────────────────────────────────
def resolve_site(site) -> tuple | None:
    """
    Map a site name (or an explicit (lat, lon)) to coordinates, else None.

    Accepts a (lat, lon) tuple straight through, or fuzzy-matches a name against
    the known SITES keys ("custer", "melissa"). Returns None when unrecognised so
    the caller can skip the weather lookup rather than guess a location.
    """
    if isinstance(site, (tuple, list)) and len(site) == 2:
        try:
            return (float(site[0]), float(site[1]))
        except (TypeError, ValueError):
            return None
    if not site:
        return None
    name = str(site).strip().lower()
    for key, coords in SITES.items():
        if key in name:
            return coords
    return None


def fetch_wind_archive(lat: float, lon: float, start_date: str, end_date: str,
                       timeout: float = 8.0) -> ArchiveWind | None:
    """
    Fetch hourly wind/sun from the Open-Meteo archive (the ONLY networked call).

    Dates are "YYYY-MM-DD". Returns an ArchiveWind, or None on any failure (no
    network, bad dates, malformed response) with the reason left in
    LAST_WEATHER_ERROR. No API key is needed. ``wind_speed_unit=ms`` is requested
    so speeds come back in m/s (the API default is km/h).
    """
    global LAST_WEATHER_ERROR
    LAST_WEATHER_ERROR = None
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": _HOURLY_VARS, "wind_speed_unit": "ms", "timezone": "auto",
    }
    url = f"{ARCHIVE_URL}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=timeout) as resp:          # noqa: S310 (fixed host)
            payload = json.loads(resp.read().decode("utf-8"))
        return parse_archive(payload)
    except Exception as exc:                                  # network / parse / shape
        LAST_WEATHER_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def wind_for_site_date(site, start_date: str, end_date: str | None = None,
                       z_target: float = 2.0) -> WindEstimate | None:
    """
    Convenience: resolve a site, fetch its archive for the date(s), summarize.

    Returns a WindEstimate, or None if the site is unrecognised or the fetch fails
    (reason in LAST_WEATHER_ERROR). ``end_date`` defaults to ``start_date`` (one
    day). Offline-safe — callers should fall back to their defaults on None.
    """
    global LAST_WEATHER_ERROR
    coords = resolve_site(site)
    if coords is None:
        LAST_WEATHER_ERROR = f"unknown site {site!r} (no coordinates)"
        return None
    if not start_date:
        LAST_WEATHER_ERROR = "no date supplied for the archive lookup"
        return None
    archive = fetch_wind_archive(coords[0], coords[1], start_date,
                                 end_date or start_date)
    if archive is None or len(archive) == 0:
        return None
    return summarize_archive(archive, z_target=z_target)

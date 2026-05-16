"""Climate ingestion and screening weather generation.

The app can run with no external services by generating a deterministic typical
hourly year from location/state defaults. When internet access is available,
NASA POWER can be used for gridded hourly meteorology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import requests

from .state_data import ABBR_TO_STATE, STATE_DEFAULTS, get_state_defaults

DataMode = Literal["Typical screening year", "NASA POWER hourly year", "NOAA USCRN station year", "NASA + NOAA tuned validation year"]


@dataclass(frozen=True)
class Site:
    state: str
    latitude: float
    longitude: float
    label: str
    zip_code: str | None = None
    source: str = "State defaults"
    mean_air_c: float = 10.0
    annual_amp_c: float = 13.0
    diurnal_amp_c: float = 6.0
    region_factor: float = 1.0

    @property
    def annual_ground_mean_c(self) -> float:
        # Shallow ground is commonly close to annual mean air temperature but is
        # slightly warmer beneath paved dark surfaces.
        return float(self.mean_air_c + 1.5)


def resolve_site(state: str, zip_code: str | None = None) -> Site:
    """Resolve a state/ZIP into a usable site object.

    ZIP lookup uses pgeocode when available. If the lookup fails, the function
    falls back gracefully to state centroid defaults.
    """
    defaults = get_state_defaults(state)
    resolved_state = state if state in STATE_DEFAULTS else ABBR_TO_STATE.get(state.upper(), "Illinois")
    lat = float(defaults["lat"])
    lon = float(defaults["lon"])
    label = resolved_state
    source = "State centroid defaults"

    clean_zip = (zip_code or "").strip()
    if clean_zip:
        try:
            import pgeocode  # type: ignore

            nomi = pgeocode.Nominatim("US")
            rec = nomi.query_postal_code(clean_zip)
            if rec is not None and not pd.isna(rec.latitude) and not pd.isna(rec.longitude):
                lat = float(rec.latitude)
                lon = float(rec.longitude)
                if isinstance(rec.state_name, str) and rec.state_name in STATE_DEFAULTS:
                    resolved_state = rec.state_name
                    defaults = get_state_defaults(resolved_state)
                label = f"ZIP {clean_zip} ({resolved_state})"
                source = "ZIP centroid via pgeocode"
        except Exception:
            label = f"{resolved_state} fallback for ZIP {clean_zip}"
            source = "State fallback; ZIP lookup unavailable"

    # Blend defaults with a latitude correction so ZIP-level centroid changes
    # matter even when only state climate defaults are available.
    lat_delta = lat - float(defaults["lat"])
    mean_air_c = float(defaults["mean_c"]) - 0.13 * lat_delta
    annual_amp_c = max(3.0, float(defaults["amp_c"]) + 0.04 * abs(lat_delta))
    diurnal_amp_c = float(defaults["diurnal_c"])
    return Site(
        state=resolved_state,
        latitude=lat,
        longitude=lon,
        label=label,
        zip_code=clean_zip or None,
        source=source,
        mean_air_c=mean_air_c,
        annual_amp_c=annual_amp_c,
        diurnal_amp_c=diurnal_amp_c,
        region_factor=float(defaults["region_factor"]),
    )


def _rng_for_site(site: Site, year: int) -> np.random.Generator:
    seed = int(abs(site.latitude * 1000) + abs(site.longitude * 1000) * 7 + year * 13) % (2**32 - 1)
    return np.random.default_rng(seed)


def _daily_ar1_noise(n_days: int, rng: np.random.Generator, sigma: float = 3.2, rho: float = 0.83) -> np.ndarray:
    noise = np.zeros(n_days)
    innovations = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n_days)
    for i in range(1, n_days):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def _solar_geometry(lat_deg: float, doy: np.ndarray, hour: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    decl = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    # Approximate local solar time. Screening mode does not model equation of time/time zone.
    hour_angle = np.radians(15.0 * (hour + 0.5 - 12.0))
    sin_alt = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(sin_alt, 0.0, None)


def generate_typical_hourly_weather(site: Site, year: int = 2025) -> pd.DataFrame:
    """Generate a deterministic, plausible hourly screening climate year.

    This is not a replacement for station data. It exists so the sales app always
    loads and lets a visitor explore the product before providing an exact ZIP or
    API credentials.
    """
    idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    rng = _rng_for_site(site, year)

    n_days = int(idx.dayofyear.max())
    daily_noise = _daily_ar1_noise(n_days, rng)
    noise_hourly = daily_noise[doy - 1] + rng.normal(0, 0.45, n)

    seasonal = site.mean_air_c - site.annual_amp_c * np.cos(2 * np.pi * (doy - 15) / 365.0)
    diurnal_strength = site.diurnal_amp_c * (0.75 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365.0))
    diurnal = diurnal_strength * np.sin(2 * np.pi * (hour - 15) / 24.0)
    air_c = seasonal + diurnal + noise_hourly

    sin_alt = _solar_geometry(site.latitude, doy, hour)
    cloud_daily = np.clip(0.72 + 0.22 * rng.normal(size=n_days), 0.25, 1.05)
    # Cloudiness is more persistent in winter for northern maritime locations.
    winter_cloud_penalty = 0.08 * np.cos(2 * np.pi * (doy - 15) / 365.0) * np.clip((site.latitude - 35) / 15, 0, 1)
    cloud = np.clip(cloud_daily[doy - 1] - winter_cloud_penalty, 0.20, 1.05)
    ghi = 980.0 * np.power(sin_alt, 1.18) * cloud

    rh_base = 67.0 - 0.9 * (air_c - site.mean_air_c) + 8.0 * (1.0 - cloud)
    rh = np.clip(rh_base + rng.normal(0, 8, n), 18, 100)

    wind_daily = np.clip(rng.lognormal(mean=np.log(2.7), sigma=0.35, size=n_days), 0.4, 10)
    wind = np.clip(wind_daily[doy - 1] + rng.normal(0, 0.45, n), 0.2, 14)

    # Screening precipitation/wetness. Calibrated only for risk timing, not hydrology.
    cold_wet_bias = np.clip((10.0 - air_c) / 20.0, 0, 1)
    event_probability = np.clip(0.018 + 0.035 * cold_wet_bias + 0.015 * (rh > 88), 0, 0.12)
    event = rng.random(n) < event_probability
    precip = np.where(event, rng.gamma(shape=1.2, scale=1.0, size=n), 0.0)
    wet = (precip > 0.05) | ((rh > 92) & (air_c < 3.0))

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "air_temp_C": air_c.astype(float),
            "ghi_W_m2": ghi.astype(float),
            "wind_m_s": wind.astype(float),
            "rh_pct": rh.astype(float),
            "precip_mm": precip.astype(float),
            "wet_flag": wet.astype(bool),
            "data_source": "Typical screening year",
        }
    ).set_index("timestamp")
    return df


def fetch_nasa_power_hourly(site: Site, year: int = 2024, timeout: int = 30) -> pd.DataFrame:
    """Fetch hourly NASA POWER data for a single year.

    Falls back to the screening generator if the API is unreachable or returns an
    unexpected payload.
    """
    start = f"{year}0101"
    end = f"{year}1231"
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": f"{site.longitude:.4f}",
        "latitude": f"{site.latitude:.4f}",
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        params_block = payload["properties"]["parameter"]
        temp = params_block["T2M"]
        keys = sorted(temp.keys())
        idx = pd.to_datetime(keys, format="%Y%m%d%H")

        def arr(name: str, default: float = np.nan) -> np.ndarray:
            block = params_block.get(name, {})
            return np.array([float(block.get(k, default)) for k in keys], dtype=float)

        ghi = arr("ALLSKY_SFC_SW_DWN", 0.0)
        # POWER radiation unit conventions can vary by endpoint/version. Keep the
        # values in physically plausible W/m2 for the surface energy model.
        if np.nanpercentile(ghi, 95) < 20:
            ghi = ghi * 1000.0
        if np.nanpercentile(ghi, 99) > 1300:
            ghi = ghi / np.nanmax([np.nanpercentile(ghi, 99) / 1050.0, 1.0])

        rh = np.clip(arr("RH2M", 65), 0, 100)
        precip = np.clip(arr("PRECTOTCORR", 0), 0, None)
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "air_temp_C": arr("T2M", site.mean_air_c),
                "ghi_W_m2": np.clip(ghi, 0, 1200),
                "wind_m_s": np.clip(arr("WS2M", 2.5), 0.1, 35),
                "rh_pct": rh,
                "precip_mm": precip,
                "wet_flag": (precip > 0.05) | ((rh > 92) & (arr("T2M", site.mean_air_c) < 3.0)),
                "data_source": "NASA POWER hourly",
            }
        ).set_index("timestamp")
        # Drop leap day for a consistent 8760-hour sales/optimization workflow.
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        if len(df) < 8700:
            raise ValueError("NASA POWER response contained too few hourly records")
        return df
    except Exception:
        return generate_typical_hourly_weather(site, year=2025)


def load_weather(site: Site, mode: DataMode, year: int = 2024) -> pd.DataFrame:
    if mode in {"NASA POWER hourly year", "NASA + NOAA tuned validation year"}:
        return fetch_nasa_power_hourly(site, year=year)
    if mode == "NOAA USCRN station year":
        try:
            from .validation import fetch_noaa_uscrn_hourly

            return fetch_noaa_uscrn_hourly(site, year=year)
        except Exception:
            return generate_typical_hourly_weather(site, year=2025)
    return generate_typical_hourly_weather(site, year=2025)

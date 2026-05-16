"""NASA/NOAA validation and model-tuning helpers.

The production app uses NASA POWER as a no-key, gridded weather source for any
U.S. location and NOAA USCRN as the observed validation backbone where a nearby
station exists. Tuning is deliberately lightweight: it calibrates a few high-
leverage screening parameters, not a site-specific engineering model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from io import StringIO
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .climate import Site
from .physics import SimulationConfig, SimulationResult, pavement_record, run_thermal_simulation
from .soil import SoilProfile
from .state_data import STATE_DEFAULTS

NOAA_USCRN_BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
USCRN_COLUMNS = [
    "WBANNO",
    "UTC_DATE",
    "UTC_TIME",
    "LST_DATE",
    "LST_TIME",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "T_CALC",
    "T_HR_AVG",
    "T_MAX",
    "T_MIN",
    "P_CALC",
    "SOLARAD",
    "SOLARAD_FLAG",
    "SOLARAD_MAX",
    "SOLARAD_MAX_FLAG",
    "SOLARAD_MIN",
    "SOLARAD_MIN_FLAG",
    "SUR_TEMP_TYPE",
    "SUR_TEMP",
    "SUR_TEMP_FLAG",
    "SUR_TEMP_MAX",
    "SUR_TEMP_MAX_FLAG",
    "SUR_TEMP_MIN",
    "SUR_TEMP_MIN_FLAG",
    "RH_HR_AVG",
    "RH_HR_AVG_FLAG",
    "SOIL_MOISTURE_5",
    "SOIL_MOISTURE_10",
    "SOIL_MOISTURE_20",
    "SOIL_MOISTURE_50",
    "SOIL_MOISTURE_100",
    "SOIL_TEMP_5",
    "SOIL_TEMP_10",
    "SOIL_TEMP_20",
    "SOIL_TEMP_50",
    "SOIL_TEMP_100",
]


@dataclass(frozen=True)
class NOAAStation:
    filename: str
    station_label: str
    latitude: float
    longitude: float
    distance_km: float


@dataclass
class ValidationScore:
    surface_rmse_C: float | None
    soil_rmse_mean_C: float | None
    soil_20cm_rmse_C: float | None
    n_surface: int
    n_soil: int
    composite_score: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "surface_rmse_C": self.surface_rmse_C,
            "soil_rmse_mean_C": self.soil_rmse_mean_C,
            "soil_20cm_rmse_C": self.soil_20cm_rmse_C,
            "n_surface": self.n_surface,
            "n_soil": self.n_soil,
            "composite_score": self.composite_score,
        }


@dataclass
class TuningResult:
    status: str
    applied: bool
    tuned_config: SimulationConfig
    tuned_soil: SoilProfile
    validation_weather: pd.DataFrame
    validation_result: SimulationResult | None
    trials: pd.DataFrame
    score: ValidationScore | None
    station: NOAAStation | None
    note: str


def _state_abbr(state: str) -> str:
    if state in STATE_DEFAULTS:
        return str(STATE_DEFAULTS[state]["abbr"])
    return state.upper()[:2]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _clean_station_name(filename: str, year: int) -> str:
    name = filename.replace(f"CRNH0203-{year}-", "").replace(".txt", "")
    return name.replace("_", " ")


def _list_uscrn_files(year: int, timeout: int = 20) -> list[str]:
    url = f"{NOAA_USCRN_BASE}/{year}/"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return sorted(set(re.findall(r'href="(CRNH0203-' + str(year) + r'-[^"]+\.txt)"', resp.text)))


def _station_from_first_line(filename: str, year: int, site: Site, timeout: int = 12) -> NOAAStation | None:
    url = f"{NOAA_USCRN_BASE}/{year}/{filename}"
    # The server honors range requests in most environments. If it does not, the
    # full 2-MB annual file is still small enough for an occasional lookup.
    resp = requests.get(url, headers={"Range": "bytes=0-600"}, timeout=timeout)
    resp.raise_for_status()
    line = resp.text.splitlines()[0] if resp.text.splitlines() else ""
    parts = line.split()
    if len(parts) < 8:
        return None
    lon = float(parts[6])
    lat = float(parts[7])
    return NOAAStation(
        filename=filename,
        station_label=_clean_station_name(filename, year),
        latitude=lat,
        longitude=lon,
        distance_km=_haversine_km(site.latitude, site.longitude, lat, lon),
    )


def find_nearest_uscrn_station(site: Site, year: int = 2024, timeout: int = 20, max_candidates: int = 28) -> NOAAStation | None:
    """Find the nearest USCRN station in the same state when possible.

    The search is state-first to limit network traffic. If the selected state has
    no station in the annual directory, a limited national search is attempted.
    """
    files = _list_uscrn_files(year, timeout=timeout)
    abbr = _state_abbr(site.state)
    state_files = [f for f in files if f"-{abbr}_" in f]
    candidate_files = state_files[:max_candidates]
    if not candidate_files:
        # Limited national fallback biased to files whose state code appears near
        # the middle of the list. This avoids downloading hundreds of headers.
        step = max(1, len(files) // max_candidates)
        candidate_files = files[::step][:max_candidates]

    stations: list[NOAAStation] = []
    for filename in candidate_files:
        try:
            station = _station_from_first_line(filename, year=year, site=site, timeout=timeout)
            if station is not None:
                stations.append(station)
        except Exception:
            continue
    if not stations:
        return None
    return min(stations, key=lambda s: s.distance_km)


def _replace_missing(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([-9999, -9999.0, -999.0, -99.0, 9999.0], np.nan)


def _wind_proxy(index: pd.DatetimeIndex, site: Site) -> np.ndarray:
    # USCRN hourly02 omits wind. This deterministic proxy keeps the surface
    # boundary usable for validation without pretending it is observed wind.
    doy = index.dayofyear.to_numpy()
    winter_boost = 0.35 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    regional = 0.15 * np.clip((abs(site.longitude) - 85.0) / 30.0, -1.0, 1.0)
    return np.clip(2.4 + winter_boost + regional, 0.6, 5.0)


def fetch_noaa_uscrn_hourly(site: Site, year: int = 2024, timeout: int = 25, station: NOAAStation | None = None) -> pd.DataFrame:
    """Fetch a nearest-station NOAA USCRN hourly02 validation dataframe.

    Output columns match the app weather schema and include observed USCRN
    surface/soil temperatures for validation overlays.
    """
    station = station or find_nearest_uscrn_station(site, year=year, timeout=timeout)
    if station is None:
        raise RuntimeError("No NOAA USCRN station could be resolved for this site/year")

    url = f"{NOAA_USCRN_BASE}/{year}/{station.filename}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(
        StringIO(resp.text),
        sep=r"\s+",
        names=USCRN_COLUMNS,
        engine="python",
        dtype={"SUR_TEMP_TYPE": str},
    )
    # Build local-standard-time index. USCRN timestamps mark the hour ending.
    lst_date = raw["LST_DATE"].astype(str).str.zfill(8)
    lst_time = raw["LST_TIME"].astype(str).str.zfill(4)
    idx = pd.to_datetime(lst_date + lst_time, format="%Y%m%d%H%M", errors="coerce")
    raw.index = idx
    raw = raw[~raw.index.isna()].sort_index()
    raw = raw[~((raw.index.month == 2) & (raw.index.day == 29))]

    air = _replace_missing(raw["T_HR_AVG"]).fillna(_replace_missing(raw["T_CALC"]))
    rh = _replace_missing(raw["RH_HR_AVG"]).clip(0, 100)
    precip = _replace_missing(raw["P_CALC"]).clip(lower=0).fillna(0.0)
    ghi = _replace_missing(raw["SOLARAD"]).clip(lower=0, upper=1300)
    surface = _replace_missing(raw["SUR_TEMP"])

    df = pd.DataFrame(index=raw.index)
    df["air_temp_C"] = air.interpolate(limit_direction="both").to_numpy(dtype=float)
    df["ghi_W_m2"] = ghi.fillna(0.0).to_numpy(dtype=float)
    df["wind_m_s"] = _wind_proxy(df.index, site)
    df["rh_pct"] = rh.interpolate(limit_direction="both").fillna(65).to_numpy(dtype=float)
    df["precip_mm"] = precip.to_numpy(dtype=float)
    df["wet_flag"] = (df["precip_mm"] > 0.05) | ((df["rh_pct"] > 92) & (df["air_temp_C"] < 3.0))
    df["observed_surface_C"] = surface.to_numpy(dtype=float)
    for depth_cm in [5, 10, 20, 50, 100]:
        src = _replace_missing(raw[f"SOIL_TEMP_{depth_cm}"])
        df[f"observed_soil_{depth_cm}cm_C"] = src.to_numpy(dtype=float)
    for depth_cm in [5, 10, 20, 50, 100]:
        src = _replace_missing(raw[f"SOIL_MOISTURE_{depth_cm}"])
        df[f"observed_soil_moisture_{depth_cm}cm"] = src.to_numpy(dtype=float)
    df["data_source"] = f"NOAA USCRN hourly ({station.station_label})"
    df.attrs["station"] = station.__dict__
    df.attrs["source_url"] = url
    return df


def interpolate_model_depth(result: SimulationResult, depth_m: float) -> pd.Series:
    values = np.array([np.interp(depth_m, result.depth_m, row) for row in result.temperature_C], dtype=float)
    return pd.Series(values, index=result.time_series.index, name=f"model_{depth_m:.2f}m_C")


def _rmse(obs: pd.Series, pred: pd.Series) -> tuple[float | None, int]:
    aligned = pd.concat([obs.astype(float), pred.astype(float)], axis=1, join="inner").dropna()
    if aligned.empty:
        return None, 0
    arr = aligned.iloc[:, 0].to_numpy(dtype=float)
    hat = aligned.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(arr) & np.isfinite(hat) & (arr > -80) & (arr < 80)
    if mask.sum() == 0:
        return None, 0
    return float(np.sqrt(np.mean((hat[mask] - arr[mask]) ** 2))), int(mask.sum())


def score_simulation_against_noaa(result: SimulationResult, noaa_weather: pd.DataFrame) -> ValidationScore:
    """Score modeled surface/soil temperature against NOAA USCRN observations."""
    surface_rmse, n_surface = _rmse(noaa_weather.get("observed_surface_C", pd.Series(dtype=float)), result.surface_C)
    soil_rmses: list[float] = []
    n_soil_total = 0
    soil_20_rmse: float | None = None
    for depth_cm in [5, 10, 20, 50, 100]:
        col = f"observed_soil_{depth_cm}cm_C"
        if col not in noaa_weather:
            continue
        pred = interpolate_model_depth(result, depth_cm / 100.0)
        rmse_val, n_val = _rmse(noaa_weather[col], pred)
        if rmse_val is not None:
            soil_rmses.append(rmse_val)
            n_soil_total += n_val
            if depth_cm == 20:
                soil_20_rmse = rmse_val
    soil_mean = float(np.mean(soil_rmses)) if soil_rmses else None
    # Composite score rewards matching surface skin temperature, but keeps soil
    # dynamics in the fit because thermosyphon performance depends on soil heat.
    components = []
    if surface_rmse is not None:
        components.append(0.55 * surface_rmse)
    if soil_20_rmse is not None:
        components.append(0.20 * soil_20_rmse)
    if soil_mean is not None:
        components.append(0.25 * soil_mean)
    score = float(np.sum(components)) if components else float("inf")
    return ValidationScore(surface_rmse, soil_mean, soil_20_rmse, n_surface, n_soil_total, score)


def _trial_grid() -> list[dict[str, float]]:
    # Small Latin-hypercube-like grid so calibration remains responsive on
    # Streamlit Cloud. Values are multipliers/offsets, not final properties.
    return [
        {"albedo_factor": 1.00, "soil_k_factor": 1.00, "ground_offset_C": 0.0, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 0.82, "soil_k_factor": 1.00, "ground_offset_C": 0.0, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 1.18, "soil_k_factor": 1.00, "ground_offset_C": 0.0, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 1.00, "soil_k_factor": 0.75, "ground_offset_C": 0.0, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 1.00, "soil_k_factor": 1.30, "ground_offset_C": 0.0, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 1.00, "soil_k_factor": 1.00, "ground_offset_C": -1.4, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 1.00, "soil_k_factor": 1.00, "ground_offset_C": 1.4, "convection_multiplier": 1.00, "sky_offset_C": 0.0},
        {"albedo_factor": 0.88, "soil_k_factor": 0.82, "ground_offset_C": -0.8, "convection_multiplier": 1.12, "sky_offset_C": -0.5},
        {"albedo_factor": 0.88, "soil_k_factor": 1.22, "ground_offset_C": 0.8, "convection_multiplier": 0.92, "sky_offset_C": 0.5},
        {"albedo_factor": 1.12, "soil_k_factor": 0.82, "ground_offset_C": -0.8, "convection_multiplier": 0.92, "sky_offset_C": -0.5},
        {"albedo_factor": 1.12, "soil_k_factor": 1.22, "ground_offset_C": 0.8, "convection_multiplier": 1.12, "sky_offset_C": 0.5},
        {"albedo_factor": 0.96, "soil_k_factor": 1.10, "ground_offset_C": 0.4, "convection_multiplier": 1.04, "sky_offset_C": 0.2},
    ]


def _config_for_trial(config: SimulationConfig, trial: dict[str, float]) -> SimulationConfig:
    pav = pavement_record(config.pavement_type)
    base_albedo = float(config.custom_albedo if config.custom_albedo is not None else pav["albedo"])
    tuned_albedo = float(np.clip(base_albedo * trial["albedo_factor"], 0.04, 0.70))
    return replace(
        config,
        custom_albedo=tuned_albedo,
        convection_multiplier=float(config.convection_multiplier * trial["convection_multiplier"]),
        sky_temperature_offset_C=float(config.sky_temperature_offset_C + trial["sky_offset_C"]),
        ground_mean_offset_C=float(config.ground_mean_offset_C + trial["ground_offset_C"]),
        spinup_years=min(max(config.spinup_years, 0), 1),
    )


def _soil_for_trial(soil: SoilProfile, trial: dict[str, float]) -> SoilProfile:
    factor = trial["soil_k_factor"]
    return replace(
        soil,
        name=f"{soil.name} (NOAA tuned)",
        k_W_mK=float(np.clip(soil.k_W_mK * factor, 0.35, 2.40)),
        source=f"{soil.source}; NOAA USCRN calibration factor {factor:.2f}",
    )


def tune_model_against_noaa_uscrn(
    site: Site,
    soil: SoilProfile,
    config: SimulationConfig,
    year: int = 2024,
    max_trials: int = 12,
    timeout: int = 25,
) -> TuningResult:
    """Tune screening parameters against nearest NOAA USCRN station data."""
    try:
        noaa = fetch_noaa_uscrn_hourly(site, year=year, timeout=timeout)
        station_dict = noaa.attrs.get("station", {})
        station = NOAAStation(**station_dict) if station_dict else None
    except Exception as exc:
        return TuningResult(
            status="NOAA validation unavailable",
            applied=False,
            tuned_config=config,
            tuned_soil=soil,
            validation_weather=pd.DataFrame(),
            validation_result=None,
            trials=pd.DataFrame(),
            score=None,
            station=None,
            note=f"Validation/tuning could not load NOAA USCRN data: {exc}",
        )

    rows: list[dict[str, float | int | str | None]] = []
    best_tuple: tuple[float, SimulationConfig, SoilProfile, SimulationResult, ValidationScore, dict[str, float]] | None = None
    for trial in _trial_grid()[:max_trials]:
        trial_config = _config_for_trial(config, trial)
        trial_soil = _soil_for_trial(soil, trial)
        try:
            result = run_thermal_simulation(noaa, site, trial_soil, trial_config, thermosyphon=None)
            score = score_simulation_against_noaa(result, noaa)
            score_value = score.composite_score
        except Exception:
            continue
        row = dict(trial)
        row.update(score.as_dict())
        row["custom_albedo"] = trial_config.custom_albedo
        row["soil_k_W_mK"] = trial_soil.k_W_mK
        rows.append(row)
        if np.isfinite(score_value) and (best_tuple is None or score_value < best_tuple[0]):
            best_tuple = (score_value, trial_config, trial_soil, result, score, trial)

    trials = pd.DataFrame(rows).sort_values("composite_score", ascending=True).reset_index(drop=True) if rows else pd.DataFrame()
    if best_tuple is None:
        return TuningResult(
            status="NOAA validation loaded, but tuning failed",
            applied=False,
            tuned_config=config,
            tuned_soil=soil,
            validation_weather=noaa,
            validation_result=None,
            trials=trials,
            score=None,
            station=station,
            note="No valid NOAA surface/soil observations were available for scoring.",
        )

    _, tuned_config, tuned_soil, result, score, best_trial = best_tuple
    station_note = f"Nearest NOAA USCRN station: {station.station_label}, {station.distance_km:.0f} km away." if station else "NOAA USCRN station selected."
    note = (
        f"NASA/NOAA tuning applied. {station_note} "
        f"Best screening fit used albedo factor {best_trial['albedo_factor']:.2f}, "
        f"soil-k factor {best_trial['soil_k_factor']:.2f}, and ground offset {best_trial['ground_offset_C']:+.1f} C."
    )
    return TuningResult(
        status="NASA/NOAA tuned",
        applied=True,
        tuned_config=tuned_config,
        tuned_soil=tuned_soil,
        validation_weather=noaa,
        validation_result=result,
        trials=trials,
        score=score,
        station=station,
        note=note,
    )


def validation_summary_table(tuning: TuningResult | None) -> pd.DataFrame:
    if tuning is None:
        return pd.DataFrame()
    rows: list[tuple[str, str]] = [("Status", tuning.status), ("Applied", "Yes" if tuning.applied else "No")]
    if tuning.station is not None:
        rows.extend(
            [
                ("NOAA station", tuning.station.station_label),
                ("Station distance", f"{tuning.station.distance_km:.0f} km"),
                ("Station coordinates", f"{tuning.station.latitude:.3f}, {tuning.station.longitude:.3f}"),
            ]
        )
    if tuning.score is not None:
        rows.extend(
            [
                ("Surface RMSE", f"{tuning.score.surface_rmse_C:.2f} C" if tuning.score.surface_rmse_C is not None else "n/a"),
                ("Mean soil RMSE", f"{tuning.score.soil_rmse_mean_C:.2f} C" if tuning.score.soil_rmse_mean_C is not None else "n/a"),
                ("20-cm soil RMSE", f"{tuning.score.soil_20cm_rmse_C:.2f} C" if tuning.score.soil_20cm_rmse_C is not None else "n/a"),
                ("Composite score", f"{tuning.score.composite_score:.2f}"),
            ]
        )
    rows.append(("Note", tuning.note))
    return pd.DataFrame(rows, columns=["Item", "Value"])

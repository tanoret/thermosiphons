"""Finite-difference driveway/soil thermal model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .climate import Site
from .soil import SoilProfile
from .thermosyphon import ThermosyphonDesign, thermosyphon_flux_per_area

PavementType = Literal["Concrete", "Asphalt", "Permeable pavers"]

PAVEMENT_LIBRARY: dict[str, dict[str, float | str]] = {
    "Concrete": {
        "k_W_mK": 1.45,
        "rho_cp_J_m3K": 2.05e6,
        "albedo": 0.32,
        "emissivity": 0.92,
        "label": "light concrete",
    },
    "Asphalt": {
        "k_W_mK": 0.78,
        "rho_cp_J_m3K": 1.85e6,
        "albedo": 0.09,
        "emissivity": 0.95,
        "label": "dark asphalt",
    },
    "Permeable pavers": {
        "k_W_mK": 1.05,
        "rho_cp_J_m3K": 1.85e6,
        "albedo": 0.24,
        "emissivity": 0.93,
        "label": "pavers over aggregate",
    },
}

BASE_GRAVEL = {"k_W_mK": 0.95, "rho_cp_J_m3K": 1.70e6}


@dataclass(frozen=True)
class SimulationConfig:
    pavement_type: PavementType = "Concrete"
    pavement_thickness_m: float = 0.11
    base_thickness_m: float = 0.15
    max_depth_m: float = 6.0
    bottom_gradient_K_m: float = 0.025
    spinup_years: int = 1
    dt_s: float = 3600.0
    custom_albedo: float | None = None
    custom_emissivity: float | None = None
    convection_multiplier: float = 1.0
    sky_temperature_offset_C: float = 0.0
    ground_mean_offset_C: float = 0.0
    precipitation_energy_enabled: bool = True
    evaporative_cooling_factor: float = 0.35
    snow_melt_factor: float = 0.85


@dataclass
class SimulationResult:
    weather: pd.DataFrame
    time_series: pd.DataFrame
    depth_m: np.ndarray
    temperature_C: np.ndarray
    layer_names: list[str]
    config: SimulationConfig
    site: Site
    soil: SoilProfile
    thermosyphon: ThermosyphonDesign | None = None

    @property
    def surface_C(self) -> pd.Series:
        return self.time_series["surface_C"]

    @property
    def hp_flux_W_m2(self) -> pd.Series:
        return self.time_series["thermosyphon_flux_W_m2"]



def pavement_record(pavement_type: str) -> dict[str, float | str]:
    return dict(PAVEMENT_LIBRARY.get(pavement_type, PAVEMENT_LIBRARY["Concrete"]))



def build_grid(max_depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell edges, centers and thicknesses for a nonuniform grid."""
    z = 0.0
    edges = [z]
    while z < max_depth_m - 1e-9:
        if z < 0.25:
            dz = 0.02
        elif z < 1.5:
            dz = 0.05
        elif z < 4.0:
            dz = 0.10
        else:
            dz = 0.20
        z = min(z + dz, max_depth_m)
        edges.append(z)
    edge_arr = np.array(edges, dtype=float)
    centers = 0.5 * (edge_arr[:-1] + edge_arr[1:])
    dz_arr = np.diff(edge_arr)
    return edge_arr, centers, dz_arr



def assign_properties(
    centers: np.ndarray,
    config: SimulationConfig,
    soil: SoilProfile,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pav = pavement_record(config.pavement_type)
    k = np.empty_like(centers)
    rho_cp = np.empty_like(centers)
    names: list[str] = []
    for i, z in enumerate(centers):
        if z <= config.pavement_thickness_m:
            k[i] = float(pav["k_W_mK"])
            rho_cp[i] = float(pav["rho_cp_J_m3K"])
            names.append(str(pav["label"]))
        elif z <= config.pavement_thickness_m + config.base_thickness_m:
            k[i] = float(BASE_GRAVEL["k_W_mK"])
            rho_cp[i] = float(BASE_GRAVEL["rho_cp_J_m3K"])
            names.append("compacted base")
        else:
            k[i] = soil.k_W_mK
            rho_cp[i] = soil.rho_cp_J_m3K
            names.append(soil.name)
    return k, rho_cp, names



def _ground_mean(site: Site, config: SimulationConfig) -> float:
    return float(site.annual_ground_mean_c + config.ground_mean_offset_C)



def _initial_temperature_profile(site: Site, centers: np.ndarray, soil: SoilProfile, config: SimulationConfig) -> np.ndarray:
    annual_mean = _ground_mean(site, config)
    omega = 2.0 * np.pi / (365.0 * 24.0 * 3600.0)
    alpha = soil.k_W_mK / soil.rho_cp_J_m3K
    damping_depth = np.sqrt(2.0 * alpha / omega)
    amp = max(site.annual_amp_c * 0.75, 1.0)
    # January initialization: surface close to winter, deep near annual mean.
    phase = 2.0 * np.pi * (1 - 15) / 365.0
    profile = annual_mean - amp * np.exp(-centers / damping_depth) * np.cos(phase - centers / damping_depth)
    return profile.astype(float)



def _internal_conductances(k: np.ndarray, dz: np.ndarray) -> np.ndarray:
    # Face conductance per unit area between adjacent cell centers.
    left_res = (dz[:-1] / 2.0) / k[:-1]
    right_res = (dz[1:] / 2.0) / k[1:]
    return 1.0 / (left_res + right_res)



def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm. lower/upper length n-1, diag/rhs length n."""
    n = len(diag)
    cprime = np.empty(n - 1, dtype=float)
    dprime = np.empty(n, dtype=float)
    denom = diag[0]
    if abs(denom) < 1e-15:
        raise FloatingPointError("Zero diagonal in tridiagonal solve")
    cprime[0] = upper[0] / denom
    dprime[0] = rhs[0] / denom
    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * cprime[i - 1]
        if abs(denom) < 1e-15:
            raise FloatingPointError("Zero diagonal in tridiagonal solve")
        cprime[i] = upper[i] / denom
        dprime[i] = (rhs[i] - lower[i - 1] * dprime[i - 1]) / denom
    denom = diag[n - 1] - lower[n - 2] * cprime[n - 2]
    dprime[n - 1] = (rhs[n - 1] - lower[n - 2] * dprime[n - 2]) / denom

    x = np.empty(n, dtype=float)
    x[n - 1] = dprime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dprime[i] - cprime[i] * x[i + 1]
    return x



def _sky_temperature_C(air_C: float, rh_pct: float, ghi_W_m2: float, config: SimulationConfig) -> float:
    # Cloudy/wet hours have a warmer effective sky. Clear dry nights are colder.
    cloud_proxy = np.clip(ghi_W_m2 / 750.0, 0.0, 1.0)
    night_clear_penalty = 4.0 * (1.0 - cloud_proxy) * np.clip((85.0 - rh_pct) / 85.0, 0.0, 1.0)
    return float(air_C - 4.5 - night_clear_penalty + config.sky_temperature_offset_C)



def _surface_coefficients(weather_row: pd.Series, albedo: float, emissivity: float, config: SimulationConfig) -> tuple[float, float, float, float]:
    """Return linearized top boundary coefficients.

    Boundary form is q_into_surface = q_const - h_total * T_surface.
    The extra terms approximate evaporation and snow/ice phase-change loads. They
    are intentionally conservative screening terms, not a full snowpack model.
    """
    air = float(weather_row["air_temp_C"])
    wind = max(float(weather_row.get("wind_m_s", 2.0)), 0.1)
    rh = float(weather_row.get("rh_pct", 65.0))
    ghi = max(float(weather_row.get("ghi_W_m2", 0.0)), 0.0)
    precip_mm_h = max(float(weather_row.get("precip_mm", 0.0)), 0.0)
    wet = bool(weather_row.get("wet_flag", False)) or precip_mm_h > 0.02

    h_conv = (4.0 + 4.0 * np.sqrt(wind)) * max(config.convection_multiplier, 0.2)
    tref_K = max(air + 273.15, 230.0)
    sigma = 5.670374419e-8
    h_rad = 4.0 * emissivity * sigma * tref_K**3
    sky_C = _sky_temperature_C(air, rh, ghi, config)
    h_total = h_conv + h_rad

    q_extra = 0.0
    q_phase = 0.0
    if config.precipitation_energy_enabled:
        # Evaporative cooling proxy for wet pavement. The humidity deficit and
        # wind dependence keep the term small during cold saturated storms and
        # larger during dry/windy wet-pavement hours.
        if wet:
            humidity_deficit = np.clip((100.0 - rh) / 100.0, 0.0, 1.0)
            q_evap = -config.evaporative_cooling_factor * humidity_deficit * (22.0 + 10.0 * wind)
            q_extra += float(np.clip(q_evap, -85.0, 0.0))

        if precip_mm_h > 0:
            mass_flux = 1000.0 * precip_mm_h * 1.0e-3 / 3600.0  # kg/m2/s
            cp_water = 4180.0
            # Precipitation sensible heat relative to freezing. Positive rain can
            # warm the surface slightly; cold rain/snow cools it.
            q_extra += mass_flux * cp_water * np.clip(air, -8.0, 8.0)
            # Snow or freezing rain creates a latent load near/below freezing.
            snow_fraction = float(np.clip((1.5 - air) / 3.0, 0.0, 1.0))
            q_phase = -config.snow_melt_factor * snow_fraction * mass_flux * 334000.0
            q_extra += q_phase

    q_solar = (1.0 - albedo) * ghi
    q_const = q_solar + h_conv * air + h_rad * sky_C + q_extra
    return float(h_total), float(q_const), float(q_solar), float(q_phase)



def _make_base_matrix_terms(k: np.ndarray, rho_cp: np.ndarray, dz: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(k)
    capdt = rho_cp * dz / dt_s
    g_faces = _internal_conductances(k, dz)
    diag = capdt.copy()
    lower = np.zeros(n - 1, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    for i in range(n):
        if i > 0:
            g = g_faces[i - 1]
            diag[i] += g
            lower[i - 1] = -g
        if i < n - 1:
            g = g_faces[i]
            diag[i] += g
            upper[i] = -g
    return capdt, diag, lower, upper



def _index_at_depth(centers: np.ndarray, depth_m: float) -> int:
    return int(np.argmin(np.abs(centers - depth_m)))



def run_thermal_simulation(
    weather: pd.DataFrame,
    site: Site,
    soil: SoilProfile,
    config: SimulationConfig,
    thermosyphon: ThermosyphonDesign | None = None,
) -> SimulationResult:
    """Run the transient finite-difference model and return the final simulated year."""
    if len(weather) == 0:
        raise ValueError("Weather dataframe is empty")
    weather = weather.copy()
    if not isinstance(weather.index, pd.DatetimeIndex):
        if "timestamp" in weather.columns:
            weather = weather.set_index(pd.to_datetime(weather["timestamp"]))
        else:
            raise ValueError("Weather must have a DatetimeIndex or timestamp column")
    weather = weather.sort_index()

    edges, centers, dz = build_grid(config.max_depth_m)
    k, rho_cp, layer_names = assign_properties(centers, config, soil)
    capdt, diag_base, lower_base, upper_base = _make_base_matrix_terms(k, rho_cp, dz, config.dt_s)
    cap_area = rho_cp * dz

    pav = pavement_record(config.pavement_type)
    albedo = float(config.custom_albedo if config.custom_albedo is not None else pav["albedo"])
    emissivity = float(config.custom_emissivity if config.custom_emissivity is not None else pav["emissivity"])

    bottom_temp_C = _ground_mean(site, config) + config.bottom_gradient_K_m * config.max_depth_m
    bottom_g = k[-1] / max(dz[-1] / 2.0, 1e-6)

    T = _initial_temperature_profile(site, centers, soil, config)
    n_hours = len(weather)
    n_total = n_hours * (config.spinup_years + 1)
    out_T = np.empty((n_hours, len(centers)), dtype=np.float32)
    surf = np.empty(n_hours, dtype=float)
    hp_flux = np.zeros(n_hours, dtype=float)
    q_solar_abs = np.empty(n_hours, dtype=float)
    q_surface_net = np.empty(n_hours, dtype=float)
    q_phase_arr = np.empty(n_hours, dtype=float)

    top_hp_idx = 0
    bot_hp_idx = 0
    if thermosyphon is not None and thermosyphon.enabled:
        top_hp_idx = _index_at_depth(centers, thermosyphon.top_depth_m)
        bot_hp_idx = _index_at_depth(centers, min(thermosyphon.depth_m, config.max_depth_m - dz[-1]))
        if bot_hp_idx <= top_hp_idx:
            bot_hp_idx = min(len(centers) - 1, top_hp_idx + 1)

    weather_rows = [row for _, row in weather.iterrows()]
    for step in range(n_total):
        hour_idx = step % n_hours
        row = weather_rows[hour_idx]
        diag = diag_base.copy()
        lower = lower_base.copy()
        upper = upper_base.copy()
        rhs = capdt * T

        h_total, q_const, q_solar, q_phase = _surface_coefficients(row, albedo=albedo, emissivity=emissivity, config=config)
        diag[0] += h_total
        rhs[0] += q_const
        diag[-1] += bottom_g
        rhs[-1] += bottom_g * bottom_temp_C

        q_hp = 0.0
        if thermosyphon is not None and thermosyphon.enabled:
            q_hp = thermosyphon_flux_per_area(
                thermosyphon,
                top_temp_C=float(T[top_hp_idx]),
                bottom_temp_C=float(T[bot_hp_idx]),
                cap_top_J_m2K=float(cap_area[top_hp_idx]),
                cap_bottom_J_m2K=float(cap_area[bot_hp_idx]),
                dt_s=config.dt_s,
                soil_k_W_mK=soil.k_W_mK,
                top_k_W_mK=float(k[top_hp_idx]),
            )
            rhs[top_hp_idx] += q_hp
            rhs[bot_hp_idx] -= q_hp

        T = _solve_tridiagonal(lower, diag, upper, rhs)

        if step >= n_total - n_hours:
            out_i = step - (n_total - n_hours)
            out_T[out_i, :] = T.astype(np.float32)
            surf[out_i] = T[0]
            hp_flux[out_i] = q_hp
            q_solar_abs[out_i] = q_solar
            q_surface_net[out_i] = q_const - h_total * T[0]
            q_phase_arr[out_i] = q_phase

    ts = pd.DataFrame(
        {
            "surface_C": surf,
            "thermosyphon_flux_W_m2": hp_flux,
            "absorbed_solar_W_m2": q_solar_abs,
            "net_surface_flux_W_m2": q_surface_net,
            "snow_ice_phase_load_W_m2": q_phase_arr,
        },
        index=weather.index,
    )
    return SimulationResult(
        weather=weather,
        time_series=ts,
        depth_m=centers,
        temperature_C=out_T,
        layer_names=layer_names,
        config=config,
        site=site,
        soil=soil,
        thermosyphon=thermosyphon,
    )

"""Parametric wickless thermosyphon model.

The model is intentionally conservative. It treats a vertical wickless pipe as a
one-way thermal diode: heat moves upward only when the deeper/evaporator region
is warmer than the near-surface/condenser region by a startup margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

import numpy as np


FLUID_LIBRARY: dict[str, dict[str, float | str]] = {
    "Methanol blend (factory sealed)": {
        "qmax_25mm_W": 65.0,
        "internal_R_K_W": 0.055,
        "startup_C": 0.6,
        "temp_min_C": -40.0,
        "temp_max_C": 85.0,
        "contact_R_K_W": 0.075,
        "spreader_multiplier": 1.0,
        "evaporator_multiplier": 1.0,
        "label": "Subzero-capable economy fluid, factory sealed",
    },
    "CO2 / refrigerant grade (factory sealed)": {
        "qmax_25mm_W": 95.0,
        "internal_R_K_W": 0.040,
        "startup_C": 0.4,
        "temp_min_C": -50.0,
        "temp_max_C": 45.0,
        "contact_R_K_W": 0.060,
        "spreader_multiplier": 1.0,
        "evaporator_multiplier": 1.0,
        "label": "Higher capacity, pressure-rated factory assembly",
    },
    "High-output CO2 + thermal grout + heat spreader": {
        "qmax_25mm_W": 165.0,
        "internal_R_K_W": 0.022,
        "startup_C": 0.22,
        "temp_min_C": -55.0,
        "temp_max_C": 48.0,
        "contact_R_K_W": 0.030,
        "spreader_multiplier": 1.65,
        "evaporator_multiplier": 1.35,
        "label": "Pressure-rated high-output concept with conductive grout and near-surface heat spreader",
    },
    "Assured 90 hybrid thermosyphon + low-power assist": {
        "qmax_25mm_W": 190.0,
        "internal_R_K_W": 0.020,
        "startup_C": 0.18,
        "temp_min_C": -55.0,
        "temp_max_C": 48.0,
        "contact_R_K_W": 0.024,
        "spreader_multiplier": 2.10,
        "evaporator_multiplier": 1.75,
        "booster_W_m2": 300.0,
        "booster_setpoint_C": 1.5,
        "booster_gain_W_m2K": 150.0,
        "booster_cop": 1.0,
        "label": "Passive thermosyphon base load with thermostat-controlled surface assist for 90%+ freeze-hour reduction",
    },
    "Water (above-freezing applications only)": {
        "qmax_25mm_W": 80.0,
        "internal_R_K_W": 0.035,
        "startup_C": 0.4,
        "temp_min_C": 0.1,
        "temp_max_C": 95.0,
        "contact_R_K_W": 0.060,
        "spreader_multiplier": 1.0,
        "evaporator_multiplier": 1.0,
        "label": "Low-cost but not recommended for freeze-risk designs",
    },
}


@dataclass(frozen=True)
class ThermosyphonDesign:
    spacing_m: float
    depth_m: float
    diameter_m: float
    top_depth_m: float = 0.10
    fluid: str = "Methanol blend (factory sealed)"
    startup_delta_C: float | None = None
    enabled: bool = True
    booster_W_m2: float | None = None
    booster_setpoint_C: float | None = None
    booster_gain_W_m2K: float | None = None
    booster_cop: float | None = None

    @property
    def tributary_area_m2(self) -> float:
        return self.spacing_m * self.spacing_m

    @property
    def diameter_mm(self) -> float:
        return self.diameter_m * 1000.0

    @property
    def fluid_record(self) -> dict[str, float | str]:
        return FLUID_LIBRARY.get(self.fluid, FLUID_LIBRARY["Methanol blend (factory sealed)"])



def condenser_length_m(design: ThermosyphonDesign) -> float:
    rec = design.fluid_record
    spreader_multiplier = float(rec.get("spreader_multiplier", 1.0))
    return float(max(0.25, min(1.15, design.depth_m * 0.18 * spreader_multiplier)))


def evaporator_length_m(design: ThermosyphonDesign) -> float:
    rec = design.fluid_record
    evaporator_multiplier = float(rec.get("evaporator_multiplier", 1.0))
    return float(max(0.50, min(3.25, design.depth_m * 0.35 * evaporator_multiplier)))


def booster_capacity_W_m2(design: ThermosyphonDesign) -> float:
    if design.booster_W_m2 is not None:
        return float(max(design.booster_W_m2, 0.0))
    return float(max(float(design.fluid_record.get("booster_W_m2", 0.0)), 0.0))


def booster_setpoint_C(design: ThermosyphonDesign) -> float:
    if design.booster_setpoint_C is not None:
        return float(design.booster_setpoint_C)
    return float(design.fluid_record.get("booster_setpoint_C", 0.55))


def booster_gain_W_m2K(design: ThermosyphonDesign) -> float:
    if design.booster_gain_W_m2K is not None:
        return float(max(design.booster_gain_W_m2K, 0.0))
    return float(max(float(design.fluid_record.get("booster_gain_W_m2K", 60.0)), 0.0))


def booster_cop(design: ThermosyphonDesign) -> float:
    if design.booster_cop is not None:
        return float(max(design.booster_cop, 0.1))
    return float(max(float(design.fluid_record.get("booster_cop", 1.0)), 0.1))


def pipe_count(area_m2: float, spacing_m: float) -> int:
    return int(np.ceil(max(area_m2, 0.1) / max(spacing_m * spacing_m, 0.01)))


def pipe_conductance_W_K(design: ThermosyphonDesign, soil_k_W_mK: float, top_k_W_mK: float = 1.2) -> float:
    """Return effective pipe-plus-contact conductance for one pipe."""
    rec = design.fluid_record
    radius = max(design.diameter_m / 2.0, 0.004)
    influence_radius = max(design.spacing_m / np.sqrt(np.pi), radius * 8.0)
    spreader_multiplier = float(rec.get("spreader_multiplier", 1.0))
    condenser_len = condenser_length_m(design)
    evaporator_len = evaporator_length_m(design)
    top_k = max(top_k_W_mK * spreader_multiplier, 0.45)
    soil_k = max(soil_k_W_mK, 0.35)

    r_top = max(log(influence_radius / radius), 0.1) / (2.0 * pi * top_k * condenser_len)
    r_bottom = max(log(influence_radius / radius), 0.1) / (2.0 * pi * soil_k * evaporator_len)
    r_internal = float(rec["internal_R_K_W"])
    # Contact/backfill resistance is explicitly lower for high-output designs
    # that include thermal grout and a near-surface spreader strip.
    default_contact = 0.055 + 0.020 * (0.025 / max(design.diameter_m, 0.012))
    r_contact = float(rec.get("contact_R_K_W", default_contact))
    conductance = 1.0 / (r_top + r_bottom + r_internal + r_contact)
    return float(np.clip(conductance, 0.5, 180.0))


def max_heat_per_pipe_W(design: ThermosyphonDesign) -> float:
    rec = design.fluid_record
    base = float(rec["qmax_25mm_W"])
    diameter_factor = (design.diameter_m / 0.025) ** 1.75
    length_factor = sqrt(max(design.depth_m, 0.5) / 2.4)
    return float(np.clip(base * diameter_factor * length_factor, 10.0, 1250.0))




def temperature_capacity_factor(design: ThermosyphonDesign, top_temp_C: float, bottom_temp_C: float) -> float:
    """Capacity derate for working-fluid operating envelope.

    Water-filled units are intentionally derated to zero below freezing; CO2 and
    methanol blends retain capacity in subzero winter screening applications.
    """
    rec = design.fluid_record
    tmin = float(rec.get("temp_min_C", -80.0))
    tmax = float(rec.get("temp_max_C", 120.0))
    tmean = 0.5 * (top_temp_C + bottom_temp_C)
    if top_temp_C < tmin or bottom_temp_C < tmin or tmean > tmax:
        return 0.0
    low_margin = np.clip((tmean - tmin) / 6.0, 0.0, 1.0)
    high_margin = np.clip((tmax - tmean) / 10.0, 0.0, 1.0)
    return float(min(low_margin, high_margin))


def activation_delta_C(design: ThermosyphonDesign) -> float:
    if design.startup_delta_C is not None:
        return float(design.startup_delta_C)
    return float(design.fluid_record["startup_C"])


def thermosyphon_flux_per_area(
    design: ThermosyphonDesign,
    top_temp_C: float,
    bottom_temp_C: float,
    cap_top_J_m2K: float,
    cap_bottom_J_m2K: float,
    dt_s: float,
    soil_k_W_mK: float,
    top_k_W_mK: float,
) -> float:
    """Compute upward heat flux per driveway area, W/m2.

    Positive means heat is added near the driveway surface and removed at depth.
    """
    if not design.enabled:
        return 0.0
    delta = bottom_temp_C - top_temp_C - activation_delta_C(design)
    if delta <= 0.0:
        return 0.0
    g = pipe_conductance_W_K(design, soil_k_W_mK=soil_k_W_mK, top_k_W_mK=top_k_W_mK)
    capacity_factor = temperature_capacity_factor(design, top_temp_C=top_temp_C, bottom_temp_C=bottom_temp_C)
    if capacity_factor <= 0.0:
        return 0.0
    q_pipe = min(g * delta, max_heat_per_pipe_W(design) * capacity_factor)
    q_area = q_pipe / design.tributary_area_m2

    # Prevent explicit source term from numerically over-equalizing the two cells.
    physical_delta = max(bottom_temp_C - top_temp_C, 0.0)
    if physical_delta > 0:
        # A high-output design includes a heat spreader and better grout, so it can
        # safely use a larger fraction of the available cell-to-cell thermal
        # disequilibrium without producing unstable explicit equalization.
        equalize_fraction = 0.96 if ("High-output" in design.fluid or "Assured 90" in design.fluid) else 0.70
        q_equalize = equalize_fraction * physical_delta / max(dt_s, 1.0) / (
            1.0 / max(cap_top_J_m2K, 1.0) + 1.0 / max(cap_bottom_J_m2K, 1.0)
        )
        q_area = min(q_area, q_equalize)
    return float(max(q_area, 0.0))


def summarize_design(design: ThermosyphonDesign, area_m2: float, soil_k_W_mK: float, top_k_W_mK: float) -> dict[str, float | str]:
    n = pipe_count(area_m2, design.spacing_m)
    return {
        "spacing_m": design.spacing_m,
        "depth_m": design.depth_m,
        "diameter_mm": design.diameter_mm,
        "pipe_count": n,
        "conductance_W_K_per_pipe": pipe_conductance_W_K(design, soil_k_W_mK, top_k_W_mK),
        "qmax_W_per_pipe": max_heat_per_pipe_W(design),
        "fluid": design.fluid,
        "activation_delta_C": activation_delta_C(design),
        "booster_capacity_W_m2": booster_capacity_W_m2(design),
        "booster_setpoint_C": booster_setpoint_C(design),
    }

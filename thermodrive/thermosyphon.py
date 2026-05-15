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
        "label": "Subzero-capable economy fluid, factory sealed",
    },
    "CO2 / refrigerant grade (factory sealed)": {
        "qmax_25mm_W": 95.0,
        "internal_R_K_W": 0.040,
        "startup_C": 0.4,
        "temp_min_C": -50.0,
        "temp_max_C": 45.0,
        "label": "Higher capacity, pressure-rated factory assembly",
    },
    "Water (above-freezing applications only)": {
        "qmax_25mm_W": 80.0,
        "internal_R_K_W": 0.035,
        "startup_C": 0.4,
        "temp_min_C": 0.1,
        "temp_max_C": 95.0,
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

    @property
    def tributary_area_m2(self) -> float:
        return self.spacing_m * self.spacing_m

    @property
    def diameter_mm(self) -> float:
        return self.diameter_m * 1000.0

    @property
    def fluid_record(self) -> dict[str, float | str]:
        return FLUID_LIBRARY.get(self.fluid, FLUID_LIBRARY["Methanol blend (factory sealed)"])


def pipe_count(area_m2: float, spacing_m: float) -> int:
    return int(np.ceil(max(area_m2, 0.1) / max(spacing_m * spacing_m, 0.01)))


def pipe_conductance_W_K(design: ThermosyphonDesign, soil_k_W_mK: float, top_k_W_mK: float = 1.2) -> float:
    """Return effective pipe-plus-contact conductance for one pipe."""
    radius = max(design.diameter_m / 2.0, 0.004)
    influence_radius = max(design.spacing_m / np.sqrt(np.pi), radius * 8.0)
    condenser_len = max(0.25, min(0.60, design.depth_m * 0.18))
    evaporator_len = max(0.50, min(1.50, design.depth_m * 0.35))
    top_k = max(top_k_W_mK, 0.45)
    soil_k = max(soil_k_W_mK, 0.35)

    r_top = max(log(influence_radius / radius), 0.1) / (2.0 * pi * top_k * condenser_len)
    r_bottom = max(log(influence_radius / radius), 0.1) / (2.0 * pi * soil_k * evaporator_len)
    r_internal = float(design.fluid_record["internal_R_K_W"])
    # Two conservative contact terms for imperfect embedment/backfill.
    r_contact = 0.055 + 0.020 * (0.025 / max(design.diameter_m, 0.012))
    conductance = 1.0 / (r_top + r_bottom + r_internal + r_contact)
    return float(np.clip(conductance, 0.5, 90.0))


def max_heat_per_pipe_W(design: ThermosyphonDesign) -> float:
    rec = design.fluid_record
    base = float(rec["qmax_25mm_W"])
    diameter_factor = (design.diameter_m / 0.025) ** 1.75
    length_factor = sqrt(max(design.depth_m, 0.5) / 2.4)
    return float(np.clip(base * diameter_factor * length_factor, 10.0, 450.0))


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
    q_pipe = min(g * delta, max_heat_per_pipe_W(design))
    q_area = q_pipe / design.tributary_area_m2

    # Prevent explicit source term from numerically over-equalizing the two cells.
    physical_delta = max(bottom_temp_C - top_temp_C, 0.0)
    if physical_delta > 0:
        q_equalize = 0.70 * physical_delta / max(dt_s, 1.0) / (
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
    }

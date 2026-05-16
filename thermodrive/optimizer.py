"""Design-space search for thermosyphon field sizing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .climate import Site
from .cost import CostEstimate, InstallType, estimate_project_cost
from .metrics import PerformanceMetrics, compare_metrics, compute_metrics
from .physics import SimulationConfig, SimulationResult, run_thermal_simulation
from .soil import SoilProfile
from .thermosyphon import (
    ThermosyphonDesign,
    activation_delta_C,
    booster_capacity_W_m2,
    booster_gain_W_m2K,
    booster_setpoint_C,
    max_heat_per_pipe_W,
    pipe_conductance_W_K,
    pipe_count,
)
from .units import FT_TO_M

Goal = Literal["Reduce freeze risk", "Reduce thermal cracking", "Overheating analysis"]


@dataclass
class OptimizationResult:
    status: str
    package_label: str
    baseline_result: SimulationResult
    baseline_metrics: PerformanceMetrics
    design: ThermosyphonDesign | None
    design_result: SimulationResult | None
    design_metrics: PerformanceMetrics | None
    comparison: dict[str, float]
    cost: CostEstimate | None
    candidates: pd.DataFrame
    recommendation_note: str


def _candidate_designs(max_depth_m: float, fluid: str, aggressive: bool = False) -> list[ThermosyphonDesign]:
    high_output = ("High-output" in fluid) or ("Assured 90" in fluid)
    hybrid = "Assured 90" in fluid
    depth_ft_options = [6, 8, 10, 12, 15, 16, 18, 20, 24, 28, 32] if hybrid else ([4, 6, 8, 10, 12, 15, 16, 18, 20, 24] if high_output else [4, 6, 8, 10, 12, 15])
    spacing_ft_options = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] if hybrid else ([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0] if high_output else [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    diameter_mm_options = [32, 38, 50, 63] if hybrid else ([25, 32, 38, 50] if high_output else ([19, 25, 32] if aggressive else [19, 25]))
    designs: list[ThermosyphonDesign] = []
    for depth_ft in depth_ft_options:
        depth_m = depth_ft * FT_TO_M
        if depth_m > max_depth_m + 1e-9:
            continue
        for spacing_ft in spacing_ft_options:
            spacing_m = spacing_ft * FT_TO_M
            for dia_mm in diameter_mm_options:
                designs.append(
                    ThermosyphonDesign(
                        spacing_m=spacing_m,
                        depth_m=depth_m,
                        diameter_m=dia_mm / 1000.0,
                        top_depth_m=0.055 if high_output else 0.10,
                        fluid=fluid,
                    )
                )
    return designs


def _depth_series(result: SimulationResult, depth_m: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(result.depth_m - depth_m)))
    return result.temperature_C[:, idx].astype(float)


def _surrogate_reduction(
    baseline: SimulationResult,
    baseline_metrics: PerformanceMetrics,
    design: ThermosyphonDesign,
    area_m2: float,
    goal: Goal,
) -> dict[str, float]:
    surface = baseline.surface_C.to_numpy(dtype=float)
    bottom = _depth_series(baseline, design.depth_m)
    dT = bottom - surface
    activation = activation_delta_C(design)
    active = dT > activation
    g = pipe_conductance_W_K(design, baseline.soil.k_W_mK, top_k_W_mK=1.2)
    rec = design.fluid_record
    tmin = float(rec.get("temp_min_C", -80.0))
    tmax = float(rec.get("temp_max_C", 120.0))
    tmean = 0.5 * (surface + bottom)
    low_margin = np.clip((tmean - tmin) / 6.0, 0.0, 1.0)
    high_margin = np.clip((tmax - tmean) / 10.0, 0.0, 1.0)
    cap = np.minimum(low_margin, high_margin)
    cap[(surface < tmin) | (bottom < tmin) | (tmean > tmax)] = 0.0
    q_pipe = np.minimum(np.maximum(g * (dT - activation), 0), max_heat_per_pipe_W(design) * cap)
    q_area_passive = q_pipe / design.tributary_area_m2
    assist_cap = booster_capacity_W_m2(design)
    q_assist = np.zeros_like(surface)
    if assist_cap > 0:
        setpoint = booster_setpoint_C(design)
        gain = booster_gain_W_m2K(design)
        air = baseline.weather.get("air_temp_C", pd.Series(surface, index=baseline.weather.index)).to_numpy(dtype=float)
        q_assist = np.minimum(assist_cap, np.maximum(0.0, (setpoint - surface) * gain))
        q_assist[(surface >= setpoint) | (air >= 4.0)] = 0.0
    q_area = q_area_passive + q_assist
    # Convert heat flux to a rough temperature lift. This is only for ranking;
    # full FD verification is run for selected candidates. Assisted designs use
    # a tighter denominator because the controller places heat close to the slab.
    if assist_cap > 0:
        lift_denominator = 42.0
        lift_cap = 16.0
    else:
        lift_denominator = 85.0 if "High-output" in design.fluid else 65.0
        lift_cap = 4.5 if "High-output" in design.fluid else 3.5
    temp_lift = np.clip(q_area / lift_denominator, 0, lift_cap)
    cold_weight = np.clip((4.0 - surface) / 8.0, 0.0, 1.0)
    if assist_cap > 0:
        cold_weight = np.clip((8.0 - surface) / 10.0, 0.15, 1.0)
    surface_hat = surface + temp_lift * cold_weight
    wet = baseline.weather.get("wet_flag", pd.Series(False, index=baseline.weather.index)).to_numpy(dtype=bool)
    freeze_base = np.sum(surface < 0)
    wet_base = np.sum((surface < 0) & wet)
    freeze_new = np.sum(surface_hat < 0)
    wet_new = np.sum((surface_hat < 0) & wet)
    crossings_base = baseline_metrics.freeze_thaw_cycles
    crossings_new = int(np.sum(np.signbit(surface_hat[1:]) != np.signbit(surface_hat[:-1])))
    freeze_red = 100.0 * (freeze_base - freeze_new) / max(freeze_base, 1)
    wet_red = 100.0 * (wet_base - wet_new) / max(wet_base, 1)
    crack_red = 100.0 * (crossings_base - crossings_new) / max(crossings_base, 1)
    annual_kwh_m2 = float(np.sum(q_area) * 3600.0 / 3.6e6)
    if goal == "Reduce thermal cracking":
        primary = max(crack_red, 0.4 * freeze_red)
    else:
        primary = freeze_red
    return {
        "estimated_reduction_pct": float(max(primary, 0.0)),
        "estimated_freeze_reduction_pct": float(max(freeze_red, 0.0)),
        "estimated_wet_freeze_reduction_pct": float(max(wet_red, 0.0)),
        "estimated_cracking_reduction_pct": float(max(crack_red, 0.0)),
        "estimated_annual_kWh_m2": annual_kwh_m2,
    }


def _candidate_table(
    baseline: SimulationResult,
    baseline_metrics: PerformanceMetrics,
    area_m2: float,
    install_type: InstallType,
    region_factor: float,
    goal: Goal,
    target_reduction_pct: float,
    max_depth_m: float,
    fluid: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    aggressive = target_reduction_pct >= 50 or goal == "Reduce freeze risk"
    for d in _candidate_designs(max_depth_m=max_depth_m, fluid=fluid, aggressive=aggressive):
        sur = _surrogate_reduction(baseline, baseline_metrics, d, area_m2, goal)
        cost = estimate_project_cost(area_m2, d, install_type, region_factor=region_factor)
        rows.append(
            {
                "spacing_ft": d.spacing_m / FT_TO_M,
                "depth_ft": d.depth_m / FT_TO_M,
                "diameter_mm": d.diameter_mm,
                "pipe_count": pipe_count(area_m2, d.spacing_m),
                "fluid": d.fluid,
                "estimated_reduction_pct": sur["estimated_reduction_pct"],
                "estimated_freeze_reduction_pct": sur["estimated_freeze_reduction_pct"],
                "estimated_wet_freeze_reduction_pct": sur["estimated_wet_freeze_reduction_pct"],
                "estimated_cracking_reduction_pct": sur["estimated_cracking_reduction_pct"],
                "estimated_annual_kWh_m2": sur["estimated_annual_kWh_m2"],
                "base_cost": cost.total_base,
                "cost_per_sqft": cost.cost_per_sqft,
                "rank_score": cost.total_base / max(sur["estimated_reduction_pct"], 1.5),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["target_gap"] = np.maximum(0.0, target_reduction_pct - df["estimated_reduction_pct"])
    df["meets_surrogate_target"] = df["estimated_reduction_pct"] >= target_reduction_pct
    df = df.sort_values(["target_gap", "rank_score", "base_cost"], ascending=[True, True, True]).reset_index(drop=True)
    return df


def _design_from_row(row: pd.Series, fluid: str) -> ThermosyphonDesign:
    high_output = ("High-output" in fluid) or ("Assured 90" in fluid)
    return ThermosyphonDesign(
        spacing_m=float(row["spacing_ft"]) * FT_TO_M,
        depth_m=float(row["depth_ft"]) * FT_TO_M,
        diameter_m=float(row["diameter_mm"]) / 1000.0,
        top_depth_m=0.055 if high_output else 0.10,
        fluid=fluid,
    )


def _primary_reduction(goal: Goal, baseline_metrics: PerformanceMetrics, comparison: dict[str, float]) -> float:
    if goal == "Reduce thermal cracking":
        return max(
            comparison.get("stress_index_reduction_pct", 0.0),
            comparison.get("freeze_thaw_reduction_pct", 0.0),
            comparison.get("freeze_degree_hour_reduction_pct", 0.0),
        )
    return comparison["freeze_hour_reduction_pct"]


def optimize_design(
    weather: pd.DataFrame,
    site: Site,
    soil: SoilProfile,
    config: SimulationConfig,
    area_m2: float,
    install_type: InstallType,
    region_factor: float,
    goal: Goal,
    target_reduction_pct: float,
    max_depth_m: float,
    fluid: str,
    baseline_result: SimulationResult | None = None,
    full_verify_count: int = 8,
) -> OptimizationResult:
    baseline = baseline_result or run_thermal_simulation(weather, site, soil, config, thermosyphon=None)
    baseline_metrics = compute_metrics(baseline)

    if goal == "Overheating analysis":
        note = (
            "A vertical wickless thermosyphon is a one-way upward heat-transfer device. "
            "It is not recommended as the primary summer driveway cooling solution. "
            "Use the overheating plots to size reflective coatings, shading, or an active/bidirectional ground loop."
        )
        return OptimizationResult(
            status="No passive thermosyphon recommendation for overheating-only goal",
            package_label="Cooling alternative recommended",
            baseline_result=baseline,
            baseline_metrics=baseline_metrics,
            design=None,
            design_result=None,
            design_metrics=None,
            comparison={},
            cost=None,
            candidates=pd.DataFrame(),
            recommendation_note=note,
        )

    if baseline_metrics.freeze_hours == 0 and goal == "Reduce freeze risk":
        note = "The screening year produced no baseline freeze hours. A thermosyphon field is not recommended for freeze-risk reduction at this location."
        return OptimizationResult(
            status="No winter freeze-risk design needed for screening year",
            package_label="No-build recommendation",
            baseline_result=baseline,
            baseline_metrics=baseline_metrics,
            design=None,
            design_result=None,
            design_metrics=None,
            comparison={},
            cost=None,
            candidates=pd.DataFrame(),
            recommendation_note=note,
        )

    candidates = _candidate_table(
        baseline,
        baseline_metrics,
        area_m2,
        install_type,
        region_factor,
        goal,
        target_reduction_pct,
        max_depth_m,
        fluid,
    )
    if candidates.empty:
        note = "No constructable candidate met the selected maximum depth. Increase maximum depth or switch to a new-driveway installation scenario."
        return OptimizationResult(
            status="No constructable candidate",
            package_label="Needs design review",
            baseline_result=baseline,
            baseline_metrics=baseline_metrics,
            design=None,
            design_result=None,
            design_metrics=None,
            comparison={},
            cost=None,
            candidates=candidates,
            recommendation_note=note,
        )

    # Verify a diverse set while keeping Streamlit response time reasonable.
    # Put the top-performance designs first so cold states such as Idaho do not
    # get stuck verifying only low-cost, low-output candidates.
    verify_budget = min(len(candidates), max(full_verify_count, 4))
    idxs: list[int] = []
    idxs.extend(candidates.sort_values("estimated_reduction_pct", ascending=False).head(2).index.tolist())
    idxs.extend(candidates.head(max(verify_budget - 2, 1)).index.tolist())
    half = candidates[candidates["estimated_reduction_pct"] >= 0.5 * target_reduction_pct]
    if not half.empty:
        idxs.extend(half.sort_values("base_cost").head(2).index.tolist())
    idxs = list(dict.fromkeys(idxs))[:verify_budget]

    verified_rows: list[dict[str, float | int | str | bool]] = []
    best: tuple[float, ThermosyphonDesign, SimulationResult, PerformanceMetrics, dict[str, float], CostEstimate] | None = None
    best_meeting: tuple[float, ThermosyphonDesign, SimulationResult, PerformanceMetrics, dict[str, float], CostEstimate] | None = None
    for idx in idxs:
        row = candidates.loc[idx]
        design = _design_from_row(row, fluid)
        design_config = SimulationConfig(
            pavement_type=config.pavement_type,
            pavement_thickness_m=config.pavement_thickness_m,
            base_thickness_m=config.base_thickness_m,
            max_depth_m=max(config.max_depth_m, min(max_depth_m, design.depth_m + 0.8)),
            bottom_gradient_K_m=config.bottom_gradient_K_m,
            spinup_years=config.spinup_years,
            dt_s=config.dt_s,
            custom_albedo=config.custom_albedo,
            custom_emissivity=config.custom_emissivity,
            convection_multiplier=config.convection_multiplier,
            sky_temperature_offset_C=config.sky_temperature_offset_C,
            ground_mean_offset_C=config.ground_mean_offset_C,
            precipitation_energy_enabled=config.precipitation_energy_enabled,
            evaporative_cooling_factor=config.evaporative_cooling_factor,
            snow_melt_factor=config.snow_melt_factor,
        )
        result = run_thermal_simulation(weather, site, soil, design_config, thermosyphon=design)
        metrics = compute_metrics(result)
        comparison = compare_metrics(baseline_metrics, metrics)
        primary = _primary_reduction(goal, baseline_metrics, comparison)
        cost = estimate_project_cost(area_m2, design, install_type, region_factor=region_factor)
        meets = primary >= target_reduction_pct
        verified_rows.append(
            {
                "spacing_ft": row["spacing_ft"],
                "depth_ft": row["depth_ft"],
                "diameter_mm": row["diameter_mm"],
                "pipe_count": row["pipe_count"],
                "base_cost": cost.total_base,
                "cost_per_sqft": cost.cost_per_sqft,
                "verified_primary_reduction_pct": primary,
                "verified_freeze_reduction_pct": comparison["freeze_hour_reduction_pct"],
                "verified_wet_freeze_reduction_pct": comparison["wet_freeze_reduction_pct"],
                "verified_stress_reduction_pct": comparison["stress_index_reduction_pct"],
                "annual_hp_kWh_m2": metrics.annual_hp_kWh_m2,
                "annual_assist_kWh_m2": metrics.annual_assist_kWh_m2,
                "total_heat_kWh_m2": metrics.total_heat_kWh_m2,
                "meets_target": meets,
            }
        )
        score = cost.total_base / max(primary, 1.0)
        candidate_tuple = (score, design, result, metrics, comparison, cost)
        if best is None or primary > _primary_reduction(goal, baseline_metrics, best[4]) or (primary == _primary_reduction(goal, baseline_metrics, best[4]) and cost.total_base < best[5].total_base):
            best = candidate_tuple
        if meets and (best_meeting is None or cost.total_base < best_meeting[5].total_base):
            best_meeting = candidate_tuple

    verified_df = pd.DataFrame(verified_rows)
    if not verified_df.empty:
        candidates = candidates.merge(
            verified_df,
            on=["spacing_ft", "depth_ft", "diameter_mm", "pipe_count", "base_cost", "cost_per_sqft"],
            how="left",
        )

    chosen = best_meeting if best_meeting is not None else best
    if chosen is None:
        note = "Candidate verification failed unexpectedly. Review inputs or reduce target aggressiveness."
        return OptimizationResult(
            status="Verification failed",
            package_label="Needs design review",
            baseline_result=baseline,
            baseline_metrics=baseline_metrics,
            design=None,
            design_result=None,
            design_metrics=None,
            comparison={},
            cost=None,
            candidates=candidates,
            recommendation_note=note,
        )

    _, design, result, metrics, comparison, cost = chosen
    primary = _primary_reduction(goal, baseline_metrics, comparison)
    if primary >= target_reduction_pct:
        status = "Target met in verified finite-difference model"
        if cost.cost_per_sqft < 20:
            package = "Economy package"
        elif cost.cost_per_sqft < 35:
            package = "Balanced package"
        else:
            package = "Maximum performance package"
        if "Assured 90" in fluid:
            note = "Recommended package meets the target using passive thermosyphon heat plus thermostat-controlled low-power assist. Passive and assisted heat are reported separately."
        else:
            note = "Recommended design is the lowest-cost verified candidate meeting the selected target."
    else:
        status = "Target not fully met; best available passive thermosyphon package shown"
        package = "Best passive package within constraints"
        note = (
            "The selected target is aggressive for a passive wickless thermosyphon field. "
            "For 90%+ freeze-hour reduction, switch to the Assured 90 hybrid package or add a hydronic/electric assist zone."
        )

    return OptimizationResult(
        status=status,
        package_label=package,
        baseline_result=baseline,
        baseline_metrics=baseline_metrics,
        design=design,
        design_result=result,
        design_metrics=metrics,
        comparison=comparison,
        cost=cost,
        candidates=candidates,
        recommendation_note=note,
    )

"""Performance metrics for driveway thermal simulations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .physics import SimulationResult


@dataclass(frozen=True)
class PerformanceMetrics:
    freeze_hours: int
    wet_freeze_hours: int
    freeze_thaw_cycles: int
    winter_p5_C: float
    summer_p95_C: float
    max_surface_C: float
    min_surface_C: float
    avg_daily_swing_C: float
    p95_daily_swing_C: float
    thermal_stress_index: float
    freeze_degree_hours_C_h: float
    active_hp_hours: int = 0
    annual_hp_kWh_m2: float = 0.0
    annual_assist_kWh_m2: float = 0.0
    total_heat_kWh_m2: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "freeze_hours": self.freeze_hours,
            "wet_freeze_hours": self.wet_freeze_hours,
            "freeze_thaw_cycles": self.freeze_thaw_cycles,
            "winter_p5_C": self.winter_p5_C,
            "summer_p95_C": self.summer_p95_C,
            "max_surface_C": self.max_surface_C,
            "min_surface_C": self.min_surface_C,
            "avg_daily_swing_C": self.avg_daily_swing_C,
            "p95_daily_swing_C": self.p95_daily_swing_C,
            "thermal_stress_index": self.thermal_stress_index,
            "freeze_degree_hours_C_h": self.freeze_degree_hours_C_h,
            "active_hp_hours": self.active_hp_hours,
            "annual_hp_kWh_m2": self.annual_hp_kWh_m2,
            "annual_assist_kWh_m2": self.annual_assist_kWh_m2,
            "total_heat_kWh_m2": self.total_heat_kWh_m2,
        }


def zero_crossings(series_C: pd.Series, threshold_C: float = 0.0) -> int:
    arr = series_C.to_numpy(dtype=float) - threshold_C
    arr = np.where(np.isclose(arr, 0.0), 1e-9, arr)
    return int(np.sum(np.signbit(arr[1:]) != np.signbit(arr[:-1])))


def daily_swing(series_C: pd.Series) -> pd.Series:
    daily = series_C.resample("D")
    return daily.max() - daily.min()


def compute_metrics(result: SimulationResult) -> PerformanceMetrics:
    surf = result.surface_C.astype(float)
    weather = result.weather.reindex(surf.index)
    wet = weather.get("wet_flag", pd.Series(False, index=surf.index)).astype(bool)
    freeze = surf < 0.0
    winter = surf.index.month.isin([12, 1, 2])
    summer = surf.index.month.isin([6, 7, 8])
    swing = daily_swing(surf)
    hp = result.hp_flux_W_m2.astype(float)
    assist = result.assist_flux_W_m2.astype(float)
    hp_kwh_m2 = float(hp.sum() * 3600.0 / 3.6e6)
    assist_kwh_m2 = float(assist.sum() * 3600.0 / 3.6e6)
    freeze_degree_hours = float(np.maximum(0.0, -surf.to_numpy(dtype=float)).sum())
    # Stress index combines freeze-thaw crossings, subfreezing severity, and large daily swings.
    # This is a screening damage index rather than a structural design criterion.
    stress = float(zero_crossings(surf) + 0.02 * freeze_degree_hours + 0.5 * np.nanpercentile(swing, 95))
    return PerformanceMetrics(
        freeze_hours=int(freeze.sum()),
        wet_freeze_hours=int((freeze & wet).sum()),
        freeze_thaw_cycles=zero_crossings(surf),
        winter_p5_C=float(np.nanpercentile(surf[winter] if winter.any() else surf, 5)),
        summer_p95_C=float(np.nanpercentile(surf[summer] if summer.any() else surf, 95)),
        max_surface_C=float(np.nanmax(surf)),
        min_surface_C=float(np.nanmin(surf)),
        avg_daily_swing_C=float(np.nanmean(swing)),
        p95_daily_swing_C=float(np.nanpercentile(swing, 95)),
        thermal_stress_index=stress,
        freeze_degree_hours_C_h=freeze_degree_hours,
        active_hp_hours=int(((hp + assist) > 0.5).sum()),
        annual_hp_kWh_m2=hp_kwh_m2,
        annual_assist_kWh_m2=assist_kwh_m2,
        total_heat_kWh_m2=hp_kwh_m2 + assist_kwh_m2,
    )


def compare_metrics(baseline: PerformanceMetrics, design: PerformanceMetrics) -> dict[str, float]:
    def reduction(base: float, new: float) -> float:
        if base <= 1e-9:
            return 0.0
        return 100.0 * (base - new) / base

    return {
        "freeze_hour_reduction_pct": reduction(baseline.freeze_hours, design.freeze_hours),
        "wet_freeze_reduction_pct": reduction(baseline.wet_freeze_hours, design.wet_freeze_hours),
        "freeze_thaw_reduction_pct": reduction(baseline.freeze_thaw_cycles, design.freeze_thaw_cycles),
        "stress_index_reduction_pct": reduction(baseline.thermal_stress_index, design.thermal_stress_index),
        "freeze_degree_hour_reduction_pct": reduction(baseline.freeze_degree_hours_C_h, design.freeze_degree_hours_C_h),
        "daily_swing_reduction_pct": reduction(baseline.p95_daily_swing_C, design.p95_daily_swing_C),
        "summer_p95_change_C": design.summer_p95_C - baseline.summer_p95_C,
    }


def risk_label(metrics: PerformanceMetrics) -> str:
    if metrics.wet_freeze_hours > 300 or metrics.freeze_thaw_cycles > 120:
        return "High"
    if metrics.wet_freeze_hours > 80 or metrics.freeze_thaw_cycles > 50:
        return "Moderate"
    return "Low"

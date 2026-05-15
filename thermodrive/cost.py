"""Transparent cost model and bill of materials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .thermosyphon import ThermosyphonDesign, pipe_count
from .units import M_TO_FT, M2_TO_SQFT

InstallType = Literal["Retrofit existing driveway", "New driveway / major replacement"]


@dataclass(frozen=True)
class CostEstimate:
    bom: pd.DataFrame
    subtotal_direct: float
    contractor_overhead_profit: float
    distribution_margin: float
    contingency: float
    total_base: float
    total_low: float
    total_high: float
    cost_per_sqft: float
    cost_per_pipe: float
    confidence: str


def _pipe_unit_cost_per_ft(diameter_m: float) -> float:
    d_mm = diameter_m * 1000.0
    if d_mm <= 20:
        return 3.85
    if d_mm <= 26:
        return 6.50
    if d_mm <= 35:
        return 9.75
    if d_mm <= 42:
        return 14.50
    return 22.00


def _factory_charge_cost(fluid: str) -> float:
    if "High-output" in fluid:
        return 135.0
    if "CO2" in fluid or "refrigerant" in fluid:
        return 82.0
    if "Water" in fluid:
        return 28.0
    return 48.0


def _line(lines: list[dict[str, float | str]], item: str, quantity: float, unit: str, unit_cost: float, note: str = "") -> None:
    lines.append(
        {
            "Item": item,
            "Quantity": quantity,
            "Unit": unit,
            "Unit cost": unit_cost,
            "Base cost": quantity * unit_cost,
            "Note": note,
        }
    )


def estimate_project_cost(
    area_m2: float,
    design: ThermosyphonDesign,
    install_type: InstallType,
    region_factor: float = 1.0,
    markup_enabled: bool = True,
) -> CostEstimate:
    """Create a sales-level installed cost estimate.

    Numbers are intentionally visible/editable. Treat as budgetary until a local
    contractor validates utility conflicts, pavement condition, access, and soil.
    """
    area_sqft = area_m2 * M2_TO_SQFT
    n_pipes = pipe_count(area_m2, design.spacing_m)
    length_ft = design.depth_m * M_TO_FT
    total_pipe_ft = n_pipes * length_ft
    existing = install_type == "Retrofit existing driveway"
    lines: list[dict[str, float | str]] = []

    _line(lines, "Thermosyphon tube stock", total_pipe_ft, "ft", _pipe_unit_cost_per_ft(design.diameter_m), f"{design.diameter_mm:.0f} mm tube")
    _line(lines, "End caps, charge port, QA fittings", n_pipes, "pipe", 16.0 if design.diameter_mm <= 26 else 23.0, "Factory-sealed assembly")
    _line(lines, "Working fluid charge + pressure test", n_pipes, "pipe", _factory_charge_cost(design.fluid), design.fluid)
    _line(lines, "External corrosion/wear coating", total_pipe_ft, "ft", 1.35, "Below-grade protection")
    high_output = "High-output" in design.fluid
    _line(lines, "Conductive grout / thermal backfill", total_pipe_ft, "ft", 3.10 if high_output else 1.65, "Improves contact resistance")
    if high_output:
        _line(lines, "Near-surface heat-spreader strip/manifold", area_sqft, "ft²", 2.35, "High-output package: spreads pipe heat into the slab")
        _line(lines, "Enhanced factory QA / pressure rating", n_pipes, "pipe", 42.0, "Higher capacity refrigerant-grade assembly")

    if existing:
        _line(lines, "Coring/drilling/excavation labor", total_pipe_ft, "vertical ft", 9.75 * region_factor, "Retrofit production rate")
        _line(lines, "Surface layout, sawcutting, traffic control", area_sqft, "ft²", 1.05 * region_factor, "Existing driveway")
        _line(lines, "Pavement restoration and seal", area_sqft, "ft²", 2.20 * region_factor, "Patch/finish allowance")
        _line(lines, "Spoils handling and disposal", n_pipes, "pipe", 14.0 * region_factor, "Local disposal varies")
        mobilization = 1050.0 * region_factor
    else:
        _line(lines, "Pipe placement labor during driveway work", total_pipe_ft, "vertical ft", 5.75 * region_factor, "Installed before paving")
        _line(lines, "Layout, protection, embedment coordination", area_sqft, "ft²", 0.65 * region_factor, "New construction")
        _line(lines, "Incremental finishing allowance", area_sqft, "ft²", 0.55 * region_factor, "Above normal paving")
        mobilization = 750.0 * region_factor

    _line(lines, "Mobilization / equipment minimum", 1, "lot", mobilization, "Crew, small rig, tools")
    _line(lines, "Utility locate and constructability review", 1, "lot", 350.0 * region_factor, "Pre-install checklist")
    _line(lines, "Design engineering and proposal package", 1, "lot", 450.0, "Sales-ready estimate")

    bom = pd.DataFrame(lines)
    direct = float(bom["Base cost"].sum())
    if markup_enabled:
        contractor = 0.15 * direct
        distribution = 0.08 * (direct + contractor)
        contingency = 0.10 * (direct + contractor + distribution)
    else:
        contractor = distribution = contingency = 0.0
    total = direct + contractor + distribution + contingency
    low = total * 0.86
    high = total * (1.24 if existing else 1.18)
    return CostEstimate(
        bom=bom,
        subtotal_direct=direct,
        contractor_overhead_profit=contractor,
        distribution_margin=distribution,
        contingency=contingency,
        total_base=total,
        total_low=low,
        total_high=high,
        cost_per_sqft=total / max(area_sqft, 1.0),
        cost_per_pipe=total / max(n_pipes, 1),
        confidence="Budgetary screening estimate; validate with local contractor before quotation.",
    )


def cost_summary_table(estimate: CostEstimate) -> pd.DataFrame:
    rows = [
        ("Direct materials + labor", estimate.subtotal_direct),
        ("Contractor overhead/profit", estimate.contractor_overhead_profit),
        ("Distribution margin", estimate.distribution_margin),
        ("Contingency", estimate.contingency),
        ("Total installed estimate", estimate.total_base),
    ]
    return pd.DataFrame(rows, columns=["Category", "Cost"])

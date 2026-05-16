"""Transparent sales-level cost model and bill of materials.

The heat-pipe hardware cost is not a single fixed value. It is anchored to the
latest vendor conversation (roughly $100 for a reference sealed thermosyphon),
then adjusted for pipe diameter, depth, fluid/package complexity, and job
quantity. This keeps the app close to the vendor guidance without pretending
that every pipe configuration costs exactly the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .thermosyphon import ThermosyphonDesign, booster_capacity_W_m2, pipe_count
from .units import M_TO_FT, M2_TO_SQFT

InstallType = Literal["Retrofit existing driveway", "New driveway / major replacement"]

# Backwards-compatible name used by the optimizer and app. This is now the
# vendor reference price for a standard factory-sealed pipe, not a literal unit
# price that applies to every diameter/depth/package.
DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST = 100.0
DEFAULT_VENDOR_REFERENCE_PIPE_COST = DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST


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
    factory_heat_pipe_unit_cost: float
    vendor_reference_unit_cost: float
    heat_pipe_material_subtotal: float
    heat_pipe_count: int
    confidence: str


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


def _round_to_nearest_five(value: float) -> float:
    return float(round(value / 5.0) * 5.0)


def _bounded(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def estimate_factory_heat_pipe_unit_cost(
    design: ThermosyphonDesign,
    heat_pipe_count: int,
    vendor_reference_unit_cost: float = DEFAULT_VENDOR_REFERENCE_PIPE_COST,
) -> float:
    """Estimate factory cost per sealed thermosyphon from a vendor anchor.

    Vendor guidance is treated as a benchmark for a reference pipe rather than a
    fixed catalog price. The scaling is intentionally modest because the vendor
    quote indicated that manufacturing can stay near $100/pipe, but larger,
    longer, pressure-rated, and hybrid-ready assemblies should cost somewhat
    more. Small/simple economy assemblies can price below the benchmark.
    """
    anchor = float(max(vendor_reference_unit_cost, 20.0))
    diameter_mm = max(design.diameter_mm, 10.0)
    depth_ft = max(design.depth_m * M_TO_FT, 3.0)
    fluid = design.fluid

    # Reference: about 25 mm diameter and 8 ft depth.
    diameter_factor = _bounded((diameter_mm / 25.0) ** 0.30, 0.84, 1.22)
    depth_factor = _bounded((depth_ft / 8.0) ** 0.18, 0.88, 1.15)

    if "Assured 90" in fluid:
        package_factor = 1.08
    elif "High-output" in fluid:
        package_factor = 1.04
    elif "CO2" in fluid or "refrigerant" in fluid:
        package_factor = 1.03
    elif "Water" in fluid:
        package_factor = 0.88
    else:
        package_factor = 0.98

    # Small jobs lose economies of scale; larger driveway fields get a modest
    # volume break. Keep it bounded so the unit price stays near the vendor
    # guidance during sales calls.
    n = max(int(heat_pipe_count), 1)
    quantity_factor = _bounded((n / 35.0) ** -0.06, 0.93, 1.10)

    raw = anchor * diameter_factor * depth_factor * package_factor * quantity_factor

    # Guardrails: stay close to the vendor conversation unless the user changes
    # the anchor. These are deliberately wide enough for package variation but
    # narrow enough to avoid reverting to the overly expensive older model.
    low = max(35.0, 0.72 * anchor)
    high = max(low + 5.0, 1.35 * anchor)
    return _round_to_nearest_five(_bounded(raw, low, high))


def _pipe_handling_cost(design: ThermosyphonDesign) -> float:
    """Small receiving/QA allowance outside the vendor's manufacturing price."""
    if "Assured 90" in design.fluid:
        return 16.0
    if "High-output" in design.fluid:
        return 12.0
    return 8.0


def estimate_project_cost(
    area_m2: float,
    design: ThermosyphonDesign,
    install_type: InstallType,
    region_factor: float = 1.0,
    markup_enabled: bool = True,
    factory_heat_pipe_unit_cost: float = DEFAULT_VENDOR_REFERENCE_PIPE_COST,
) -> CostEstimate:
    """Create a sales-level installed cost estimate.

    ``factory_heat_pipe_unit_cost`` is the editable vendor benchmark for a
    reference sealed thermosyphon. The BOM line uses a design-adjusted factory
    unit price derived from the benchmark, so the app stays close to the vendor
    quote without making every pipe exactly $100.
    """
    area_sqft = area_m2 * M2_TO_SQFT
    n_pipes = pipe_count(area_m2, design.spacing_m)
    length_ft = design.depth_m * M_TO_FT
    total_pipe_ft = n_pipes * length_ft
    existing = install_type == "Retrofit existing driveway"
    high_output = ("High-output" in design.fluid) or ("Assured 90" in design.fluid)
    hybrid = "Assured 90" in design.fluid or booster_capacity_W_m2(design) > 0
    vendor_anchor = float(max(factory_heat_pipe_unit_cost, 0.0))
    estimated_factory_cost = estimate_factory_heat_pipe_unit_cost(
        design,
        heat_pipe_count=n_pipes,
        vendor_reference_unit_cost=vendor_anchor,
    )
    lines: list[dict[str, float | str]] = []

    _line(
        lines,
        "Factory-manufactured sealed heat pipe",
        n_pipes,
        "pipe",
        estimated_factory_cost,
        (
            f"Design-adjusted from ${vendor_anchor:,.0f}/pipe vendor benchmark; "
            f"{design.diameter_mm:.0f} mm x {length_ft:.0f} ft, {design.fluid}"
        ),
    )
    _line(
        lines,
        "Receiving QA, labeling, protective handling",
        n_pipes,
        "pipe",
        _pipe_handling_cost(design),
        "Incoming inspection and jobsite protection outside factory cost",
    )

    # Thermal contact and distribution materials remain separate from the factory
    # heat-pipe cost because they are site/installation dependent.
    _line(lines, "Conductive grout / thermal backfill", total_pipe_ft, "vertical ft", 1.85 if hybrid else (1.65 if high_output else 0.95), "Improves soil/pipe contact")
    if high_output:
        _line(lines, "Near-surface heat-spreader strip/manifold", area_sqft, "ft2", 1.45 if hybrid else 1.15, "Spreads pipe heat into the slab")
    if hybrid:
        _line(lines, "Thermostat-controlled low-power assist mat/cable", area_sqft, "ft2", 4.25, "Assured 90 package; supplements during high-risk freeze hours")
        _line(lines, "Controls, sensors, relay panel", 1, "lot", 925.0 * region_factor, "Surface sensor + weather-aware controller")
        _line(lines, "Electrical rough-in and GFCI protection", area_sqft, "ft2", 1.25 * region_factor, "Budgetary allowance; electrician to verify")

    if existing:
        _line(lines, "Coring/drilling/excavation labor", total_pipe_ft, "vertical ft", 8.25 * region_factor, "Retrofit production rate")
        _line(lines, "Surface layout, sawcutting, traffic control", area_sqft, "ft2", 0.85 * region_factor, "Existing driveway")
        _line(lines, "Pavement restoration and seal", area_sqft, "ft2", 1.80 * region_factor, "Patch/finish allowance")
        _line(lines, "Spoils handling and disposal", n_pipes, "pipe", 11.0 * region_factor, "Local disposal varies")
        mobilization = 900.0 * region_factor
    else:
        _line(lines, "Pipe placement labor during driveway work", total_pipe_ft, "vertical ft", 3.85 * region_factor, "Installed before paving")
        _line(lines, "Layout, protection, embedment coordination", area_sqft, "ft2", 0.45 * region_factor, "New construction")
        _line(lines, "Incremental finishing allowance", area_sqft, "ft2", 0.35 * region_factor, "Above normal paving")
        mobilization = 650.0 * region_factor

    _line(lines, "Mobilization / equipment minimum", 1, "lot", mobilization, "Crew, small rig, tools")
    _line(lines, "Utility locate and constructability review", 1, "lot", 300.0 * region_factor, "Pre-install checklist")
    _line(lines, "Design engineering and proposal package", 1, "lot", 425.0, "Sales-ready estimate")

    bom = pd.DataFrame(lines)
    heat_pipe_material_subtotal = float(
        bom.loc[bom["Item"].eq("Factory-manufactured sealed heat pipe"), "Base cost"].sum()
    )
    direct = float(bom["Base cost"].sum())
    if markup_enabled:
        contractor = 0.13 * direct
        distribution = 0.06 * (direct + contractor)
        contingency = 0.08 * (direct + contractor + distribution)
    else:
        contractor = distribution = contingency = 0.0
    total = direct + contractor + distribution + contingency
    low = total * 0.88
    high = total * (1.18 if existing else 1.14)
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
        factory_heat_pipe_unit_cost=estimated_factory_cost,
        vendor_reference_unit_cost=vendor_anchor,
        heat_pipe_material_subtotal=heat_pipe_material_subtotal,
        heat_pipe_count=n_pipes,
        confidence="Budgetary screening estimate using an editable vendor benchmark and design-adjusted factory heat-pipe unit cost; validate installation labor and local conditions before quotation.",
    )


def cost_summary_table(estimate: CostEstimate) -> pd.DataFrame:
    rows = [
        ("Factory heat pipes", estimate.heat_pipe_material_subtotal),
        ("Other direct materials + labor", estimate.subtotal_direct - estimate.heat_pipe_material_subtotal),
        ("Contractor overhead/profit", estimate.contractor_overhead_profit),
        ("Distribution margin", estimate.distribution_margin),
        ("Contingency", estimate.contingency),
        ("Total installed estimate", estimate.total_base),
    ]
    return pd.DataFrame(rows, columns=["Category", "Cost"])

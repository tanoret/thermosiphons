"""Proposal/export helpers."""
from __future__ import annotations

import io
from dataclasses import asdict

import pandas as pd

from .metrics import PerformanceMetrics
from .optimizer import OptimizationResult
from .thermosyphon import summarize_design
from .units import M2_TO_SQFT, M_TO_FT, currency


def proposal_markdown(result: OptimizationResult, area_m2: float) -> str:
    base = result.baseline_metrics
    lines = [
        "# ThermoDrive Driveway Freeze-Risk Screening Proposal",
        "",
        f"**Location:** {result.baseline_result.site.label}",
        f"**Driveway area:** {area_m2 * M2_TO_SQFT:,.0f} ft²",
        f"**Status:** {result.status}",
        f"**Package:** {result.package_label}",
        "",
        "## Baseline risk",
        f"- Freeze hours: {base.freeze_hours:,}",
        f"- Wet-freeze hours: {base.wet_freeze_hours:,}",
        f"- Freeze-thaw cycles: {base.freeze_thaw_cycles:,}",
        f"- Winter 5th-percentile surface temperature: {base.winter_p5_C:.1f} °C",
    ]
    if result.design is not None and result.design_metrics is not None and result.cost is not None:
        design = result.design
        metrics = result.design_metrics
        comp = result.comparison
        lines.extend(
            [
                "",
                "## Recommended design",
                f"- Pipe spacing: {design.spacing_m * M_TO_FT:.1f} ft",
                f"- Pipe depth: {design.depth_m * M_TO_FT:.1f} ft",
                f"- Pipe diameter: {design.diameter_mm:.0f} mm",
                f"- Working fluid package: {design.fluid}",
                f"- Estimated installed cost: {currency(result.cost.total_base)} ({currency(result.cost.total_low)}–{currency(result.cost.total_high)})",
                f"- Cost per ft²: ${result.cost.cost_per_sqft:,.2f}",
                f"- Factory heat-pipe hardware: {currency(result.cost.heat_pipe_material_subtotal)} ({result.cost.heat_pipe_count:,} pipes × est. ${result.cost.factory_heat_pipe_unit_cost:,.0f}/pipe; vendor benchmark ${result.cost.vendor_reference_unit_cost:,.0f}/reference pipe)",
                "",
                "## Expected performance",
                f"- Freeze-hour reduction: {comp.get('freeze_hour_reduction_pct', 0):.0f}%",
                f"- Wet-freeze-hour reduction: {comp.get('wet_freeze_reduction_pct', 0):.0f}%",
                f"- Freeze-thaw-cycle reduction: {comp.get('freeze_thaw_reduction_pct', 0):.0f}%",
                f"- Passive thermosyphon heat delivered: {metrics.annual_hp_kWh_m2:.1f} kWh/m²",
                f"- Thermostat assist heat delivered: {metrics.annual_assist_kWh_m2:.1f} kWh/m²",
                f"- Total heat delivered: {metrics.total_heat_kWh_m2:.1f} kWh/m²",
            ]
        )
    lines.extend(
        [
            "",
            "## Important notes",
            result.recommendation_note,
            "This screening model is for concept sizing and sales qualification. Final proposals require local contractor pricing, utility locate, soil verification, electrical review for assisted packages, and product-specific thermosyphon testing.",
        ]
    )
    return "\n".join(lines)


def workbook_bytes(result: OptimizationResult, area_m2: float) -> bytes:
    """Return a multi-sheet XLSX-style byte stream if openpyxl is available; CSV zip fallback otherwise."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("proposal.md", proposal_markdown(result, area_m2))
        zf.writestr("baseline_metrics.csv", pd.DataFrame([result.baseline_metrics.as_dict()]).to_csv(index=False))
        if result.design_metrics is not None:
            zf.writestr("design_metrics.csv", pd.DataFrame([result.design_metrics.as_dict()]).to_csv(index=False))
        if result.cost is not None:
            zf.writestr("bill_of_materials.csv", result.cost.bom.to_csv(index=False))
        if result.candidates is not None and not result.candidates.empty:
            zf.writestr("candidate_search.csv", result.candidates.to_csv(index=False))
    return buffer.getvalue()

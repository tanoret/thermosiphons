from thermodrive.cost import (
    DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST,
    estimate_factory_heat_pipe_unit_cost,
    estimate_project_cost,
)
from thermodrive.thermosyphon import ThermosyphonDesign, pipe_count
from thermodrive.units import SQFT_TO_M2


def test_cost_has_bom_and_positive_total():
    design = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    cost = estimate_project_cost(500 * SQFT_TO_M2, design, "Retrofit existing driveway", region_factor=1.0)
    assert cost.total_base > 0
    assert len(cost.bom) >= 5
    assert cost.cost_per_sqft > 0


def test_vendor_benchmark_creates_design_adjusted_front_line_item():
    area_m2 = 500 * SQFT_TO_M2
    design = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    cost = estimate_project_cost(area_m2, design, "New driveway / major replacement", region_factor=1.0)
    n_pipes = pipe_count(area_m2, design.spacing_m)
    row = cost.bom.loc[cost.bom["Item"].eq("Factory-manufactured sealed heat pipe")].iloc[0]
    assert cost.vendor_reference_unit_cost == DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST
    assert row["Unit cost"] == cost.factory_heat_pipe_unit_cost
    assert 0.72 * DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST <= cost.factory_heat_pipe_unit_cost <= 1.35 * DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST
    assert row["Base cost"] == n_pipes * cost.factory_heat_pipe_unit_cost
    assert cost.heat_pipe_count == n_pipes
    assert cost.heat_pipe_material_subtotal == n_pipes * cost.factory_heat_pipe_unit_cost


def test_factory_unit_cost_scales_near_vendor_anchor():
    standard = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    high_output = ThermosyphonDesign(
        spacing_m=0.9,
        depth_m=7.3,
        diameter_m=0.05,
        fluid="Assured 90 hybrid thermosyphon + low-power assist",
    )
    standard_unit = estimate_factory_heat_pipe_unit_cost(standard, heat_pipe_count=35, vendor_reference_unit_cost=100)
    high_output_unit = estimate_factory_heat_pipe_unit_cost(high_output, heat_pipe_count=45, vendor_reference_unit_cost=100)
    assert 70 <= standard_unit <= 120
    assert high_output_unit > standard_unit
    assert high_output_unit <= 135


def test_vendor_benchmark_is_editable_and_design_adjustment_changes_nonreference_designs():
    reference = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    advanced = ThermosyphonDesign(
        spacing_m=0.9,
        depth_m=7.3,
        diameter_m=0.05,
        fluid="Assured 90 hybrid thermosyphon + low-power assist",
    )
    low = estimate_project_cost(500 * SQFT_TO_M2, reference, "New driveway / major replacement", factory_heat_pipe_unit_cost=85)
    high = estimate_project_cost(500 * SQFT_TO_M2, reference, "New driveway / major replacement", factory_heat_pipe_unit_cost=125)
    advanced_cost = estimate_project_cost(500 * SQFT_TO_M2, advanced, "New driveway / major replacement", factory_heat_pipe_unit_cost=100)
    assert low.vendor_reference_unit_cost == 85
    assert high.vendor_reference_unit_cost == 125
    assert high.total_base > low.total_base
    assert advanced_cost.factory_heat_pipe_unit_cost != advanced_cost.vendor_reference_unit_cost

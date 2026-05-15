from thermodrive.cost import estimate_project_cost
from thermodrive.thermosyphon import ThermosyphonDesign
from thermodrive.units import SQFT_TO_M2


def test_cost_has_bom_and_positive_total():
    design = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    cost = estimate_project_cost(500 * SQFT_TO_M2, design, "Retrofit existing driveway", region_factor=1.0)
    assert cost.total_base > 0
    assert len(cost.bom) >= 5
    assert cost.cost_per_sqft > 0

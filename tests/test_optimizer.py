from thermodrive.climate import generate_typical_hourly_weather, resolve_site
from thermodrive.optimizer import optimize_design
from thermodrive.physics import SimulationConfig
from thermodrive.soil import default_soil
from thermodrive.units import SQFT_TO_M2, FT_TO_M


def test_optimizer_returns_result_for_cold_state():
    site = resolve_site("Minnesota", "")
    weather = generate_typical_hourly_weather(site, 2025)
    soil = default_soil("Auto / balanced loam")
    cfg = SimulationConfig(max_depth_m=4.0, spinup_years=0)
    result = optimize_design(
        weather=weather,
        site=site,
        soil=soil,
        config=cfg,
        area_m2=600 * SQFT_TO_M2,
        install_type="New driveway / major replacement",
        region_factor=1.0,
        goal="Reduce freeze risk",
        target_reduction_pct=20,
        max_depth_m=8 * FT_TO_M,
        fluid="Methanol blend (factory sealed)",
        full_verify_count=4,
    )
    assert result.baseline_metrics.freeze_hours > 0
    assert result.status
    assert result.candidates is not None

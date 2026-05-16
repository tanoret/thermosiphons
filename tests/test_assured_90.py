from thermodrive.climate import generate_typical_hourly_weather, resolve_site
from thermodrive.metrics import compute_metrics, compare_metrics
from thermodrive.physics import SimulationConfig, run_thermal_simulation
from thermodrive.soil import default_soil
from thermodrive.thermosyphon import ThermosyphonDesign, booster_capacity_W_m2
from thermodrive.units import FT_TO_M


def test_assured_90_package_reduces_idaho_freeze_hours_above_90_percent():
    site = resolve_site("Idaho", "")
    weather = generate_typical_hourly_weather(site, 2025)
    soil = default_soil("Auto / balanced loam")
    cfg = SimulationConfig(max_depth_m=24 * FT_TO_M, spinup_years=0)
    baseline = run_thermal_simulation(weather, site, soil, cfg)
    design = ThermosyphonDesign(
        spacing_m=4.0 * FT_TO_M,
        depth_m=18.0 * FT_TO_M,
        diameter_m=0.032,
        top_depth_m=0.055,
        fluid="Assured 90 hybrid thermosyphon + low-power assist",
    )
    result = run_thermal_simulation(weather, site, soil, cfg, thermosyphon=design)
    comp = compare_metrics(compute_metrics(baseline), compute_metrics(result))
    assert booster_capacity_W_m2(design) > 0
    assert comp["freeze_hour_reduction_pct"] >= 90.0
    assert result.assist_flux_W_m2.sum() > 0

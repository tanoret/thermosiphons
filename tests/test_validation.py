from thermodrive.climate import generate_typical_hourly_weather, resolve_site
from thermodrive.physics import SimulationConfig, run_thermal_simulation
from thermodrive.soil import default_soil
from thermodrive.validation import score_simulation_against_noaa, validation_summary_table, TuningResult


def test_validation_score_accepts_noaa_style_observations():
    site = resolve_site("Illinois", "")
    weather = generate_typical_hourly_weather(site, 2025).iloc[:240].copy()
    soil = default_soil("Auto / balanced loam")
    cfg = SimulationConfig(max_depth_m=2.0, spinup_years=0)
    result = run_thermal_simulation(weather, site, soil, cfg)
    obs = weather.copy()
    obs["observed_surface_C"] = result.surface_C + 0.25
    obs["observed_soil_20cm_C"] = result.temperature_C[:, 10] + 0.20
    score = score_simulation_against_noaa(result, obs)
    assert score.surface_rmse_C is not None
    assert score.surface_rmse_C < 0.5
    assert score.composite_score < 1.0


def test_validation_summary_empty_safe():
    assert validation_summary_table(None).empty

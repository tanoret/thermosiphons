from thermodrive.climate import generate_typical_hourly_weather, resolve_site
from thermodrive.metrics import compute_metrics
from thermodrive.physics import SimulationConfig, run_thermal_simulation
from thermodrive.soil import default_soil
from thermodrive.thermosyphon import ThermosyphonDesign


def test_baseline_simulation_runs_one_year():
    site = resolve_site("Illinois", "")
    weather = generate_typical_hourly_weather(site, 2025)
    soil = default_soil("Auto / balanced loam")
    cfg = SimulationConfig(max_depth_m=4.0, spinup_years=0)
    result = run_thermal_simulation(weather, site, soil, cfg)
    assert len(result.time_series) == 8760
    assert result.temperature_C.shape[0] == 8760
    assert result.surface_C.notna().all()
    metrics = compute_metrics(result)
    assert metrics.max_surface_C > metrics.min_surface_C


def test_thermosyphon_adds_nonnegative_heat_flux():
    site = resolve_site("Minnesota", "")
    weather = generate_typical_hourly_weather(site, 2025)
    soil = default_soil("Auto / balanced loam")
    cfg = SimulationConfig(max_depth_m=4.0, spinup_years=0)
    design = ThermosyphonDesign(spacing_m=1.2, depth_m=2.4, diameter_m=0.025)
    result = run_thermal_simulation(weather, site, soil, cfg, thermosyphon=design)
    assert (result.hp_flux_W_m2 >= -1e-9).all()
    assert result.hp_flux_W_m2.sum() >= 0

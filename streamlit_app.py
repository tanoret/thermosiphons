from __future__ import annotations

import pandas as pd
import streamlit as st

from thermodrive.climate import load_weather, resolve_site
try:
    from thermodrive.cost import DEFAULT_VENDOR_REFERENCE_PIPE_COST, InstallType
except ImportError:  # Backward compatibility for older checked-out cost.py files.
    from thermodrive.cost import InstallType
    try:
        from thermodrive.cost import DEFAULT_FACTORY_HEAT_PIPE_UNIT_COST as DEFAULT_VENDOR_REFERENCE_PIPE_COST
    except ImportError:
        DEFAULT_VENDOR_REFERENCE_PIPE_COST = 100.0
from thermodrive.metrics import risk_label
from thermodrive.optimizer import Goal, OptimizationResult, optimize_design
from thermodrive.physics import PAVEMENT_LIBRARY, SimulationConfig, run_thermal_simulation
from thermodrive.plots import (
    candidate_scatter,
    climate_temperature_plot,
    cost_waterfall,
    freeze_calendar_plot,
    hp_flux_plot,
    monthly_performance_plot,
    pipe_layout_plot,
    seasonal_depth_profiles,
    soil_heatmap,
    soil_validation_plot,
    source_comparison_plot,
    tuning_score_plot,
    validation_overlay_plot,
)
from thermodrive.report import proposal_markdown, workbook_bytes
from thermodrive.soil import SoilProfile, choose_soil_profile
from thermodrive.state_data import STATE_NAMES, get_state_defaults
from thermodrive.thermosyphon import FLUID_LIBRARY, pipe_count, summarize_design
from thermodrive.units import FT_TO_M, M2_TO_SQFT, SQFT_TO_M2, c_to_f, currency
from thermodrive.validation import TuningResult, tune_model_against_noaa_uscrn, validation_summary_table

st.set_page_config(
    page_title="ThermoDrive | Passive Driveway Thermosyphon Design",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
  --blue: #1E88E5;
  --navy: #102033;
  --muted: #5B6B7C;
  --bg: #F7F9FC;
  --card: #FFFFFF;
  --line: rgba(16,32,51,0.10);
  --green: #2E7D32;
  --orange: #F39C12;
  --red: #C62828;
}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px;}
.hero {
  background: linear-gradient(135deg, #102033 0%, #174C80 52%, #1E88E5 100%);
  color: white;
  padding: 1.75rem 2rem;
  border-radius: 24px;
  margin-bottom: 1rem;
  box-shadow: 0 22px 60px rgba(16, 32, 51, .18);
}
.hero h1 {font-size: 2.35rem; margin: 0 0 .3rem 0; letter-spacing: -0.04em;}
.hero p {font-size: 1.05rem; opacity: .92; max-width: 1040px; margin: 0;}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.05rem 1.1rem;
  box-shadow: 0 12px 30px rgba(16,32,51,.06);
  min-height: 116px;
}
.card .label {color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700;}
.card .value {color: var(--navy); font-size: 1.7rem; font-weight: 800; line-height: 1.1; margin-top: .35rem;}
.card .note {color: var(--muted); font-size: .9rem; margin-top: .4rem;}
.badge {display:inline-block; padding:.35rem .65rem; border-radius:999px; font-weight:700; font-size:.82rem; margin:.10rem .15rem .10rem 0;}
.badge-blue {background:rgba(30,136,229,.12); color:#0D5EA8;}
.badge-green {background:rgba(46,125,50,.12); color:#1B5E20;}
.badge-orange {background:rgba(243,156,18,.14); color:#9B5D00;}
.badge-red {background:rgba(198,40,40,.12); color:#9E1B1B;}
.small-muted {color: var(--muted); font-size: .92rem;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
[data-testid="stMetric"] {
  background: white; padding: 1rem; border: 1px solid var(--line); border-radius: 16px;
  box-shadow: 0 8px 20px rgba(16,32,51,.04);
}
hr {border: none; border-top: 1px solid var(--line); margin: 1.4rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, note: str = "", badge_class: str = "badge-blue") -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_site(state: str, zip_code: str):
    return resolve_site(state, zip_code)


@st.cache_data(show_spinner=False)
def cached_weather(state: str, zip_code: str, mode: str, year: int):
    site = resolve_site(state, zip_code)
    return load_weather(site, mode, year=year)


@st.cache_data(show_spinner=False)
def cached_soil(state: str, zip_code: str, texture: str, use_usda: bool):
    site = resolve_site(state, zip_code)
    return choose_soil_profile(site, texture, use_usda=use_usda)


@st.cache_data(show_spinner=False)
def cached_tuning(state: str, zip_code: str, soil: SoilProfile, config: SimulationConfig, year: int, enabled: bool) -> TuningResult | None:
    if not enabled:
        return None
    site = resolve_site(state, zip_code)
    return tune_model_against_noaa_uscrn(site, soil, config, year=year)


@st.cache_data(show_spinner=False)
def cached_optimization(
    weather: pd.DataFrame,
    state: str,
    zip_code: str,
    soil: SoilProfile,
    config: SimulationConfig,
    area_m2: float,
    install_type: InstallType,
    goal: Goal,
    target_reduction_pct: float,
    max_depth_m: float,
    fluid: str,
    full_verify_count: int,
    vendor_reference_pipe_cost: float,
):
    site = resolve_site(state, zip_code)
    baseline = run_thermal_simulation(weather, site, soil, config, thermosyphon=None)
    region_factor = float(get_state_defaults(site.state)["region_factor"])
    return optimize_design(
        weather=weather,
        site=site,
        soil=soil,
        config=config,
        area_m2=area_m2,
        install_type=install_type,
        region_factor=region_factor,
        goal=goal,
        target_reduction_pct=target_reduction_pct,
        max_depth_m=max_depth_m,
        fluid=fluid,
        baseline_result=baseline,
        full_verify_count=full_verify_count,
        factory_heat_pipe_unit_cost=vendor_reference_pipe_cost,
    )


st.markdown(
    """
    <div class="hero">
      <h1>ThermoDrive™ passive driveway thermosyphon designer</h1>
      <p>Screen winter freeze risk, tune the physics with NASA POWER and NOAA USCRN validation data, size a low-cost thermosyphon field, and export a sales-ready installed-cost proposal.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

CLIMATE_MODES = [
    "Typical screening year",
    "NASA POWER hourly year",
    "NOAA USCRN station year",
    "NASA + NOAA tuned validation year",
]
YEARS = [2024, 2023, 2022, 2021, 2020, 2019]

with st.sidebar:
    st.subheader("1. Location")
    state = st.selectbox("State", STATE_NAMES, index=STATE_NAMES.index("Idaho"))
    zip_code = st.text_input("ZIP code (optional)", value="", placeholder="e.g., 60614")
    climate_mode = st.selectbox("Climate and validation source", CLIMATE_MODES, index=0)
    weather_year = st.selectbox("Weather / validation year", YEARS, index=0, disabled=climate_mode == "Typical screening year")

    st.divider()
    st.subheader("2. Driveway")
    area_sqft = st.number_input("Driveway area (ft²)", min_value=150, max_value=20000, value=720, step=25)
    pavement_type = st.selectbox("Pavement type", list(PAVEMENT_LIBRARY.keys()), index=0)
    pavement_thickness_in = st.slider("Pavement thickness (in)", 2.0, 10.0, 4.5, 0.5)
    install_type: InstallType = st.radio("Installation scenario", ["Retrofit existing driveway", "New driveway / major replacement"], index=1)

    st.divider()
    st.subheader("3. Goal")
    goal: Goal = st.radio("Customer goal", ["Reduce freeze risk", "Reduce thermal cracking", "Overheating analysis"], index=0)
    if goal == "Reduce thermal cracking":
        target_default = 25
        target_help = "Target reduction in freeze-thaw / thermal stress index."
    elif goal == "Overheating analysis":
        target_default = 0
        target_help = "Thermosyphon optimization is disabled for overheating-only goals."
    else:
        target_default = 90
        target_help = "Target reduction in wet-freeze or freeze hours. Passive-only designs may not reach 90% in cold states; the Assured 90 hybrid package adds transparent thermostat-controlled assist."
    target_reduction_pct = st.slider("Target reduction (%)", 0, 95, target_default, 5, help=target_help, disabled=goal == "Overheating analysis")

    fluid_options = list(FLUID_LIBRARY.keys())
    fluid = st.selectbox(
        "Thermosyphon package",
        fluid_options,
        index=fluid_options.index("Assured 90 hybrid thermosyphon + low-power assist") if "Assured 90 hybrid thermosyphon + low-power assist" in fluid_options else (fluid_options.index("High-output CO2 + thermal grout + heat spreader") if "High-output CO2 + thermal grout + heat spreader" in fluid_options else 0),
        help="The passive packages expand to larger diameters, closer spacing, deeper boreholes, thermal grout, and a near-surface heat spreader. Assured 90 adds thermostat-controlled assist when passive-only heat cannot meet high freeze-hour targets.",
    )
    max_depth_ft = st.slider("Maximum constructable depth (ft)", 4.0, 32.0, 24.0, 1.0)

    st.divider()
    st.subheader("4. Pricing")
    vendor_reference_pipe_cost = st.number_input(
        "Vendor benchmark cost ($/reference pipe)",
        min_value=20.0,
        max_value=1000.0,
        value=float(DEFAULT_VENDOR_REFERENCE_PIPE_COST),
        step=5.0,
        help="Vendor benchmark for a reference factory-manufactured sealed thermosyphon. The app adjusts the actual pipe unit cost for diameter, depth, package complexity, and quantity, while keeping it close to this benchmark.",
    )

    st.divider()
    with st.expander("Advanced model, soil, and validation", expanded=False):
        soil_texture = st.selectbox("Soil assumption", ["Auto / balanced loam", "Sandy / well drained", "Clay / high plasticity", "Wet / high water table", "Gravelly / engineered fill"])
        use_usda = st.toggle("Try USDA Soil Data Access lookup", value=False)
        custom_albedo_enabled = st.toggle("Override pavement albedo", value=False)
        custom_albedo = st.slider("Albedo", 0.05, 0.65, 0.32, 0.01, disabled=not custom_albedo_enabled)
        spinup_years = st.slider("Thermal spin-up years", 0, 3, 0, 1)
        precipitation_energy = st.toggle("Include rain/snow energy loads", value=True)
        evap_factor = st.slider("Wet-pavement evaporation factor", 0.0, 1.0, 0.35, 0.05)
        enable_tuning_default = climate_mode == "NASA + NOAA tuned validation year"
        enable_tuning = st.toggle("Tune model using nearest NOAA USCRN station", value=enable_tuning_default)
        full_verify_count = st.slider("Verified candidate simulations", 3, 20, 5, 1)

    run_button = st.button("Generate proposal", type="primary", use_container_width=True)

if "has_run" not in st.session_state:
    st.session_state.has_run = True
if run_button:
    st.session_state.has_run = True

area_m2 = area_sqft * SQFT_TO_M2
max_depth_m = max_depth_ft * FT_TO_M
site = cached_site(state, zip_code)
raw_soil = cached_soil(state, zip_code, soil_texture, use_usda)
base_config = SimulationConfig(
    pavement_type=pavement_type,
    pavement_thickness_m=pavement_thickness_in / 39.3701,
    base_thickness_m=0.15,
    max_depth_m=max(max_depth_m + 0.8, 4.0),
    spinup_years=spinup_years,
    custom_albedo=custom_albedo if custom_albedo_enabled else None,
    precipitation_energy_enabled=precipitation_energy,
    evaporative_cooling_factor=evap_factor,
)

if not st.session_state.has_run:
    st.stop()

with st.spinner("Loading weather, validating/tuning model, and generating design proposal..."):
    weather = cached_weather(state, zip_code, climate_mode, weather_year)
    tuning = cached_tuning(state, zip_code, raw_soil, base_config, weather_year, enable_tuning or climate_mode == "NASA + NOAA tuned validation year")
    soil = tuning.tuned_soil if tuning is not None and tuning.applied else raw_soil
    config = tuning.tuned_config if tuning is not None and tuning.applied else base_config
    opt: OptimizationResult = cached_optimization(
        weather,
        state,
        zip_code,
        soil,
        config,
        area_m2,
        install_type,
        goal,
        float(target_reduction_pct),
        max_depth_m,
        fluid,
        full_verify_count,
        float(vendor_reference_pipe_cost),
    )

baseline = opt.baseline_result
design_result = opt.design_result
base_metrics = opt.baseline_metrics
design_metrics = opt.design_metrics
risk = risk_label(base_metrics)
site_line = f"{site.label} · {site.latitude:.2f}, {site.longitude:.2f} · {site.source}"
source_line = str(weather["data_source"].iloc[0]) if "data_source" in weather else "Climate data"
soil_line = f"{soil.name} · k={soil.k_W_mK:.2f} W/m-K · {soil.source}"

badges = [
    f"<span class='badge badge-blue'>{site_line}</span>",
    f"<span class='badge badge-orange'>{source_line}</span>",
]
if tuning is not None and tuning.applied:
    badges.append("<span class='badge badge-green'>NASA/NOAA tuned</span>")
elif tuning is not None and not tuning.applied:
    badges.append("<span class='badge badge-red'>NOAA tuning unavailable</span>")
st.markdown(" ".join(badges), unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>{soil_line}</div>", unsafe_allow_html=True)

st.subheader("Cost-first estimate")
cost_cols = st.columns(5)
with cost_cols[0]:
    if opt.cost is not None:
        card("Installed estimate", currency(opt.cost.total_base), f"{currency(opt.cost.total_low)}–{currency(opt.cost.total_high)}")
    else:
        card("Installed estimate", "—", "No thermosyphon package selected")
with cost_cols[1]:
    if opt.cost is not None:
        card("Installed cost / ft²", f"${opt.cost.cost_per_sqft:,.2f}", f"{area_sqft:,.0f} ft² driveway")
    else:
        card("Installed cost / ft²", "—", "No thermosyphon package selected")
with cost_cols[2]:
    if opt.cost is not None:
        card("Est. factory unit", f"${opt.cost.factory_heat_pipe_unit_cost:,.0f}/pipe", f"Benchmark ${opt.cost.vendor_reference_unit_cost:,.0f}/reference pipe")
    else:
        card("Est. factory unit", "—", f"Benchmark ${vendor_reference_pipe_cost:,.0f}/reference pipe")
with cost_cols[3]:
    if opt.cost is not None:
        card("Factory pipe subtotal", currency(opt.cost.heat_pipe_material_subtotal), f"{opt.cost.heat_pipe_count:,} pipes")
    else:
        card("Factory pipe subtotal", "—", "No thermosyphon package selected")
with cost_cols[4]:
    if opt.design is not None:
        n = pipe_count(area_m2, opt.design.spacing_m)
        card("Recommended field", f"{n:,} pipes", f"{opt.design.depth_m/FT_TO_M:.0f} ft deep · {opt.design.spacing_m/FT_TO_M:.1f} ft spacing")
    else:
        card("Recommended field", "No-build", opt.package_label)

st.caption("Cost model uses the vendor benchmark up front, then estimates each pipe near that value based on diameter, depth, package, and quantity. Installation, grout, heat-spreader, controls/electrical, contractor overhead, distribution, and contingency are still included in the installed estimate.")

st.subheader("Performance snapshot")
p1, p2, p3, p4 = st.columns(4)
with p1:
    badge = "High" if risk == "High" else "Moderate" if risk == "Moderate" else "Low"
    card("Baseline freeze risk", badge, f"{base_metrics.wet_freeze_hours:,} wet-freeze hours · {base_metrics.freeze_thaw_cycles:,} freeze-thaw cycles")
with p2:
    if opt.comparison:
        primary = opt.comparison.get("wet_freeze_reduction_pct", opt.comparison.get("freeze_hour_reduction_pct", 0))
        if goal == "Reduce thermal cracking":
            primary = max(
                opt.comparison.get("stress_index_reduction_pct", 0),
                opt.comparison.get("freeze_thaw_reduction_pct", 0),
                opt.comparison.get("freeze_degree_hour_reduction_pct", 0),
            )
        card("Verified reduction", f"{primary:.0f}%", opt.status)
    else:
        card("Verified reduction", "N/A", opt.status)
with p3:
    if design_metrics is not None:
        card("Passive heat", f"{design_metrics.annual_hp_kWh_m2:.1f} kWh/m²", "Annual thermosyphon heat delivered")
    else:
        card("Passive heat", "—", "No thermosyphon package selected")
with p4:
    if design_metrics is not None:
        card("Assist heat", f"{design_metrics.annual_assist_kWh_m2:.1f} kWh/m²", "Annual thermostat assist energy")
    else:
        card("Assist heat", "—", "No assist package selected")

st.info(opt.recommendation_note)
if "Assured 90" in fluid:
    st.warning(
        "Assured 90 mode is enabled: passive thermosyphons carry the base heat load, but a thermostat-controlled assist layer is allowed to meet high freeze-hour targets. The dashboard reports passive and assisted heat separately so the claim stays transparent."
    )
elif "High-output" in fluid:
    st.warning(
        "High-output concept mode is enabled: results assume a pressure-rated CO2 assembly, conductive grout, closer spacing, and a near-surface heat spreader. Treat this as an aggressive concept for customer qualification until lab/field testing confirms the package constants."
    )
if tuning is not None and tuning.applied:
    st.success(tuning.note)

(tab_overview, tab_design, tab_technical, tab_validation, tab_cost, tab_assumptions) = st.tabs(
    ["Overview", "Recommended design", "Technical plots", "Validation & tuning", "Cost & proposal", "Assumptions"]
)

with tab_overview:
    left, right = st.columns([2.1, 1.0])
    with left:
        st.plotly_chart(climate_temperature_plot(weather, baseline, design_result), use_container_width=True)
    with right:
        st.subheader("Screening metrics")
        st.metric("Baseline freeze hours", f"{base_metrics.freeze_hours:,}")
        st.metric("Baseline wet-freeze hours", f"{base_metrics.wet_freeze_hours:,}")
        st.metric("Winter 5th-percentile surface", f"{base_metrics.winter_p5_C:.1f} °C", f"{c_to_f(base_metrics.winter_p5_C):.1f} °F")
        if design_metrics is not None:
            comp = opt.comparison
            st.metric("Freeze-hour reduction", f"{comp.get('freeze_hour_reduction_pct', 0):.0f}%")
            st.metric("Wet-freeze reduction", f"{comp.get('wet_freeze_reduction_pct', 0):.0f}%")
    st.plotly_chart(monthly_performance_plot(baseline, design_result), use_container_width=True)
    st.plotly_chart(freeze_calendar_plot(baseline, design_result), use_container_width=True)

with tab_design:
    if opt.design is None:
        st.warning(opt.status)
        st.write(opt.recommendation_note)
        st.plotly_chart(pipe_layout_plot(area_m2, None), use_container_width=True)
    else:
        dsum = summarize_design(opt.design, area_m2, soil.k_W_mK, top_k_W_mK=1.2)
        design_cols = st.columns(5)
        design_cols[0].metric("Pipe count", f"{dsum['pipe_count']:,}")
        design_cols[1].metric("Spacing", f"{opt.design.spacing_m/FT_TO_M:.1f} ft")
        design_cols[2].metric("Depth", f"{opt.design.depth_m/FT_TO_M:.0f} ft")
        design_cols[3].metric("Diameter", f"{opt.design.diameter_mm:.0f} mm")
        design_cols[4].metric("Capacity", f"{dsum['qmax_W_per_pipe']:.0f} W/pipe")
        st.plotly_chart(pipe_layout_plot(area_m2, opt.design), use_container_width=True)

        st.subheader("Performance comparison")
        perf = pd.DataFrame(
            [
                {"Metric": "Freeze hours", "Baseline": base_metrics.freeze_hours, "Design": design_metrics.freeze_hours if design_metrics else None, "Reduction": f"{opt.comparison.get('freeze_hour_reduction_pct', 0):.0f}%"},
                {"Metric": "Wet-freeze hours", "Baseline": base_metrics.wet_freeze_hours, "Design": design_metrics.wet_freeze_hours if design_metrics else None, "Reduction": f"{opt.comparison.get('wet_freeze_reduction_pct', 0):.0f}%"},
                {"Metric": "Freeze-thaw cycles", "Baseline": base_metrics.freeze_thaw_cycles, "Design": design_metrics.freeze_thaw_cycles if design_metrics else None, "Reduction": f"{opt.comparison.get('freeze_thaw_reduction_pct', 0):.0f}%"},
                {"Metric": "Freeze degree-hours (°C·h)", "Baseline": round(base_metrics.freeze_degree_hours_C_h, 0), "Design": round(design_metrics.freeze_degree_hours_C_h, 0) if design_metrics else None, "Reduction": f"{opt.comparison.get('freeze_degree_hour_reduction_pct', 0):.0f}%"},
                {"Metric": "95th-percentile daily swing (°C)", "Baseline": round(base_metrics.p95_daily_swing_C, 1), "Design": round(design_metrics.p95_daily_swing_C, 1) if design_metrics else None, "Reduction": f"{opt.comparison.get('daily_swing_reduction_pct', 0):.0f}%"},
                {"Metric": "Passive heat delivered (kWh/m²)", "Baseline": 0, "Design": round(design_metrics.annual_hp_kWh_m2, 1) if design_metrics else None, "Reduction": "—"},
                {"Metric": "Assist heat delivered (kWh/m²)", "Baseline": 0, "Design": round(design_metrics.annual_assist_kWh_m2, 1) if design_metrics else None, "Reduction": "—"},
                {"Metric": "Total heat delivered (kWh/m²)", "Baseline": 0, "Design": round(design_metrics.total_heat_kWh_m2, 1) if design_metrics else None, "Reduction": "—"},
            ]
        )
        st.dataframe(perf, hide_index=True, use_container_width=True)

        st.subheader("Candidate search")
        st.plotly_chart(candidate_scatter(opt.candidates), use_container_width=True)
        if not opt.candidates.empty:
            display_candidates = opt.candidates.copy()
            money_cols = [c for c in display_candidates.columns if "cost" in c or c == "base_cost"]
            for col in money_cols:
                display_candidates[col] = display_candidates[col].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            st.dataframe(display_candidates.head(20), hide_index=True, use_container_width=True)

with tab_technical:
    st.subheader("Temperature distribution")
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(soil_heatmap(baseline, "Baseline driveway/soil temperature"), use_container_width=True)
    with h2:
        if design_result is not None:
            st.plotly_chart(soil_heatmap(design_result, "With thermosyphon field"), use_container_width=True)
        else:
            st.plotly_chart(soil_heatmap(baseline, "Baseline only"), use_container_width=True)
    st.plotly_chart(seasonal_depth_profiles(baseline, design_result), use_container_width=True)
    if design_result is not None:
        st.plotly_chart(hp_flux_plot(design_result), use_container_width=True)

with tab_validation:
    st.subheader("NASA / NOAA validation and tuning")
    if tuning is None and climate_mode != "NOAA USCRN station year":
        st.info("Enable 'Tune model using nearest NOAA USCRN station' in Advanced settings, or choose the NASA + NOAA tuned validation mode, to run calibration.")
    elif tuning is not None:
        st.dataframe(validation_summary_table(tuning), hide_index=True, use_container_width=True)
        noaa_weather = tuning.validation_weather
        nasa_weather = weather if "NASA" in source_line or climate_mode == "NASA + NOAA tuned validation year" else None
        st.plotly_chart(source_comparison_plot(nasa_weather, noaa_weather), use_container_width=True)
        st.plotly_chart(validation_overlay_plot(tuning.validation_result, noaa_weather), use_container_width=True)
        st.plotly_chart(soil_validation_plot(tuning.validation_result, noaa_weather), use_container_width=True)
        st.plotly_chart(tuning_score_plot(tuning.trials), use_container_width=True)
        if not tuning.trials.empty:
            st.subheader("Calibration trials")
            st.dataframe(tuning.trials.round(3), hide_index=True, use_container_width=True)
    else:
        # NOAA climate mode without separate calibration still exposes observed data in the weather frame.
        st.plotly_chart(validation_overlay_plot(baseline, weather), use_container_width=True)
        st.plotly_chart(soil_validation_plot(baseline, weather), use_container_width=True)

with tab_cost:
    if opt.cost is None:
        st.warning("No thermosyphon cost estimate is shown because the selected goal does not produce a thermosyphon package.")
    else:
        st.info(f"Factory heat-pipe hardware is estimated at ${opt.cost.factory_heat_pipe_unit_cost:,.0f}/pipe from a ${opt.cost.vendor_reference_unit_cost:,.0f}/reference-pipe vendor benchmark. The installed estimate adds installation, thermal materials, controls/electrical if applicable, overhead, distribution, and contingency.")
        cost_cols = st.columns(5)
        cost_cols[0].metric("Base installed", currency(opt.cost.total_base))
        cost_cols[1].metric("Low–high range", f"{currency(opt.cost.total_low)} – {currency(opt.cost.total_high)}")
        cost_cols[2].metric("Per ft²", f"${opt.cost.cost_per_sqft:,.2f}")
        cost_cols[3].metric("Installed / pipe", currency(opt.cost.cost_per_pipe))
        cost_cols[4].metric("Factory subtotal", currency(opt.cost.heat_pipe_material_subtotal), f"est. ${opt.cost.factory_heat_pipe_unit_cost:,.0f}/pipe")
        st.plotly_chart(cost_waterfall(opt.cost), use_container_width=True)
        st.subheader("Bill of materials")
        bom = opt.cost.bom.copy()
        bom["Unit cost"] = bom["Unit cost"].map(lambda x: f"${x:,.2f}")
        bom["Base cost"] = opt.cost.bom["Base cost"].map(lambda x: f"${x:,.0f}")
        st.dataframe(bom, hide_index=True, use_container_width=True)
        st.caption(opt.cost.confidence)

    st.subheader("Export proposal package")
    md = proposal_markdown(opt, area_m2)
    st.download_button(
        "Download proposal markdown",
        data=md.encode("utf-8"),
        file_name="thermodrive_proposal.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        "Download proposal data package (ZIP)",
        data=workbook_bytes(opt, area_m2),
        file_name="thermodrive_proposal_data.zip",
        mime="application/zip",
        use_container_width=True,
    )

with tab_assumptions:
    st.subheader("Model and sales assumptions")
    st.markdown(
        """
        **What the model does:** solves a 1D transient finite-difference driveway/base/soil heat equation with hourly air temperature, wind, humidity, solar radiation, rain/snow energy loads, and a linearized surface energy balance. The thermosyphon field is represented as a one-way heat-transfer source/sink distributed over a finite evaporator length and a near-surface condenser/heat-spreader zone.

        **NASA/NOAA tuning:** NASA POWER provides gridded nationwide hourly weather; NOAA USCRN provides high-quality observed validation data where nearby stations exist. The tuning routine fits albedo, soil conductivity, ground mean offset, convection, and sky-temperature correction within conservative bounds.

        **What the model does not promise:** guaranteed passive-only snow melting during design blizzards. A wickless vertical thermosyphon requires gravity return of condensate, so it is useful for upward winter heat transport but is not a summer cooling device. The Assured 90 package is intentionally hybrid: thermosyphons reduce the base load and a controlled assist layer closes the remaining freeze-hour gap.

        **Use this app for:** screening, sales qualification, package comparison, preliminary cost range, and identifying locations where a passive product is likely or unlikely to be attractive.

        **Use engineering review for:** final quotes, stamped drawings, utility conflicts, frost-heave-prone soils, groundwater, pressure-rated working fluid selection, and field/lab validation of the factory thermosyphon assembly.
        """
    )
    st.subheader("Current inputs")
    inputs = pd.DataFrame(
        [
            ["State", state],
            ["ZIP", zip_code or "not provided"],
            ["Climate source", source_line],
            ["NASA/NOAA tuning", tuning.status if tuning is not None else "not requested"],
            ["Driveway area", f"{area_sqft:,.0f} ft²"],
            ["Pavement", pavement_type],
            ["Pavement thickness", f"{pavement_thickness_in:.1f} in"],
            ["Install type", install_type],
            ["Goal", goal],
            ["Target", f"{target_reduction_pct}%"],
            ["Soil", soil_line],
            ["Thermosyphon package", fluid],
            ["Vendor benchmark cost", f"${vendor_reference_pipe_cost:,.0f}/reference pipe"],
            ["Max depth", f"{max_depth_ft:.0f} ft"],
            ["Rain/snow loads", "enabled" if precipitation_energy else "disabled"],
        ],
        columns=["Input", "Value"],
    )
    st.dataframe(inputs, hide_index=True, use_container_width=True)

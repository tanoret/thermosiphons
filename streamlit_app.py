from __future__ import annotations

import pandas as pd
import streamlit as st

from thermodrive.climate import load_weather, resolve_site
from thermodrive.cost import InstallType
from thermodrive.metrics import compare_metrics, compute_metrics, risk_label
from thermodrive.optimizer import Goal, OptimizationResult, optimize_design
from thermodrive.physics import PAVEMENT_LIBRARY, SimulationConfig, run_thermal_simulation
from thermodrive.plots import (
    candidate_scatter,
    climate_temperature_plot,
    cost_waterfall,
    freeze_calendar_plot,
    hp_flux_plot,
    pipe_layout_plot,
    seasonal_depth_profiles,
    soil_heatmap,
)
from thermodrive.report import proposal_markdown, workbook_bytes
from thermodrive.soil import choose_soil_profile
from thermodrive.state_data import STATE_NAMES, get_state_defaults
from thermodrive.thermosyphon import FLUID_LIBRARY, pipe_count, summarize_design
from thermodrive.units import FT_TO_M, M2_TO_SQFT, SQFT_TO_M2, c_to_f, currency

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
.hero p {font-size: 1.05rem; opacity: .92; max-width: 980px; margin: 0;}
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
.badge {display:inline-block; padding:.35rem .65rem; border-radius:999px; font-weight:700; font-size:.82rem;}
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
def cached_baseline(weather: pd.DataFrame, state: str, zip_code: str, soil_texture: str, use_usda: bool, config: SimulationConfig):
    site = resolve_site(state, zip_code)
    soil = choose_soil_profile(site, soil_texture, use_usda=use_usda)
    return run_thermal_simulation(weather, site, soil, config, thermosyphon=None)


@st.cache_data(show_spinner=False)
def cached_optimization(
    weather: pd.DataFrame,
    state: str,
    zip_code: str,
    soil_texture: str,
    use_usda: bool,
    config: SimulationConfig,
    area_m2: float,
    install_type: InstallType,
    goal: Goal,
    target_reduction_pct: float,
    max_depth_m: float,
    fluid: str,
    full_verify_count: int,
):
    site = resolve_site(state, zip_code)
    soil = choose_soil_profile(site, soil_texture, use_usda=use_usda)
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
    )


st.markdown(
    """
    <div class="hero">
      <h1>ThermoDrive™ passive driveway thermosyphon designer</h1>
      <p>Screen winter freeze risk, size a low-cost vertical thermosyphon field, and produce a sales-ready installed-cost proposal from local climate, pavement, and soil assumptions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("1. Location")
    state = st.selectbox("State", STATE_NAMES, index=STATE_NAMES.index("Illinois"))
    zip_code = st.text_input("ZIP code (optional)", value="", placeholder="e.g., 60614")
    climate_mode = st.selectbox("Climate source", ["Typical screening year", "NASA POWER hourly year"])
    nasa_year = st.selectbox("NASA POWER year", [2024, 2023, 2022, 2021, 2020], index=0, disabled=climate_mode != "NASA POWER hourly year")

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
        target_default = 10
        target_help = "Target reduction in wet-freeze or freeze hours."
    target_reduction_pct = st.slider("Target reduction (%)", 0, 80, target_default, 5, help=target_help, disabled=goal == "Overheating analysis")

    fluid = st.selectbox("Thermosyphon package", list(FLUID_LIBRARY.keys()), index=0)
    max_depth_ft = st.slider("Maximum constructable depth (ft)", 4.0, 15.0, 10.0, 1.0)

    st.divider()
    with st.expander("Advanced assumptions", expanded=False):
        soil_texture = st.selectbox("Soil assumption", ["Auto / balanced loam", "Sandy / well drained", "Clay / high plasticity", "Wet / high water table", "Gravelly / engineered fill"])
        use_usda = st.toggle("Try USDA Soil Data Access lookup", value=False)
        custom_albedo_enabled = st.toggle("Override pavement albedo", value=False)
        custom_albedo = st.slider("Albedo", 0.05, 0.65, 0.32, 0.01, disabled=not custom_albedo_enabled)
        spinup_years = st.slider("Thermal spin-up years", 0, 3, 0, 1)
        full_verify_count = st.slider("Verified candidate simulations", 3, 14, 4, 1)

    run_button = st.button("Generate proposal", type="primary", use_container_width=True)

# Auto-run on first load. The button remains useful after changing inputs.
if "has_run" not in st.session_state:
    st.session_state.has_run = True
if run_button:
    st.session_state.has_run = True

area_m2 = area_sqft * SQFT_TO_M2
max_depth_m = max_depth_ft * FT_TO_M
site = cached_site(state, zip_code)
weather = cached_weather(state, zip_code, climate_mode, nasa_year)
soil = cached_soil(state, zip_code, soil_texture, use_usda)
config = SimulationConfig(
    pavement_type=pavement_type,
    pavement_thickness_m=pavement_thickness_in / 39.3701,
    base_thickness_m=0.15,
    max_depth_m=max(max_depth_m + 0.8, 4.0),
    spinup_years=spinup_years,
    custom_albedo=custom_albedo if custom_albedo_enabled else None,
)

if st.session_state.has_run:
    with st.spinner("Running finite-difference model, sizing candidates, and building proposal..."):
        opt: OptimizationResult = cached_optimization(
            weather,
            state,
            zip_code,
            soil_texture,
            use_usda,
            config,
            area_m2,
            install_type,
            goal,
            float(target_reduction_pct),
            max_depth_m,
            fluid,
            full_verify_count,
        )
else:
    st.stop()

baseline = opt.baseline_result
design_result = opt.design_result
base_metrics = opt.baseline_metrics
design_metrics = opt.design_metrics
risk = risk_label(base_metrics)
site_line = f"{site.label} · {site.latitude:.2f}, {site.longitude:.2f} · {site.source}"
source_line = str(weather["data_source"].iloc[0]) if "data_source" in weather else "Climate data"
soil_line = f"{soil.name} · k={soil.k_W_mK:.2f} W/m-K · {soil.source}"

st.markdown(f"<span class='badge badge-blue'>{site_line}</span> <span class='badge badge-orange'>{source_line}</span>", unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>{soil_line}</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    badge = "High" if risk == "High" else "Moderate" if risk == "Moderate" else "Low"
    card("Baseline freeze risk", badge, f"{base_metrics.wet_freeze_hours:,} wet-freeze hours · {base_metrics.freeze_thaw_cycles:,} freeze-thaw cycles")
with c2:
    if opt.design is not None:
        n = pipe_count(area_m2, opt.design.spacing_m)
        card("Recommended field", f"{n:,} pipes", f"{opt.design.depth_m/FT_TO_M:.0f} ft deep · {opt.design.spacing_m/FT_TO_M:.1f} ft spacing")
    else:
        card("Recommended field", "No-build", opt.package_label)
with c3:
    if opt.cost is not None:
        card("Installed estimate", currency(opt.cost.total_base), f"{currency(opt.cost.total_low)}–{currency(opt.cost.total_high)} · ${opt.cost.cost_per_sqft:,.2f}/ft²")
    else:
        card("Installed estimate", "—", "No thermosyphon package selected")
with c4:
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

st.info(opt.recommendation_note)

tab_overview, tab_design, tab_technical, tab_cost, tab_assumptions = st.tabs(
    ["Overview", "Recommended design", "Technical plots", "Cost & proposal", "Assumptions"]
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
                {"Metric": "Annual heat delivered (kWh/m²)", "Baseline": 0, "Design": round(design_metrics.annual_hp_kWh_m2, 1) if design_metrics else None, "Reduction": "—"},
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

with tab_cost:
    if opt.cost is None:
        st.warning("No thermosyphon cost estimate is shown because the selected goal does not produce a thermosyphon package.")
    else:
        cost_cols = st.columns(4)
        cost_cols[0].metric("Base installed", currency(opt.cost.total_base))
        cost_cols[1].metric("Low–high range", f"{currency(opt.cost.total_low)} – {currency(opt.cost.total_high)}")
        cost_cols[2].metric("Per ft²", f"${opt.cost.cost_per_sqft:,.2f}")
        cost_cols[3].metric("Per pipe", currency(opt.cost.cost_per_pipe))
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
        **What the model does:** solves a 1D transient finite-difference driveway/base/soil heat equation with hourly air temperature, wind, humidity, solar radiation, and a linearized surface energy balance. The thermosyphon field is represented as a conservative one-way heat-transfer source/sink between a deep soil cell and a near-surface cell.

        **What the model does not promise:** guaranteed snow melting during design blizzards. A wickless vertical thermosyphon requires gravity return of condensate, so it is useful for upward winter heat transport but is not a summer cooling device.

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
            ["Driveway area", f"{area_sqft:,.0f} ft²"],
            ["Pavement", pavement_type],
            ["Pavement thickness", f"{pavement_thickness_in:.1f} in"],
            ["Install type", install_type],
            ["Goal", goal],
            ["Target", f"{target_reduction_pct}%"],
            ["Soil", soil_line],
            ["Thermosyphon package", fluid],
            ["Max depth", f"{max_depth_ft:.0f} ft"],
        ],
        columns=["Input", "Value"],
    )
    st.dataframe(inputs, hide_index=True, use_container_width=True)

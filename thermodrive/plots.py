"""Plotly visualizations for the Streamlit sales dashboard."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .cost import CostEstimate, cost_summary_table
from .physics import SimulationResult
from .thermosyphon import ThermosyphonDesign, pipe_count
from .units import M2_TO_SQFT, M_TO_FT

BRAND_BLUE = "#1E88E5"
BRAND_NAVY = "#102033"
BRAND_ORANGE = "#F39C12"
BRAND_GREEN = "#2E7D32"
BRAND_RED = "#C62828"
GRID = "rgba(16,32,51,0.12)"


def _base_layout(fig: go.Figure, title: str, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Arial, sans-serif", color=BRAND_NAVY),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, title=ytitle)
    return fig


def climate_temperature_plot(weather: pd.DataFrame, baseline: SimulationResult | None = None, design: SimulationResult | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weather.index, y=weather["air_temp_C"], name="Air", line=dict(color="#78909C", width=1)))
    if "observed_surface_C" in weather:
        obs = weather["observed_surface_C"].replace([-9999, -9999.0], np.nan)
        fig.add_trace(go.Scatter(x=weather.index, y=obs, name="NOAA observed surface", line=dict(color="#263238", width=1.2, dash="dot"), visible="legendonly"))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=baseline.time_series.index, y=baseline.surface_C, name="Baseline driveway", line=dict(color=BRAND_ORANGE, width=1.6)))
    if design is not None:
        fig.add_trace(go.Scatter(x=design.time_series.index, y=design.surface_C, name="With thermosyphons", line=dict(color=BRAND_BLUE, width=1.8)))
    fig.add_hrect(y0=-40, y1=0, fillcolor="rgba(198,40,40,0.06)", line_width=0)
    fig.add_hline(y=0, line=dict(color=BRAND_RED, dash="dash"), annotation_text="Freezing", annotation_position="bottom right")
    fig = _base_layout(fig, "Hourly temperature profile", "Temperature (°C)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05))
    return fig


def freeze_calendar_plot(baseline: SimulationResult, design: SimulationResult | None = None) -> go.Figure:
    base = baseline.surface_C.copy()
    df = pd.DataFrame({"Baseline": (base < 0).astype(int)}, index=base.index)
    if design is not None:
        df["With thermosyphons"] = (design.surface_C < 0).astype(int)
    daily = df.resample("D").sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily.index, y=daily["Baseline"], name="Baseline freeze hours", marker_color=BRAND_ORANGE))
    if "With thermosyphons" in daily:
        fig.add_trace(go.Bar(x=daily.index, y=daily["With thermosyphons"], name="Design freeze hours", marker_color=BRAND_BLUE))
    return _base_layout(fig, "Daily freeze-hour calendar", "Hours below 0°C")


def soil_heatmap(result: SimulationResult, title: str = "Soil/driveway temperature map") -> go.Figure:
    stride = max(1, len(result.time_series) // 700)
    z = result.temperature_C[::stride, :].T
    x = result.time_series.index[::stride]
    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=result.depth_m * M_TO_FT,
            z=z,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="°C"),
            hovertemplate="%{x}<br>Depth %{y:.1f} ft<br>%{z:.1f} °C<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title="Depth (ft)")
    return _base_layout(fig, title, "Depth (ft)")


def seasonal_depth_profiles(baseline: SimulationResult, design: SimulationResult | None = None) -> go.Figure:
    fig = go.Figure()
    labels = [("Winter", 15), ("Spring", 105), ("Summer", 198), ("Fall", 290)]
    for label, day in labels:
        idx = min(max((day - 1) * 24 + 12, 0), len(baseline.time_series) - 1)
        fig.add_trace(
            go.Scatter(
                x=baseline.temperature_C[idx, :],
                y=baseline.depth_m * M_TO_FT,
                mode="lines",
                name=f"Baseline {label}",
                line=dict(width=1.6, dash="dot"),
            )
        )
        if design is not None:
            fig.add_trace(
                go.Scatter(
                    x=design.temperature_C[idx, :],
                    y=design.depth_m * M_TO_FT,
                    mode="lines",
                    name=f"Design {label}",
                    line=dict(width=2.0),
                )
            )
    fig.add_vline(x=0, line=dict(color=BRAND_RED, dash="dash"))
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, "Seasonal temperature profiles", "Depth (ft)")


def hp_flux_plot(design: SimulationResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=design.time_series.index, y=design.hp_flux_W_m2, name="Passive thermosyphon heat", fill="tozeroy", line=dict(color=BRAND_BLUE)))
    if "assist_flux_W_m2" in design.time_series and design.assist_flux_W_m2.max() > 0:
        fig.add_trace(go.Scatter(x=design.time_series.index, y=design.assist_flux_W_m2, name="Thermostat assist heat", fill="tozeroy", line=dict(color=BRAND_GREEN)))
    return _base_layout(fig, "Heat delivered to driveway", "W/m²")


def cost_waterfall(estimate: CostEstimate) -> go.Figure:
    table = cost_summary_table(estimate)
    measures = ["relative"] * max(len(table) - 1, 0) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            name="Cost",
            orientation="v",
            measure=measures,
            x=table["Category"],
            y=table["Cost"],
            connector={"line": {"color": GRID}},
        )
    )
    return _base_layout(fig, "Installed cost build-up", "USD")


def candidate_scatter(candidates: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if candidates.empty:
        return _base_layout(fig, "Candidate design search", "Verified reduction (%)")
    y_col = "verified_primary_reduction_pct" if "verified_primary_reduction_pct" in candidates.columns else "estimated_reduction_pct"
    hover_cols = ["spacing_ft", "depth_ft", "diameter_mm", "pipe_count", "cost_per_sqft"]
    df = candidates.copy()
    df[y_col] = df[y_col].fillna(df.get("estimated_reduction_pct", 0))
    fig = px.scatter(
        df,
        x="base_cost",
        y=y_col,
        size="pipe_count",
        color="depth_ft",
        hover_data=hover_cols,
        labels={"base_cost": "Installed cost ($)", y_col: "Reduction (%)", "depth_ft": "Depth (ft)"},
        title="Candidate design search",
    )
    fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.5, color="white")))
    return _base_layout(fig, "Candidate design search", "Reduction (%)")


def pipe_layout_plot(area_m2: float, design: ThermosyphonDesign | None) -> go.Figure:
    fig = go.Figure()
    area_sqft = area_m2 * M2_TO_SQFT
    if design is None:
        length_ft = math.sqrt(area_sqft * 2.4)
        width_ft = area_sqft / length_ft
        fig.add_shape(type="rect", x0=0, y0=0, x1=length_ft, y1=width_ft, line=dict(color=BRAND_NAVY), fillcolor="rgba(30,136,229,0.08)")
        fig.add_annotation(x=length_ft/2, y=width_ft/2, text="No thermosyphon layout", showarrow=False)
    else:
        spacing_ft = design.spacing_m * M_TO_FT
        length_ft = math.sqrt(area_sqft * 2.4)
        width_ft = area_sqft / length_ft
        xs = np.arange(spacing_ft / 2, length_ft, spacing_ft)
        ys = np.arange(spacing_ft / 2, width_ft, spacing_ft)
        xx, yy = np.meshgrid(xs, ys)
        fig.add_shape(type="rect", x0=0, y0=0, x1=length_ft, y1=width_ft, line=dict(color=BRAND_NAVY), fillcolor="rgba(30,136,229,0.06)")
        fig.add_trace(go.Scatter(x=xx.ravel(), y=yy.ravel(), mode="markers", name="Thermosyphon", marker=dict(size=9, color=BRAND_BLUE)))
        fig.add_annotation(
            x=length_ft / 2,
            y=width_ft + 0.8,
            text=f"Conceptual rectangular layout: {pipe_count(area_m2, design.spacing_m)} pipes @ {spacing_ft:.1f} ft spacing",
            showarrow=False,
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=420)
    return _base_layout(fig, "Conceptual pipe layout", "Width (ft)").update_xaxes(title="Length (ft)")


def monthly_performance_plot(baseline: SimulationResult, design: SimulationResult | None = None) -> go.Figure:
    """Monthly freeze-risk and heat-delivery summary."""
    df = pd.DataFrame(index=baseline.time_series.index)
    df["Baseline freeze hours"] = (baseline.surface_C < 0.0).astype(int)
    if design is not None:
        df["Design freeze hours"] = (design.surface_C < 0.0).astype(int)
        df["Passive kWh/m2"] = design.hp_flux_W_m2 * 3600.0 / 3.6e6
        df["Assist kWh/m2"] = design.assist_flux_W_m2 * 3600.0 / 3.6e6
    monthly = df.resample("MS").sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly.index, y=monthly["Baseline freeze hours"], name="Baseline freeze hours", marker_color=BRAND_ORANGE))
    if design is not None:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["Design freeze hours"], name="Design freeze hours", marker_color=BRAND_BLUE))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Passive kWh/m2"], name="Passive kWh/m2", yaxis="y2", line=dict(color=BRAND_GREEN, width=2.4)))
        if "Assist kWh/m2" in monthly and monthly["Assist kWh/m2"].sum() > 0:
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Assist kWh/m2"], name="Assist kWh/m2", yaxis="y2", line=dict(color=BRAND_NAVY, width=2.2, dash="dot")))
        fig.update_layout(yaxis2=dict(title="kWh/m2", overlaying="y", side="right", showgrid=False))
    return _base_layout(fig, "Monthly risk reduction and delivered heat", "Freeze hours")


def source_comparison_plot(nasa_weather: pd.DataFrame | None, noaa_weather: pd.DataFrame | None) -> go.Figure:
    """Compare NASA gridded weather with nearest NOAA station observations."""
    fig = go.Figure()
    if nasa_weather is not None and not nasa_weather.empty:
        daily = nasa_weather[["air_temp_C", "ghi_W_m2"]].resample("D").mean()
        fig.add_trace(go.Scatter(x=daily.index, y=daily["air_temp_C"], name="NASA air temp", line=dict(color=BRAND_BLUE, width=1.8)))
    if noaa_weather is not None and not noaa_weather.empty:
        daily2 = noaa_weather[["air_temp_C", "ghi_W_m2"]].resample("D").mean()
        fig.add_trace(go.Scatter(x=daily2.index, y=daily2["air_temp_C"], name="NOAA air temp", line=dict(color=BRAND_ORANGE, width=1.8)))
    fig.add_hline(y=0, line=dict(color=BRAND_RED, dash="dash"))
    return _base_layout(fig, "NASA POWER vs NOAA USCRN daily air temperature", "Daily mean air temp (C)")


def validation_overlay_plot(validation_result: SimulationResult | None, noaa_weather: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    if validation_result is not None:
        fig.add_trace(go.Scatter(x=validation_result.time_series.index, y=validation_result.surface_C, name="Modeled surface", line=dict(color=BRAND_BLUE, width=1.8)))
    if noaa_weather is not None and not noaa_weather.empty and "observed_surface_C" in noaa_weather:
        obs = noaa_weather["observed_surface_C"].replace([-9999, -9999.0], np.nan)
        fig.add_trace(go.Scatter(x=noaa_weather.index, y=obs, name="NOAA IR surface", line=dict(color=BRAND_ORANGE, width=1.3)))
    fig.add_hline(y=0, line=dict(color=BRAND_RED, dash="dash"))
    return _base_layout(fig, "Validation: modeled vs NOAA observed surface temperature", "Surface temperature (C)")


def soil_validation_plot(validation_result: SimulationResult | None, noaa_weather: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    if noaa_weather is None or noaa_weather.empty:
        return _base_layout(fig, "Validation: soil temperatures", "Soil temperature (C)")
    depths = [(5, 0.05), (20, 0.20), (50, 0.50), (100, 1.00)]
    for cm, m in depths:
        obs_col = f"observed_soil_{cm}cm_C"
        if obs_col in noaa_weather:
            obs = noaa_weather[obs_col].replace([-9999, -9999.0], np.nan).resample("D").mean()
            fig.add_trace(go.Scatter(x=obs.index, y=obs, name=f"NOAA {cm} cm", line=dict(width=1.2, dash="dot")))
        if validation_result is not None and validation_result.depth_m.max() >= m:
            values = np.array([np.interp(m, validation_result.depth_m, row) for row in validation_result.temperature_C], dtype=float)
            model = pd.Series(values, index=validation_result.time_series.index).resample("D").mean()
            fig.add_trace(go.Scatter(x=model.index, y=model, name=f"Model {cm} cm", line=dict(width=1.8)))
    return _base_layout(fig, "Validation: daily soil-temperature profiles", "Soil temperature (C)")


def tuning_score_plot(trials: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    if trials is None or trials.empty:
        return _base_layout(fig, "Tuning trials", "Composite RMSE score")
    df = trials.reset_index(drop=True).copy()
    df["trial"] = np.arange(1, len(df) + 1)
    fig = px.scatter(
        df,
        x="trial",
        y="composite_score",
        size="soil_k_W_mK" if "soil_k_W_mK" in df else None,
        color="custom_albedo" if "custom_albedo" in df else None,
        hover_data=[c for c in ["surface_rmse_C", "soil_rmse_mean_C", "albedo_factor", "soil_k_factor", "ground_offset_C"] if c in df.columns],
        title="NOAA calibration trial scores",
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="white")))
    return _base_layout(fig, "NOAA calibration trial scores", "Composite RMSE score")

"""Unit conversion helpers."""
from __future__ import annotations

FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
SQFT_TO_M2 = 0.09290304
M2_TO_SQFT = 1.0 / SQFT_TO_M2
IN_TO_M = 0.0254
BTU_H_FT2_TO_W_M2 = 3.15459075


def f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def ft_to_m(length_ft: float) -> float:
    return length_ft * FT_TO_M


def m_to_ft(length_m: float) -> float:
    return length_m * M_TO_FT


def sqft_to_m2(area_sqft: float) -> float:
    return area_sqft * SQFT_TO_M2


def m2_to_sqft(area_m2: float) -> float:
    return area_m2 * M2_TO_SQFT


def inches_to_m(length_in: float) -> float:
    return length_in * IN_TO_M


def currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    if value >= 10_000:
        return f"${value/1000:,.0f}k"
    return f"${value:,.0f}"

# U.S. customary display conversions. The thermal model remains SI internally.
W_TO_BTU_H = 3.412141633
W_M2_TO_BTU_H_FT2 = 1.0 / BTU_H_FT2_TO_W_M2
KWH_M2_TO_KWH_FT2 = 1.0 / M2_TO_SQFT
MM_TO_IN = 1.0 / 25.4
KM_TO_MI = 0.6213711922
W_MK_TO_BTU_H_FT_F = 0.577789318


def c_delta_to_f_delta(delta_c: float) -> float:
    return delta_c * 9.0 / 5.0


def c_degree_hours_to_f_degree_hours(value_c_h: float) -> float:
    return value_c_h * 9.0 / 5.0


def w_to_btu_h(power_w: float) -> float:
    return power_w * W_TO_BTU_H


def w_m2_to_btu_h_ft2(flux_w_m2: float) -> float:
    return flux_w_m2 * W_M2_TO_BTU_H_FT2


def kwh_m2_to_kwh_ft2(energy_kwh_m2: float) -> float:
    return energy_kwh_m2 * KWH_M2_TO_KWH_FT2


def mm_to_in(length_mm: float) -> float:
    return length_mm * MM_TO_IN


def km_to_miles(distance_km: float) -> float:
    return distance_km * KM_TO_MI


def conductivity_w_mk_to_btu_h_ft_f(k_w_mk: float) -> float:
    return k_w_mk * W_MK_TO_BTU_H_FT_F

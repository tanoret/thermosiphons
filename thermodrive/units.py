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

"""Soil property defaults and optional USDA Soil Data Access lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import requests

from .climate import Site

SoilTexture = Literal["Auto / balanced loam", "Sandy / well drained", "Clay / high plasticity", "Wet / high water table", "Gravelly / engineered fill"]


@dataclass(frozen=True)
class SoilProfile:
    name: str
    k_W_mK: float
    rho_cp_J_m3K: float
    moisture_label: str
    frost_susceptibility: str
    source: str
    notes: str = ""


DEFAULT_SOILS: dict[str, SoilProfile] = {
    "Auto / balanced loam": SoilProfile(
        name="Balanced loam",
        k_W_mK=1.15,
        rho_cp_J_m3K=2.25e6,
        moisture_label="average",
        frost_susceptibility="medium",
        source="Internal screening default",
        notes="Typical mixed mineral soil with moderate moisture.",
    ),
    "Sandy / well drained": SoilProfile(
        name="Sandy / well drained",
        k_W_mK=0.75,
        rho_cp_J_m3K=1.75e6,
        moisture_label="low",
        frost_susceptibility="low to medium",
        source="Internal screening default",
        notes="Lower heat capacity and lower thermal conductivity unless wet.",
    ),
    "Clay / high plasticity": SoilProfile(
        name="Clay / high plasticity",
        k_W_mK=1.05,
        rho_cp_J_m3K=2.45e6,
        moisture_label="medium-high",
        frost_susceptibility="high",
        source="Internal screening default",
        notes="Potential frost heave and cracking risk; installation review recommended.",
    ),
    "Wet / high water table": SoilProfile(
        name="Wet / high water table",
        k_W_mK=1.65,
        rho_cp_J_m3K=2.85e6,
        moisture_label="high",
        frost_susceptibility="high",
        source="Internal screening default",
        notes="Better thermal contact but higher constructability risk.",
    ),
    "Gravelly / engineered fill": SoilProfile(
        name="Gravelly / engineered fill",
        k_W_mK=0.95,
        rho_cp_J_m3K=1.85e6,
        moisture_label="variable",
        frost_susceptibility="low",
        source="Internal screening default",
        notes="Thermal conductivity depends strongly on compaction and moisture.",
    ),
}


def default_soil(texture: SoilTexture) -> SoilProfile:
    return DEFAULT_SOILS.get(texture, DEFAULT_SOILS["Auto / balanced loam"])


def _texture_from_sand_clay(sand: float | None, clay: float | None) -> tuple[str, float, float, str]:
    sand = 40.0 if sand is None or np.isnan(sand) else float(sand)
    clay = 20.0 if clay is None or np.isnan(clay) else float(clay)
    if sand > 65:
        return "Sandy soil from USDA lookup", 0.80, 1.85e6, "low to medium"
    if clay > 35:
        return "Clay-rich soil from USDA lookup", 1.05, 2.50e6, "high"
    if sand < 35 and clay < 20:
        return "Silt/loam soil from USDA lookup", 1.10, 2.30e6, "medium to high"
    return "Loam soil from USDA lookup", 1.15, 2.25e6, "medium"


def fetch_usda_soil_profile(site: Site, timeout: int = 12) -> SoilProfile | None:
    """Try to estimate soil properties from USDA Soil Data Access.

    This is best-effort and intentionally fails closed to internal defaults.
    """
    point = f"point({site.longitude:.6f} {site.latitude:.6f})"
    url = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
    query_component = f"""
        SELECT TOP 1 c.cokey, c.compname, c.comppct_r, c.mukey
        FROM component AS c
        WHERE c.mukey IN
          (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{point}'))
        ORDER BY c.comppct_r DESC
    """
    try:
        resp = requests.post(url, data={"query": query_component, "format": "JSON"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("Table", [])
        if not rows:
            return None
        row = rows[0]
        cokey = str(row.get("cokey") or row.get("COKEY") or "")
        compname = str(row.get("compname") or row.get("COMPNAME") or "USDA soil")
        if not cokey:
            return None
        query_horizon = f"""
            SELECT TOP 3 hzdept_r, hzdepb_r, sandtotal_r, claytotal_r, dbthirdbar_r, awc_r
            FROM chorizon
            WHERE cokey = '{cokey}'
            ORDER BY hzdept_r ASC
        """
        resp2 = requests.post(url, data={"query": query_horizon, "format": "JSON"}, timeout=timeout)
        resp2.raise_for_status()
        hrows = resp2.json().get("Table", [])
        if not hrows:
            return None

        weights = []
        sands = []
        clays = []
        bulk = []
        awc = []
        for h in hrows:
            top = float(h.get("hzdept_r") or h.get("HZDEPT_R") or 0)
            bot = float(h.get("hzdepb_r") or h.get("HZDEPB_R") or 30)
            w = max(bot - top, 1.0)
            weights.append(w)
            sands.append(float(h.get("sandtotal_r") or h.get("SANDTOTAL_R") or np.nan))
            clays.append(float(h.get("claytotal_r") or h.get("CLAYTOTAL_R") or np.nan))
            bulk.append(float(h.get("dbthirdbar_r") or h.get("DBTHIRDBAR_R") or np.nan))
            awc.append(float(h.get("awc_r") or h.get("AWC_R") or np.nan))
        weights_arr = np.array(weights, dtype=float)
        sand = float(np.nansum(np.array(sands) * weights_arr) / np.nansum(weights_arr))
        clay = float(np.nansum(np.array(clays) * weights_arr) / np.nansum(weights_arr))
        name, base_k, rho_cp, frost = _texture_from_sand_clay(sand, clay)
        awc_mean = float(np.nanmean(awc)) if not np.all(np.isnan(awc)) else 0.15
        bulk_mean = float(np.nanmean(bulk)) if not np.all(np.isnan(bulk)) else 1.45
        # Adjust conductivity for available water capacity and density, gently.
        k = base_k * np.clip(0.85 + 1.2 * awc_mean, 0.75, 1.35) * np.clip(0.90 + 0.10 * bulk_mean, 0.85, 1.15)
        return SoilProfile(
            name=f"{name}: {compname.title()}",
            k_W_mK=float(np.clip(k, 0.55, 1.85)),
            rho_cp_J_m3K=float(np.clip(rho_cp * (0.92 + 0.08 * bulk_mean), 1.6e6, 3.0e6)),
            moisture_label="USDA-estimated",
            frost_susceptibility=frost,
            source="USDA Soil Data Access best-effort lookup",
            notes=f"Weighted near-surface sand={sand:.0f}%, clay={clay:.0f}%, AWC={awc_mean:.2f}.",
        )
    except Exception:
        return None


def choose_soil_profile(site: Site, texture: SoilTexture, use_usda: bool = False) -> SoilProfile:
    if use_usda:
        prof = fetch_usda_soil_profile(site)
        if prof is not None:
            return prof
    return default_soil(texture)

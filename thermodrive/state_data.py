"""State-level defaults used when a ZIP-level lookup is unavailable.

Values are intentionally conservative screening defaults. ZIP/weather API data should
be preferred for project estimates.
"""
from __future__ import annotations

STATE_DEFAULTS: dict[str, dict[str, float | str]] = {
    "Alabama": {"abbr": "AL", "lat": 32.8067, "lon": -86.7911, "mean_c": 17.5, "amp_c": 10.5, "diurnal_c": 6.0, "region_factor": 0.92},
    "Alaska": {"abbr": "AK", "lat": 61.3707, "lon": -152.4044, "mean_c": -2.0, "amp_c": 17.0, "diurnal_c": 5.5, "region_factor": 1.35},
    "Arizona": {"abbr": "AZ", "lat": 33.7298, "lon": -111.4312, "mean_c": 18.5, "amp_c": 13.0, "diurnal_c": 9.0, "region_factor": 1.00},
    "Arkansas": {"abbr": "AR", "lat": 34.9697, "lon": -92.3731, "mean_c": 15.8, "amp_c": 12.0, "diurnal_c": 6.5, "region_factor": 0.88},
    "California": {"abbr": "CA", "lat": 36.1162, "lon": -119.6816, "mean_c": 16.0, "amp_c": 8.0, "diurnal_c": 7.0, "region_factor": 1.28},
    "Colorado": {"abbr": "CO", "lat": 39.0598, "lon": -105.3111, "mean_c": 8.0, "amp_c": 13.0, "diurnal_c": 8.0, "region_factor": 1.06},
    "Connecticut": {"abbr": "CT", "lat": 41.5978, "lon": -72.7554, "mean_c": 10.8, "amp_c": 13.5, "diurnal_c": 6.0, "region_factor": 1.18},
    "Delaware": {"abbr": "DE", "lat": 39.3185, "lon": -75.5071, "mean_c": 13.0, "amp_c": 12.5, "diurnal_c": 6.0, "region_factor": 1.08},
    "District of Columbia": {"abbr": "DC", "lat": 38.9072, "lon": -77.0369, "mean_c": 14.5, "amp_c": 12.5, "diurnal_c": 6.5, "region_factor": 1.22},
    "Florida": {"abbr": "FL", "lat": 27.7663, "lon": -81.6868, "mean_c": 22.0, "amp_c": 6.0, "diurnal_c": 5.5, "region_factor": 1.00},
    "Georgia": {"abbr": "GA", "lat": 33.0406, "lon": -83.6431, "mean_c": 17.0, "amp_c": 10.5, "diurnal_c": 6.0, "region_factor": 0.96},
    "Hawaii": {"abbr": "HI", "lat": 21.0943, "lon": -157.4983, "mean_c": 24.0, "amp_c": 3.0, "diurnal_c": 4.0, "region_factor": 1.45},
    "Idaho": {"abbr": "ID", "lat": 44.2405, "lon": -114.4788, "mean_c": 7.0, "amp_c": 14.5, "diurnal_c": 8.0, "region_factor": 0.96},
    "Illinois": {"abbr": "IL", "lat": 40.3495, "lon": -88.9861, "mean_c": 10.5, "amp_c": 14.5, "diurnal_c": 6.5, "region_factor": 1.03},
    "Indiana": {"abbr": "IN", "lat": 39.8494, "lon": -86.2583, "mean_c": 11.2, "amp_c": 14.0, "diurnal_c": 6.5, "region_factor": 0.96},
    "Iowa": {"abbr": "IA", "lat": 42.0115, "lon": -93.2105, "mean_c": 9.0, "amp_c": 15.5, "diurnal_c": 7.0, "region_factor": 0.94},
    "Kansas": {"abbr": "KS", "lat": 38.5266, "lon": -96.7265, "mean_c": 12.5, "amp_c": 15.0, "diurnal_c": 8.0, "region_factor": 0.90},
    "Kentucky": {"abbr": "KY", "lat": 37.6681, "lon": -84.6701, "mean_c": 13.5, "amp_c": 12.5, "diurnal_c": 6.5, "region_factor": 0.92},
    "Louisiana": {"abbr": "LA", "lat": 31.1695, "lon": -91.8678, "mean_c": 19.5, "amp_c": 9.0, "diurnal_c": 5.5, "region_factor": 0.92},
    "Maine": {"abbr": "ME", "lat": 44.6939, "lon": -69.3819, "mean_c": 6.0, "amp_c": 15.0, "diurnal_c": 6.0, "region_factor": 1.12},
    "Maryland": {"abbr": "MD", "lat": 39.0639, "lon": -76.8021, "mean_c": 13.0, "amp_c": 12.5, "diurnal_c": 6.0, "region_factor": 1.15},
    "Massachusetts": {"abbr": "MA", "lat": 42.2302, "lon": -71.5301, "mean_c": 10.0, "amp_c": 13.5, "diurnal_c": 5.5, "region_factor": 1.24},
    "Michigan": {"abbr": "MI", "lat": 43.3266, "lon": -84.5361, "mean_c": 7.0, "amp_c": 15.0, "diurnal_c": 6.5, "region_factor": 1.02},
    "Minnesota": {"abbr": "MN", "lat": 45.6945, "lon": -93.9002, "mean_c": 5.0, "amp_c": 17.5, "diurnal_c": 7.0, "region_factor": 1.00},
    "Mississippi": {"abbr": "MS", "lat": 32.7416, "lon": -89.6787, "mean_c": 18.0, "amp_c": 10.0, "diurnal_c": 6.0, "region_factor": 0.88},
    "Missouri": {"abbr": "MO", "lat": 38.4561, "lon": -92.2884, "mean_c": 12.5, "amp_c": 14.0, "diurnal_c": 7.0, "region_factor": 0.91},
    "Montana": {"abbr": "MT", "lat": 46.9219, "lon": -110.4544, "mean_c": 5.5, "amp_c": 16.0, "diurnal_c": 8.0, "region_factor": 1.02},
    "Nebraska": {"abbr": "NE", "lat": 41.1254, "lon": -98.2681, "mean_c": 9.5, "amp_c": 15.5, "diurnal_c": 7.5, "region_factor": 0.90},
    "Nevada": {"abbr": "NV", "lat": 38.3135, "lon": -117.0554, "mean_c": 10.5, "amp_c": 15.0, "diurnal_c": 9.0, "region_factor": 1.10},
    "New Hampshire": {"abbr": "NH", "lat": 43.4525, "lon": -71.5639, "mean_c": 7.5, "amp_c": 15.0, "diurnal_c": 6.0, "region_factor": 1.13},
    "New Jersey": {"abbr": "NJ", "lat": 40.2989, "lon": -74.5210, "mean_c": 12.0, "amp_c": 13.0, "diurnal_c": 5.8, "region_factor": 1.22},
    "New Mexico": {"abbr": "NM", "lat": 34.8405, "lon": -106.2485, "mean_c": 12.5, "amp_c": 13.5, "diurnal_c": 9.0, "region_factor": 0.96},
    "New York": {"abbr": "NY", "lat": 42.1657, "lon": -74.9481, "mean_c": 8.5, "amp_c": 14.5, "diurnal_c": 6.0, "region_factor": 1.18},
    "North Carolina": {"abbr": "NC", "lat": 35.6301, "lon": -79.8064, "mean_c": 15.5, "amp_c": 11.5, "diurnal_c": 6.5, "region_factor": 0.96},
    "North Dakota": {"abbr": "ND", "lat": 47.5289, "lon": -99.7840, "mean_c": 4.0, "amp_c": 18.0, "diurnal_c": 7.5, "region_factor": 0.96},
    "Ohio": {"abbr": "OH", "lat": 40.3888, "lon": -82.7649, "mean_c": 10.8, "amp_c": 13.8, "diurnal_c": 6.0, "region_factor": 0.98},
    "Oklahoma": {"abbr": "OK", "lat": 35.5653, "lon": -96.9289, "mean_c": 15.0, "amp_c": 13.5, "diurnal_c": 8.0, "region_factor": 0.88},
    "Oregon": {"abbr": "OR", "lat": 44.5720, "lon": -122.0709, "mean_c": 10.0, "amp_c": 9.0, "diurnal_c": 6.5, "region_factor": 1.08},
    "Pennsylvania": {"abbr": "PA", "lat": 40.5908, "lon": -77.2098, "mean_c": 10.0, "amp_c": 13.5, "diurnal_c": 6.0, "region_factor": 1.08},
    "Rhode Island": {"abbr": "RI", "lat": 41.6809, "lon": -71.5118, "mean_c": 10.5, "amp_c": 13.0, "diurnal_c": 5.5, "region_factor": 1.18},
    "South Carolina": {"abbr": "SC", "lat": 33.8569, "lon": -80.9450, "mean_c": 17.0, "amp_c": 10.5, "diurnal_c": 6.0, "region_factor": 0.94},
    "South Dakota": {"abbr": "SD", "lat": 44.2998, "lon": -99.4388, "mean_c": 7.0, "amp_c": 17.0, "diurnal_c": 7.5, "region_factor": 0.90},
    "Tennessee": {"abbr": "TN", "lat": 35.7478, "lon": -86.6923, "mean_c": 15.0, "amp_c": 11.5, "diurnal_c": 6.5, "region_factor": 0.91},
    "Texas": {"abbr": "TX", "lat": 31.0545, "lon": -97.5635, "mean_c": 19.0, "amp_c": 11.0, "diurnal_c": 7.5, "region_factor": 0.95},
    "Utah": {"abbr": "UT", "lat": 40.1500, "lon": -111.8624, "mean_c": 9.5, "amp_c": 14.5, "diurnal_c": 8.5, "region_factor": 1.02},
    "Vermont": {"abbr": "VT", "lat": 44.0459, "lon": -72.7107, "mean_c": 6.5, "amp_c": 15.5, "diurnal_c": 6.0, "region_factor": 1.12},
    "Virginia": {"abbr": "VA", "lat": 37.7693, "lon": -78.1700, "mean_c": 13.5, "amp_c": 12.0, "diurnal_c": 6.5, "region_factor": 1.02},
    "Washington": {"abbr": "WA", "lat": 47.4009, "lon": -121.4905, "mean_c": 9.5, "amp_c": 8.0, "diurnal_c": 5.5, "region_factor": 1.16},
    "West Virginia": {"abbr": "WV", "lat": 38.4912, "lon": -80.9545, "mean_c": 11.5, "amp_c": 12.5, "diurnal_c": 6.0, "region_factor": 0.92},
    "Wisconsin": {"abbr": "WI", "lat": 44.2685, "lon": -89.6165, "mean_c": 6.5, "amp_c": 16.0, "diurnal_c": 6.5, "region_factor": 0.98},
    "Wyoming": {"abbr": "WY", "lat": 42.7560, "lon": -107.3025, "mean_c": 5.5, "amp_c": 15.5, "diurnal_c": 8.5, "region_factor": 0.98},
}

ABBR_TO_STATE = {str(v["abbr"]): k for k, v in STATE_DEFAULTS.items()}
STATE_NAMES = list(STATE_DEFAULTS.keys())


def get_state_defaults(state: str) -> dict[str, float | str]:
    """Return a copy of defaults for a state name or two-letter abbreviation."""
    if state in STATE_DEFAULTS:
        return dict(STATE_DEFAULTS[state])
    state_upper = state.upper().strip()
    if state_upper in ABBR_TO_STATE:
        return dict(STATE_DEFAULTS[ABBR_TO_STATE[state_upper]])
    return dict(STATE_DEFAULTS["Illinois"])

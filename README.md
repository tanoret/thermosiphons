# ThermoDrive™ Driveway Freeze-Risk Designer

ThermoDrive is a Streamlit sales and engineering screening application for sizing vertical wickless thermosyphon fields and optional controlled assist packages beneath residential driveways. It combines a transient finite-difference driveway/soil thermal model, NASA/NOAA climate validation, a conservative thermosyphon performance model, a mixed discrete design search, and a transparent installed-cost/BOM estimate.

> Important: a vertical wickless thermosyphon is modeled as a one-way upward heat-transfer device. It can reduce winter freeze risk when deeper soil is warmer than the driveway surface. The Assured 90 package is explicitly hybrid: passive thermosyphons reduce the base load, while thermostat-controlled assist closes the remaining freeze-hour gap.

## New in this package

- Added **Assured 90 hybrid thermosyphon + low-power assist** mode for >90% freeze-hour-reduction targets in cold states such as Idaho. The dashboard reports passive thermosyphon heat and assist heat separately.
- Replaced the previous single-cell thermosyphon coupling with a **distributed evaporator/condenser model** that uses finite evaporator length, thermal grout, and near-surface heat-spreader behavior.
- Expanded the optimizer to target freeze-hour reduction directly, search deeper/larger hybrid candidates, and show whether 90% is achieved by passive-only or hybrid operation.

- Added **NOAA USCRN hourly validation** mode with observed air temperature, precipitation, solar radiation, RH, infrared surface temperature, soil moisture, and soil temperature at 5/10/20/50/100 cm when a nearby station is available.
- Added **NASA + NOAA tuned validation year** mode: NASA POWER provides the gridded project weather; nearest NOAA USCRN data are used to tune albedo, soil conductivity, ground-temperature offset, convection, and sky-temperature correction within conservative screening bounds.
- Added a **Validation & tuning** dashboard tab with NASA-vs-NOAA comparison, observed-vs-modeled surface temperature, observed-vs-modeled soil temperature, and calibration trial scores.
- Improved the finite-difference model with rain/snow energy loads, wet-pavement evaporation, ground-mean offset tuning, and working-fluid temperature derating.
- Improved dashboard plots with a monthly performance chart, freeze threshold shading, observed-data overlays, and richer validation summaries.

## What is included

- Streamlit sales dashboard with polished interactive Plotly visuals.
- State and optional ZIP-code location inputs.
- Self-contained synthetic screening climate year that works without API keys.
- Optional NASA POWER hourly weather mode.
- Optional NOAA USCRN station-year validation mode.
- Optional NASA + NOAA tuned validation mode.
- Optional USDA Soil Data Access best-effort soil lookup.
- 1D implicit finite-difference driveway/base/soil temperature solver.
- Directional thermosyphon source/sink model with distributed evaporator/condenser coupling.
- Optimizer for freeze-risk and thermal-cracking goals.
- Cost model with material, installation, controlled-assist/electrical allowances, distribution, overhead, and contingency line items.
- Proposal markdown and data-package export.
- Unit tests for solver, optimizer, cost model, and validation scoring.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to:

```text
streamlit_app.py
```

4. Deploy. Streamlit will install dependencies from `requirements.txt`.

## Data modes

### Typical screening year

The default mode uses deterministic state/ZIP climate assumptions. It is fast, self-contained, and useful for live demos. It should be labeled as a screening result.

### NASA POWER hourly year

The app requests hourly gridded meteorological data from NASA POWER. If the request fails or the response is incomplete, the app falls back to the screening year.

### NOAA USCRN station year

The app finds a nearby NOAA USCRN station for the selected state/year and uses the station's hourly observations as the weather forcing. Observed surface and soil temperatures are retained for validation plots.

### NASA + NOAA tuned validation year

The app uses NASA POWER as the project weather source and NOAA USCRN as a validation/tuning source. The tuning procedure runs a small set of calibration trials and selects the set that minimizes a composite RMSE between modeled and observed surface/soil temperatures.

### USDA soil lookup

The optional soil lookup uses USDA Soil Data Access. It is best-effort and falls back to internal soil defaults.

## Repository structure

```text
thermodrive_app/
  streamlit_app.py
  requirements.txt
  runtime.txt
  README.md
  .streamlit/config.toml
  thermodrive/
    climate.py
    cost.py
    metrics.py
    optimizer.py
    physics.py
    plots.py
    report.py
    soil.py
    state_data.py
    thermosyphon.py
    units.py
    validation.py
  tests/
    test_cost.py
    test_fd_model.py
    test_optimizer.py
    test_validation.py
```

## Running tests

```bash
pytest -q
```

## Model summary

The finite-difference model solves a transient 1D heat equation through pavement, compacted base, and soil layers:

```text
rho_cp(z) dT/dt = d/dz(k(z) dT/dz) + S_hp(z,t)
```

The top boundary uses a linearized surface energy balance with solar absorption, convection, longwave exchange, wet-pavement evaporation, and rain/snow phase-change loads. The lower boundary uses annual mean ground temperature plus a small geothermal-gradient term. NASA/NOAA tuning can adjust the effective surface/soil parameters within conservative bounds.

The thermosyphon model adds an upward heat flux between a distributed evaporator zone and a distributed near-surface condenser/heat-spreader zone when the deeper soil is warmer than the slab by the startup margin. Heat flux is limited by pipe conductance, pipe capacity, fluid temperature envelope, spacing/tributary area, and a numerical equalization limiter. Hybrid designs also include a thermostat-controlled surface-assist heat flux with an explicit peak W/m² limit, setpoint, and annual kWh/m² reporting.

## Cost model

The cost model is intentionally transparent and budgetary. It includes:

- tube stock;
- factory sealing, charging, and pressure test allowance;
- end caps and fittings;
- conductive grout/backfill;
- installation labor;
- controlled-assist mat/cable, sensors, controls, and electrical rough-in for hybrid packages;
- surface restoration;
- mobilization;
- utility/constructability review;
- engineering/proposal allowance;
- contractor overhead/profit;
- distribution margin;
- contingency.

Before using for a binding quote, replace defaults with supplier quotes, local contractor rates, and product-specific QA/manufacturing costs.

## Engineering caveats

This app is designed for sales qualification and preliminary concept sizing. Final engineering requires:

- validated thermosyphon test data for the selected fluid/diameter/fill ratio;
- local soil verification and frost-heave review;
- local utility locate;
- driveway structural review;
- installation method review;
- pressure-rated factory assembly documentation;
- contractor pricing.

The passive-only model should not be used to promise guaranteed snow melting in severe storms. The Assured 90 package should be sold as a hybrid freeze-hour-reduction system, not as a passive-only thermosyphon claim.


## v3 high-output concept mode

This build adds a **High-output CO2 + thermal grout + heat spreader** package intended for cold locations such as Idaho where the base wickless thermosyphon assumptions can look weak. The model does not fake performance with a blanket multiplier; it expands the physical design space to include:

- larger pressure-rated diameters,
- tighter pipe spacing,
- deeper constructable boreholes,
- lower contact resistance using thermal grout,
- a near-surface heat-spreader strip/manifold, and
- verification of more high-performance candidates in the finite-difference model.

Use this mode for sales qualification and concept comparison. It still requires lab or pilot-field calibration before final customer guarantees.


## v4 Assured 90 mode

This build adds a **verified 90% freeze-hour target path**. In Idaho screening cases, passive-only thermosyphons reduce freeze severity but typically do not eliminate enough freeze hours. The new Assured 90 package uses thermosyphons for base heat transfer and a controlled assist layer during the coldest risk hours. This is more honest and more sales-ready than inflating thermosyphon efficiency with an arbitrary multiplier.

Key v4 calculation changes:

- distributed evaporator/source coupling over the lower pipe length;
- distributed condenser/sink coupling over the slab heat-spreader zone;
- separate `assist_flux_W_m2` time series;
- annual passive, assist, and total kWh/m² metrics;
- candidate table columns for verified assist energy;
- target slider extended to 95%;
- default Idaho demo set to the Assured 90 package.

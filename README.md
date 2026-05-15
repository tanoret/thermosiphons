# ThermoDrive™ Passive Driveway Thermosyphon Designer

ThermoDrive is a Streamlit sales and engineering screening application for sizing vertical wickless thermosyphon fields beneath residential driveways. It combines a transient finite-difference driveway/soil thermal model, a conservative thermosyphon performance model, a mixed discrete design search, and a transparent installed-cost/BOM estimate.

> Important: a vertical wickless thermosyphon is modeled as a one-way upward heat-transfer device. It can reduce winter freeze risk when deeper soil is warmer than the driveway surface. It is not modeled as a summer driveway cooling solution.

## What is included

- Streamlit sales dashboard with polished interactive Plotly visuals.
- State and optional ZIP-code location inputs.
- Self-contained synthetic screening climate year that works without API keys.
- Optional NASA POWER hourly weather mode.
- Optional USDA Soil Data Access best-effort soil lookup.
- 1D implicit finite-difference driveway/base/soil temperature solver.
- Conservative directional thermosyphon source/sink model.
- Optimizer for freeze-risk and thermal-cracking goals.
- Cost model with material, installation, distribution, overhead, and contingency line items.
- Proposal markdown and data-package export.
- Unit tests for solver, optimizer, and cost model.

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

The app can request hourly gridded meteorological data from NASA POWER. If the request fails or the response is incomplete, the app falls back to the screening year.

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
  tests/
    test_cost.py
    test_fd_model.py
    test_optimizer.py
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

The top boundary uses a linearized surface energy balance with solar absorption, convection, and longwave exchange. The lower boundary uses annual mean ground temperature plus a small geothermal-gradient term.

The thermosyphon model adds a conservative upward heat flux between a deeper soil cell and a near-surface cell when the deeper cell is warmer than the top cell by the startup margin. Heat flux is limited by pipe conductance, pipe capacity, spacing/tributary area, and a numerical equalization limiter.

## Cost model

The cost model is intentionally transparent and budgetary. It includes:

- tube stock;
- factory sealing, charging, and pressure test allowance;
- end caps and fittings;
- conductive grout/backfill;
- installation labor;
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

The model should not be used to promise guaranteed snow melting in severe storms.

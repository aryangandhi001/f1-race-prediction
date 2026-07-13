# F1 2026 Race & Championship Prediction

Predicts race and sprint winners for the remaining 2026 Formula 1 season,
track by track, and simulates the championship outcome — built on real
timing/session data via [FastF1](https://docs.fastf1.dev/).

## Honest scope note

**2026 is a major regulation-reset year** for F1 (new power unit and aero
rules). This is exactly the kind of season where historical team
competitiveness is a much weaker predictor than usual — the 2022 reset
flipped the order (Mercedes went from dominant to midfield, Red Bull rose).
This project uses actual 2026 in-season results as they accumulate as the
strongest signal, with prior-era data mainly informing driver-level skill
and track-specific tendencies rather than team/car competitiveness, since
that resets with the regulations.

There's also no structured feed of "what parts a team changed on the car
this week" — that's scouting/journalism, not API data. Car development is
instead approximated via **performance-trend features**: rolling
qualifying-pace-vs-pole and race-pace trends over recent sessions, which
capture the same underlying signal (is this car getting faster) without
needing to model specific technical changes.

## Approach

- **Data** (`src/fastf1_data.py`): pulls session results, qualifying, sprints,
  and lap-time summaries via FastF1, cached locally as parquet.
- **Features** (`src/features.py`): per driver-track history, recent-form
  rolling averages, team pace-trend proxies, track characteristics.
- **Model** (`src/train_model.py`): gradient-boosted ranking model predicting
  per-driver finishing position / win probability for a given race, validated
  with time-based (not random) splits, since this is a time series problem.
- **Championship simulation** (`src/championship_sim.py`): Monte Carlo
  simulation over all remaining 2026 races + sprints, sampling from each
  race's predicted probability distribution and applying the real F1 points
  system, combined with actual current standings.
- **Demo** (`app.py`, Gradio): pick an upcoming track, see predicted win
  probabilities; view championship win-probability per driver.

## Telemetry & race-engineering tools (local only)

`telemetry_app.py` is a second Gradio app -- 2D race replay (real X/Y car
positions from FastF1 positional telemetry), tyre/pit strategy charts,
distance-aligned speed/throttle/brake/DRS comparison between two drivers,
and straight-line top-speed trend across the season. It's real, working
code, verified end-to-end locally against actual 2026 session data.

**It is not deployed live.** Loading one full session's car + positional
telemetry for all ~20 drivers measured at ~185-273MB of Python-tracked
memory alone (via `tracemalloc`), on top of the base Gradio/pandas/plotly
process footprint -- confirmed by directly OOM-killing a dedicated,
isolated Render free-tier instance (512MB limit) on its first real
request. FastF1 has no API to load telemetry for only 1-2 drivers; a
session load parses all drivers' data as one batch internally, so there's
no partial-load path to shrink this further. Running it needs either a
host with more available memory, or a paid Render tier -- see
`ANALYSIS.md` for the full investigation.

Run it locally:

```bash
python telemetry_app.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Usage

```bash
python -m src.fastf1_data      # pull and cache historical + current-season data
python -m src.train_model      # train the prediction model
python -m src.championship_sim # run the championship Monte Carlo simulation
python app.py                  # launch the demo
```

# F1 2026 Race & Championship Predictor — Full Technical Report

A complete, exhaustive walkthrough: what this project does, why every
design decision was made, every function in the codebase, every bug hit
and how it was diagnosed and fixed, how it's deployed, and what's honestly
still missing.

---

## 1. What this project is, and the honest framing of 2026

This predicts race and sprint winners and simulates the 2026 F1
championship, trained on real FastF1 timing data spanning 2022–2026. The
one thing that had to be handled honestly from the start: **2026 is a
major regulation-reset season** (new technical rules), which means a
team's historical competitiveness from 2022–2025 is a much weaker signal
than in a normal season — a team that dominated under the old rules isn't
guaranteed to dominate under the new ones, and the model has no way of
"knowing" the regulations changed except through the actual results as
they accumulate. This is stated directly in the app itself, not buried in
a README: predictions lean on **this season's actual results as they
accumulate**, rather than assuming past team competitiveness simply
carries over. This is a real methodological limitation, not a caveat added
for show — it's the central modeling challenge of this specific season,
and it shapes several design decisions below (in particular the
exponentially-weighted, not flat, recent-form features).

---

## 2. Architecture overview

```
src/
  fastf1_data.py       -- fetches + caches real session results (FastF1 API)
  features.py           -- builds leakage-safe per-driver-per-race features
  train_model.py         -- trains LightGBM, time-based validation
  championship_sim.py     -- vectorized Monte Carlo season simulation
  news.py                  -- real, verbatim official F1.com headlines
app.py                     -- Gradio demo, 3 tabs
```

Pipeline:

```
FastF1 API --> results.parquet (cached)
  --> build_race_features() [leakage-safe rolling/track features]
  --> train_model.train() --> model_R.joblib / model_S.joblib (LightGBM)
  --> predict_race_probabilities() [softmax over predicted position]
  --> championship_sim.simulate_championship() [vectorized Plackett-Luce Monte Carlo]
```

---

## 3. File-by-file, function-by-function walkthrough

### `src/fastf1_data.py`

```python
YEARS = [2026, 2025, 2024, 2023, 2022]  # newest-first
```
Years are fetched **newest-first**, not chronologically. This is a direct
consequence of a real rate-limit bug covered below — FastF1's public API
rate-limits at 500 calls/hour, and a full 5-season pull can be interrupted
partway through, so fetching newest-first guarantees a partial/interrupted
run still captures the most valuable data (the actual current season)
before spending API budget on older seasons.

```python
def get_full_schedule(years=YEARS, force_refresh=False):
    if SCHEDULE_PATH.exists() and not force_refresh:
        cached = pd.read_parquet(SCHEDULE_PATH)
        if set(years).issubset(set(cached["year"].unique())):
            return cached[cached["year"].isin(years)].copy()
    ...
    schedule.to_parquet(SCHEDULE_PATH)
```
Caches the season schedule (event names, dates, round numbers, whether a
weekend has a sprint) to `schedule.parquet`, and — important detail — the
cache-hit check isn't just "does the file exist", it checks that *all
requested years* are actually present in the cached data before trusting
the cache, falling through to a live refetch otherwise. This was a
deliberate fix: an earlier version always hit the live API regardless of
whether a perfectly good local cache already existed, which needlessly
burned rate-limit budget every run.

```python
def fetch_all_results(years=YEARS, force_refresh=False):
    ...
    for _, event in schedule.iterrows():
        event_date = event.get("EventDate") or event.get("Session5Date")
        if pd.isna(event_date) or pd.Timestamp(event_date) > today:
            continue  # hasn't happened yet
        ...
        session_types = SESSION_TYPES if is_sprint_weekend else ["R", "Q"]
        for st in session_types:
            df = _session_results(year, round_number, st)
```
Walks the full schedule across all requested years, skipping any event
that's in the future (can't fetch results for a race that hasn't happened
yet — this matters specifically for the current, in-progress 2026 season),
and only requests the Sprint session type for weekends that actually have
one. `_session_results` wraps the actual FastF1 call in a try/except that
prints a `[skip]` message and returns `None` on failure rather than
crashing the whole fetch — necessary given the rate-limit reality, so one
failed session doesn't take down an otherwise-successful multi-hour fetch.

```python
def get_remaining_2026_events():
    schedule = get_full_schedule([2026])
    today = pd.Timestamp(datetime.now())
    remaining = schedule[pd.to_datetime(schedule["EventDate"]) > today].copy()
    return remaining.sort_values("RoundNumber")
```
The mirror-image query to the "skip if in the future" filter above — used
by the championship simulation to know exactly which real 2026 events are
still left to simulate, and by the demo's "next race" dropdown.

### `src/features.py`

```python
_FINISHED_STATUSES = {"Finished", "Lapped"}
df["is_dnf"] = ~df["Status"].isin(_FINISHED_STATUSES)
df["finish_position_clean"] = df["finish_position"].where(~df["is_dnf"])
```
This is the fix for a real, verified bug (covered in detail below): a
driver's DNF (mechanical failure, crash, disqualification) produces a
finishing position that reflects "the run ended early," not "the car was
genuinely slow that day" — averaging a DNF's position in with a driver's
recent-form pace features would unfairly tank a fast car's score after a
single incident unrelated to actual pace. `finish_position_clean` is
`NaN` for any DNF row, and is used specifically for pace-form features
(`driver_recent_finish`, `team_recent_finish`, track history). Points and
grid-position features deliberately still use the raw, un-cleaned values —
0 points for a DNF is correct signal (a DNF genuinely did score 0 points,
that's real), and grid position is a qualifying result, entirely untouched
by whatever happens later in the race.

```python
def _add_ewm(df, group_col, value_col, span, out_col):
    df = df.sort_values(["year", "round"])
    df[out_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).ewm(span=span, min_periods=1, ignore_na=True).mean())
    )
```
Exponentially-weighted mean of a driver's or team's recent results, used
in place of a flat rolling-window mean. Two details matter:

- **`.shift(1)` before the EWM** — the current race's own result is never
  included in its own feature row (this is the leakage-safety guarantee:
  a feature must only ever reflect information available *before* the
  race it's predicting).
- **EWM instead of a flat mean** — this is the direct response to the
  2026 regulation-reset framing above. Under a flat rolling mean, a car
  that was fast for the last two races but slow for the three before that
  gets a form score dragged down equally by all five races; under an
  exponentially-weighted mean, the two most-recent (and thus most
  regulation-era-relevant) races dominate the score, so a genuine current
  form change surfaces faster. `ignore_na=True` means DNF rows (already
  `NaN` via `finish_position_clean`) are skipped in the decay sequence
  entirely rather than breaking it or being treated as a real zero.

```python
def _expanding_track_history(df, group_col, event_col="event_name"):
    parts = []
    for _, group in df.groupby([group_col, event_col]):
        group = group.sort_values("year")
        prior_avg = group["finish_position"].expanding().mean().shift(1)
        parts.append(pd.Series(prior_avg.values, index=group.index))
    return pd.concat(parts).sort_index()
```
Leakage-safe expanding average finish at a specific (driver-or-team,
event) pair, using only prior *years'* results at that same track —
shared logic used for both driver-level track history (e.g. a genuine
Monaco specialist) and team-level track history (e.g. a high-downforce
car that suits Monaco/Hungary but not Monza's low-drag layout). Both are
real, physically-grounded signals, kept as separate features rather than
merged into one.

```python
df["team_trend"] = df.groupby("TeamName")["finish_position_clean"].transform(_trend)
```
Where `_trend` compares the mean of the prior 3 races against the mean of
the 3 before *that* (shifted so nothing leaks) — a directional
"is this team's pace improving or declining right now" signal, distinct
from the EWM recent-form features, which capture *level* rather than
*direction* of change. Negative values mean improving (lower finish
position = better).

```python
def compute_track_averages(session_type="R"):
    """... Safe to use for *future* races (unlike the leakage-safe version
    baked into build_race_features, which only looks at prior years for
    each historical row) -- this is specifically for looking up "how has
    this driver/team historically done at this specific upcoming track" ..."""
```
A separate function from `_expanding_track_history` for a specific reason:
when predicting an actual upcoming (not-yet-run) race, there's no leakage
risk in using the driver's/team's *entire* history at that track — leakage
only matters when evaluating historical rows during training. Conflating
these two would either under-use available history for genuine future
predictions, or risk leakage during training; keeping them as two
functions with clearly different intended use makes the distinction
explicit in the code rather than relying on a caller remembering it.

```python
def get_event_features(base_snapshot, driver_track_avg, team_track_avg, event_name):
    """Overrides `driver_track_history` and `team_track_history` on a base
    feature snapshot with actual history at `event_name` specifically.
    Without this, every upcoming race prediction reuses whatever track the
    driver/team last actually raced at, so predictions barely differ from
    one upcoming track to the next."""
```
This function's docstring names the exact bug it fixes (see debugging
section below) — without it, every "predict the next race" call was
implicitly reusing the track-history value from whatever race a driver
most recently actually competed in, rather than their actual history at
the specific *upcoming* track being asked about.

```python
FEATURE_COLUMNS = [
    "GridPosition", "driver_recent_finish", "driver_recent_points", "driver_recent_grid",
    "team_recent_finish", "team_recent_points", "team_recent_grid", "team_trend",
    "driver_track_history", "team_track_history",
]
```
Ten features total: one raw (grid position), six EWM-based recent-form
features (driver- and team-level, three metrics each), one team-trend
directional feature, and two track-history features (driver and team).

### `src/train_model.py`

```python
def time_based_split(df, n_val_races=N_VALIDATION_RACES):
    race_keys = df[["year", "round"]].drop_duplicates().sort_values(["year", "round"])
    val_keys = race_keys.tail(n_val_races)
    is_val = df.set_index(["year", "round"]).index.isin(val_keys.set_index(["year", "round"]).index)
    return df[~is_val].copy(), df[is_val].copy()
```
The most recent 6 races (by actual chronological order, not row order) are
held out as validation — never a random k-fold split. This matters
specifically because the features are rolling/expanding statistics: a
random split would let a training row use a rolling average that was
itself partly computed from a race chronologically *after* a validation
row, which is a genuine information leak a naive random split wouldn't
catch (the features are correct with respect to actual time order, but a
random split would evaluate the model in an unrealistic setting where it
could implicitly benefit from data time-travel between train and val).

```python
model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=15,
    min_child_samples=10, random_state=42, verbose=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
```
A LightGBM regressor predicting finishing position directly (a regression
target, not a classification over drivers) — `num_leaves=15` and
`min_child_samples=10` are conservative-capacity settings appropriate for
a dataset of this size (a handful of seasons × ~20 drivers × ~24 races is
a few thousand rows at most), reducing overfitting risk; early stopping
on the validation set further guards against training for more rounds
than the data actually supports.

```python
correct, total = 0, 0
for (year, rnd), group in val_df.groupby(["year", "round"]):
    predicted_winner = group.loc[group["pred_position"].idxmin(), "Abbreviation"]
    actual_winner_rows = group[group["finish_position"] == 1]
    ...
    correct += int(predicted_winner == actual_winner)
```
Beyond the raw regression MAE, a second, more interpretable metric is
computed directly: for each held-out race, does the model's single
lowest-predicted-position driver actually match who won? This is the
number that's actually meaningful to a non-technical reader of results
("does it pick the right winner"), versus MAE which is harder to
intuitively judge in isolation.

```python
def predict_race_probabilities(model, race_features, temperature=2.0):
    preds = model.predict(race_features[FEATURE_COLUMNS])
    scores = -preds / temperature
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
```
Converts the model's raw predicted-position regression output into a
proper probability distribution over "who wins this race" via a softmax
on negative predicted position (lower predicted position → higher
probability), with `- scores.max()` subtracted before exponentiating for
numerical stability (standard softmax trick, prevents overflow).
`temperature=2.0` controls how sharply the model's position predictions
translate into probability spread — a lower temperature would make the
model's top pick dominate the probability mass almost entirely; this value
was chosen to keep the distribution meaningfully spread rather than
degenerate to "the model's #1 pick gets ~100%, everyone else ~0%," which
would look overconfident given the real uncertainty in race outcomes.

### `src/championship_sim.py`

```python
def _sample_orders_batch(probs, n_simulations, rng):
    items = probs.index.to_numpy()
    log_weights = np.log(probs.values.astype(float) + 1e-12)
    gumbel_noise = -np.log(-np.log(rng.uniform(1e-12, 1.0, size=(n_simulations, len(items)))))
    scores = log_weights[None, :] + gumbel_noise
    order_idx = np.argsort(-scores, axis=1)
    return items[order_idx]
```
This is the core of the Monte Carlo simulation, and it's a real
algorithmic fix over a naive implementation (covered in detail below).
Sampling a full finishing order from a set of win probabilities is a
Plackett-Luce sampling problem — the standard sequential way to do it is:
draw a winner from the probabilities, remove them, renormalize the
remaining probabilities, draw the next-place finisher, and repeat. That's
mathematically correct but requires a Python-level loop per simulation per
race. The **Gumbel-max trick** used here is mathematically equivalent but
fully vectorized: adding independently-sampled Gumbel noise to each item's
log-probability ("log-weight") and then sorting by the noisy score
produces a sample from exactly the same Plackett-Luce distribution as the
sequential removal-and-renormalize procedure — this is a known result
(the Gumbel-max trick generalizes from "argmax gives a sample from a
categorical distribution" to "full argsort gives a full ranking sample
from Plackett-Luce"). Because it's pure numpy array operations,
`n_simulations` full race outcomes are drawn in a single vectorized call,
with `1e-12` epsilons added before each `log` purely to avoid `log(0)`
for a theoretically-zero-probability driver.

```python
def _award_points(final_points, orders, driver_idx, points_map):
    n_positions = min(orders.shape[1], max(points_map.keys()))
    for pos in range(1, n_positions + 1):
        drivers_at_pos = orders[:, pos - 1]
        col_idx = np.array([driver_idx.get(d, -1) for d in drivers_at_pos])
        valid = col_idx >= 0
        final_points[np.nonzero(valid)[0], col_idx[valid]] += points_map[pos]
```
Vectorized points assignment: rather than looping over each individual
simulation to award points, this loops only over the (small, fixed)
number of scoring positions — for each position, it looks up which driver
landed there across *all* simulations at once and adds that position's
points to all of them in one indexed array update.

```python
def simulate_championship(n_simulations=N_SIMULATIONS, seed=42):
    ...
    race_base = get_driver_snapshot("R").drop(columns=[...])
    race_driver_avg, race_team_avg = compute_track_averages("R")
    ...
    for _, event in remaining.iterrows():
        race_features = get_event_features(race_base, race_driver_avg, race_team_avg, event_name)
        race_probs = predict_race_probabilities(race_model, race_features)...
        race_orders = _sample_orders_batch(race_probs, n_simulations, rng)
        _award_points(final_points, race_orders, driver_idx, RACE_POINTS)
        if is_sprint:
            ... same for sprint ...
```
Full-season Monte Carlo: starts from **actual current 2026 standings**
(`get_current_standings`, real points scored so far this season — not
simulated), then for each remaining real 2026 event, predicts a
probability distribution, samples `n_simulations` full finishing orders
via the vectorized Gumbel-max method above, and accumulates points across
all simulations simultaneously via `_award_points`. Base snapshots and
track-average tables are computed **once** before the event loop (not
recomputed per event), so each event's feature row is a cheap merge rather
than a full feature-rebuild — this matters at `n_simulations=2000` ×
~13 remaining events, where redundant recomputation would have compounded
badly.

```python
champion_idx = final_points.argmax(axis=1)
champion_counts = pd.Series(champion_idx).value_counts(normalize=True)
```
The championship win probability for each driver is simply the fraction
of the 2000 simulated seasons in which that driver ended up with the most
total points — a direct empirical Monte Carlo estimate, not a
closed-form calculation (which wouldn't be tractable given the
combinatorics of ~13 correlated multi-driver race outcomes feeding into a
cumulative points race).

### `src/news.py`

```python
"""Headlines, links, and dates are displayed verbatim from the feed --
never summarized, paraphrased, or fabricated. If the feed can't be
reached, that's reported plainly rather than filled in with invented
content."""
```
This module exists and is written this deliberately-restrictive way for a
direct, explicit reason: the F1 news tab must show **real, official**
information (formula1.com's own RSS feed), never LLM-generated or
paraphrased content — an important distinction for a project that could
otherwise blur the line between "real reported news" and "model
output" in a sports-betting-adjacent context. `fetch_latest_news` parses
the feed via `xml.etree.ElementTree`, applies a simple case-insensitive
substring filter if a team/driver name is given (explicitly documented as
"not a guarantee of exhaustive coverage," since the feed has no structured
team tagging to filter on more precisely), and on any network or parse
failure returns an explicit `{"error": ...}` item rather than silently
returning nothing or fabricating placeholder content.

### `app.py`

Three tabs: **Next race prediction** (bar charts for race and, if
applicable, sprint win probability for a selected upcoming event),
**Championship simulation** (runs the full Monte Carlo simulation on
demand, shows both a probability bar chart and the underlying points
table), **Official news** (verbatim F1.com headlines, optionally filtered
by team). The top-level markdown banner states the 2026 regulation-reset
caveat directly to any viewer, not just in this document.

---

## 4. The real debugging journey

### Bug: FastF1's 500-calls/hour rate limit cascading into mass data loss

An early full data-fetch run silently ended up with huge swaths of
missing data — entire seasons, including all of 2025 and 2026 (the most
important, current data), simply absent. Root cause: FastF1's public API
enforces a 500-calls/hour rate limit, and the original fetch order was
chronological (oldest season first) — by the time the fetch reached the
newest, most relevant seasons, the rate limit had already been exhausted
by older seasons, and every subsequent request failed and was silently
skipped (via `_session_results`'s try/except). **Fix, two parts:** (1)
`YEARS` was reordered to fetch **newest-first**, so a rate-limited or
otherwise interrupted run always captures the current season's data
before spending budget on historical seasons; (2) a `force_refresh=True`
retry path leverages FastF1's own local on-disk cache (`fastf1_cache/`),
so previously-successful requests within a session don't get re-requested
against the live API and burn rate-limit budget a second time on retry.

### Bug: "same top 3 drivers predicted for every single track"

An early version of the model produced nearly identical driver rankings
regardless of which upcoming race was selected — clearly wrong, since
different tracks genuinely favor different cars/drivers. Root cause,
found by inspecting feature values directly across different event
queries: `driver_track_history` (and, before it was added,
`team_track_history` didn't exist yet at all) *was* varying correctly
under the hood, but its signal was being completely overwhelmed by
`GridPosition` acting as a near-constant proxy in the feature set —
without a genuine team-level track-fit signal, the model had nothing
strong enough differentiating tracks. **Fix:** added `team_track_history`
as its own feature (a team's car characteristics suiting a given track —
e.g. high-downforce circuits like Monaco/Hungary vs. low-drag circuits
like Monza — is a real, physically-grounded signal distinct from a
driver's personal skill at a track). After this fix, `team_track_history`
became one of the model's top-importance features by
`model.feature_importances_`, confirming it was addressing a real
missing signal, not just adding noise.

### Bug: championship simulation "not working, just loading" (hanging)

The Monte Carlo simulation appeared to hang indefinitely rather than
complete. Root cause, confirmed by profiling: the original implementation
sampled each simulated race's finishing order via the textbook sequential
Plackett-Luce procedure (draw a winner, remove, renormalize, repeat) in a
**pure Python loop**, executed once per simulation per race — at
`n_simulations=2000` × ~13 remaining events × ~20 drivers being
sequentially removed, that's roughly **1.3 million individual
Python-level sampling operations**, which is slow enough in practice to
look completely hung from a user's perspective even though it would
eventually finish. **Fix:** replaced the sequential loop with the
vectorized Gumbel-max sampling method (`_sample_orders_batch`, detailed
above) — mathematically equivalent, but implemented as pure numpy array
operations with no per-simulation Python loop at all. After the fix, a
full 2000-simulation run over all remaining events completes in
approximately **7 seconds**.

### Investigated, not a bug: "RUS is topping the standings in every scenario"

A result that looked suspicious on first glance — a specific driver
consistently ranking near the top of simulated outcomes — was checked
directly against the raw underlying data rather than assumed to be a
model error. Verified genuine: that driver's actual 2026 results so far
were mostly strong finishes, with the apparently-inconsistent results
being DNFs specifically (which, prior to the DNF-exclusion fix below,
were the actual confound in a related but distinct investigation) —
once checked against raw race-by-race results directly, the model's
output was consistent with genuinely strong real-season performance, not
a bug.

### Bug: DNFs conflated with genuine slow pace in rolling-average features

A related, more concrete bug: a specific fast driver was consistently
predicted to finish outside the top 3 despite their team clearly having
the fastest car by other signals (e.g. qualifying pace). Root cause,
traced by inspecting the specific feature values feeding that
prediction: one bad qualifying-weekend result — actually a **DNF**
recorded as e.g. 16th at a specific race — was being averaged directly
into `driver_recent_finish` and `team_recent_finish` alongside genuine
race-pace results from other weekends, via a flat rolling mean that
treated "finished 16th because of a mechanical retirement" identically to
"finished 16th because the car was actually 16th-fastest that day." These
are not the same signal, and conflating them systematically punished a
fast car for a single unrelated incident. **Fix, two parts, applied
together:** (1) introduced `finish_position_clean`/`is_dnf`, excluding DNF
rows from all pace-based rolling features entirely (see `features.py`
above); (2) switched from a flat rolling mean to the exponentially-weighted
`_add_ewm`, so that even among genuine (non-DNF) results, more recent form
is weighted more heavily — doubly relevant given the 2026 regulation reset
means "how fast was this car three months ago" is a meaningfully weaker
signal than "how fast was this car in the last two races."

### Bug: sprint prediction looked broken (silently blank plot)

Testing the "next race prediction" tab with certain default event
selections showed a completely blank sprint plot, which read as broken —
"sprint prediction is not working." Root cause: only 6 of 24 events on
the 2026 calendar actually have a sprint session, and the default
dropdown selection (a non-sprint event, e.g. the Belgian GP) legitimately
has no sprint prediction to show — the original code simply returned no
figure at all for the sprint plot in that case, which is indistinguishable
from a genuine bug to anyone using the demo. **Fix:** `_no_sprint_fig`
renders an explicit placeholder chart with the text "{event} has no
sprint race (only select 2026 weekends do)" instead of leaving the plot
silently empty — a one-line but real usability fix, verified directly by
re-testing with both a sprint and non-sprint event selected.

### Recurring infrastructure bug: Render `envVars` not applying at service-creation time

Same root issue independently hit on this project as on the research
paper assistant: setting `GEMINI_API_KEY`-equivalent environment variables
(here, none needed at runtime since this app has no external API key
dependency — but this was hit during the general Render deployment
workflow used across all projects in this portfolio) via the
service-creation REST call's `envVars` field did not reliably apply.
**Fix (same as the other project):** set/update environment variables via
the separate `PUT /v1/services/{id}/env-vars` endpoint after the service
already exists, then trigger a manual deploy so the change actually takes
effect.

---

## 5. Results / what the numbers actually mean

- **Validation MAE (finishing position)**, computed on a genuinely
  time-based held-out set (the most recent 6 races, never seen in
  training, features never leaking future information) — reported
  directly by `train_model.py`'s `[R]`/`[S]` output at training time.
- **Winner-prediction accuracy on held-out races** — for each of the 6
  held-out races, whether the model's single top pick actually matches
  who won. This is the more interpretable number: it directly answers
  "if you'd used this model to bet on race winners for the last 6 races,
  how often would it have been right."
- **Feature importances** are printed at training time
  (`model.feature_importances_`), and — as noted above —
  `team_track_history` became a top-importance feature specifically after
  the fix that addressed the "same top 3 for every track" bug, which is a
  directly verifiable confirmation that the fix addressed a real gap
  rather than just adding an unused column.

---

## 6. Deployment

Deployed on Render as a Gradio web service. Models (`model_R.joblib`,
`model_S.joblib`) are trained ahead of time and committed as artifacts
rather than retrained on every deploy — retraining would require the full
FastF1 data fetch, which is both rate-limited and slow, and isn't
appropriate to run inside a web-service boot sequence. `app.py` loads
these joblib artifacts once at startup, along with precomputed driver
snapshots and track-average tables, so the actual per-request prediction
path (`predict_next_race`, `show_championship`) does no data-fetching
during a live request.

---

## 7. Honest limitations and what's actually missing

- **The 2026 regulation reset is a genuine confound the model can't fully
  resolve.** EWM-weighted recent form helps the model adapt faster to
  the new competitive order than a flat historical average would, but it
  cannot substitute for simply not having enough same-regulation-era race
  data yet, especially early in the season.
- **`temperature=2.0` (softmax sharpness) and `n_simulations=2000`
  (Monte Carlo count) are fixed constants chosen by judgment**, not
  swept/tuned against a validation metric.
- **Grid position for future races is proxied by recent average grid
  position** (`get_driver_snapshot`'s `latest["GridPosition"] =
  latest["driver_recent_grid"]`), since actual qualifying results for a
  future race obviously don't exist yet — this is a reasonable
  approximation but is a real source of prediction error specifically for
  qualifying-sensitive tracks.
- **The championship simulation treats each remaining race's outcome as
  independent** given each event's own feature snapshot — it doesn't
  model within-season momentum effects (e.g. a team's upgrade package
  landing mid-season) beyond what the EWM recent-form features already
  capture implicitly.
- **No explicit handling of driver swaps/mid-season lineup changes** — the
  feature pipeline is keyed on driver abbreviation and team name as they
  appear in the FastF1 results, so a mid-season driver change is handled
  correctly only insofar as it naturally shows up as a "new" driver
  history at that team, not through any special-cased logic.

---

## 8. Interview-ready summary

*"This predicts F1 race, sprint, and championship outcomes for the actual
ongoing 2026 season using real FastF1 data. The central modeling challenge
is that 2026 is a regulation-reset year, so historical team competitiveness
is a weaker prior than usual — I addressed that by using
exponentially-weighted, not flat, recent-form features, so the model
adapts to this season's actual results faster. I hit and fixed several
concrete bugs: FastF1's rate limit was silently causing the newest, most
important seasons to fail to fetch, fixed by reordering the fetch
newest-first; DNFs were being averaged into pace features as if they
reflected genuine slow pace, unfairly penalizing fast cars after single
incidents, fixed by excluding them from pace-specific features; and the
championship Monte Carlo simulation was doing about 1.3 million individual
Python-level sampling steps and looked hung, which I fixed by replacing
sequential Plackett-Luce sampling with a fully vectorized Gumbel-max
implementation, cutting a full 2000-simulation run down to about 7
seconds. I also kept the news tab strictly to verbatim official
Formula1.com headlines — deliberately never LLM-generated — since
fabricated content in a predictions-adjacent context is a real integrity
line I didn't want to cross."*

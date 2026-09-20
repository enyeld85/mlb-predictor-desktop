# MLB Apex Desktop Predictor & Betting Analytics Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, production-grade MLB prediction and betting analytics Tkinter desktop application that ingests official MLB Stats data, executes a 10,000-iteration vectorized Monte Carlo simulation, calculates odds-independent fundamental predictions and calibrated uncertainty intervals, layers on de-vigged market edge analytics, and presents an 8-tab dark sportsbook GUI.

**Architecture:** A decoupled, five-module desktop application: SQLite persistence layer (`database.py`), background threaded MLB API ingestion with caching and defensive parsing (`data_fetch.py`), a vectorized inning-aware Monte Carlo simulation and statistical prop engine (`models.py`), a dark sportsbook Tkinter UI with 8 tabs, inline editable Treeviews, and embedded Matplotlib figures (`gui.py`), orchestrated by `main.py`.

**Tech Stack:** Python 3.11+, Tkinter/ttk, SQLite (`sqlite3`), NumPy, SciPy, Pandas, Scikit-Learn, Requests, Matplotlib (`FigureCanvasTkAgg`), Pytest.

## Global Constraints
- Target workspace: `C:\Users\enyel\.gemini\antigravity\scratch\mlb-predictor-desktop`.
- Python 3.11+ compatibility.
- Never fabricate missing statistics: missing values must be `None` and display `"Data unavailable"` or `"-"` with conservative priors and `LOW` Data Confidence.
- Monte Carlo must be vectorized using NumPy arrays `(10000, 9)` and execute in under 1.5 seconds for a full 15-game daily slate on a background thread.
- Decoupled picks: Model predictions, win %, fair odds, and +1.5 spread deltas must generate without requiring sportsbook odds.
- Formula for predictive uncertainty:
  ```
  SE_MC = sqrt(P * (1.0 - P) / Nsim)
  sigma_model = sqrt(sigma_baseline^2 + sigma_sample^2 + sigma_lineup^2)
  SE_total = sqrt(SE_MC^2 + sigma_model^2)
  U_95 = 1.96 * SE_total
  Lower = max(0.0, P - U_95)
  Upper = min(1.0, P + U_95)
  ```
- Weather factor formula:
  $$F_{\text{weather}} = 1.0 + 0.012 \times \left(\frac{\text{Temp} - 72}{10}\right) + 0.015 \times \max(0, \text{WindOutMph} - 5) - 0.012 \times \max(0, \text{WindInMph} - 5)$$
  (Domes/closed roofs: $F_{\text{weather}} = 1.0$).
- Dynamic season parameters must reside in a `model_constants` SQLite table and be recalibrated during backtesting/retraining rather than hardcoded.

---

### Task 1: Project Scaffolding & Dependencies Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: Installed virtual environment/packages (`pytest`, `requests`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`) and shared test fixtures.

- [ ] **Step 1: Write requirements.txt**
Create `requirements.txt` with pinned compatible versions:
```text
requests>=2.31.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
pytest>=7.4.0
```

- [ ] **Step 2: Create test fixtures in tests/conftest.py**
Write fixtures providing sample game data, mock schedule responses, and a temporary in-memory SQLite database connection.

- [ ] **Step 3: Run pytest verification**
Run: `pytest tests -v`
Expected: PASS (0 collected or initial fixture sanity check passes).

---

### Task 2: SQLite Persistence Layer & Dynamic Model Constants (`database.py`)

**Files:**
- Create: `database.py`
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: Python standard library `sqlite3`, `json`, `os`.
- Produces:
  - `init_db(db_path: str = "mlb_analytics.db") -> sqlite3.Connection`
  - `get_db_connection(db_path: str = "mlb_analytics.db") -> sqlite3.Connection`
  - `get_model_constants(conn) -> dict[str, float]`
  - `update_model_constants(conn, constants: dict[str, float]) -> None`
  - `save_slate_cache(conn, game_date: str, games: list[dict]) -> None`
  - `get_cached_slate(conn, game_date: str) -> list[dict] | None`
  - `log_pick(conn, pick_data: dict) -> int`
  - `get_picks_history(conn, limit: int = 500) -> list[dict]`
  - `get_delta_buckets_history(conn) -> list[dict]`

- [ ] **Step 1: Write the failing test for database initialization & schema creation**
Create `tests/test_database.py` testing table creation (`teams`, `pitchers`, `batters`, `park_factors`, `games_cache`, `model_constants`, `picks_history`), default model constants insertion, caching, and pick logging.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_database.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'database'`).

- [ ] **Step 3: Implement database.py**
Implement `database.py` with:
  - Tables creation with proper indexes on `game_date`, `game_pk`, `team_id`, and `pick_id`.
  - Default `model_constants`: `dispersion_alpha=0.12`, `home_field_advantage_runs=0.18`, `temp_coefficient=0.012`, `wind_coefficient_out=0.015`, `wind_coefficient_in=0.012`, `cfip_constant=3.15`, `hr_fb_baseline=0.115`, `baseline_league_runs=4.40`.
  - JSON serialization for complex nested lineup and odds payloads.
  - Safe transactional context managers.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_database.py -v`
Expected: PASS.

---

### Task 3: MLB Stats API Client, Defensive Ingestion & Background Threading (`data_fetch.py`)

**Files:**
- Create: `data_fetch.py`
- Test: `tests/test_data_fetch.py`

**Interfaces:**
- Consumes: `database.py`, `requests`.
- Produces:
  - `MLBDataFetcher` class with:
    - `fetch_schedule_for_date(date_str: str, force_refresh: bool = False) -> tuple[list[dict], bool, str]`
      (Returns `(games_list, is_cached, last_updated_str)`)
    - `fetch_pitcher_metrics(pitcher_id: int) -> dict`
    - `fetch_team_platoon_splits(team_id: int) -> dict`
    - `compute_bullpen_fatigue(recent_boxscores: list) -> float`
    - `get_park_factor(venue_id: int, venue_name: str) -> dict`
    - `fetch_slate_async(date_str: str, on_success, on_error, on_progress) -> threading.Thread`

- [ ] **Step 1: Write failing tests for data fetching, fallback, and missing data**
Create `tests/test_data_fetch.py` testing:
  - Parsing of schedule payload with probable pitchers, confirmed vs projected lineups, and weather.
  - Verification that missing stats return `None` and are labeled `"Data unavailable"` without crashing.
  - Caching behavior: returns SQLite cached slate if API call times out or encounters network error.
  - Background thread execution with mock callback invocation.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_data_fetch.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'data_fetch'`).

- [ ] **Step 3: Implement data_fetch.py**
Implement `data_fetch.py`:
  - Uses `statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,lineups,weather,linescore`.
  - Defensive key extraction with safe `.get()` defaults.
  - In-engine FIP, xFIP, and bullpen 3-day fatigue score calculations.
  - Curated park factors database covering all 30 MLB stadiums with roof types and altitude.
  - Background worker thread with queue / callback hooks.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_data_fetch.py -v`
Expected: PASS.

---

### Task 4: Vectorized Monte Carlo Simulator, Statistical Models & Calibration (`models.py`)

**Files:**
- Create: `models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `database.py`, `numpy`, `scipy.stats`.
- Produces:
  - `simulate_game(home_stats: dict, away_stats: dict, park_weather: dict, constants: dict, n_sim: int = 10000) -> dict`:
    Returns:
    - `p_home_win`, `p_away_win`, `expected_home_score`, `expected_away_score`
    - `p_home_cover_plus_1_5`, `p_away_cover_plus_1_5`, `p_home_cover_minus_1_5`, `p_away_cover_minus_1_5`
    - `spread_delta_home` ($P(+1.5) - P(\text{ML})$), `spread_delta_away`
    - `totals_distribution` (probabilities for 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5)
    - `p_nrfi`, `p_yrfi`
    - `home_uncertainty_95`, `away_uncertainty_95`
    - `data_confidence` ('HIGH', 'MEDIUM', 'LOW')
  - `model_player_props(batter_list: list, pitcher_stats: dict, park_weather: dict) -> list[dict]`:
    Returns Hits (0, 1+, 2+, 3+), Total Bases (0.5, 1.5, 2.5, 3.5), HR %, H+R+RBI, Pitcher Ks, Pitcher Outs.
  - `american_to_prob(american_odds: int | float) -> float`
  - `prob_to_american(prob: float) -> str`
  - `devig_two_way(line1: float, line2: float) -> tuple[float, float]`
  - `calculate_edge(model_prob: float, market_line: float | None, market_opp_line: float | None = None) -> dict`:
    Returns `fair_odds`, `market_prob_novig`, `edge_pct`, `ev_pct`, `recommendation`.
  - `compute_calibration_metrics(predictions: list[dict]) -> dict`:
    Returns `brier_score`, `log_loss`, `calibration_buckets` (7 buckets with over/underconfidence flags), `historical_delta_buckets`.

- [ ] **Step 1: Write failing tests for statistical simulator and calibration**
Create `tests/test_models.py` testing:
  - 10,000 Monte Carlo iterations execution speed benchmark (< 1.5 seconds for 15 games).
  - Proper Negative Binomial parameters and variance.
  - Dynamic starter vs bullpen inning allocation (e.g. 2.0 IP for opener vs 6.1 IP for ace).
  - Inning 1 NRFI evaluation consistency.
  - Exact quadrature uncertainty interval calculation:
    `U_95 = 1.96 * sqrt(SE_MC^2 + sigma_model^2)`.
  - Odds-independent picks: ensure fair odds and win % generate when market odds are `None`.
  - Brier score, Log Loss, and 7-bucket calibration categorization.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_models.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'models'`).

- [ ] **Step 3: Implement models.py**
Implement `models.py` with:
  - Vectorized NumPy arrays `np.random.negative_binomial` of shape `(10000, 9)`.
  - Dimensionless weather multiplier $F_{\text{weather}}$.
  - Discrete multinomial and Poisson models for Hits, TB, HR, H+R+RBI, Ks, and Outs.
  - Two-way market multiplicative devigging and EV calculations.
  - 7 calibration buckets and Historical Spread Delta performance bucket grouping ($0\text{--}5\%$, $5\text{--}10\%$, $10\text{--}15\%$, $15\text{--}20\%$, $20\%+$).

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_models.py -v`
Expected: PASS (All tests pass and execution time < 1.5s).

---

### Task 5: Desktop Tkinter GUI with 8 Tabs, Dark Theme & Matplotlib (`gui.py` & `main.py`)

**Files:**
- Create: `gui.py`
- Create: `main.py`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `database.py`, `data_fetch.py`, `models.py`, `tkinter`, `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`.
- Produces:
  - `MLBPredictorApp(tk.Tk)` with 8 Notebook tabs:
    1. Today's Slate (Game Matchup cards, starters, projected scores, win %, run lines, NRFI, best pick)
    2. Moneylines (Treeview table, inline double-click editable odds, Edge %, EV, Confidence)
    3. Run Lines (Treeview table, +1.5 / -1.5, Spread Delta, Historical Delta Performance bucket table)
    4. Player Props (Treeview table, category dropdown filter, team filter, search, Over/Under %, Edge)
    5. NRFI / YRFI (Matchup table & cards, 1st inning metrics, fair vs market lines)
    6. Model Picks (Auto-ranked value bets, probability slider, edge slider, confidence filter)
    7. Backtesting & Settings (Weight sliders, retrain button, backtest filter parameters, results)
    8. Performance & Calibration (7-bucket calibration table, embedded Matplotlib reliability & bankroll curves)
  - Non-blocking asynchronous refresh with animated progress bar.
  - Responsible gaming disclaimer footer.

- [ ] **Step 1: Write GUI smoke and event handling tests**
Create `tests/test_gui.py` testing:
  - Headless initialization of `MLBPredictorApp` (using `Tcl/Tk` virtual display or unit mock).
  - Population of all 8 tabs with sample slate data.
  - Inline editing of an odds cell in Moneylines and verification that Edge % and recommendation update instantaneously.
  - Dynamic filtering in Player Props and Model Picks.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_gui.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'gui'`).

- [ ] **Step 3: Implement gui.py and main.py**
Implement `gui.py` and `main.py`:
  - Complete Dark Sportsbook Theme styles (Slate `#0f172a`, `#1e293b`, emerald green `#10b981`, amber `#f59e0b`, cyan `#06b6d4`, red `#ef4444`).
  - Treeviews configured with alternating row tags and cell double-click entry overlays for editing.
  - Embedded Matplotlib `FigureCanvasTkAgg` plots for calibration reliability and cumulative bankroll unit growth.
  - Asynchronous thread orchestration using `root.after()` for thread-safe UI updates.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_gui.py -v`
Expected: PASS.

---

### Task 6: Comprehensive End-to-End System Verification & Packaging

**Files:**
- Test: `tests/test_e2e.py`
- Create: `README.md`

- [ ] **Step 1: Write End-to-End integration test**
Verify the complete flow:
  1. Initialize fresh database in temporary directory.
  2. Ingest daily MLB slate via mock/cached API data.
  3. Run 10,000-iteration Monte Carlo simulation and verify all 8 markets calculate correctly.
  4. Edit odds line and verify de-vigged edge recalculation.
  5. Log picks, run backtest query, and calculate Brier score, Log Loss, and Spread Delta bucket performance.

- [ ] **Step 2: Execute full test suite**
Run: `pytest tests -v`
Expected: PASS (All test suites pass 100%).

- [ ] **Step 3: Write user documentation in README.md**
Document installation, launching with `python main.py`, tab guide, odds editing instructions, model retraining, and data provenance.

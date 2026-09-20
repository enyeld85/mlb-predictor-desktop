# MLB Apex Desktop Predictor & Betting Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Tkinter%20Desktop-slate)](https://docs.python.org/3/library/tkinter.html)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Simulation](https://img.shields.io/badge/Simulation-10%2C000%20Run%20Vectorized%20Monte%20Carlo-emerald)](https://numpy.org)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing%20(Pytest)-brightgreen)](https://pytest.org)

**MLB Apex Desktop Predictor & Betting Analytics Platform** is a professional-grade, standalone desktop analytics application built with Python and Tkinter. It autonomously ingests official MLB Stats API feeds, executes a 10,000-iteration vectorized Monte Carlo simulation with Negative Binomial draws, calculates odds-independent fundamental predictions with calibrated uncertainty intervals, layers on de-vigged market edge analytics, and presents everything inside an 8-tab dark sportsbook interface.

---

## Key Highlights

- **Decoupled Fundamental Projections**: Model win rates, projected scores, run lines, and fair odds are computed purely from empirical baseball factors (pitcher underlying metrics, platoon splits, bullpen fatigue, weather, and ballpark factors) without requiring sportsbook odds.
- **10,000-Iteration Vectorized Simulation**: Executes a full 15-game daily slate simulation in under 1.5 seconds via vectorized NumPy Negative Binomial arrays `(10000, 9)` with ghost-runner extra innings.
- **Quadrature Predictive Uncertainty Intervals**: Rigorously reports dynamic 95% confidence intervals ($U_{95}$) combining Monte Carlo standard error, baseline model error, pitcher sample size decay, and lineup confirmation status.
- **Interactive Sportsbook Interface**: Dark theme (`#0f172a`, `#1e293b`, `#10b981`, `#06b6d4`, `#ef4444`) featuring double-click inline odds editing, real-time edge recalculation, dynamic sliders, and embedded Matplotlib reliability/bankroll curves.
- **Defensive Data Provenance**: Ingests official MLB Stats API data with SQLite caching and an automatic offline fallback mode. Missing statistics are never fabricated (`None` / "Data unavailable" with conservative priors).
- **Dynamic Season Constants**: Model parameters (dispersion $\alpha$, home field advantage runs, weather coefficients, cFIP baseline) reside in SQLite `model_constants` and can be re-estimated dynamically.

---

## System Architecture

The application follows a decoupled five-module design:

```
mlb-predictor-desktop/
├── main.py                  # CLI argument parsing, DB init & Tkinter application launch
├── database.py              # SQLite schemas, caching, pick tracking, delta buckets, dynamic constants
├── data_fetch.py            # MLB Stats API client, 30-ballpark factors, weather, bullpen fatigue, async worker
├── models.py                # Vectorized Monte Carlo engine, props, devigging, calibration, uncertainty
├── gui.py                   # 8-tab dark sportsbook GUI, editable Treeviews, embedded Matplotlib canvas
├── requirements.txt         # Pinned production dependencies
├── tests/
│   ├── conftest.py          # Shared pytest fixtures, mock game data, and schedule payloads
│   ├── test_sanity.py       # Fixture sanity verification
│   ├── test_database.py     # SQLite persistence layer and dynamic constants tests
│   ├── test_data_fetch.py   # API ingestion, weather parsing, fatigue index, and caching tests
│   ├── test_models.py       # Monte Carlo speed benchmark, props, devigging, and calibration tests
│   ├── test_gui.py          # Headless GUI smoke, 8-tab rendering, and event handling tests
│   └── test_e2e.py          # Comprehensive end-to-end integration and packaging tests
└── docs/
    └── superpowers/         # Architectural specifications and implementation plans
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.11 or higher
- Windows, macOS, or Linux with Tk/Tcl support

### 1. Clone & Navigate
```bash
git clone <repo-url> mlb-predictor-desktop
cd mlb-predictor-desktop
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Application
```bash
python main.py
```

### CLI Options
`main.py` accepts the following optional command-line flags:
```text
--db PATH            Path to SQLite database file (default: mlb_analytics.db)
--date YYYY-MM-DD    Initial slate date to load (default: today's date)
--no-auto-load       Launch without immediately querying external schedule feed
```

**Example:**
```bash
python main.py --date 2026-09-20 --db test_slate.db
```

---

## The 8 Dedicated Tabs Guide

### Tab 1: Today's Slate
- **Matchup Cards**: Modern overview cards for every game on the scheduled slate.
- **Starting Pitchers**: Handedness, ERA, WHIP, sample innings, and lineup confirmation badges (`CONFIRMED` in green vs `PROJECTED` in amber).
- **Projections**: Projected final scores, win probabilities, fair American odds, and run line cover rates (+1.5).
- **Key Banner**: Highlights the top model pick and first-inning NRFI probability.

### Tab 2: Moneylines
- **Table Columns**: Game ID, Matchup, Team (Away/Home), Starter, Model Win %, Fair American Odds, Book Odds, Edge %, EV %, Data Confidence, Recommendation.
- **Inline Editing**: Double-click any **Book Odds** cell to enter updated lines (e.g., `+135` or `-120`). The system instantly recomputes de-vigged market probability, Edge %, Expected Value %, and the recommendation badge.

### Tab 3: Run Lines & Historical Delta Performance
- **Run Lines Table**: Shows +1.5 and -1.5 cover probabilities, baseline win %, and the **Spread Delta** ($P(+1.5) - P(\text{ML})$).
- **Historical Delta Performance Table**: Evaluates out-of-sample persistence across 5 calibrated delta tiers:
  - `0–5% Delta`
  - `5–10% Delta`
  - `10–15% Delta`
  - `15–20% Delta`
  - `20%+ Delta`
  Displays Total Bets, Wins, Cover %, Net Units Won, and ROI % for each tier.

### Tab 4: Player Props
- **Markets Covered**:
  - **Hits**: Over/Under 0.5 (1+ hits), 2+, 3+
  - **Total Bases**: Over/Under 0.5, 1.5, 2.5, 3.5
  - **Home Runs**: Discrete PA-based Poisson/multinomial HR probability
  - **Hits + Runs + RBIs**: Compound production expectation
  - **Pitcher Strikeouts (Ks)**: Line 5.5 Over/Under based on CSW% and opponent K%
  - **Pitcher Outs Recorded**: Line 17.5 Over/Under based on pitch efficiency
- **Interactive Filters**: Filter by Category dropdown, Team dropdown, or real-time player name search. Double-click odds cells to update sportsbook prices.

### Tab 5: NRFI / YRFI
- **First Inning Breakdown**: Tailored specifically for the popular No Run First Inning / Yes Run First Inning market.
- **Inputs**: Pitcher 1st-inning ERA, 1st-inning WHIP, top-of-the-order hitting metrics, ballpark run factor, and weather suppression.
- **Displays**: NRFI %, YRFI %, Fair Odds, Market Odds, Edge %, and recommendation.

### Tab 6: Model Picks
- **Ranked Conviction List**: Auto-aggregates and ranks all positive-expectation opportunities across Moneylines, Run Lines, Totals, NRFI, and Props.
- **Sliders & Selectors**:
  - `Min Prob %`: Filter bets requiring a minimum simulated probability.
  - `Min Edge %`: Filter bets requiring a minimum edge over market.
  - `Confidence`: Filter by `HIGH`, `MEDIUM`, or `LOW` data confidence.
- **Action**: Highlight any pick and click **"💾 Log Selected Pick to DB"** to record it in SQLite for subsequent backtesting and calibration tracking.

### Tab 7: Backtesting & Settings
- **Dynamic Constant Sliders**:
  - `dispersion_alpha`: Negative binomial dispersion parameter (default: `0.12`).
  - `home_field_advantage_runs`: Home field run expectancy boost (default: `0.18`).
  - `temp_coefficient`: Temperature sensitivity per 10°F from 72°F (default: `0.012`).
  - `wind_coefficient_out`: Outfield wind boost per mph > 5 (default: `0.015`).
  - `wind_coefficient_in`: Infield wind suppression per mph > 5 (default: `0.012`).
  - `baseline_league_runs`: Baseline runs per team per game (default: `4.40`).
  - `cfip_constant`: Constant in FIP formula (default: `3.15`).
  - `hr_fb_baseline`: League average HR/FB ratio (default: `0.115`).
- **Actions**: Save customized parameters directly to SQLite or restore defaults.
- **Historical Backtesting Engine**: Evaluates all settled bets in `picks_history` and computes Total Bets, Record, Win %, Net Units, ROI %, Brier Score, and Log Loss.

### Tab 8: Performance & Calibration
- **7-Bucket Calibration Table**: Compares model predicted probability against actual empirical win rate across 7 intervals:
  - `50–54%`, `55–59%`, `60–64%`, `65–69%`, `70–74%`, `75–79%`, `80%+`
  - Automatically flags whether each bucket is `Well Calibrated`, `Overconfident`, or `Underconfident`.
- **Embedded Matplotlib Plots**:
  1. **Reliability Curve**: Empirical win rate plotted against perfect calibration diagonal.
  2. **Cumulative Bankroll Growth Curve**: Unit progression tracking net profit over settled bets.

---

## Mathematical Formulas & Methodology

### 1. Vectorized Negative Binomial Inning Simulation
Runs scored per half-inning $i \in \{1, \dots, 9\}$ follow a Negative Binomial distribution:
$$\text{Variance}: \quad \sigma^2 = \mu_i + \alpha \mu_i^2$$
$$\text{Parameters}: \quad p = \frac{\mu_i}{\sigma^2} = \frac{1}{1 + \alpha \mu_i}, \quad n = \frac{1}{\alpha}$$
Where $\mu_i$ is dynamically parameterized by the starter or bullpen run suppression, opponent platoon offense (wRC+ / wOBA), park factors, and weather.

### 2. Dimensionless Weather Multiplier ($F_{\text{weather}}$)
$$F_{\text{weather}} = 1.0 + 0.012 \times \left(\frac{\text{Temp} - 72}{10}\right) + 0.015 \times \max(0, \text{WindOut} - 5) - 0.012 \times \max(0, \text{WindIn} - 5)$$
*(For domes and closed retractable roofs, $F_{\text{weather}} = 1.0$ exactly).*

### 3. Quadrature Predictive Uncertainty ($U_{95}$)
Combining simulation sampling standard error and out-of-sample model volatility in quadrature:
$$SE_{\text{MC}} = \sqrt{\frac{P(1 - P)}{N_{\text{sim}}}}$$
$$\sigma_{\text{model}} = \sqrt{\sigma_{\text{baseline}}^2 + \sigma_{\text{sample}}^2 + \sigma_{\text{lineup}}^2}$$
$$SE_{\text{total}} = \sqrt{SE_{\text{MC}}^2 + \sigma_{\text{model}}^2}$$
$$U_{95} = 1.96 \times SE_{\text{total}}$$
Where:
- $\sigma_{\text{baseline}} = 0.025$ (2.5% baseline volatility).
- $\sigma_{\text{sample}} = 0.030 \times \max\left(0, 1 - \frac{\text{IP}}{50.0}\right)$ (sample size penalty for $< 50$ IP).
- $\sigma_{\text{lineup}} = 0.015$ if lineup is `PROJECTED` ($0.0$ when `CONFIRMED`).

### 4. Market De-vigging & Expected Value (+EV)
Two-way multiplicative normalization removes bookmaker juice from American lines:
$$P_{\text{novig}, 1} = \frac{P_{\text{raw}, 1}}{P_{\text{raw}, 1} + P_{\text{raw}, 2}}$$
$$\text{Edge } \% = (P_{\text{model}} - P_{\text{novig}}) \times 100$$
$$\text{EV } \% = \left[ P_{\text{model}} \times b - (1 - P_{\text{model}}) \right] \times 100$$
Where $b$ is the decimal net payout per unit wagered.

### 5. Rolling 3-Day Bullpen Fatigue Index ($F$)
$$F = \min\left(1.0, \frac{\text{Pitches}_{d-1} \times 1.0 + \text{Pitches}_{d-2} \times 0.6 + \text{Pitches}_{d-3} \times 0.3}{75.0}\right)$$

### 6. Model Evaluation Metrics
$$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$
$$\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right]$$

---

## Inline Odds Editing Guide

You can easily test custom market prices without editing code:

1. **Locate Target Cell**: Navigate to **Moneylines**, **Run Lines**, **Player Props**, or **NRFI / YRFI**.
2. **Double-Click**: Double-click on any `Book Odds` cell. An in-place editing overlay appears.
3. **Enter Odds**:
   - American format: `+145`, `-120`, `+200`, `-350`
   - Numeric shorthand: `145` (parsed as `+145`) or `-150`
4. **Commit**: Press <kbd>Enter</kbd> or click outside the cell.
5. **Instant Update**: The row immediately recalculates Edge %, EV %, and updates the recommendation badge (`Strong Edge`, `Slight Edge`, `Pass`, `Fade`).

---

## Data Provenance & Offline Operation

| Feature | Source / Calculation | Fallback Strategy |
| :--- | :--- | :--- |
| **Schedule & Probable Starters** | `statsapi.mlb.com/api/v1/schedule` | Cached slate loaded from SQLite |
| **Starting Lineups** | Official MLB batting order payload | Fallback to projected order; labeled `PROJECTED` |
| **Pitcher Metrics (ERA, FIP, xFIP)** | Season boxscore aggregates | League average baseline (4.25 ERA) with `LOW` Data Confidence |
| **Ballpark Factors** | Curated 30-ballpark database (elevation, roof, run/HR factors) | Neutral 1.00 run and HR factor |
| **Weather Conditions** | Hourly stadium weather data | 72°F, calm wind (neutral $F_{\text{weather}} = 1.0$) |

*Zero Fabrication Policy*: Missing values are marked as `None`, displayed as `"-"` or `"Data unavailable"`, and lower the game's Data Confidence to `LOW`.

---

## Running the Test Suite

The test suite contains 56 unit and end-to-end integration tests with 100% pass rate:

```bash
# Run the complete test suite:
python -m pytest tests -v

# Run only the end-to-end integration tests:
python -m pytest tests/test_e2e.py -v

# Run tests with execution duration benchmarking:
python -m pytest tests --durations=10
```

### Test Suite Overview
- `tests/test_database.py`: Verifies schema creation, default model constant seeding, slate caching, and pick logging/settlement.
- `tests/test_data_fetch.py`: Verifies API parsing, bullpen fatigue decay, park factors lookup, and offline fallback.
- `tests/test_models.py`: Verifies vectorized Monte Carlo simulation, quadrature uncertainty, props, devigging, and calibration metrics.
- `tests/test_gui.py`: Verifies headless GUI initialization, 8 tabs existence, inline editing, and filter responsiveness.
- `tests/test_e2e.py`: Verifies the complete end-to-end lifecycle from ingestion to prediction, settlement, calibration, and GUI packaging.

---

## Responsible Gaming Disclaimer

> **DISCLAIMER**: The MLB Apex Desktop Predictor is intended solely for educational, research, and sports analytics evaluation purposes. The simulations and statistical recommendations do not guarantee financial returns. If you or someone you know has a gambling problem and wants help, call **1-800-GAMBLER** or visit [ncpgambling.org](https://www.ncpgambling.org/).

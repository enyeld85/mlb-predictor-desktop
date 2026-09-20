# System Design Specification: MLB Apex Desktop Predictor & Betting Analytics Platform

**Date:** 2026-09-20  
**Status:** Approved (Refined Edition v4)  
**Target Workspace:** `C:\Users\enyel\.gemini\antigravity\scratch\mlb-predictor-desktop`  
**Platform:** Python 3.11+ / Tkinter / SQLite / Scikit-Learn / Matplotlib / NumPy  

---

## 1. Executive Summary & Core Objectives

The **MLB Apex Desktop Predictor & Betting Analytics Platform** is a standalone, professional-grade desktop sports analytics and betting evaluation application. It autonomously ingests official MLB data, executes a 10,000-iteration vectorized Monte Carlo simulation, and computes empirical outcome probabilities across all major betting markets:
* Moneyline (Win %)
* Run Lines (+1.5 and -1.5 cover probabilities, spread deltas, and historical delta performance)
* Game Totals (Over/Under distributions) & Team Totals
* NRFI / YRFI (No Run / Yes Run First Inning)
* Individual Player Props (Hits: 0, 1+, 2+, 3+; Total Bases: 0.5, 1.5, 2.5, 3.5; Home Runs; Hits+Runs+RBIs; Pitcher Strikeouts; Pitcher Outs Recorded)

### Core Design Philosophy: Decoupled Fundamental Picks
1. Projections and picks are generated **strictly from fundamental baseball performance factors** (pitcher underlying metrics, platoon splits, bullpen fatigue, lineup confirmation, weather, and park factors) without requiring betting odds.
2. Every prediction displays: **Model Probability**, **Empirically Calibrated Uncertainty Interval (±X.X%)**, and **Data Confidence (High / Medium / Low)**.
3. When sportsbook odds are available (from default consensus lines, inline table editing, or optional The Odds API integration), the platform layers on betting market analytics: no-vig market fair values, Edge %, Expected Value (+EV), and Plain-English recommendation tiers ("Strong Edge", "Slight Edge", "Pass", "Fade").

---

## 2. System Architecture & Directory Layout

```
mlb-predictor-desktop/
├── main.py                  # Application entry point, DB init & Tkinter event loop
├── database.py              # SQLite storage: caching, schema creation, prediction logs, backtests, model_constants
├── data_fetch.py            # Official MLB Stats API client, park factors, weather, async worker threads
├── models.py                # Decoupled statistical engine: Inning Monte Carlo sim, ML, props, calibration, delta buckets
├── gui.py                   # Tkinter/ttk dark sportsbook GUI, 8 tabs, editable Treeviews, Matplotlib charts
├── requirements.txt         # Project dependencies: requests, pandas, numpy, scipy, scikit-learn, matplotlib
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-09-20-mlb-predictor-desktop-design.md
```

---

## 3. Data Layer & Feature Provenance (`database.py` & `data_fetch.py`)

### 3.1 Data Dependency Risk & Fallback Strategy
* **API Risk Disclosure**: `statsapi.mlb.com` is an unofficial, community-utilized endpoint backing MLB.com. It lacks an SLA and could alter response schemas or rate-limit requests.
* **Mitigation Strategy**:
  1. **Persistent SQLite Caching with TTL**: All schedule, team, pitcher, and game data is cached locally with a TTL check (default 30 minutes). Once a slate is downloaded, the app operates 100% offline.
  2. **Defensive Schema Parsing**: Dictionary lookups with fallback defaults; missing keys will never crash the ingestion pipeline.
  3. **Offline Fallback Banner**: If the network is unavailable or the endpoint fails, the system automatically loads the latest cached slate from SQLite and displays: `"Displaying Cached Slate (Updated: YYYY-MM-DD HH:MM)"`.

### 3.2 Dynamic Season Constants vs Hardcoded Values
Rather than permanently hardcoding fixed values, baseline constants are stored in a dedicated `model_constants` SQLite table and are dynamically re-estimated during the Retraining/Backtesting routine on historical season data:
* `dispersion_alpha`: Initially $0.12$ (re-estimated per season from $\frac{\text{Var}(\text{Runs}) - \mu}{\mu^2}$).
* `home_field_advantage_runs`: Initially $+0.18$ runs / $53.5\%$ win rate (re-estimated from season home splits).
* `temp_coefficient`: Multiplicative factor $+0.012$ ($+1.2\%$ run scoring per $10^\circ\text{F}$ relative to $72^\circ\text{F}$).
* `wind_coefficient_out`: Multiplicative factor $+0.015$ per mph blowing out to center field above 5 mph.
* `wind_coefficient_in`: Multiplicative factor $+0.012$ per mph blowing in toward home plate above 5 mph.
* `cfip_constant`: $3.15$ baseline (re-calibrated per season as $\text{lgERA} - \frac{13\text{lgHR} + 3(\text{lgBB} + \text{lgHBP}) - 2\text{lgK}}{\text{lgIP}}$).
* `hr_fb_baseline`: $0.115$ ($11.5\%$ league average HR per Fly Ball).

### 3.3 Explicit Feature Provenance & Calculation Rules
Never fabricate statistics. The table below defines the exact source or mathematical formula for every feature:

| Feature | Source / Endpoint | Calculation / Derivation Formula | Fallback if Unavailable |
| :--- | :--- | :--- | :--- |
| **Schedule, Probable Starters, Lineups** | `statsapi.mlb.com/api/v1/schedule` | Directly parsed (`probablePitcher`, `lineups`, `weather`) | Lineup marked `PROJECTED`; starter marked `UNCONFIRMED` |
| **Traditional Pitching (ERA, WHIP, K, BB, IP, HR, HBP)** | `statsapi.mlb.com/api/v1/people/{id}/stats` | Directly parsed from pitching season stats | `None` (Shows "Data unavailable") |
| **FIP (Fielding Independent Pitching)** | Calculated in-engine | $\text{FIP} = \frac{13\text{HR} + 3(\text{BB} + \text{HBP}) - 2\text{K}}{\text{IP}} + c_{\text{FIP}}$ | Defaults to league average ERA (4.25) with `LOW` Data Confidence |
| **xFIP (Expected FIP)** | Calculated in-engine | Replaces actual HR with normalized Fly Balls ($\text{FB}$) and league HR/FB rate ($11.5\%$): $\text{xFIP} = \frac{13(\text{FB} \times 0.115) + 3(\text{BB} + \text{HBP}) - 2\text{K}}{\text{IP}} + c_{\text{FIP}}$ | Defaults to FIP or league baseline |
| **SIERA (Skill-Interactive ERA)** | v1.0 Architectural Strategy | **v1.0 explicitly defers to xFIP as the core FIP metric** (preventing truncated polynomial errors). Full empirical multi-term SIERA is documented for v1.1 when play-by-play batted ball vectorization is integrated. | xFIP |
| **CSW% (Called Strikes + Whiffs)** | MLB live/boxscore pitch tracking feeds | $\text{CSW}\% = \frac{\text{Called Strikes} + \text{Swinging Strikes}}{\text{Total Pitches}}$ | `None` / omitted from K-prop modifier |
| **Team Platoon Offense (wOBA, OPS vs L/R)** | `statsapi.mlb.com/api/v1/teams/{id}/stats?group=hitting&split=vsL/vsR` | Directly parsed; wOBA calculated via standard weights ($0.69\text{BB} + 0.72\text{HBP} + 0.89\text{1B} + 1.27\text{2B} + 1.62\text{3B} + 2.10\text{HR}$) | Team overall season wOBA |
| **wRC+** | Calculated in-engine | Normalized: $\left(\frac{\text{wOBA} - \text{Lg\_wOBA}}{\text{wOBA\_scale}} + \frac{\text{Lg\_R}}{\text{PA}}\right) / \text{ParkFactor}$ scaled to 100 base | 100 (League Average) |
| **Bullpen Fatigue Index (0.0 to 1.0)** | Calculated from recent boxscores (last 3 days) | Exponential decay of pitches thrown over previous 72 hours: $F = \min\left(1.0, \frac{\text{Pitches}_{d-1} \times 1.0 + \text{Pitches}_{d-2} \times 0.6 + \text{Pitches}_{d-3} \times 0.3}{75.0}\right)$ | 0.30 (Average Rest) |
| **Statcast Hard-Hit% & Barrel%** | MLB Statcast data / savant cache | Exit velocity $\ge 95\text{ mph}$ rate; Barrel classification | Marked "Data unavailable", regression-to-mean prior applied, Data Confidence lowered to `LOW` |
| **Park Factors & Altitude** | Curated database (`park_factors`) | Annual refresh cadence: 3-year park run and HR factors (e.g. Coors 1.30 run, 1.25 HR; Petco 0.92 run; Great American 1.18 HR) with `last_updated_season` | Neutral 1.00 |

### 3.4 SQLite Database Schema (`database.py`)
The local SQLite database (`mlb_analytics.db`) contains:
* `teams`: IDs, names, platoon splits (vs LHP/RHP), bullpen metrics, rolling 3-day fatigue index.
* `pitchers`: Handedness, metrics (ERA, FIP, xFIP, WHIP, K%, BB%, 1st-inning ERA/WHIP, rolling median IP/start).
* `batters`: Handedness, AVG, OBP, SLG, ISO, K%, BB%, wOBA vs LHP/RHP, hard-hit/barrel where available.
* `park_factors`: Ballpark run & HR multipliers, altitude (ft), roof type, `last_updated_season`.
* `games_cache`: Scheduled games, probable starters, lineup confirmation status (`CONFIRMED` / `PROJECTED`), temperature, wind, weather conditions, fetch timestamp.
* `model_constants`: Dynamic season parameters ($\alpha$, HFA runs, $c_{\text{FIP}}$, `temp_coefficient`, `wind_coefficient_out`, `wind_coefficient_in`, fatigue weights).
* `picks_history`: Logged predictions (market, pick, model prob, uncertainty interval, fair odds, market odds, edge, confidence, actual outcome, closing odds, units won/lost).

---

## 4. Mathematical & Statistical Engine (`models.py`)

### 4.1 Weather Adjustment Multiplier Formulation
The weather factor $F_{\text{weather}}$ is applied as a dimensionless multiplier to expected runs:
$$F_{\text{weather}} = 1.0 + 0.012 \times \left(\frac{\text{Temp} - 72}{10}\right) + w_{\text{out}} \times \max(0, \text{WindOutMph} - 5) - w_{\text{in}} \times \max(0, \text{WindInMph} - 5)$$
Where:
* $+0.012$ per $10^\circ\text{F}$ deviation represents a $+1.2\%$ relative change in run expectancy (e.g., $92^\circ\text{F}$ yields $1.0 + 0.024 = 1.024$ multiplier).
* $w_{\text{out}} = 0.015$ (`wind_coefficient_out`): $+1.5\%$ per mph blowing out to center field above 5 mph.
* $w_{\text{in}} = 0.012$ (`wind_coefficient_in`): $-1.2\%$ run suppression per mph blowing in toward home plate above 5 mph.
* If a game is played in a dome or with a closed retractable roof: $F_{\text{weather}} = 1.0$ exactly.

### 4.2 Dynamic Starter vs Bullpen Inning Allocation
The model does not hardcode a static 60/40 split:
1. For each starting pitcher, projected innings $IP_{\text{proj}}$ are dynamically estimated from rolling median IP per start, pitch counts, and role:
   * Opener: $IP_{\text{proj}} = 1.0\text{--}2.0$ IP.
   * Standard Starter: $IP_{\text{proj}} = \text{median}(\text{last 5 starts IP})$, typically $4.2\text{--}6.2$ IP.
   * Ace / Deep Starter: $6.1\text{--}7.1$ IP.
2. Bullpen innings required: $IP_{\text{bullpen}} = \max(1.0, 9.0 - IP_{\text{proj}})$.
3. Inning-level run expectations are split between the starter phase and bullpen relief phase.

### 4.3 Inning-by-Inning Vectorized Monte Carlo Simulation
The game simulation executes an **inning-aware simulation**:
1. **Innings 1 through $\lfloor IP_{\text{proj}} \rfloor$**:
   * Expected half-inning runs $\lambda_{\text{starter}}$ calculated from the starting pitcher's run-suppression metric (xFIP scaled to league) against the specific batting order (top of order in Inning 1, middle in later innings) and platoon advantage.
   * **Inning 1 Evaluation**: Directly computes the **NRFI / YRFI** probability:
     $$P(\text{NRFI}) = P(R_{\text{away}, 1} = 0 \text{ and } R_{\text{home}, 1} = 0)$$
2. **Innings $\lceil IP_{\text{proj}} \rceil$ through 9**:
   * Expected half-inning runs $\lambda_{\text{bullpen}}$ calculated from team bullpen FIP/WHIP, adjusted upwards if bullpen fatigue score $F$ is elevated.
3. **Extra Innings (Inning 10+)**:
   * Tied games after 9 innings enter extra innings with MLB's ghost runner on 2nd base, using an empirical run-expectancy distribution ($E[\text{runs}] \approx 1.10$ per half-inning) until a winner is decided.
4. **Vectorized NumPy Array Execution**:
   * All 10,000 game iterations are simulated simultaneously as a NumPy array of shape `(10000, 9)` for home and away teams using vectorized Negative Binomial draws:
     $$\text{Variance}: \quad \sigma^2 = \mu + \alpha \mu^2 \quad \text{with dispersion } \alpha = 0.12$$
     $$\text{NegBin Parameters}: \quad p = \frac{\mu}{\sigma^2}, \quad n = \frac{\mu^2}{\sigma^2 - \mu}$$
   * **Performance Benchmark**: Execution for 10,000 iterations across all 15 daily games is **under 1.5 seconds**, running entirely in a background worker thread alongside data ingestion.

### 4.4 Empirically Calibrated Predictive Uncertainty
Predictive uncertainty combines simulation standard error with empirical model standard error in quadrature:

```
SE_MC = sqrt(P * (1.0 - P) / Nsim)
sigma_model = sqrt(sigma_baseline^2 + sigma_sample^2 + sigma_lineup^2)
SE_total = sqrt(SE_MC^2 + sigma_model^2)
U_95 = 1.96 * SE_total

Lower = max(0.0, P - U_95)
Upper = min(1.0, P + U_95)
```

Where:
* At $N_{\text{sim}} = 10,000$, Monte Carlo sampling standard error is minimal ($SE_{\text{MC}} \le 0.0050$, meaning $1.96 \times SE_{\text{MC}} \le \pm 0.98\%$).
* $\sigma_{\text{baseline}} = 0.025$ ($2.5\%$ standard error representing baseline out-of-sample volatility).
* $\sigma_{\text{sample}} = 0.030 \times \max\left(0.0, 1.0 - \frac{IP_{\text{season}}}{50.0}\right)$ (increases when starter sample is $< 50$ IP).
* $\sigma_{\text{lineup}} = 0.015$ if lineup is `PROJECTED` ($0.0$ when `CONFIRMED`).
* **Illustrative Runtime Display**: A game with a confirmed ace ($\sigma_{\text{baseline}} = 0.025$, $\sigma_{\text{sample}} = 0$, $\sigma_{\text{lineup}} = 0$) and $P = 0.642$ yields $SE_{\text{MC}} \approx 0.0048$, $SE_{\text{total}} \approx 0.0255$, and $U_{95} \approx \pm 5.0\%$, displayed as:
  `Model: 64.2% | Uncertainty: ±5.0% | Data Confidence: High`
  *(Note: $\pm 5.0\%$ is an illustrative example computed dynamically from actual runtime parameters).*

### 4.5 Calibration & Evaluation Metrics Over Raw Win Rate
Calibration is prioritized over nominal win rate:
* **Log Loss**:
  $$\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right]$$
* **Brier Score**:
  $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$
* **7 Calibration Buckets**: $50\text{--}54\%$, $55\text{--}59\%$, $60\text{--}64\%$, $65\text{--}69\%$, $70\text{--}74\%$, $75\text{--}79\%$, and $80\%+$. The system compares predicted win % against actual win %, automatically flagging if the model is overconfident ($P_{\text{pred}} > P_{\text{actual}}$) or underconfident ($P_{\text{pred}} < P_{\text{actual}}$).

### 4.6 Player Prop Projections
* **Hits (0, 1+, 2+, 3+)**: Negative Binomial / Poisson model parameterized by projected PA (from batting order 1–9 and game run environment) $\times$ Batter Contact/BABIP $\times$ Pitcher Hit Suppression $\times$ Park Factor.
* **Total Bases (0.5, 1.5, 2.5, 3.5)**: Discrete multinomial distribution of 1B, 2B, 3B, HR based on ISO, SLG, and opposing pitcher extra-base vulnerability.
* **Home Runs**:
  $$P(\text{HR} \ge 1) = 1 - (1 - p_{\text{HR}})^{\text{PA}}$$
* **Hits + Runs + RBIs**: Compound distribution based on PA, batting order slot, on-base probability, and team run expectation.
* **Pitcher Strikeouts & Outs**: Expected batters faced $\times$ swinging-strike% / CSW% against opponent team K% vs starter handedness. Outs recorded parameterized by expected pitch count (median 88–95 pitches) and pitch efficiency.

---

## 5. Desktop GUI Structure (`gui.py` & `main.py`)

### 5.1 Sportsbook Dark Theme Palette
* Background: Deep Slate `#0f172a`
* Container / Cards: Raised Slate `#1e293b`
* Borders: Slate `#334155`
* Value / High Confidence / Wins: Emerald Green `#10b981`
* Slight Edge / Projected Lineup: Amber `#f59e0b`
* Projections / Fair Odds: Cyan `#06b6d4`
* Fade / Negative Edge: Rose / Red `#ef4444`

### 5.2 Header & Date Navigation Bar
* Title Banner: "MLB APEX PREDICTOR & BETTING ANALYTICS".
* Date Navigator (`< Prev Day`, `Today`, `Next Day >`, Date Picker).
* `Refresh Data` button + non-blocking `ttk.Progressbar` + Status Label.
* Summary Counters: Total Slate Games, Confirmed Lineups, Best Value Picks.

### 5.3 Tab Specifications (`ttk.Notebook`)
1. **Today's Slate**: Matchup cards with starting pitchers, confirmed/projected badges, projected scores, win %, run line cover %, NRFI %, uncertainty interval, and key model pick banner.
2. **Moneylines**: Multi-column table with editable book odds cells. Displays Matchup, Team, Model Win %, Uncertainty (±%), Fair Odds, Book Odds, No-Vig %, Edge %, EV, Confidence, Recommendation.
3. **Run Lines & Historical Delta Performance**:
   * Table showing Team, +1.5 Cover %, -1.5 Cover %, Spread Delta ($P(+1.5) - P(\text{ML})$), Editable Book +1.5 Odds, Edge %, Pick.
   * **Historical Delta Performance Table**: Evaluates out-of-sample persistence across Spread Delta buckets:
     * `0–5% Delta`
     * `5–10% Delta`
     * `10–15% Delta`
     * `15–20% Delta`
     * `20%+ Delta`
     Displays: Bucket | Sample Games | Avg ML % | Avg +1.5 % | Historical +1.5 Cover % | Actual ROI | Delta Persistence Edge.
4. **Player Props**: Filterable table by prop category (`Hits`, `Total Bases`, `Home Runs`, `H+R+RBI`, `Pitcher Ks`, `Pitcher Outs`), Team, and Player. Displays Over %, Under %, Fair Odds, Editable Book Line/Odds, Edge %, Confidence, Recommendation.
5. **NRFI / YRFI**: Inning 1 breakdown table and cards: Starter 1st-inning stats, Top 3 hitters' xwOBA, NRFI % vs YRFI %, Fair Odds, Market Odds, Edge.
6. **Model Picks**: Auto-ranked highest-conviction opportunities across all markets, filterable by min probability slider, min edge slider, and data confidence level.
7. **Backtesting & Settings**: Weight adjustment sliders (Starter, Bullpen, Platoon, Form, Home field), retrain button, backtest filter parameters, and complete metrics (Bets, Wins, Losses, Win%, Units P&L, ROI%, Brier Score, Log Loss, CLV).
8. **Performance & Calibration**: 7-bucket calibration table comparing model predicted % vs actual win %, plus embedded Matplotlib charts (Reliability Curve and Bankroll Growth Curve).

---

## 6. Verification & Test Suite

1. **`tests/test_models.py`**:
   * Vectorized NumPy Monte Carlo speed test (< 1.5s for 15 games $\times$ 10,000 iterations).
   * Inning-by-inning starter/bullpen allocation and ghost runner extra-inning convergence.
   * Negative Binomial dispersion consistency ($\alpha = 0.12$).
   * Log loss and Brier score calculation accuracy.
   * Quadrature uncertainty interval computation (`U_95 = 1.96 * sqrt(SE_MC^2 + sigma_model^2)`) and bounds check ($0 \le \text{Lower} \le \text{Upper} \le 1$).
   * Spread Delta bucket aggregation logic.
2. **`tests/test_data_fetch.py`**:
   * Graceful handling of missing stats (returns `None` / "Data unavailable" without crashing).
   * SQLite caching verification and offline fallback simulation.
   * Lineup status parsing (`CONFIRMED` vs `PROJECTED`).
3. **`tests/test_gui.py`**:
   * Tkinter app initialization and 8-tab switching.
   * Inline editing event handling and immediate Edge recalculation.
   * Background thread non-blocking UI behavior.

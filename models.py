"""Vectorized Monte Carlo Simulator, Statistical Models & Calibration Engine.

Provides:
- Inning-by-inning vectorized Monte Carlo game simulation with Negative Binomial draws.
- Ghost-runner extra innings tiebreaker resolution.
- Dimensionless weather multiplier formulation.
- Empirical and Monte Carlo quadrature uncertainty interval calculation.
- Odds conversions, two-way market multiplicative devigging, and EV calculations.
- Discrete player prop distributions (Hits, Total Bases, HR, H+R+RBI, Ks, Outs).
- Model evaluation metrics: Log loss, Brier score, and 7 calibration buckets.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# =====================================================================
# 1. Odds Conversions, Devigging & Edge Layering
# =====================================================================

def american_to_prob(american_odds: int | float | str) -> float:
    """Convert American odds to implied probability."""
    if isinstance(american_odds, str):
        cleaned = american_odds.strip().replace("+", "")
        if not cleaned or cleaned in ("-", "None", "0"):
            return 0.50
        try:
            odds = float(cleaned)
        except ValueError:
            return 0.50
    else:
        odds = float(american_odds)

    if odds == 0:
        return 0.50

    if odds > 0:
        prob = 100.0 / (odds + 100.0)
    else:
        prob = abs(odds) / (abs(odds) + 100.0)

    return round(float(prob), 4)


def prob_to_american(prob: float) -> str:
    """Convert probability to American odds format string (e.g. '+130', '-150')."""
    p = float(prob)
    if p <= 0.0001:
        return "+10000"
    if p >= 0.9999:
        return "-10000"

    if p >= 0.50:
        if math.isclose(p, 0.50, abs_tol=1e-5):
            return "-100"
        odds = -round((p / (1.0 - p)) * 100.0)
        return f"{int(odds)}"
    else:
        odds = round(((1.0 - p) / p) * 100.0)
        return f"+{int(odds)}"


def devig_two_way(
    line1: float | int | str,
    line2: float | int | str
) -> tuple[float, float]:
    """Perform two-way multiplicative devigging from American lines or raw probabilities."""
    def _parse(line: float | int | str) -> float:
        if isinstance(line, str):
            cleaned = line.strip().replace("+", "")
            try:
                val = float(cleaned)
            except ValueError:
                return 0.50
        else:
            val = float(line)

        if 0.0 < val < 1.0:
            return val
        return american_to_prob(val)

    p1 = _parse(line1)
    p2 = _parse(line2)
    total = p1 + p2

    if total <= 0:
        return (0.50, 0.50)

    novig1 = p1 / total
    novig2 = p2 / total
    return (round(novig1, 4), round(novig2, 4))


def calculate_edge(
    model_prob: float,
    market_line: float | int | str | None,
    market_opp_line: float | int | str | None = None
) -> dict[str, Any]:
    """Calculate betting edge, EV%, and recommendation; handles odds-independent picks."""
    fair_odds = prob_to_american(model_prob)

    # Odds-independent decoupled pick mode
    if market_line is None or market_line == 0 or str(market_line).strip() in ("", "0", "None", "-"):
        return {
            "fair_odds": fair_odds,
            "market_prob_novig": None,
            "edge_pct": None,
            "ev_pct": None,
            "recommendation": "Model Pick",
        }

    # Devig market odds
    if market_opp_line is not None and str(market_opp_line).strip() not in ("", "0", "None", "-"):
        market_prob_novig, _ = devig_two_way(market_line, market_opp_line)
    else:
        market_prob_novig = american_to_prob(market_line)

    # Edge % = (model_prob - market_prob_novig) * 100
    edge_pct = round((model_prob - market_prob_novig) * 100.0, 2)

    # Expected Value %: EV = p * b - (1 - p) where b is decimal net profit per unit
    try:
        if isinstance(market_line, str):
            num_line = float(market_line.strip().replace("+", ""))
        else:
            num_line = float(market_line)

        if num_line > 0:
            b = num_line / 100.0
        elif num_line < 0:
            b = 100.0 / abs(num_line)
        else:
            b = 1.0

        ev = (model_prob * b) - (1.0 - model_prob)
        ev_pct = round(ev * 100.0, 2)
    except (ValueError, ZeroDivisionError):
        ev_pct = edge_pct

    # Plain-English recommendation tiers
    if edge_pct >= 4.0:
        recommendation = "Strong Edge"
    elif edge_pct >= 1.0:
        recommendation = "Slight Edge"
    elif edge_pct >= -2.0:
        recommendation = "Pass"
    else:
        recommendation = "Fade"

    return {
        "fair_odds": fair_odds,
        "market_prob_novig": market_prob_novig,
        "edge_pct": edge_pct,
        "ev_pct": ev_pct,
        "recommendation": recommendation,
    }


# =====================================================================
# 2. Weather Adjustment Multiplier
# =====================================================================

def calculate_weather_factor(
    weather: dict[str, Any] | None,
    roof_type: str | None = None,
    constants: dict[str, float] | None = None
) -> float:
    """Calculate dimensionless weather factor F_weather for run scoring.

    Formula:
    F_weather = 1.0 + temp_coeff * ((Temp - 72) / 10)
                + w_out * max(0, WindOut - 5) - w_in * max(0, WindIn - 5)
    Closed roofs and domes return 1.0 exactly.
    """
    if weather is None:
        weather = {}
    if constants is None:
        constants = {}

    roof = roof_type or weather.get("roof_type") or weather.get("roof_status") or "Open"
    if str(roof).strip().title() in ["Dome", "Closed"] or weather.get("is_dome") or weather.get("roof_status") == "Closed":
        return 1.0

    temp_coeff = float(constants.get("temp_coefficient", 0.012))
    w_out = float(constants.get("wind_coefficient_out", 0.015))
    w_in = float(constants.get("wind_coefficient_in", 0.012))

    try:
        temp = float(weather.get("temp", 72.0) or 72.0)
    except (ValueError, TypeError):
        temp = 72.0

    try:
        wind_speed = float(weather.get("wind_speed", 0.0) or 0.0)
    except (ValueError, TypeError):
        wind_speed = 0.0

    wind_dir = str(weather.get("wind_dir", "") or "").lower()

    if "out" in wind_dir:
        wind_out = wind_speed
        wind_in = 0.0
    elif "in" in wind_dir:
        wind_in = wind_speed
        wind_out = 0.0
    else:
        wind_out = float(weather.get("wind_out", 0.0) or 0.0)
        wind_in = float(weather.get("wind_in", 0.0) or 0.0)

    f_weather = (
        1.0
        + temp_coeff * ((temp - 72.0) / 10.0)
        + w_out * max(0.0, wind_out - 5.0)
        - w_in * max(0.0, wind_in - 5.0)
    )

    return round(float(f_weather), 4)


# =====================================================================
# 3. Quadrature Predictive Uncertainty Calculation
# =====================================================================

def calculate_uncertainty(
    p: float,
    n_sim: int = 10000,
    sample_ip: float | None = 50.0,
    lineup_status: str | None = "CONFIRMED"
) -> float:
    """Calculate 95% predictive uncertainty interval using quadrature combination.

    SE_MC = sqrt(p * (1.0 - p) / n_sim)
    sigma_baseline = 0.025
    sigma_sample = 0.030 * max(0.0, 1.0 - (sample_ip / 50.0))
    sigma_lineup = 0.015 if lineup_status != "CONFIRMED" else 0.0
    sigma_model = sqrt(sigma_baseline^2 + sigma_sample^2 + sigma_lineup^2)
    SE_total = sqrt(SE_MC^2 + sigma_model^2)
    U_95 = 1.96 * SE_total
    """
    p_clamped = min(max(float(p), 0.0), 1.0)
    n_sim = max(int(n_sim), 1)
    ip = float(sample_ip or 0.0)
    lineup = str(lineup_status or "PROJECTED").strip().upper()

    se_mc = math.sqrt(p_clamped * (1.0 - p_clamped) / n_sim)
    sigma_baseline = 0.025
    sigma_sample = 0.030 * max(0.0, 1.0 - (ip / 50.0))
    sigma_lineup = 0.015 if lineup != "CONFIRMED" else 0.0

    sigma_model = math.sqrt(sigma_baseline**2 + sigma_sample**2 + sigma_lineup**2)
    se_total = math.sqrt(se_mc**2 + sigma_model**2)
    u_95 = 1.96 * se_total

    return round(float(u_95), 4)


# =====================================================================
# 4. Vectorized Monte Carlo Game Simulator
# =====================================================================

def simulate_game(
    home_team: dict[str, Any],
    away_team: dict[str, Any],
    venue: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    constants: dict[str, float] | None = None,
    n_sim: int = 10000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute 10,000-run vectorized inning-by-inning Monte Carlo simulation.

    Simulates 9 innings for home and away via Negative Binomial distribution,
    resolves tied games with MLB ghost-runner extra innings, and computes
    moneyline, run lines (+/- 1.5), spread deltas, totals distribution,
    NRFI/YRFI, uncertainties, and data confidence.
    """
    # Defensive signature parsing: support park_weather combined dict
    if venue is not None and "run_factor" in venue and "temp" in venue:
        constants = weather or constants
        weather = venue

    if constants is None:
        constants = {}
    if venue is None:
        venue = {}
    if weather is None:
        weather = {}

    alpha = float(constants.get("dispersion_alpha", 0.12))
    hfa_runs = float(constants.get("home_field_advantage_runs", 0.18))
    baseline_league_runs = float(constants.get("baseline_league_runs", 4.40))
    mu_base_inning = baseline_league_runs / 9.0

    # Park & Weather Multipliers
    roof_type = venue.get("roof_type", "Open")
    run_factor = float(venue.get("run_factor", 1.0) or 1.0)
    weather_factor = calculate_weather_factor(weather, roof_type, constants)
    env_mult = run_factor * weather_factor

    # Starters & Bullpens Extraction
    away_starter = away_team.get("starter") or {}
    home_starter = home_team.get("starter") or {}

    def _get_ip(starter: dict[str, Any]) -> float:
        raw_ip = starter.get("median_ip")
        if raw_ip is None:
            return 5.2
        try:
            val = float(raw_ip)
            # Convert baseball .1 / .2 notation (e.g. 5.1 -> 5.333, 5.2 -> 5.667)
            whole = math.floor(val)
            frac = val - whole
            if 0.09 < frac < 0.15:
                return whole + 1.0 / 3.0
            elif 0.19 < frac < 0.25:
                return whole + 2.0 / 3.0
            return val
        except (ValueError, TypeError):
            return 5.2

    away_starter_ip = min(max(_get_ip(away_starter), 1.0), 8.5)
    home_starter_ip = min(max(_get_ip(home_starter), 1.0), 8.5)

    away_starter_metric = float(
        away_starter.get("xfip") or away_starter.get("fip") or away_starter.get("era") or baseline_league_runs
    )
    home_starter_metric = float(
        home_starter.get("xfip") or home_starter.get("fip") or home_starter.get("era") or baseline_league_runs
    )

    away_starter_1st = float(away_starter.get("first_inning_era") or away_starter_metric)
    home_starter_1st = float(home_starter.get("first_inning_era") or home_starter_metric)

    # Bullpen metrics & fatigue adjustment
    away_bp_metric = float(away_team.get("bullpen_fip") or away_team.get("bullpen_era") or 4.10)
    away_bp_fatigue = float(away_team.get("bullpen_fatigue", 0.0) or 0.0)
    away_bp_factor = (away_bp_metric / baseline_league_runs) * (1.0 + 0.15 * away_bp_fatigue)

    home_bp_metric = float(home_team.get("bullpen_fip") or home_team.get("bullpen_era") or 4.10)
    home_bp_fatigue = float(home_team.get("bullpen_fatigue", 0.0) or 0.0)
    home_bp_factor = (home_bp_metric / baseline_league_runs) * (1.0 + 0.15 * home_bp_fatigue)

    # Offense factors vs pitcher hand
    away_hand = str(away_starter.get("hand", "R")).upper()
    home_hand = str(home_starter.get("hand", "R")).upper()

    away_wrc = float(
        away_team.get(f"wrc_plus_vs_{home_hand.lower()}hp")
        or away_team.get("wrc_plus_vs_rhp")
        or 100.0
    )
    home_wrc = float(
        home_team.get(f"wrc_plus_vs_{away_hand.lower()}hp")
        or home_team.get("wrc_plus_vs_rhp")
        or 100.0
    )
    offense_away = away_wrc / 100.0
    offense_home = home_wrc / 100.0

    # Inning-by-inning expected runs lambda (1..9)
    # Away bats against Home pitching; Home bats against Away pitching
    lambda_away = np.zeros(9, dtype=np.float64)
    lambda_home = np.zeros(9, dtype=np.float64)

    hfa_per_inning = hfa_runs / 9.0

    for i in range(9):
        inn_num = i + 1  # 1 to 9

        # Away batting vs Home starter/bullpen
        if inn_num < home_starter_ip:
            w_starter = 1.0
        elif inn_num - 1 < home_starter_ip <= inn_num:
            w_starter = home_starter_ip - (inn_num - 1)
        else:
            w_starter = 0.0
        w_bullpen = 1.0 - w_starter

        if inn_num == 1:
            starter_suppression_home = (home_starter_1st / baseline_league_runs)
            top_order_mult = 1.05
        else:
            starter_suppression_home = (home_starter_metric / baseline_league_runs)
            top_order_mult = 1.0

        def_home = w_starter * starter_suppression_home + w_bullpen * home_bp_factor
        lam_away = mu_base_inning * offense_away * def_home * env_mult * top_order_mult
        lambda_away[i] = max(0.04, lam_away)

        # Home batting vs Away starter/bullpen
        if inn_num < away_starter_ip:
            w_starter_a = 1.0
        elif inn_num - 1 < away_starter_ip <= inn_num:
            w_starter_a = away_starter_ip - (inn_num - 1)
        else:
            w_starter_a = 0.0
        w_bullpen_a = 1.0 - w_starter_a

        if inn_num == 1:
            starter_suppression_away = (away_starter_1st / baseline_league_runs)
            top_order_mult_h = 1.05
        else:
            starter_suppression_away = (away_starter_metric / baseline_league_runs)
            top_order_mult_h = 1.0

        def_away = w_starter_a * starter_suppression_away + w_bullpen_a * away_bp_factor
        lam_home = (mu_base_inning * offense_home * def_away * env_mult * top_order_mult_h) + hfa_per_inning
        lambda_home[i] = max(0.04, lam_home)

    # Vectorized Inning Simulation: Negative Binomial shape (n_sim, 9)
    # Variance: sigma^2 = mu + alpha * mu^2
    # p = mu / sigma^2 = 1 / (1 + alpha * mu)
    # n = mu^2 / (sigma^2 - mu) = 1 / alpha
    if alpha > 0:
        n_param = 1.0 / alpha
        p_away = 1.0 / (1.0 + alpha * lambda_away)
        p_home = 1.0 / (1.0 + alpha * lambda_home)
        away_innings = np.random.negative_binomial(n_param, p_away, size=(n_sim, 9))
        home_innings = np.random.negative_binomial(n_param, p_home, size=(n_sim, 9))
    else:
        away_innings = np.random.poisson(lambda_away, size=(n_sim, 9))
        home_innings = np.random.poisson(lambda_home, size=(n_sim, 9))

    # NRFI / YRFI probability evaluation
    p_nrfi = float(np.mean((away_innings[:, 0] == 0) & (home_innings[:, 0] == 0)))
    p_yrfi = float(1.0 - p_nrfi)

    # 9-inning total scores
    away_totals = np.sum(away_innings, axis=1)
    home_totals = np.sum(home_innings, axis=1)

    # Tiebreaker Ghost Runner Extra Innings (E[runs] approx 1.10 per half-inning)
    ties = np.where(away_totals == home_totals)[0]
    mu_extra = 1.10
    if alpha > 0:
        p_extra = 1.0 / (1.0 + alpha * mu_extra)
        n_extra = 1.0 / alpha
        while len(ties) > 0:
            extra_away = np.random.negative_binomial(n_extra, p_extra, size=len(ties))
            extra_home = np.random.negative_binomial(n_extra, p_extra, size=len(ties))
            away_totals[ties] += extra_away
            home_totals[ties] += extra_home
            ties = np.where(away_totals == home_totals)[0]
    else:
        while len(ties) > 0:
            extra_away = np.random.poisson(mu_extra, size=len(ties))
            extra_home = np.random.poisson(mu_extra, size=len(ties))
            away_totals[ties] += extra_away
            home_totals[ties] += extra_home
            ties = np.where(away_totals == home_totals)[0]

    # Probabilities & Projected Scores
    home_wins = (home_totals > away_totals)
    away_wins = (away_totals > home_totals)
    home_win_prob = round(float(np.mean(home_wins)), 4)
    away_win_prob = round(float(1.0 - home_win_prob), 4)

    expected_home_score = round(float(np.mean(home_totals)), 2)
    expected_away_score = round(float(np.mean(away_totals)), 2)
    projected_total = round(expected_home_score + expected_away_score, 2)

    # Run Line Covers (+1.5 and -1.5)
    home_cover_plus_1_5 = round(float(np.mean((home_totals + 1.5) > away_totals)), 4)
    away_cover_plus_1_5 = round(float(np.mean((away_totals + 1.5) > home_totals)), 4)
    home_cover_minus_1_5 = round(float(np.mean((home_totals - 1.5) > away_totals)), 4)
    away_cover_minus_1_5 = round(float(np.mean((away_totals - 1.5) > home_totals)), 4)

    spread_delta_home = round(home_cover_plus_1_5 - home_win_prob, 4)
    spread_delta_away = round(away_cover_plus_1_5 - away_win_prob, 4)

    # Totals Distribution
    combined_scores = home_totals + away_totals
    totals_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]
    totals_distribution: dict[Any, dict[str, float]] = {}
    for line in totals_lines:
        o = round(float(np.mean(combined_scores > line)), 4)
        u = round(float(np.mean(combined_scores < line)), 4)
        entry = {"over": o, "under": u}
        totals_distribution[line] = entry
        totals_distribution[str(line)] = entry

    # Data Confidence & Uncertainty Calculation
    home_sample_ip = float(home_starter.get("sample_ip", 0.0) or 0.0) if home_starter else 0.0
    away_sample_ip = float(away_starter.get("sample_ip", 0.0) or 0.0) if away_starter else 0.0
    home_lineup_status = str(home_team.get("lineup_status", "PROJECTED")).strip().upper()
    away_lineup_status = str(away_team.get("lineup_status", "PROJECTED")).strip().upper()

    starters_available = bool(home_starter and away_starter and home_sample_ip > 0 and away_sample_ip > 0)
    if starters_available and home_sample_ip >= 35.0 and away_sample_ip >= 35.0:
        if home_lineup_status == "CONFIRMED" and away_lineup_status == "CONFIRMED":
            data_confidence = "HIGH"
        else:
            data_confidence = "MEDIUM"
    else:
        data_confidence = "LOW"

    home_uncertainty_95 = calculate_uncertainty(home_win_prob, n_sim, home_sample_ip, home_lineup_status)
    away_uncertainty_95 = calculate_uncertainty(away_win_prob, n_sim, away_sample_ip, away_lineup_status)

    # Fair Odds
    home_fair_odds = prob_to_american(home_win_prob)
    away_fair_odds = prob_to_american(away_win_prob)
    nrfi_fair_odds = prob_to_american(p_nrfi)
    yrfi_fair_odds = prob_to_american(p_yrfi)

    return {
        # Core Moneylines
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "p_home_win": home_win_prob,
        "p_away_win": away_win_prob,
        # Expected Scores & Totals
        "expected_home_score": expected_home_score,
        "expected_away_score": expected_away_score,
        "projected_total": projected_total,
        # Spread Covers
        "home_cover_plus_1_5": home_cover_plus_1_5,
        "away_cover_plus_1_5": away_cover_plus_1_5,
        "p_home_cover_plus_1_5": home_cover_plus_1_5,
        "p_away_cover_plus_1_5": away_cover_plus_1_5,
        "home_cover_minus_1_5": home_cover_minus_1_5,
        "away_cover_minus_1_5": away_cover_minus_1_5,
        "p_home_cover_minus_1_5": home_cover_minus_1_5,
        "p_away_cover_minus_1_5": away_cover_minus_1_5,
        # Spread Deltas
        "spread_delta_home": spread_delta_home,
        "spread_delta_away": spread_delta_away,
        # Distributions & First Inning
        "totals_distribution": totals_distribution,
        "p_nrfi": round(p_nrfi, 4),
        "p_yrfi": round(p_yrfi, 4),
        # Uncertainties & Confidence
        "home_uncertainty_95": home_uncertainty_95,
        "away_uncertainty_95": away_uncertainty_95,
        "data_confidence": data_confidence,
        # Fair Odds
        "home_fair_odds": home_fair_odds,
        "away_fair_odds": away_fair_odds,
        "nrfi_fair_odds": nrfi_fair_odds,
        "yrfi_fair_odds": yrfi_fair_odds,
    }


# =====================================================================
# 5. Player Prop Projections
# =====================================================================

def model_player_props(
    batter_list: list[dict[str, Any]],
    pitcher: dict[str, Any] | None = None,
    venue: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    **kwargs: Any
) -> list[dict[str, Any]]:
    """Generate discrete player prop distributions for batters and starting pitcher."""
    if venue is None:
        venue = {}
    if weather is None:
        weather = {}

    run_factor = float(venue.get("run_factor", 1.0) or 1.0)
    hr_factor = float(venue.get("hr_factor", 1.0) or 1.0)
    roof_type = venue.get("roof_type", "Open")
    weather_factor = calculate_weather_factor(weather, roof_type)

    props_results: list[dict[str, Any]] = []

    # Opposing pitcher suppression factor
    p_whip = float(pitcher.get("whip", 1.25) or 1.25) if pitcher else 1.25
    p_k_pct = float(pitcher.get("k_pct", 0.22) or 0.22) if pitcher else 0.22
    p_hit_suppression = min(max(p_whip / 1.25, 0.70), 1.40)

    # 1. Batter Props
    for i, batter in enumerate(batter_list):
        order = int(batter.get("order", i + 1) or (i + 1))
        # Expected PA by batting slot
        pa = max(3.4, 4.65 - 0.12 * order)

        avg = float(batter.get("avg", 0.250) or 0.250)
        obp = float(batter.get("obp", 0.320) or 0.320)
        slg = float(batter.get("slg", 0.410) or 0.410)
        iso = float(batter.get("iso", max(0.05, slg - avg)) or max(0.05, slg - avg))
        woba = float(batter.get("woba", 0.315) or 0.315)

        # Expected Hits (Poisson / Negative Binomial expectation)
        exp_hits = pa * avg * p_hit_suppression * run_factor * weather_factor
        p0 = round(math.exp(-exp_hits), 4)
        p1_exact = round(exp_hits * p0, 4)
        p2_exact = round((exp_hits**2 / 2.0) * p0, 4)

        p1_plus = round(min(max(1.0 - p0, 0.0), 1.0), 4)
        p2_plus = round(min(max(1.0 - p0 - p1_exact, 0.0), 1.0), 4)
        p3_plus = round(min(max(1.0 - p0 - p1_exact - p2_exact, 0.0), 1.0), 4)

        hits_dist = {
            "0": p0, 0: p0,
            "1+": p1_plus, 1: p1_plus,
            "2+": p2_plus, 2: p2_plus,
            "3+": p3_plus, 3: p3_plus,
        }

        # Expected Total Bases
        exp_tb = pa * (avg + 1.2 * iso) * p_hit_suppression * run_factor * weather_factor
        tb_p0 = math.exp(-exp_tb)
        tb_p1 = exp_tb * tb_p0
        tb_p2 = (exp_tb**2 / 2.0) * tb_p0
        tb_p3 = (exp_tb**3 / 6.0) * tb_p0

        tb_05 = round(min(max(1.0 - tb_p0, 0.0), 1.0), 4)
        tb_15 = round(min(max(1.0 - tb_p0 - tb_p1, 0.0), 1.0), 4)
        tb_25 = round(min(max(1.0 - tb_p0 - tb_p1 - tb_p2, 0.0), 1.0), 4)
        tb_35 = round(min(max(1.0 - tb_p0 - tb_p1 - tb_p2 - tb_p3, 0.0), 1.0), 4)

        tb_dist = {
            "0.5": tb_05, 0.5: tb_05,
            "1.5": tb_15, 1.5: tb_15,
            "2.5": tb_25, 2.5: tb_25,
            "3.5": tb_35, 3.5: tb_35,
        }

        # Home Run probability: P(HR >= 1) = 1 - (1 - p_HR)^PA
        p_hr_per_pa = max(0.010, min(0.085, (iso / 4.0) * hr_factor * weather_factor))
        p_hr = round(min(max(1.0 - (1.0 - p_hr_per_pa)**pa, 0.0), 1.0), 4)

        # Hits + Runs + RBIs
        exp_hrr = exp_hits + (pa * obp * 0.65) + (pa * slg * 0.45)
        hrr_p0 = math.exp(-exp_hrr)
        hrr_p1 = exp_hrr * hrr_p0
        hrr_over_1_5 = round(min(max(1.0 - hrr_p0 - hrr_p1, 0.0), 1.0), 4)
        hrr_p2 = (exp_hrr**2 / 2.0) * hrr_p0
        hrr_over_2_5 = round(min(max(1.0 - hrr_p0 - hrr_p1 - hrr_p2, 0.0), 1.0), 4)

        props_results.append({
            "player_id": batter.get("id"),
            "player_name": batter.get("name", f"Batter {order}"),
            "role": "batter",
            "order": order,
            "expected_pa": round(pa, 2),
            "expected_hits": round(exp_hits, 2),
            "hits_dist": hits_dist,
            "expected_tb": round(exp_tb, 2),
            "tb_dist": tb_dist,
            "hr_prob": p_hr,
            "hr_prob_over_0_5": p_hr,
            "hrr_over_1_5": hrr_over_1_5,
            "hrr_over_2_5": hrr_over_2_5,
            "hrr_dist": {"1.5": hrr_over_1_5, "2.5": hrr_over_2_5},
        })

    # 2. Starting Pitcher Props (Ks and Outs)
    if pitcher:
        try:
            med_ip = float(pitcher.get("median_ip", 5.2) or 5.2)
        except (ValueError, TypeError):
            med_ip = 5.2

        expected_bf = max(15.0, med_ip * 4.25)
        expected_ks = expected_bf * p_k_pct
        expected_outs = med_ip * 3.0

        # Strikeouts distribution across common betting lines
        k_lines = [3.5, 4.5, 5.5, 6.5, 7.5]
        k_dist = {}
        for kl in k_lines:
            # Over line kl means >= int(kl) + 1
            threshold = int(math.floor(kl))
            p_under = stats.poisson.cdf(threshold, expected_ks)
            p_over = round(float(1.0 - p_under), 4)
            k_dist[kl] = p_over
            k_dist[str(kl)] = p_over

        # Pitcher Outs distribution across common betting lines
        out_lines = [14.5, 15.5, 16.5, 17.5, 18.5]
        outs_dist = {}
        for ol in out_lines:
            thresh_out = int(math.floor(ol))
            p_under_out = stats.poisson.cdf(thresh_out, expected_outs)
            p_over_out = round(float(1.0 - p_under_out), 4)
            outs_dist[ol] = p_over_out
            outs_dist[str(ol)] = p_over_out

        props_results.append({
            "player_id": pitcher.get("id"),
            "player_name": pitcher.get("name", "Starting Pitcher"),
            "role": "pitcher",
            "expected_ks": round(expected_ks, 2),
            "strikeouts_dist": k_dist,
            "expected_outs": round(expected_outs, 2),
            "outs_dist": outs_dist,
        })

    return props_results


# =====================================================================
# 6. Model Evaluation, Log Loss, Brier Score & Calibration
# =====================================================================

def compute_calibration_metrics(
    predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate Log Loss, Brier Score, and 7-bucket calibration categorization.

    Buckets:
    50-54%, 55-59%, 60-64%, 65-69%, 70-74%, 75-79%, 80%+
    Flags: Overconfident, Underconfident, or Well Calibrated.
    """
    bucket_definitions = [
        ("50-54%", 0.50, 0.55),
        ("55-59%", 0.55, 0.60),
        ("60-64%", 0.60, 0.65),
        ("65-69%", 0.65, 0.70),
        ("70-74%", 0.70, 0.75),
        ("75-79%", 0.75, 0.80),
        ("80%+", 0.80, 1.01),
    ]

    valid_pairs: list[tuple[float, float, dict[str, Any]]] = []

    for item in predictions:
        p_val = item.get("model_prob") or item.get("pred_prob") or item.get("p")
        res_val = item.get("result") or item.get("actual") or item.get("outcome") or item.get("y")

        if p_val is None or res_val is None:
            continue

        try:
            p = float(p_val)
        except (ValueError, TypeError):
            continue

        # Parse outcome
        if isinstance(res_val, str):
            res_str = res_val.strip().upper()
            if res_str in ("WIN", "W", "1", "TRUE"):
                y = 1.0
            elif res_str in ("LOSS", "L", "0", "FALSE"):
                y = 0.0
            else:
                continue  # Skip PENDING / PUSH
        elif isinstance(res_val, (int, float, bool)):
            y = 1.0 if res_val else 0.0
        else:
            continue

        # Map to favorite perspective if underdog logged (< 0.50)
        if p < 0.50:
            eval_p = 1.0 - p
            eval_y = 1.0 - y
        else:
            eval_p = p
            eval_y = y

        valid_pairs.append((eval_p, eval_y, item))

    # Empty predictions defense
    if not valid_pairs:
        empty_buckets = [
            {
                "bucket": name,
                "count": 0,
                "avg_predicted_prob": 0.0,
                "actual_win_pct": 0.0,
                "status": "Well Calibrated",
            }
            for name, _, _ in bucket_definitions
        ]
        return {
            "total_predictions": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "status": "Well Calibrated",
            "calibration_buckets": empty_buckets,
        }

    # Brier Score & Log Loss computation
    n = len(valid_pairs)
    probs = np.array([p for p, _, _ in valid_pairs], dtype=np.float64)
    actuals = np.array([y for _, y, _ in valid_pairs], dtype=np.float64)

    brier_score = float(np.mean((probs - actuals) ** 2))

    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    log_loss = float(-np.mean(actuals * np.log(probs_clipped) + (1.0 - actuals) * np.log(1.0 - probs_clipped)))

    # Bucket grouping
    buckets_output: list[dict[str, Any]] = []
    bucket_diffs: list[float] = []

    for name, low, high in bucket_definitions:
        in_bucket = [
            (p, y) for p, y, _ in valid_pairs
            if low <= p < high or (high >= 1.0 and p >= low)
        ]
        count = len(in_bucket)
        if count > 0:
            avg_p = float(np.mean([p for p, _ in in_bucket]))
            act_y = float(np.mean([y for _, y in in_bucket]))
            diff = avg_p - act_y
            bucket_diffs.append(diff)
            if diff > 0.03:
                b_status = "Overconfident"
            elif diff < -0.03:
                b_status = "Underconfident"
            else:
                b_status = "Well Calibrated"
        else:
            avg_p = 0.0
            act_y = 0.0
            b_status = "Well Calibrated"

        buckets_output.append({
            "bucket": name,
            "count": count,
            "avg_predicted_prob": round(avg_p, 4),
            "actual_win_pct": round(act_y, 4),
            "status": b_status,
        })

    # Overall Model Calibration Status
    overall_avg_pred = float(np.mean(probs))
    overall_actual = float(np.mean(actuals))
    overall_diff = overall_avg_pred - overall_actual

    if overall_diff > 0.03:
        status = "Overconfident"
    elif overall_diff < -0.03:
        status = "Underconfident"
    else:
        status = "Well Calibrated"

    return {
        "total_predictions": n,
        "brier_score": round(brier_score, 4),
        "log_loss": round(log_loss, 4),
        "status": status,
        "calibration_buckets": buckets_output,
    }

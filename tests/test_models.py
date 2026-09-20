"""Tests for models.py: Vectorized Monte Carlo simulation, statistical models & calibration.

Covers:
- Vectorized Monte Carlo speed benchmark (< 1.5s for 15 games x 10,000 iterations).
- Dynamic starter vs bullpen inning allocation.
- Inning-by-inning Negative Binomial simulation and ghost runner extra innings.
- NRFI / YRFI probability evaluation.
- Quadrature uncertainty interval computation.
- Odds conversions, two-way devigging, and odds-independent edge calculations.
- Discrete player prop distributions (Hits, Total Bases, HR, H+R+RBI, Ks, Outs).
- Calibration metrics: Log loss, Brier score, and 7 calibration buckets.
"""

from __future__ import annotations

import math
import time
import pytest
import numpy as np

import models


# =====================================================================
# 1. Odds Conversions, Devigging & Edge Calculations
# =====================================================================

def test_american_to_prob():
    """Verify American odds conversion to implied probability."""
    # Favorites (negative odds)
    assert pytest.approx(models.american_to_prob(-100), abs=1e-4) == 0.50
    assert pytest.approx(models.american_to_prob(-110), abs=1e-4) == 0.5238
    assert pytest.approx(models.american_to_prob(-150), abs=1e-4) == 0.6000
    assert pytest.approx(models.american_to_prob(-200), abs=1e-4) == 0.6667
    assert pytest.approx(models.american_to_prob(-300), abs=1e-4) == 0.7500

    # Underdogs (positive odds)
    assert pytest.approx(models.american_to_prob(100), abs=1e-4) == 0.50
    assert pytest.approx(models.american_to_prob(110), abs=1e-4) == 0.4762
    assert pytest.approx(models.american_to_prob(150), abs=1e-4) == 0.4000
    assert pytest.approx(models.american_to_prob(200), abs=1e-4) == 0.3333

    # String format input defense
    assert pytest.approx(models.american_to_prob("+150"), abs=1e-4) == 0.4000
    assert pytest.approx(models.american_to_prob("-150"), abs=1e-4) == 0.6000


def test_prob_to_american():
    """Verify implied probability conversion to American odds format string."""
    assert models.prob_to_american(0.50) in ["-100", "+100"]
    assert models.prob_to_american(0.60) == "-150"
    assert models.prob_to_american(0.40) == "+150"
    assert models.prob_to_american(0.6667) == "-200"
    assert models.prob_to_american(0.3333) == "+200"
    assert models.prob_to_american(0.75) == "-300"

    # Edge cases
    assert models.prob_to_american(0.0) == "+10000"
    assert models.prob_to_american(1.0) == "-10000"


def test_devig_two_way():
    """Verify multiplicative two-way market devigging."""
    # Standard -110 / -110 market
    p1, p2 = models.devig_two_way(-110, -110)
    assert pytest.approx(p1, abs=1e-4) == 0.50
    assert pytest.approx(p2, abs=1e-4) == 0.50
    assert pytest.approx(p1 + p2, abs=1e-4) == 1.0

    # Asymmetric market: -150 (raw 0.60) / +130 (raw ~0.4348)
    p1, p2 = models.devig_two_way(-150, 130)
    assert pytest.approx(p1 + p2, abs=1e-4) == 1.0
    assert p1 > 0.57 and p1 < 0.59
    assert p2 > 0.41 and p2 < 0.43

    # Probability inputs directly
    p1, p2 = models.devig_two_way(0.55, 0.55)
    assert pytest.approx(p1, abs=1e-4) == 0.50
    assert pytest.approx(p2, abs=1e-4) == 0.50


def test_calculate_edge_with_and_without_market_odds():
    """Verify decoupled odds-independent picks and market edge layering."""
    # Decoupled case: market_line is None (must not fail!)
    no_odds = models.calculate_edge(model_prob=0.585, market_line=None)
    assert no_odds["fair_odds"] == "-141"
    assert no_odds["market_prob_novig"] is None
    assert no_odds["edge_pct"] is None
    assert no_odds["ev_pct"] is None
    assert no_odds["recommendation"] == "Model Pick"

    # Decoupled case: market_line is 0 or empty string
    no_odds_zero = models.calculate_edge(model_prob=0.585, market_line=0)
    assert no_odds_zero["recommendation"] == "Model Pick"

    # Strong edge case: Model 62%, market -110 (novig 50%) -> edge = +12.0%
    strong = models.calculate_edge(model_prob=0.62, market_line=-110, market_opp_line=-110)
    assert strong["fair_odds"] == "-163"
    assert pytest.approx(strong["market_prob_novig"], abs=1e-3) == 0.50
    assert strong["edge_pct"] > 10.0
    assert strong["ev_pct"] > 15.0
    assert strong["recommendation"] == "Strong Edge"

    # Slight edge case: Model 53%, market -110 (novig 50%) -> edge = +3.0%
    slight = models.calculate_edge(model_prob=0.53, market_line=-110, market_opp_line=-110)
    assert pytest.approx(slight["edge_pct"], abs=1e-2) == 3.0
    assert slight["recommendation"] == "Slight Edge"

    # Pass case: Model 50.5%, market -110 (novig 50%) -> edge = +0.5%
    neutral = models.calculate_edge(model_prob=0.505, market_line=-110, market_opp_line=-110)
    assert neutral["recommendation"] == "Pass"

    # Fade / Negative edge case: Model 42%, market -110 (novig 50%) -> edge = -8.0%
    fade = models.calculate_edge(model_prob=0.42, market_line=-110, market_opp_line=-110)
    assert fade["edge_pct"] < -5.0
    assert fade["recommendation"] == "Fade"


# =====================================================================
# 2. Weather Factor Calculation
# =====================================================================

def test_calculate_weather_factor():
    """Verify dimensionless weather multiplier against roof types, temp and wind."""
    constants = {
        "temp_coefficient": 0.012,
        "wind_coefficient_out": 0.015,
        "wind_coefficient_in": 0.012,
    }

    # Closed dome / roof suppresses weather completely
    assert models.calculate_weather_factor({"temp": 95, "wind_speed": 25, "wind_dir": "Out"}, "Dome", constants) == 1.0
    assert models.calculate_weather_factor({"temp": 40, "wind_speed": 20, "wind_dir": "In"}, "Closed", constants) == 1.0
    assert models.calculate_weather_factor({"is_dome": True, "temp": 90}, "Open", constants) == 1.0

    # Neutral baseline: 72F and calm / <= 5 mph wind
    neutral = models.calculate_weather_factor({"temp": 72, "wind_speed": 4, "wind_dir": "Out"}, "Open", constants)
    assert pytest.approx(neutral, abs=1e-4) == 1.0

    # High temperature: 82F (+10F) -> 1.0 + 0.012 * 1 = 1.012
    hot = models.calculate_weather_factor({"temp": 82, "wind_speed": 0, "wind_dir": "Calm"}, "Open", constants)
    assert pytest.approx(hot, abs=1e-4) == 1.012

    # Low temperature: 52F (-20F) -> 1.0 - 0.012 * 2 = 0.976
    cold = models.calculate_weather_factor({"temp": 52, "wind_speed": 0, "wind_dir": "Calm"}, "Open", constants)
    assert pytest.approx(cold, abs=1e-4) == 0.976

    # Strong wind blowing out: 15 mph Out (+10 above 5) -> +0.015 * 10 = +0.150
    wind_out = models.calculate_weather_factor({"temp": 72, "wind_speed": 15, "wind_dir": "Out to CF"}, "Open", constants)
    assert pytest.approx(wind_out, abs=1e-4) == 1.150

    # Strong wind blowing in: 15 mph In (+10 above 5) -> -0.012 * 10 = -0.120
    wind_in = models.calculate_weather_factor({"temp": 72, "wind_speed": 15, "wind_dir": "In from LF"}, "Open", constants)
    assert pytest.approx(wind_in, abs=1e-4) == 0.880

    # Crosswind does not trigger wind_out or wind_in multipliers
    crosswind = models.calculate_weather_factor({"temp": 72, "wind_speed": 18, "wind_dir": "L to R"}, "Open", constants)
    assert pytest.approx(crosswind, abs=1e-4) == 1.0


# =====================================================================
# 3. Quadrature Uncertainty Interval Calculation
# =====================================================================

def test_calculate_uncertainty():
    """Verify empirical and Monte Carlo uncertainty interval formula with sqrt in quadrature."""
    # Ace with confirmed lineup and large sample (sample_ip >= 50, lineup CONFIRMED)
    # sigma_baseline = 0.025, sigma_sample = 0.0, sigma_lineup = 0.0 -> sigma_model = 0.025
    # p = 0.642, n_sim = 10000 -> SE_MC = sqrt(0.642 * 0.358 / 10000) = 0.004794
    # SE_total = sqrt(0.004794^2 + 0.025^2) = 0.025455
    # U_95 = 1.96 * 0.025455 = 0.04989 -> rounded to 0.0499 (approx 5.0%)
    u_ace = models.calculate_uncertainty(p=0.642, n_sim=10000, sample_ip=160.0, lineup_status="CONFIRMED")
    assert pytest.approx(u_ace, abs=1e-3) == 0.0499

    # Unconfirmed lineup adds sigma_lineup = 0.015 -> uncertainty increases
    u_proj = models.calculate_uncertainty(p=0.642, n_sim=10000, sample_ip=160.0, lineup_status="PROJECTED")
    assert u_proj > u_ace

    # Low sample IP (e.g. 10 IP) adds sample uncertainty -> uncertainty increases further
    u_rookie = models.calculate_uncertainty(p=0.642, n_sim=10000, sample_ip=10.0, lineup_status="CONFIRMED")
    assert u_rookie > u_ace

    # Bounds: uncertainty is non-negative and finite
    assert 0.0 < u_ace < 0.20
    assert 0.0 < u_rookie < 0.20


# =====================================================================
# 4. Monte Carlo Game Simulation & Negative Binomial Modeling
# =====================================================================

def test_simulate_game_output_structure_and_bounds(sample_game_data):
    """Verify complete simulate_game outputs, bounds, and probability consistency."""
    constants = {
        "dispersion_alpha": 0.12,
        "home_field_advantage_runs": 0.18,
        "temp_coefficient": 0.012,
        "wind_coefficient_out": 0.015,
        "wind_coefficient_in": 0.012,
        "baseline_league_runs": 4.40,
    }

    sim = models.simulate_game(
        home_team=sample_game_data["home_team"],
        away_team=sample_game_data["away_team"],
        venue=sample_game_data["venue"],
        weather=sample_game_data["weather"],
        constants=constants,
        n_sim=5000,
    )

    # 1. Probabilities sum to 1.0 (no ties after ghost runner extra innings)
    assert pytest.approx(sim["home_win_prob"] + sim["away_win_prob"], abs=1e-4) == 1.0
    assert 0.0 < sim["home_win_prob"] < 1.0
    assert 0.0 < sim["away_win_prob"] < 1.0

    # 2. Expected scores & totals
    assert sim["expected_home_score"] > 2.0 and sim["expected_home_score"] < 8.0
    assert sim["expected_away_score"] > 2.0 and sim["expected_away_score"] < 8.0
    assert pytest.approx(sim["projected_total"], abs=1e-3) == sim["expected_home_score"] + sim["expected_away_score"]

    # 3. Spread covers (+1.5 and -1.5)
    assert sim["home_cover_plus_1_5"] >= sim["home_win_prob"]
    assert sim["away_cover_plus_1_5"] >= sim["away_win_prob"]
    assert sim["home_cover_minus_1_5"] <= sim["home_win_prob"]
    assert sim["away_cover_minus_1_5"] <= sim["away_win_prob"]

    # 4. Spread deltas: P(+1.5) - P(ML)
    assert pytest.approx(sim["spread_delta_home"], abs=1e-4) == sim["home_cover_plus_1_5"] - sim["home_win_prob"]
    assert pytest.approx(sim["spread_delta_away"], abs=1e-4) == sim["away_cover_plus_1_5"] - sim["away_win_prob"]
    assert sim["spread_delta_home"] >= 0.0
    assert sim["spread_delta_away"] >= 0.0

    # 5. NRFI / YRFI
    assert pytest.approx(sim["p_nrfi"] + sim["p_yrfi"], abs=1e-4) == 1.0
    assert 0.30 < sim["p_nrfi"] < 0.70

    # 6. Totals distribution
    totals = sim["totals_distribution"]
    for line in [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]:
        assert line in totals or str(line) in totals
        entry = totals.get(line) or totals.get(str(line))
        assert "over" in entry and "under" in entry
        assert 0.0 <= entry["over"] <= 1.0
        assert 0.0 <= entry["under"] <= 1.0

    # 7. Uncertainty & Fair Odds
    assert 0.0 < sim["home_uncertainty_95"] < 0.15
    assert 0.0 < sim["away_uncertainty_95"] < 0.15
    assert sim["home_fair_odds"].startswith(("+", "-"))
    assert sim["away_fair_odds"].startswith(("+", "-"))
    assert sim["nrfi_fair_odds"].startswith(("+", "-"))
    assert sim["yrfi_fair_odds"].startswith(("+", "-"))

    # 8. Data confidence: Both starters confirmed with >= 35 IP and confirmed lineup
    assert sim["data_confidence"] == "HIGH"


def test_dynamic_starter_vs_bullpen_allocation(sample_game_data):
    """Verify simulation adjusts dynamically to opener vs deep starter."""
    constants = {"dispersion_alpha": 0.12, "baseline_league_runs": 4.40}

    # Case A: Opener pitching 1.2 innings with mediocre bullpen
    opener_game = sample_game_data.copy()
    opener_away = dict(sample_game_data["away_team"])
    opener_starter = dict(opener_away["starter"])
    opener_starter["median_ip"] = 1.2
    opener_away["starter"] = opener_starter
    opener_away["bullpen_era"] = 5.50
    opener_away["bullpen_fip"] = 5.60
    opener_away["bullpen_fatigue"] = 0.80

    # Case B: Ace pitching 7.0 innings with elite metrics
    ace_away = dict(sample_game_data["away_team"])
    ace_starter = dict(ace_away["starter"])
    ace_starter["median_ip"] = 7.0
    ace_starter["xfip"] = 2.80
    ace_starter["era"] = 2.50
    ace_away["starter"] = ace_starter

    sim_opener = models.simulate_game(
        home_team=sample_game_data["home_team"],
        away_team=opener_away,
        venue=sample_game_data["venue"],
        weather=sample_game_data["weather"],
        constants=constants,
        n_sim=5000,
    )

    sim_ace = models.simulate_game(
        home_team=sample_game_data["home_team"],
        away_team=ace_away,
        venue=sample_game_data["venue"],
        weather=sample_game_data["weather"],
        constants=constants,
        n_sim=5000,
    )

    # When Ace pitches deep, Home scores fewer expected runs than against Opener + tired bullpen
    assert sim_ace["expected_home_score"] < sim_opener["expected_home_score"]
    # Ace team has higher win probability than Opener team
    assert sim_ace["away_win_prob"] > sim_opener["away_win_prob"]


def test_data_confidence_levels(sample_game_data):
    """Verify HIGH, MEDIUM, and LOW data confidence assignments."""
    constants = {"dispersion_alpha": 0.12, "baseline_league_runs": 4.40}

    # HIGH: confirmed starters >= 35 IP and confirmed lineups
    sim_high = models.simulate_game(
        sample_game_data["home_team"], sample_game_data["away_team"],
        sample_game_data["venue"], sample_game_data["weather"], constants, n_sim=1000
    )
    assert sim_high["data_confidence"] == "HIGH"

    # MEDIUM: confirmed starters but PROJECTED lineup
    proj_home = dict(sample_game_data["home_team"])
    proj_home["lineup_status"] = "PROJECTED"
    sim_med = models.simulate_game(
        proj_home, sample_game_data["away_team"],
        sample_game_data["venue"], sample_game_data["weather"], constants, n_sim=1000
    )
    assert sim_med["data_confidence"] == "MEDIUM"

    # LOW: low starter sample IP (< 35 IP) or missing starter
    low_away = dict(sample_game_data["away_team"])
    low_starter = dict(low_away["starter"])
    low_starter["sample_ip"] = 12.0
    low_away["starter"] = low_starter
    sim_low = models.simulate_game(
        sample_game_data["home_team"], low_away,
        sample_game_data["venue"], sample_game_data["weather"], constants, n_sim=1000
    )
    assert sim_low["data_confidence"] == "LOW"


# =====================================================================
# 5. Performance Benchmark (< 1.5s for 15 games x 10,000 iterations)
# =====================================================================

def test_monte_carlo_performance_benchmark(sample_game_data):
    """Verify 15 games x 10,000 iterations execute in < 1.5 seconds."""
    constants = {
        "dispersion_alpha": 0.12,
        "home_field_advantage_runs": 0.18,
        "temp_coefficient": 0.012,
        "wind_coefficient_out": 0.015,
        "wind_coefficient_in": 0.012,
        "baseline_league_runs": 4.40,
    }

    n_games = 15
    n_sim = 10000

    start_time = time.perf_counter()
    for _ in range(n_games):
        models.simulate_game(
            home_team=sample_game_data["home_team"],
            away_team=sample_game_data["away_team"],
            venue=sample_game_data["venue"],
            weather=sample_game_data["weather"],
            constants=constants,
            n_sim=n_sim,
        )
    elapsed = time.perf_counter() - start_time

    print(f"\n[BENCHMARK] 15 games x 10,000 iterations completed in: {elapsed:.3f} seconds")
    assert elapsed < 1.5, f"Simulation took {elapsed:.3f}s, exceeding 1.5s benchmark!"


# =====================================================================
# 6. Player Prop Projections
# =====================================================================

def test_model_player_props(sample_game_data):
    """Verify player prop distributions for Hit, TB, HR, H+R+RBI, Pitcher Ks, and Outs."""
    batters = sample_game_data["away_team"]["lineup"]
    pitcher = sample_game_data["home_team"]["starter"]
    venue = sample_game_data["venue"]
    weather = sample_game_data["weather"]

    props = models.model_player_props(
        batter_list=batters,
        pitcher=pitcher,
        venue=venue,
        weather=weather
    )

    assert isinstance(props, list)
    assert len(props) > 0

    # Verify batter props
    batter_props = [p for p in props if p.get("role") != "pitcher"]
    assert len(batter_props) == len(batters)

    for bp in batter_props:
        # Hits distribution: 0, 1+, 2+, 3+
        hits = bp["hits_dist"]
        assert 0.0 <= hits["0"] <= 1.0
        assert 0.0 <= hits["1+"] <= 1.0
        assert 0.0 <= hits["2+"] <= 1.0
        assert 0.0 <= hits["3+"] <= 1.0
        # Monotonic: P(1+) >= P(2+) >= P(3+)
        assert hits["1+"] >= hits["2+"] >= hits["3+"]

        # Total bases distribution: 0.5, 1.5, 2.5, 3.5
        tb = bp["tb_dist"]
        assert tb["0.5"] >= tb["1.5"] >= tb["2.5"] >= tb["3.5"]

        # Home Run probability
        assert 0.0 < bp["hr_prob"] < 0.60

        # Hits + Runs + RBIs
        assert "hrr_dist" in bp or "hrr_over_1_5" in bp

    # Verify pitcher props
    pitcher_props = [p for p in props if p.get("role") == "pitcher"]
    assert len(pitcher_props) == 1
    pp = pitcher_props[0]
    assert pp["player_name"] == pitcher["name"]

    # Strikeouts distribution
    k_dist = pp["strikeouts_dist"]
    assert k_dist["4.5"] >= k_dist["5.5"] >= k_dist["6.5"]
    assert 0.0 < pp["expected_ks"] < 15.0

    # Outs recorded distribution
    outs_dist = pp["outs_dist"]
    assert outs_dist["15.5"] >= outs_dist["17.5"]
    assert 0.0 < pp["expected_outs"] < 27.0


# =====================================================================
# 7. Calibration Metrics, Log Loss, Brier Score & 7 Buckets
# =====================================================================

def test_compute_calibration_metrics():
    """Verify Brier Score, Log Loss, 7-bucket grouping and over/underconfidence flags."""
    # Synthetic well-calibrated dataset
    predictions = [
        {"model_prob": 0.52, "result": "WIN", "spread_delta": 0.04, "units_won": 0.91},
        {"model_prob": 0.53, "result": "LOSS", "spread_delta": 0.03, "units_won": -1.0},
        {"model_prob": 0.58, "result": "WIN", "spread_delta": 0.08, "units_won": 0.85},
        {"model_prob": 0.63, "result": "WIN", "spread_delta": 0.12, "units_won": 0.80},
        {"model_prob": 0.68, "result": "WIN", "spread_delta": 0.14, "units_won": 0.75},
        {"model_prob": 0.72, "result": "WIN", "spread_delta": 0.18, "units_won": 0.70},
        {"model_prob": 0.78, "result": "WIN", "spread_delta": 0.19, "units_won": 0.65},
        {"model_prob": 0.83, "result": "WIN", "spread_delta": 0.22, "units_won": 0.60},
    ]

    metrics = models.compute_calibration_metrics(predictions)

    # Brier Score & Log Loss
    assert metrics["brier_score"] > 0.0
    assert metrics["log_loss"] > 0.0
    assert metrics["total_predictions"] == 8

    # 7 Calibration Buckets
    buckets = metrics["calibration_buckets"]
    assert len(buckets) == 7
    bucket_labels = [b["bucket"] for b in buckets]
    assert bucket_labels == ["50-54%", "55-59%", "60-64%", "65-69%", "70-74%", "75-79%", "80%+"]

    # Check 50-54% bucket: 2 predictions, 1 win -> 50% actual vs ~52.5% predicted
    b50 = [b for b in buckets if b["bucket"] == "50-54%"][0]
    assert b50["count"] == 2
    assert pytest.approx(b50["actual_win_pct"], abs=1e-3) == 0.50
    assert pytest.approx(b50["avg_predicted_prob"], abs=1e-3) == 0.525
    assert b50["status"] == "Well Calibrated"

    # Status auto-flag verification
    assert metrics["status"] in ["Well Calibrated", "Overconfident", "Underconfident"]


def test_calibration_overconfidence_and_underconfidence_detection():
    """Verify auto-detection of overconfident and underconfident model predictions."""
    # Model predicted 80%+ on everything, but lost every game -> Overconfident!
    overconfident_preds = [
        {"model_prob": 0.85, "result": "LOSS"},
        {"model_prob": 0.82, "result": "LOSS"},
        {"model_prob": 0.88, "result": "LOSS"},
    ]
    oc_metrics = models.compute_calibration_metrics(overconfident_preds)
    assert oc_metrics["status"] == "Overconfident"
    b80 = [b for b in oc_metrics["calibration_buckets"] if b["bucket"] == "80%+"][0]
    assert b80["status"] == "Overconfident"

    # Model predicted 51% on everything, but won every game -> Underconfident!
    underconfident_preds = [
        {"model_prob": 0.51, "result": "WIN"},
        {"model_prob": 0.52, "result": "WIN"},
        {"model_prob": 0.53, "result": "WIN"},
    ]
    uc_metrics = models.compute_calibration_metrics(underconfident_preds)
    assert uc_metrics["status"] == "Underconfident"
    b50 = [b for b in uc_metrics["calibration_buckets"] if b["bucket"] == "50-54%"][0]
    assert b50["status"] == "Underconfident"


def test_calibration_empty_predictions():
    """Verify graceful handling when predictions list is empty."""
    empty_metrics = models.compute_calibration_metrics([])
    assert empty_metrics["total_predictions"] == 0
    assert empty_metrics["brier_score"] == 0.0
    assert empty_metrics["log_loss"] == 0.0
    assert len(empty_metrics["calibration_buckets"]) == 7
    assert empty_metrics["status"] == "Well Calibrated"

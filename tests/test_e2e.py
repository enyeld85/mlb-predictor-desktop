"""Comprehensive End-to-End Integration Tests for MLB Apex Desktop Platform.

Validates the full system lifecycle:
1. SQLite Database initialization and dynamic model constants seeding.
2. Defensive data ingestion from MLB Stats API (live fetch & offline SQLite cache fallback).
3. 10,000-iteration vectorized Monte Carlo simulation across all 8 core prediction markets:
   - Moneylines, Run Lines (+/- 1.5), Spread Deltas, Totals distribution (6.5 to 10.5),
   - 1st-inning NRFI/YRFI, Quadrature predictive uncertainty intervals, Data confidence tiers,
   - Discrete player prop distributions (Hits, Total Bases, HR, H+R+RBI, Ks, Outs).
4. Odds-independent decoupled pick generation vs market odds two-way devigging, Edge %, and EV %.
5. Pick logging into SQLite `picks_history`, settlement of picks, and Historical Delta Performance buckets.
6. Calibration metrics computation (Brier Score, Log Loss, and 7 calibration buckets).
7. Dynamic model parameter recalibration and persistence.
8. Headless Tkinter GUI initialization, dark sportsbook theme verification, and rendering
   across all 8 tabs with inline odds editing, dynamic filtering, pick logging, and Matplotlib charts.
"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import MagicMock
import pytest

from database import (
    init_db,
    get_db_connection,
    get_model_constants,
    update_model_constants,
    save_slate_cache,
    get_cached_slate,
    log_pick,
    update_pick_result,
    get_picks_history,
    get_delta_buckets_history,
)
from data_fetch import MLBDataFetcher
from models import (
    simulate_game,
    model_player_props,
    american_to_prob,
    prob_to_american,
    devig_two_way,
    calculate_edge,
    calculate_uncertainty,
    calculate_weather_factor,
    compute_calibration_metrics,
)
from gui import MLBPredictorApp


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def e2e_db(temp_db_path):
    """Fixture providing initialized database path and connection."""
    conn = init_db(temp_db_path)
    yield temp_db_path, conn
    conn.close()


@pytest.fixture
def multi_game_slate():
    """Fixture providing a multi-game slate with varying stadium and weather environments."""
    return [
        {
            "game_pk": 700101,
            "game_date": "2026-09-20",
            "game_time": "13:05",
            "status": "Scheduled",
            "away_team": {
                "id": 147,
                "name": "New York Yankees",
                "abbrev": "NYY",
                "record": "90-62",
                "wrc_plus_vs_rhp": 120,
                "wrc_plus_vs_lhp": 115,
                "woba_vs_rhp": 0.338,
                "woba_vs_lhp": 0.328,
                "bullpen_era": 3.20,
                "bullpen_whip": 1.12,
                "bullpen_fip": 3.40,
                "bullpen_fatigue": 0.20,
                "starter": {
                    "id": 543037,
                    "name": "Gerrit Cole",
                    "hand": "R",
                    "era": 3.05,
                    "fip": 3.15,
                    "xfip": 3.20,
                    "whip": 1.02,
                    "k_pct": 0.295,
                    "bb_pct": 0.058,
                    "median_ip": 6.2,
                    "first_inning_era": 2.40,
                    "first_inning_whip": 0.90,
                    "sample_ip": 175.0,
                },
                "lineup_status": "CONFIRMED",
                "lineup": [
                    {"id": 1, "name": "Gleyber Torres", "pos": "2B", "order": 1, "woba": 0.330, "bats": "R"},
                    {"id": 2, "name": "Juan Soto", "pos": "RF", "order": 2, "woba": 0.420, "bats": "L"},
                    {"id": 3, "name": "Aaron Judge", "pos": "CF", "order": 3, "woba": 0.450, "bats": "R"},
                    {"id": 4, "name": "Giancarlo Stanton", "pos": "DH", "order": 4, "woba": 0.350, "bats": "R"},
                ],
            },
            "home_team": {
                "id": 111,
                "name": "Boston Red Sox",
                "abbrev": "BOS",
                "record": "79-73",
                "wrc_plus_vs_rhp": 105,
                "wrc_plus_vs_lhp": 99,
                "woba_vs_rhp": 0.319,
                "woba_vs_lhp": 0.309,
                "bullpen_era": 4.15,
                "bullpen_whip": 1.30,
                "bullpen_fip": 4.10,
                "bullpen_fatigue": 0.35,
                "starter": {
                    "id": 678394,
                    "name": "Brayan Bello",
                    "hand": "R",
                    "era": 4.15,
                    "fip": 4.10,
                    "xfip": 4.00,
                    "whip": 1.30,
                    "k_pct": 0.215,
                    "bb_pct": 0.078,
                    "median_ip": 5.2,
                    "first_inning_era": 4.20,
                    "first_inning_whip": 1.35,
                    "sample_ip": 150.0,
                },
                "lineup_status": "CONFIRMED",
                "lineup": [
                    {"id": 11, "name": "Jarren Duran", "pos": "CF", "order": 1, "woba": 0.355, "bats": "L"},
                    {"id": 12, "name": "Rafael Devers", "pos": "3B", "order": 2, "woba": 0.385, "bats": "L"},
                    {"id": 13, "name": "Tyler O'Neill", "pos": "LF", "order": 3, "woba": 0.360, "bats": "R"},
                    {"id": 14, "name": "Triston Casas", "pos": "1B", "order": 4, "woba": 0.365, "bats": "L"},
                ],
            },
            "venue": {
                "id": 3,
                "venue_id": 3,
                "name": "Fenway Park",
                "venue_name": "Fenway Park",
                "altitude": 20,
                "roof_type": "Open",
                "run_factor": 1.08,
                "hr_factor": 1.05,
            },
            "weather": {
                "temp": 76,
                "wind_speed": 10,
                "wind_dir": "Out to CF",
                "condition": "Clear",
            },
        },
        {
            "game_pk": 700102,
            "game_date": "2026-09-20",
            "game_time": "16:10",
            "status": "Scheduled",
            "away_team": {
                "id": 119,
                "name": "Los Angeles Dodgers",
                "abbrev": "LAD",
                "record": "95-57",
                "wrc_plus_vs_rhp": 125,
                "wrc_plus_vs_lhp": 118,
                "woba_vs_rhp": 0.345,
                "woba_vs_lhp": 0.332,
                "bullpen_era": 3.10,
                "bullpen_whip": 1.10,
                "bullpen_fip": 3.25,
                "bullpen_fatigue": 0.15,
                "starter": {
                    "id": 669373,
                    "name": "Yoshinobu Yamamoto",
                    "hand": "R",
                    "era": 2.95,
                    "fip": 3.00,
                    "xfip": 3.05,
                    "whip": 1.00,
                    "k_pct": 0.300,
                    "bb_pct": 0.050,
                    "median_ip": 6.1,
                    "first_inning_era": 2.10,
                    "first_inning_whip": 0.85,
                    "sample_ip": 140.0,
                },
                "lineup_status": "CONFIRMED",
                "lineup": [
                    {"id": 21, "name": "Shohei Ohtani", "pos": "DH", "order": 1, "woba": 0.445, "bats": "L"},
                    {"id": 22, "name": "Mookie Betts", "pos": "SS", "order": 2, "woba": 0.390, "bats": "R"},
                    {"id": 23, "name": "Freddie Freeman", "pos": "1B", "order": 3, "woba": 0.400, "bats": "L"},
                    {"id": 24, "name": "Teoscar Hernandez", "pos": "LF", "order": 4, "woba": 0.355, "bats": "R"},
                ],
            },
            "home_team": {
                "id": 115,
                "name": "Colorado Rockies",
                "abbrev": "COL",
                "record": "58-94",
                "wrc_plus_vs_rhp": 88,
                "wrc_plus_vs_lhp": 85,
                "woba_vs_rhp": 0.295,
                "woba_vs_lhp": 0.288,
                "bullpen_era": 5.40,
                "bullpen_whip": 1.52,
                "bullpen_fip": 5.20,
                "bullpen_fatigue": 0.45,
                "starter": {
                    "id": 608344,
                    "name": "Kyle Freeland",
                    "hand": "L",
                    "era": 5.10,
                    "fip": 4.95,
                    "xfip": 4.80,
                    "whip": 1.48,
                    "k_pct": 0.165,
                    "bb_pct": 0.075,
                    "median_ip": 5.0,
                    "first_inning_era": 5.40,
                    "first_inning_whip": 1.55,
                    "sample_ip": 130.0,
                },
                "lineup_status": "CONFIRMED",
                "lineup": [
                    {"id": 31, "name": "Charlie Blackmon", "pos": "DH", "order": 1, "woba": 0.320, "bats": "L"},
                    {"id": 32, "name": "Ezequiel Tovar", "pos": "SS", "order": 2, "woba": 0.315, "bats": "R"},
                    {"id": 33, "name": "Ryan McMahon", "pos": "3B", "order": 3, "woba": 0.330, "bats": "L"},
                    {"id": 34, "name": "Brenton Doyle", "pos": "CF", "order": 4, "woba": 0.325, "bats": "R"},
                ],
            },
            "venue": {
                "id": 19,
                "venue_id": 19,
                "name": "Coors Field",
                "venue_name": "Coors Field",
                "altitude": 5200,
                "roof_type": "Open",
                "run_factor": 1.30,
                "hr_factor": 1.25,
            },
            "weather": {
                "temp": 82,
                "wind_speed": 12,
                "wind_dir": "Out to LF",
                "condition": "Sunny",
            },
        },
        {
            "game_pk": 700103,
            "game_date": "2026-09-20",
            "game_time": "19:10",
            "status": "Scheduled",
            "away_team": {
                "id": 141,
                "name": "Toronto Blue Jays",
                "abbrev": "TOR",
                "record": "74-78",
                "wrc_plus_vs_rhp": 102,
                "wrc_plus_vs_lhp": 98,
                "woba_vs_rhp": 0.312,
                "woba_vs_lhp": 0.305,
                "bullpen_era": 4.30,
                "bullpen_whip": 1.32,
                "bullpen_fip": 4.25,
                "bullpen_fatigue": 0.28,
                "starter": {
                    "id": 592332,
                    "name": "Kevin Gausman",
                    "hand": "R",
                    "era": 3.85,
                    "fip": 3.70,
                    "xfip": 3.65,
                    "whip": 1.22,
                    "k_pct": 0.260,
                    "bb_pct": 0.065,
                    "median_ip": 5.8,
                    "first_inning_era": 3.60,
                    "first_inning_whip": 1.20,
                    "sample_ip": 165.0,
                },
                "lineup_status": "PROJECTED",
                "lineup": [
                    {"id": 41, "name": "George Springer", "pos": "RF", "order": 1, "woba": 0.320, "bats": "R"},
                    {"id": 42, "name": "Vladimir Guerrero Jr.", "pos": "1B", "order": 2, "woba": 0.395, "bats": "R"},
                    {"id": 43, "name": "Bo Bichette", "pos": "SS", "order": 3, "woba": 0.335, "bats": "R"},
                    {"id": 44, "name": "Daulton Varsho", "pos": "CF", "order": 4, "woba": 0.310, "bats": "L"},
                ],
            },
            "home_team": {
                "id": 139,
                "name": "Tampa Bay Rays",
                "abbrev": "TB",
                "record": "76-76",
                "wrc_plus_vs_rhp": 97,
                "wrc_plus_vs_lhp": 104,
                "woba_vs_rhp": 0.308,
                "woba_vs_lhp": 0.318,
                "bullpen_era": 3.75,
                "bullpen_whip": 1.21,
                "bullpen_fip": 3.80,
                "bullpen_fatigue": 0.32,
                "starter": {
                    "id": 663556,
                    "name": "Shane Baz",
                    "hand": "R",
                    "era": 3.60,
                    "fip": 3.75,
                    "xfip": 3.80,
                    "whip": 1.18,
                    "k_pct": 0.255,
                    "bb_pct": 0.082,
                    "median_ip": 5.1,
                    "first_inning_era": 3.80,
                    "first_inning_whip": 1.25,
                    "sample_ip": 85.0,
                },
                "lineup_status": "PROJECTED",
                "lineup": [
                    {"id": 51, "name": "Yandy Diaz", "pos": "1B", "order": 1, "woba": 0.345, "bats": "R"},
                    {"id": 52, "name": "Brandon Lowe", "pos": "2B", "order": 2, "woba": 0.340, "bats": "L"},
                    {"id": 53, "name": "Christopher Morel", "pos": "DH", "order": 3, "woba": 0.325, "bats": "R"},
                    {"id": 54, "name": "Jose Siri", "pos": "CF", "order": 4, "woba": 0.295, "bats": "R"},
                ],
            },
            "venue": {
                "id": 12,
                "venue_id": 12,
                "name": "Tropicana Field",
                "venue_name": "Tropicana Field",
                "altitude": 44,
                "roof_type": "Dome",
                "run_factor": 0.93,
                "hr_factor": 0.89,
            },
            "weather": {
                "temp": 72,
                "wind_speed": 0,
                "wind_dir": "None",
                "condition": "Indoor Dome",
            },
        },
    ]


# =====================================================================
# 1. Complete Data-to-Prediction E2E Lifecycle
# =====================================================================

def test_e2e_complete_data_to_prediction_lifecycle(temp_db_path, multi_game_slate, monkeypatch):
    """Verify complete end-to-end flow from DB init, API ingestion, Monte Carlo simulation,

    market de-vigging, pick logging, settlement, delta buckets, and calibration analysis.
    """
    # 1. Initialize SQLite Database schemas and verify tables
    conn = init_db(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"model_constants", "games_cache", "picks_history", "teams", "pitchers", "batters", "park_factors"}.issubset(tables)

    # Verify seeded default model constants
    constants = get_model_constants(conn)
    assert constants["dispersion_alpha"] == 0.12
    assert constants["home_field_advantage_runs"] == 0.18
    assert constants["temp_coefficient"] == 0.012
    assert constants["wind_coefficient_out"] == 0.015

    # 2. Schedule Ingestion: Mock Live API fetch into SQLite cache
    fetcher = MLBDataFetcher(db_path=temp_db_path, timeout=5)

    mock_api_payload = {
        "dates": [
            {
                "date": "2026-09-20",
                "games": [
                    {
                        "gamePk": g["game_pk"],
                        "gameDate": f"{g['game_date']}T{g['game_time']}:00Z",
                        "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                        "teams": {
                            "away": {
                                "team": {"id": g["away_team"]["id"], "name": g["away_team"]["name"]},
                                "probablePitcher": {"id": g["away_team"]["starter"]["id"], "fullName": g["away_team"]["starter"]["name"]},
                            },
                            "home": {
                                "team": {"id": g["home_team"]["id"], "name": g["home_team"]["name"]},
                                "probablePitcher": {"id": g["home_team"]["starter"]["id"], "fullName": g["home_team"]["starter"]["name"]},
                            },
                        },
                        "venue": {"id": g["venue"]["id"], "name": g["venue"]["name"]},
                        "weather": {
                            "condition": g["weather"]["condition"],
                            "temp": str(g["weather"]["temp"]),
                            "wind": f"{g['weather']['wind_speed']} mph, {g['weather']['wind_dir']}",
                        },
                    }
                    for g in multi_game_slate
                ],
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_api_payload
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **k: mock_resp)

    # Ingest live schedule
    games, is_cached, status_msg = fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True, conn=conn)
    assert is_cached is False
    assert len(games) == 3
    assert "Live Feed" in status_msg or "Connected" in status_msg

    # Verify SQLite cache was populated
    cached_slate = get_cached_slate(conn, "2026-09-20")
    assert cached_slate is not None
    assert len(cached_slate) == 3
    assert cached_slate[0]["game_pk"] == 700101

    # Ingest again with force_refresh=False to verify offline SQLite cached retrieval without network
    def mock_broken_get(*a, **k):
        raise ConnectionError("Network down")

    monkeypatch.setattr(fetcher.session, "get", mock_broken_get)
    cached_games, is_cached_offline, offline_msg = fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=False, conn=conn)
    assert is_cached_offline is True
    assert len(cached_games) == 3
    assert "Cached Slate" in offline_msg

    # 3. Vectorized Monte Carlo Simulation (10,000 iterations per game)
    nyy_bos_game = multi_game_slate[0]
    sim_result = simulate_game(
        home_team=nyy_bos_game["home_team"],
        away_team=nyy_bos_game["away_team"],
        venue=nyy_bos_game["venue"],
        weather=nyy_bos_game["weather"],
        constants=constants,
        n_sim=10000,
    )

    # Verify all 8 core prediction markets
    # Market 1: Moneylines
    p_h = sim_result["home_win_prob"]
    p_a = sim_result["away_win_prob"]
    assert pytest.approx(p_h + p_a, abs=1e-4) == 1.0
    assert 0.0 < p_h < 1.0
    assert 0.0 < p_a < 1.0
    assert sim_result["home_fair_odds"].startswith(("+", "-"))
    assert sim_result["away_fair_odds"].startswith(("+", "-"))

    # Expected scores
    assert sim_result["expected_home_score"] > 0
    assert sim_result["expected_away_score"] > 0
    assert sim_result["projected_total"] > 0

    # Market 2: Run Lines (+/- 1.5)
    rl_h_plus = sim_result["home_cover_plus_1_5"]
    rl_a_plus = sim_result["away_cover_plus_1_5"]
    rl_h_minus = sim_result["home_cover_minus_1_5"]
    rl_a_minus = sim_result["away_cover_minus_1_5"]
    assert pytest.approx(rl_h_plus + rl_a_minus, abs=1e-3) == 1.0
    assert pytest.approx(rl_a_plus + rl_h_minus, abs=1e-3) == 1.0
    assert rl_h_plus > p_h  # Covering +1.5 must have higher probability than outright win

    # Market 3: Spread Deltas
    delta_h = sim_result["spread_delta_home"]
    delta_a = sim_result["spread_delta_away"]
    assert pytest.approx(delta_h, abs=1e-4) == round(rl_h_plus - p_h, 4)
    assert pytest.approx(delta_a, abs=1e-4) == round(rl_a_plus - p_a, 4)
    assert delta_h > 0
    assert delta_a > 0

    # Market 4: Totals distribution (6.5 to 10.5)
    totals_dist = sim_result["totals_distribution"]
    expected_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]
    for line in expected_lines:
        assert line in totals_dist
        p_over = totals_dist[line]["over"]
        p_under = totals_dist[line]["under"]
        assert 0.0 <= p_over <= 1.0
        assert 0.0 <= p_under <= 1.0

    # Over probability should decrease as total line increases
    for i in range(len(expected_lines) - 1):
        l1, l2 = expected_lines[i], expected_lines[i + 1]
        assert totals_dist[l1]["over"] >= totals_dist[l2]["over"] - 1e-4

    # Market 5: 1st Inning NRFI / YRFI
    p_nrfi = sim_result["p_nrfi"]
    p_yrfi = sim_result["p_yrfi"]
    assert pytest.approx(p_nrfi + p_yrfi, abs=1e-4) == 1.0
    assert 0.20 <= p_nrfi <= 0.85
    assert sim_result["nrfi_fair_odds"].startswith(("+", "-"))
    assert sim_result["yrfi_fair_odds"].startswith(("+", "-"))

    # Market 6: Quadrature Uncertainty Intervals
    u_h = sim_result["home_uncertainty_95"]
    u_a = sim_result["away_uncertainty_95"]
    assert 0.01 <= u_h <= 0.15
    assert 0.01 <= u_a <= 0.15

    # Formula check: U_95 = 1.96 * sqrt(SE_MC^2 + sigma_baseline^2 + sigma_sample^2 + sigma_lineup^2)
    expected_u = calculate_uncertainty(p_h, n_sim=10000, sample_ip=150.0, lineup_status="CONFIRMED")
    assert pytest.approx(u_h, abs=0.01) == expected_u

    # Market 7: Data Confidence Tag
    assert sim_result["data_confidence"] in ("HIGH", "MEDIUM", "LOW")
    # Both starters have >= 100 IP sample and lineups are confirmed -> should be HIGH
    assert sim_result["data_confidence"] == "HIGH"

    # Market 8: Player Prop Modeling
    batter_props = model_player_props(
        batter_list=nyy_bos_game["away_team"]["lineup"],
        pitcher=nyy_bos_game["home_team"]["starter"],
        venue=nyy_bos_game["venue"],
        weather=nyy_bos_game["weather"],
    )
    assert len(batter_props) >= 4
    first_b = batter_props[0]
    assert first_b["role"] == "batter"
    assert "hits_dist" in first_b and "1+" in first_b["hits_dist"]
    assert "tb_dist" in first_b and "1.5" in first_b["tb_dist"]
    assert "hr_prob" in first_b
    assert "hrr_over_1_5" in first_b

    # Pitcher props
    sp_props = model_player_props(
        batter_list=[],
        pitcher=nyy_bos_game["away_team"]["starter"],
        venue=nyy_bos_game["venue"],
        weather=nyy_bos_game["weather"],
    )
    assert len(sp_props) >= 1
    cole_prop = sp_props[0]
    assert cole_prop["role"] == "pitcher"
    assert "strikeouts_dist" in cole_prop and "5.5" in cole_prop["strikeouts_dist"]
    assert "outs_dist" in cole_prop and "17.5" in cole_prop["outs_dist"]

    # 4. Odds-Independent Pick Generation vs Market Odds Layering
    # Case A: Decoupled / No market odds provided
    decoupled_pick = calculate_edge(p_h, market_line=None)
    assert decoupled_pick["fair_odds"] == sim_result["home_fair_odds"]
    assert decoupled_pick["market_prob_novig"] is None
    assert decoupled_pick["edge_pct"] is None
    assert decoupled_pick["ev_pct"] is None
    assert decoupled_pick["recommendation"] == "Model Pick"

    # Case B: Market odds provided with two-way devigging
    # Assume BOS is priced at +125, NYY at -145 in sportsbook
    market_edge_h = calculate_edge(p_h, market_line="+125", market_opp_line="-145")
    assert market_edge_h["fair_odds"] == sim_result["home_fair_odds"]
    assert market_edge_h["market_prob_novig"] is not None
    assert isinstance(market_edge_h["edge_pct"], float)
    assert isinstance(market_edge_h["ev_pct"], float)
    assert market_edge_h["recommendation"] in ("Strong Edge", "Slight Edge", "Pass", "Fade")

    # 5. Logging Picks into SQLite & Pick Settlement
    logged_picks = [
        # Bucket 0-5% (Delta = 0.03) -> Settled WIN (+0.91u)
        {
            "game_pk": 700101,
            "game_date": "2026-09-20",
            "market": "RL +1.5",
            "selection": "BOS +1.5",
            "model_prob": 0.65,
            "uncertainty_interval": 0.035,
            "fair_odds": "-186",
            "market_odds": "-110",
            "edge_pct": 12.6,
            "ev_pct": 24.1,
            "confidence": "HIGH",
            "spread_delta": 0.03,
            "result": "WIN",
            "closing_odds": "-110",
            "units_won": 0.91,
        },
        # Bucket 5-10% (Delta = 0.08) -> Settled LOSS (-1.00u)
        {
            "game_pk": 700101,
            "game_date": "2026-09-20",
            "market": "ML",
            "selection": "NYY ML",
            "model_prob": 0.58,
            "uncertainty_interval": 0.038,
            "fair_odds": "-138",
            "market_odds": "-125",
            "edge_pct": 3.4,
            "ev_pct": 4.4,
            "confidence": "HIGH",
            "spread_delta": 0.08,
            "result": "LOSS",
            "closing_odds": "-130",
            "units_won": -1.00,
        },
        # Bucket 10-15% (Delta = 0.12) -> Settled WIN (+1.20u)
        {
            "game_pk": 700102,
            "game_date": "2026-09-20",
            "market": "RL +1.5",
            "selection": "COL +1.5",
            "model_prob": 0.56,
            "uncertainty_interval": 0.040,
            "fair_odds": "-127",
            "market_odds": "+120",
            "edge_pct": 10.5,
            "ev_pct": 23.2,
            "confidence": "HIGH",
            "spread_delta": 0.12,
            "result": "WIN",
            "closing_odds": "+115",
            "units_won": 1.20,
        },
        # Bucket 15-20% (Delta = 0.17) -> Settled WIN (+0.80u)
        {
            "game_pk": 700102,
            "game_date": "2026-09-20",
            "market": "NRFI",
            "selection": "LAD @ COL NRFI",
            "model_prob": 0.52,
            "uncertainty_interval": 0.042,
            "fair_odds": "-108",
            "market_odds": "+100",
            "edge_pct": 2.0,
            "ev_pct": 4.0,
            "confidence": "MEDIUM",
            "spread_delta": 0.17,
            "result": "WIN",
            "closing_odds": "-105",
            "units_won": 0.80,
        },
        # Bucket 20%+ (Delta = 0.22) -> Settled WIN (+1.50u)
        {
            "game_pk": 700103,
            "game_date": "2026-09-20",
            "market": "RL +1.5",
            "selection": "TOR +1.5",
            "model_prob": 0.72,
            "uncertainty_interval": 0.045,
            "fair_odds": "-257",
            "market_odds": "-150",
            "edge_pct": 12.0,
            "ev_pct": 20.0,
            "confidence": "MEDIUM",
            "spread_delta": 0.22,
            "result": "WIN",
            "closing_odds": "-160",
            "units_won": 1.50,
        },
        # Pending Pick
        {
            "game_pk": 700103,
            "game_date": "2026-09-20",
            "market": "ML",
            "selection": "TB ML",
            "model_prob": 0.54,
            "uncertainty_interval": 0.045,
            "fair_odds": "-117",
            "market_odds": "-105",
            "edge_pct": 2.8,
            "ev_pct": 5.3,
            "confidence": "MEDIUM",
            "spread_delta": 0.06,
            "result": "PENDING",
            "closing_odds": "",
            "units_won": 0.0,
        },
    ]

    for p in logged_picks:
        log_pick(conn, p)

    history = get_picks_history(conn)
    assert len(history) == 6
    assert history[0]["selection"] == "TB ML"
    assert history[0]["result"] == "PENDING"

    # 6. Historical Delta Performance Buckets
    delta_buckets = get_delta_buckets_history(conn)
    assert len(delta_buckets) == 5
    bucket_map = {b["bucket"]: b for b in delta_buckets}

    # Verify bucket classifications
    assert bucket_map["0-5%"]["total_bets"] == 1
    assert bucket_map["0-5%"]["wins"] == 1
    assert bucket_map["0-5%"]["cover_pct"] == 100.0
    assert pytest.approx(bucket_map["0-5%"]["units_won"]) == 0.91

    assert bucket_map["5-10%"]["total_bets"] == 1
    assert bucket_map["5-10%"]["wins"] == 0
    assert bucket_map["5-10%"]["cover_pct"] == 0.0
    assert pytest.approx(bucket_map["5-10%"]["units_won"]) == -1.00

    assert bucket_map["10-15%"]["total_bets"] == 1
    assert bucket_map["10-15%"]["wins"] == 1

    assert bucket_map["15-20%"]["total_bets"] == 1
    assert bucket_map["15-20%"]["wins"] == 1

    assert bucket_map["20%+"]["total_bets"] == 1
    assert bucket_map["20%+"]["wins"] == 1
    assert pytest.approx(bucket_map["20%+"]["units_won"]) == 1.50

    # 7. Model Calibration Analysis (Brier score, Log Loss, 7 buckets)
    cal_results = compute_calibration_metrics(history)
    assert cal_results["total_predictions"] == 5  # 5 settled bets (pending omitted)
    assert 0.0 <= cal_results["brier_score"] <= 1.0
    assert cal_results["log_loss"] >= 0.0
    assert cal_results["status"] in ("Well Calibrated", "Overconfident", "Underconfident")

    cal_buckets = cal_results["calibration_buckets"]
    assert len(cal_buckets) == 7
    expected_cal_names = ["50-54%", "55-59%", "60-64%", "65-69%", "70-74%", "75-79%", "80%+"]
    assert [b["bucket"] for b in cal_buckets] == expected_cal_names

    for cb in cal_buckets:
        assert cb["status"] in ("Well Calibrated", "Overconfident", "Underconfident")

    conn.close()


# =====================================================================
# 2. Dynamic Model Parameter Recalibration & Persistence
# =====================================================================

def test_e2e_model_constants_dynamic_recalibration(temp_db_path, multi_game_slate):
    """Verify modifying model constants in SQLite affects subsequent Monte Carlo simulations."""
    conn = init_db(temp_db_path)
    initial_constants = get_model_constants(conn)

    game = multi_game_slate[0]

    # Baseline simulation
    base_sim = simulate_game(
        home_team=game["home_team"],
        away_team=game["away_team"],
        venue=game["venue"],
        weather=game["weather"],
        constants=initial_constants,
        n_sim=5000,
    )
    base_hfa = base_sim["expected_home_score"] - base_sim["expected_away_score"]

    # Massive boost to home field advantage runs (+1.5 runs)
    modified_constants = dict(initial_constants)
    modified_constants["home_field_advantage_runs"] = 1.50
    update_model_constants(conn, modified_constants)

    # Confirm persistence in DB
    reloaded_constants = get_model_constants(conn)
    assert pytest.approx(reloaded_constants["home_field_advantage_runs"], abs=1e-4) == 1.50

    # Re-run simulation with reloaded constants
    boosted_sim = simulate_game(
        home_team=game["home_team"],
        away_team=game["away_team"],
        venue=game["venue"],
        weather=game["weather"],
        constants=reloaded_constants,
        n_sim=5000,
    )
    boosted_hfa = boosted_sim["expected_home_score"] - boosted_sim["expected_away_score"]

    # Home score margin must be significantly higher with increased home field advantage runs
    assert boosted_hfa > base_hfa
    assert boosted_sim["home_win_prob"] > base_sim["home_win_prob"]

    conn.close()


# =====================================================================
# 3. Headless Tkinter GUI End-to-End Workflow across All 8 Tabs
# =====================================================================

def test_e2e_headless_gui_full_workflow(temp_db_path, multi_game_slate):
    """Verify headless startup of MLBPredictorApp, dark theme styling, slate rendering across

    all 8 tabs, inline cell odds editing, instant recalculation, dynamic filtering,
    pick logging from UI to SQLite, backtesting, and Matplotlib canvas rendering.
    """
    # 1. Initialize SQLite Database and seed sample settled picks for calibration & delta tabs
    conn = init_db(temp_db_path)
    initial_picks = [
        {"game_pk": 700101, "game_date": "2026-09-19", "market": "ML", "selection": "NYY ML", "model_prob": 0.60, "fair_odds": "-150", "market_odds": "-130", "edge_pct": 3.5, "ev_pct": 4.5, "confidence": "HIGH", "spread_delta": 0.08, "result": "WIN", "units_won": 0.77},
        {"game_pk": 700102, "game_date": "2026-09-19", "market": "RL +1.5", "selection": "COL +1.5", "model_prob": 0.58, "fair_odds": "-138", "market_odds": "+110", "edge_pct": 10.4, "ev_pct": 21.8, "confidence": "HIGH", "spread_delta": 0.14, "result": "WIN", "units_won": 1.10},
        {"game_pk": 700103, "game_date": "2026-09-19", "market": "NRFI", "selection": "TOR @ TB NRFI", "model_prob": 0.54, "fair_odds": "-117", "market_odds": "-105", "edge_pct": 2.8, "ev_pct": 5.3, "confidence": "LOW", "spread_delta": 0.02, "result": "LOSS", "units_won": -1.00},
    ]
    for p in initial_picks:
        log_pick(conn, p)
    conn.close()

    # 2. Launch MLBPredictorApp in headless mode (auto_load=False)
    app = MLBPredictorApp(db_path=temp_db_path, auto_load=False)
    app.withdraw()  # Headless mode
    app.update_idletasks()

    try:
        # Check window properties
        assert "MLB Apex Predictor" in app.title()
        assert app.notebook is not None

        # Verify exact 8 notebook tabs
        tab_names = [app.notebook.tab(i, "text") for i in range(app.notebook.index("end"))]
        assert len(tab_names) == 8
        expected_tabs = [
            "Today's Slate",
            "Moneylines",
            "Run Lines & Historical Delta Performance",
            "Player Props",
            "NRFI / YRFI",
            "Model Picks",
            "Backtesting & Settings",
            "Performance & Calibration",
        ]
        for exp in expected_tabs:
            assert exp in tab_names, f"Tab {exp} missing from {tab_names}"

        # 3. Ingest multi-game slate into GUI
        app.load_games(multi_game_slate, date_str="2026-09-20", status_msg="E2E Slate Loaded")
        app.update_idletasks()

        # Header summary badges
        assert app.badge_games_var.get() == "3"
        assert "Confirmed" in app.badge_lineups_var.get()
        assert int(app.badge_picks_var.get()) >= 6

        # Tab 1: Today's Slate cards rendered
        slate_cards = app.frame_cards.winfo_children()
        assert len(slate_cards) >= 3

        # Tab 2: Moneylines Treeview (3 games x 2 rows = 6 rows)
        ml_rows = app.tree_moneylines.get_children()
        assert len(ml_rows) == 6
        first_ml = app.tree_moneylines.item(ml_rows[0])
        assert len(first_ml["values"]) >= 8

        # Tab 3: Run Lines Treeview & Delta Buckets
        rl_rows = app.tree_runlines.get_children()
        assert len(rl_rows) == 6
        delta_rows = app.tree_deltas.get_children()
        assert len(delta_rows) == 5

        # Tab 4: Player Props Treeview
        prop_rows = app.tree_props.get_children()
        total_props_count = len(prop_rows)
        assert total_props_count > 0

        # Tab 5: NRFI / YRFI Treeview (3 games = 3 rows)
        nrfi_rows = app.tree_nrfi.get_children()
        assert len(nrfi_rows) == 3

        # Tab 6: Model Picks Treeview
        pick_rows = app.tree_picks.get_children()
        assert len(pick_rows) >= 1
        assert len(app.active_picks) >= 6

        # Tab 8: Performance & Calibration Tab
        cal_rows = app.tree_calibration.get_children()
        assert len(cal_rows) == 7
        assert app.fig_canvas is not None
        assert len(app.fig.axes) >= 2

        # 4. Inline Cell Odds Editing on Tab 2 Moneylines
        target_ml_item = ml_rows[0]
        col_names = app.tree_moneylines["columns"]
        odds_idx = col_names.index("book_odds")
        edge_idx = col_names.index("edge_pct")
        rec_idx = col_names.index("recommendation")

        # Edit to high underdog line (+200) -> strong edge
        app.update_cell_odds(tree_name="moneylines", item_id=target_ml_item, new_odds="+200")
        app.update_idletasks()

        updated_vals_dog = app.tree_moneylines.item(target_ml_item)["values"]
        assert str(updated_vals_dog[odds_idx]) in ("+200", "200")
        edge_dog = float(str(updated_vals_dog[edge_idx]).replace("%", "").replace("+", ""))
        assert edge_dog > 0.0
        assert updated_vals_dog[rec_idx] in ("Strong Edge", "Slight Edge")

        # Edit to heavy favorite line (-350) -> negative edge / fade
        app.update_cell_odds(tree_name="moneylines", item_id=target_ml_item, new_odds="-350")
        app.update_idletasks()

        updated_vals_fav = app.tree_moneylines.item(target_ml_item)["values"]
        assert str(updated_vals_fav[odds_idx]) == "-350"
        edge_fav = float(str(updated_vals_fav[edge_idx]).replace("%", "").replace("+", ""))
        assert edge_fav < edge_dog
        assert updated_vals_fav[rec_idx] in ("Pass", "Fade")

        # 5. Dynamic Filtering on Tab 4 Player Props
        # Filter by "Hits"
        app.prop_category_var.set("Hits")
        app.apply_prop_filters()
        app.update_idletasks()
        hits_props = app.tree_props.get_children()
        assert 0 < len(hits_props) < total_props_count
        for item_id in hits_props:
            row_vals = app.tree_props.item(item_id)["values"]
            assert "Hits" in str(row_vals)

        # Filter by player search
        app.prop_category_var.set("All")
        app.prop_search_var.set("Ohtani")
        app.apply_prop_filters()
        app.update_idletasks()
        ohtani_props = app.tree_props.get_children()
        assert len(ohtani_props) > 0
        for item_id in ohtani_props:
            assert "Ohtani" in str(app.tree_props.item(item_id)["values"])

        # Reset prop filters
        app.prop_search_var.set("")
        app.apply_prop_filters()
        app.update_idletasks()
        assert len(app.tree_props.get_children()) == total_props_count

        # 6. Dynamic Sliders and Filter on Tab 6 Model Picks
        # Restrict min edge to 80% -> 0 picks
        app.slider_min_edge.set(80.0)
        app.apply_picks_filters()
        app.update_idletasks()
        assert len(app.tree_picks.get_children()) == 0

        # Relax min edge to -50% -> all picks visible
        app.slider_min_edge.set(-50.0)
        app.slider_min_prob.set(0.0)
        app.picks_confidence_var.set("All")
        app.apply_picks_filters()
        app.update_idletasks()
        assert len(app.tree_picks.get_children()) == len(app.active_picks)

        # 7. Action: Log Pick from GUI to SQLite
        initial_pick_count = len(get_picks_history(app.conn))
        all_rendered_picks = app.tree_picks.get_children()
        app.tree_picks.selection_set(all_rendered_picks[0])

        app.log_selected_pick()
        app.update_idletasks()

        new_history = get_picks_history(app.conn)
        assert len(new_history) == initial_pick_count + 1

        # 8. Tab 7: Settings & Backtesting Actions
        # Change dispersion parameter and save
        app.settings_vars["dispersion_alpha"].set(0.22)
        app.save_settings()
        app.update_idletasks()
        assert pytest.approx(get_model_constants(app.conn)["dispersion_alpha"], abs=1e-4) == 0.22

        # Run Backtest on existing settled picks
        app.run_backtest()
        app.update_idletasks()
        assert int(app.bt_bets_var.get()) >= 3
        assert app.bt_roi_var.get().endswith("%")

        # Reset defaults
        app.reset_settings_defaults()
        app.update_idletasks()
        assert pytest.approx(get_model_constants(app.conn)["dispersion_alpha"], abs=1e-4) == 0.12

        # 9. Date Navigation
        curr_d = app.current_date
        app.on_next_date()
        assert app.current_date > curr_d
        app.on_prev_date()
        assert app.current_date == curr_d

    finally:
        app.destroy()


# =====================================================================
# 4. CLI Entry Point & Packaging Verification
# =====================================================================

def test_e2e_cli_entry_point(temp_db_path, monkeypatch):
    """Verify CLI entry point in main.py correctly parses flags and initializes database."""
    import main as main_mod

    test_args = ["main.py", "--db", temp_db_path, "--date", "2026-09-20", "--no-auto-load"]
    monkeypatch.setattr("sys.argv", test_args)

    created_apps = []

    def mock_mainloop(self):
        created_apps.append(self)
        self.withdraw()
        self.update_idletasks()

    monkeypatch.setattr(MLBPredictorApp, "mainloop", mock_mainloop)

    main_mod.main()

    assert len(created_apps) == 1
    app = created_apps[0]
    try:
        assert app.current_date == "2026-09-20"
        assert app.db_path == temp_db_path
        # Confirm database was initialized
        conn = get_db_connection(temp_db_path)
        constants = get_model_constants(conn)
        assert len(constants) >= 8
        conn.close()
    finally:
        app.destroy()


import sqlite3
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
    save_team,
    get_team,
    save_pitcher,
    get_pitcher,
    save_park_factor,
    get_park_factor,
)

def test_init_db_creates_tables_and_indexes(temp_db_path):
    """Verify init_db initializes all required schema tables and indexes."""
    conn = init_db(temp_db_path)
    cursor = conn.cursor()

    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    required_tables = {
        "model_constants",
        "games_cache",
        "teams",
        "pitchers",
        "batters",
        "park_factors",
        "picks_history",
    }
    assert required_tables.issubset(tables), f"Missing tables: {required_tables - tables}"

    # Verify indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_games_cache_date" in indexes
    assert "idx_picks_history_date" in indexes
    assert "idx_picks_history_game_pk" in indexes
    conn.close()

def test_default_model_constants_seeded(temp_db_path):
    """Verify default model constants are automatically seeded on init_db."""
    conn = init_db(temp_db_path)
    constants = get_model_constants(conn)

    expected_defaults = {
        "dispersion_alpha": 0.12,
        "home_field_advantage_runs": 0.18,
        "temp_coefficient": 0.012,
        "wind_coefficient_out": 0.015,
        "wind_coefficient_in": 0.012,
        "cfip_constant": 3.15,
        "hr_fb_baseline": 0.115,
        "baseline_league_runs": 4.40,
    }

    for key, expected_val in expected_defaults.items():
        assert key in constants, f"Missing constant: {key}"
        assert pytest.approx(constants[key], rel=1e-5) == expected_val

    conn.close()

def test_update_and_reseed_model_constants(temp_db_path):
    """Verify model constants can be updated and re-init doesn't overwrite customized values."""
    conn = init_db(temp_db_path)

    # Update constants
    update_model_constants(conn, {
        "home_field_advantage_runs": 0.22,
        "dispersion_alpha": 0.15,
        "custom_param": 1.23,
    })

    updated = get_model_constants(conn)
    assert pytest.approx(updated["home_field_advantage_runs"]) == 0.22
    assert pytest.approx(updated["dispersion_alpha"]) == 0.15
    assert pytest.approx(updated["custom_param"]) == 1.23
    conn.close()

    # Re-run init_db on the same database path - customized values must persist (INSERT OR IGNORE)
    conn2 = init_db(temp_db_path)
    persisted = get_model_constants(conn2)
    assert pytest.approx(persisted["home_field_advantage_runs"]) == 0.22
    assert pytest.approx(persisted["dispersion_alpha"]) == 0.15
    assert pytest.approx(persisted["custom_param"]) == 1.23
    conn2.close()

def test_save_and_get_cached_slate(temp_db_path, sample_game_data):
    """Verify saving and retrieving cached game slates as JSON."""
    conn = init_db(temp_db_path)

    # Cache miss
    slate = get_cached_slate(conn, "2026-09-20")
    assert slate is None

    # Save slate
    save_slate_cache(conn, "2026-09-20", [sample_game_data])

    # Cache hit
    cached_slate = get_cached_slate(conn, "2026-09-20")
    assert cached_slate is not None
    assert len(cached_slate) == 1
    assert cached_slate[0]["game_pk"] == 748123
    assert cached_slate[0]["away_team"]["name"] == "New York Yankees"
    assert cached_slate[0]["home_team"]["name"] == "Boston Red Sox"

    # Overwrite slate cache for same date
    modified_game = dict(sample_game_data)
    modified_game["status"] = "Final"
    save_slate_cache(conn, "2026-09-20", [modified_game])

    cached_updated = get_cached_slate(conn, "2026-09-20")
    assert len(cached_updated) == 1
    assert cached_updated[0]["status"] == "Final"
    conn.close()

def test_log_pick_and_get_history(temp_db_path):
    """Verify logging picks and retrieving pick history."""
    conn = init_db(temp_db_path)

    pick_data_1 = {
        "game_pk": 748123,
        "game_date": "2026-09-20",
        "market": "Moneyline",
        "selection": "New York Yankees",
        "model_prob": 0.582,
        "uncertainty_interval": 0.041,
        "fair_odds": "-139",
        "market_odds": "-115",
        "edge_pct": 4.7,
        "ev_pct": 8.9,
        "confidence": "HIGH",
        "spread_delta": 0.12,
        "result": "PENDING",
        "closing_odds": "",
        "units_won": 0.0,
    }

    pick_id_1 = log_pick(conn, pick_data_1)
    assert pick_id_1 > 0

    pick_data_2 = {
        "game_pk": 748124,
        "game_date": "2026-09-20",
        "market": "Run Line +1.5",
        "selection": "Boston Red Sox +1.5",
        "model_prob": 0.640,
        "uncertainty_interval": 0.038,
        "fair_odds": "-178",
        "market_odds": "-140",
        "edge_pct": 5.6,
        "ev_pct": 9.7,
        "confidence": "HIGH",
        "spread_delta": 0.16,
    }

    pick_id_2 = log_pick(conn, pick_data_2)
    assert pick_id_2 > pick_id_1

    history = get_picks_history(conn, limit=10)
    assert len(history) == 2
    # Should be ordered descending by pick_id
    assert history[0]["pick_id"] == pick_id_2
    assert history[0]["market"] == "Run Line +1.5"
    assert history[0]["result"] == "PENDING"
    assert history[0]["units_won"] == 0.0

    assert history[1]["pick_id"] == pick_id_1
    assert history[1]["market"] == "Moneyline"
    assert history[1]["selection"] == "New York Yankees"

    # Test limit parameter
    limited = get_picks_history(conn, limit=1)
    assert len(limited) == 1
    assert limited[0]["pick_id"] == pick_id_2
    conn.close()

def test_update_pick_result(temp_db_path):
    """Verify updating a logged pick's result, units won, and closing odds."""
    conn = init_db(temp_db_path)

    pick_id = log_pick(conn, {
        "game_pk": 748123,
        "game_date": "2026-09-20",
        "market": "Moneyline",
        "selection": "New York Yankees",
        "model_prob": 0.58,
        "uncertainty_interval": 0.04,
        "fair_odds": "-138",
        "market_odds": "-115",
        "edge_pct": 4.5,
        "ev_pct": 8.5,
        "confidence": "HIGH",
        "spread_delta": 0.10,
    })

    # Update pick to settled WIN
    update_pick_result(conn, pick_id=pick_id, result="WIN", units_won=0.87, closing_odds="-125")

    history = get_picks_history(conn)
    assert len(history) == 1
    assert history[0]["result"] == "WIN"
    assert pytest.approx(history[0]["units_won"]) == 0.87
    assert history[0]["closing_odds"] == "-125"

    # Update again without closing_odds to test optional argument preservation
    update_pick_result(conn, pick_id=pick_id, result="WIN", units_won=1.0)
    history2 = get_picks_history(conn)
    assert history2[0]["result"] == "WIN"
    assert pytest.approx(history2[0]["units_won"]) == 1.0
    assert history2[0]["closing_odds"] == "-125"
    conn.close()

def test_get_delta_buckets_history(temp_db_path):
    """Verify grouping settled picks into spread delta buckets and calculating metrics."""
    conn = init_db(temp_db_path)

    # Empty history returns 5 initialized buckets
    empty_buckets = get_delta_buckets_history(conn)
    assert len(empty_buckets) == 5
    bucket_names = [b["bucket"] for b in empty_buckets]
    assert bucket_names == ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    for b in empty_buckets:
        assert b["total_bets"] == 0
        assert b["wins"] == 0
        assert b["cover_pct"] == 0.0
        assert b["units_won"] == 0.0
        assert b["roi_pct"] == 0.0

    # Insert test picks across different buckets
    picks = [
        # 0-5% bucket: 2 bets, 1 win, 1 loss -> 50% cover, units: +0.90, -1.0 = -0.10, roi: -5.0%
        {"game_pk": 1, "market": "+1.5", "selection": "A", "spread_delta": 0.03, "result": "WIN", "units_won": 0.90},
        {"game_pk": 2, "market": "+1.5", "selection": "B", "spread_delta": 0.04, "result": "LOSS", "units_won": -1.00},

        # 5-10% bucket: 1 bet, 1 win -> 100% cover, units: +1.10, roi: +110.0%
        {"game_pk": 3, "market": "+1.5", "selection": "C", "spread_delta": 0.08, "result": "WIN", "units_won": 1.10},

        # 10-15% bucket: 1 bet, 1 win -> 100% cover, units: +0.80, roi: +80.0%
        {"game_pk": 4, "market": "+1.5", "selection": "D", "spread_delta": 0.12, "result": "WIN", "units_won": 0.80},

        # 15-20% bucket: 1 bet, 1 loss -> 0% cover, units: -1.00, roi: -100.0%
        {"game_pk": 5, "market": "+1.5", "selection": "E", "spread_delta": 0.18, "result": "LOSS", "units_won": -1.00},

        # 20%+ bucket: 2 bets, 2 wins -> 100% cover, units: +2.50, roi: +125.0%
        {"game_pk": 6, "market": "+1.5", "selection": "F", "spread_delta": 0.22, "result": "WIN", "units_won": 1.50},
        {"game_pk": 7, "market": "+1.5", "selection": "G", "spread_delta": 0.25, "result": "WIN", "units_won": 1.00},

        # Unsettled / PENDING pick: should be ignored in bucket history
        {"game_pk": 8, "market": "+1.5", "selection": "H", "spread_delta": 0.09, "result": "PENDING", "units_won": 0.0},
    ]

    for p in picks:
        log_pick(conn, p)

    buckets = get_delta_buckets_history(conn)
    assert len(buckets) == 5
    b_map = {b["bucket"]: b for b in buckets}

    # 0-5%
    assert b_map["0-5%"]["total_bets"] == 2
    assert b_map["0-5%"]["wins"] == 1
    assert pytest.approx(b_map["0-5%"]["cover_pct"]) == 50.0
    assert pytest.approx(b_map["0-5%"]["units_won"]) == -0.10
    assert pytest.approx(b_map["0-5%"]["roi_pct"]) == -5.0

    # 5-10% (PENDING pick must not be counted)
    assert b_map["5-10%"]["total_bets"] == 1
    assert b_map["5-10%"]["wins"] == 1
    assert pytest.approx(b_map["5-10%"]["cover_pct"]) == 100.0
    assert pytest.approx(b_map["5-10%"]["units_won"]) == 1.10
    assert pytest.approx(b_map["5-10%"]["roi_pct"]) == 110.0

    # 10-15%
    assert b_map["10-15%"]["total_bets"] == 1
    assert b_map["10-15%"]["wins"] == 1
    assert pytest.approx(b_map["10-15%"]["cover_pct"]) == 100.0
    assert pytest.approx(b_map["10-15%"]["units_won"]) == 0.80
    assert pytest.approx(b_map["10-15%"]["roi_pct"]) == 80.0

    # 15-20%
    assert b_map["15-20%"]["total_bets"] == 1
    assert b_map["15-20%"]["wins"] == 0
    assert pytest.approx(b_map["15-20%"]["cover_pct"]) == 0.0
    assert pytest.approx(b_map["15-20%"]["units_won"]) == -1.00
    assert pytest.approx(b_map["15-20%"]["roi_pct"]) == -100.0

    # 20%+
    assert b_map["20%+"]["total_bets"] == 2
    assert b_map["20%+"]["wins"] == 2
    assert pytest.approx(b_map["20%+"]["cover_pct"]) == 100.0
    assert pytest.approx(b_map["20%+"]["units_won"]) == 2.50
    assert pytest.approx(b_map["20%+"]["roi_pct"]) == 125.0

    conn.close()

def test_teams_pitchers_and_park_factors_helpers(temp_db_path):
    """Verify storing and retrieving team, pitcher, and park factor entities."""
    conn = init_db(temp_db_path)

    # Team
    team_data = {
        "team_id": 147,
        "name": "New York Yankees",
        "abbrev": "NYY",
        "wrc_plus_vs_rhp": 118.0,
        "wrc_plus_vs_lhp": 112.0,
        "woba_vs_rhp": 0.334,
        "woba_vs_lhp": 0.325,
        "bullpen_era": 3.42,
        "bullpen_whip": 1.18,
        "bullpen_fip": 3.55,
        "bullpen_fatigue": 0.25,
    }
    save_team(conn, team_data)
    retrieved_team = get_team(conn, 147)
    assert retrieved_team is not None
    assert retrieved_team["name"] == "New York Yankees"
    assert pytest.approx(retrieved_team["wrc_plus_vs_rhp"]) == 118.0

    # Pitcher
    pitcher_data = {
        "pitcher_id": 543037,
        "name": "Gerrit Cole",
        "team_id": 147,
        "hand": "R",
        "era": 3.12,
        "fip": 3.25,
        "xfip": 3.30,
        "whip": 1.05,
        "k_pct": 0.285,
        "bb_pct": 0.062,
        "median_ip": 6.1,
        "first_inning_era": 2.80,
        "first_inning_whip": 0.95,
        "sample_ip": 160.0,
    }
    save_pitcher(conn, pitcher_data)
    retrieved_pitcher = get_pitcher(conn, 543037)
    assert retrieved_pitcher is not None
    assert retrieved_pitcher["name"] == "Gerrit Cole"
    assert pytest.approx(retrieved_pitcher["era"]) == 3.12

    # Park factor
    park_data = {
        "venue_id": 3,
        "venue_name": "Fenway Park",
        "altitude": 20.0,
        "roof_type": "Open",
        "run_factor": 1.08,
        "hr_factor": 1.05,
        "last_updated_season": 2026,
    }
    save_park_factor(conn, park_data)
    retrieved_park = get_park_factor(conn, 3)
    assert retrieved_park is not None
    assert retrieved_park["venue_name"] == "Fenway Park"
    assert pytest.approx(retrieved_park["run_factor"]) == 1.08

    conn.close()

def test_get_db_connection(temp_db_path):
    """Verify get_db_connection connects to existing or new database with Row factory."""
    init_db(temp_db_path)
    conn = get_db_connection(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT param_name, param_value FROM model_constants LIMIT 1")
    row = cursor.fetchone()
    assert row is not None
    # Row factory allows dict/attribute-like or key indexing
    assert row["param_name"] is not None
    conn.close()

def test_delta_buckets_percentage_scale_and_edge_deltas(temp_db_path):
    """Verify spread_delta values provided as percentages (e.g., 7.5) or edge thresholds map accurately."""
    conn = init_db(temp_db_path)
    picks = [
        {"game_pk": 101, "spread_delta": 2.5, "result": "WIN", "units_won": 1.0},    # 0-5%
        {"game_pk": 102, "spread_delta": 7.5, "result": "WIN", "units_won": 1.0},    # 5-10%
        {"game_pk": 103, "spread_delta": 14.9, "result": "WIN", "units_won": 1.0},   # 10-15%
        {"game_pk": 104, "spread_delta": 15.0, "result": "WIN", "units_won": 1.0},   # 15-20%
        {"game_pk": 105, "spread_delta": 35.0, "result": "WIN", "units_won": 1.0},   # 20%+
    ]
    for p in picks:
        log_pick(conn, p)

    buckets = get_delta_buckets_history(conn)
    b_map = {b["bucket"]: b for b in buckets}
    assert b_map["0-5%"]["wins"] == 1
    assert b_map["5-10%"]["wins"] == 1
    assert b_map["10-15%"]["wins"] == 1
    assert b_map["15-20%"]["wins"] == 1
    assert b_map["20%+"]["wins"] == 1
    conn.close()

def test_empty_and_fallback_helpers(temp_db_path):
    """Verify helper behavior for missing entities and edge inputs."""
    conn = init_db(temp_db_path)

    # Empty picks history
    assert get_picks_history(conn) == []

    # Non-existent entities
    assert get_team(conn, 99999) is None
    assert get_pitcher(conn, 99999) is None
    assert get_park_factor(conn, 99999) is None

    # Slate cache with game missing game_pk
    save_slate_cache(conn, "2026-09-21", [{"away_team": "Team A", "home_team": "Team B"}])
    slate = get_cached_slate(conn, "2026-09-21")
    assert slate is not None
    assert len(slate) == 1
    assert slate[0]["away_team"] == "Team A"

    conn.close()


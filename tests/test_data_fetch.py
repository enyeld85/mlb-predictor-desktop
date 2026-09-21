import time
import pytest
import requests
from unittest.mock import MagicMock

from database import init_db, save_slate_cache, save_pitcher
from data_fetch import MLBDataFetcher


@pytest.fixture
def data_fetcher(temp_db_path):
    """Fixture providing an MLBDataFetcher with an initialized temporary DB."""
    init_db(temp_db_path)
    return MLBDataFetcher(db_path=temp_db_path, timeout=5)


# =====================================================================
# 1. Bullpen Fatigue Calculations & Edge Cases
# =====================================================================

def test_bullpen_fatigue_edge_cases(data_fetcher):
    """Verify 3-day exponential decay bullpen fatigue formula and boundary conditions."""
    # 0 pitches thrown -> 0.0 fatigue
    assert data_fetcher.compute_bullpen_fatigue([0, 0, 0]) == 0.0
    assert data_fetcher.compute_bullpen_fatigue([]) == 0.0
    assert data_fetcher.compute_bullpen_fatigue(None) == 0.0

    # Moderate usage: 30 yesterday, 20 two days ago, 10 three days ago
    # Formula: (30*1.0 + 20*0.6 + 10*0.3) / 75.0 = (30 + 12 + 3) / 75 = 45 / 75 = 0.60
    fatigue = data_fetcher.compute_bullpen_fatigue([30, 20, 10])
    assert pytest.approx(fatigue, rel=1e-3) == 0.60

    # Heavy usage: capped at 1.0
    # (50*1.0 + 40*0.6 + 30*0.3) / 75.0 = (50 + 24 + 9) / 75 = 83 / 75 = 1.1067 -> 1.0
    assert data_fetcher.compute_bullpen_fatigue([50, 40, 30]) == 1.0

    # Single-day heavy usage (> 75)
    assert data_fetcher.compute_bullpen_fatigue([80]) == 1.0

    # Dict format compatibility
    dict_history = [{"pitches": 30}, {"pitches": 20}, {"pitches": 10}]
    assert pytest.approx(data_fetcher.compute_bullpen_fatigue(dict_history), rel=1e-3) == 0.60


# =====================================================================
# 2. Park Factors Lookup and Fallback
# =====================================================================

def test_park_factors_lookup_and_fallback(data_fetcher):
    """Verify park factors lookup for MLB stadiums and fallback for unknown venues."""
    # Known venue by ID (Fenway Park, ID=3)
    fenway = data_fetcher.get_park_factor(3, "Fenway Park")
    assert fenway["run_factor"] > 1.0
    assert fenway["roof_type"] == "Open"
    assert fenway["altitude"] == 20

    # Known venue (Coors Field, ID=19)
    coors = data_fetcher.get_park_factor(19, "Coors Field")
    assert coors["altitude"] == 5200
    assert coors["run_factor"] >= 1.25

    # Known venue by name fallback when ID is 0 or unmapped
    tropicana = data_fetcher.get_park_factor(9999, "Tropicana Field")
    assert tropicana["roof_type"] == "Dome"

    # Unknown venue fallback: neutral 1.00 run and HR factor
    unknown = data_fetcher.get_park_factor(88888, "Imaginary Stadium")
    assert unknown["run_factor"] == 1.00
    assert unknown["hr_factor"] == 1.00
    assert unknown["roof_type"] == "Open"


# =====================================================================
# 3. Schedule Parsing with Mock Payload
# =====================================================================

def test_fetch_schedule_success(data_fetcher, mock_schedule_response, monkeypatch):
    """Verify schedule parsing with probable pitchers, team records, venue, and weather."""
    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_schedule_response
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get)

    games, is_cached, status_msg = data_fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True)

    assert is_cached is False
    assert "Live Feed" in status_msg or "Connected" in status_msg
    assert len(games) == 1

    game = games[0]
    assert game["game_pk"] == 748123
    assert game["game_date"] == "2026-09-20"
    assert game["away_team"]["name"] == "New York Yankees"
    assert game["away_team"]["abbrev"] == "NYY"
    assert game["away_team"]["starter"]["name"] == "Gerrit Cole"
    assert game["home_team"]["name"] == "Boston Red Sox"
    assert game["home_team"]["abbrev"] == "BOS"
    assert game["home_team"]["starter"]["name"] == "Brayan Bello"
    assert game["venue"]["name"] == "Fenway Park"
    assert game["weather"]["temp"] == 74
    assert game["weather"]["wind_speed"] == 8
    assert "Out to CF" in game["weather"]["wind_dir"]


# =====================================================================
# 4. Lineup Confirmation Detection (CONFIRMED vs PROJECTED)
# =====================================================================

def test_lineup_confirmation_detection(data_fetcher, mock_schedule_response, monkeypatch):
    """Verify lineup status is CONFIRMED when batting orders are published, else PROJECTED."""
    # First: response without official batting order -> PROJECTED
    def mock_get_no_lineup(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_schedule_response
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get_no_lineup)
    games, _, _ = data_fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True)
    assert games[0]["away_team"]["lineup_status"] == "PROJECTED"
    assert games[0]["home_team"]["lineup_status"] == "PROJECTED"

    # Second: response with official lineup batting order -> CONFIRMED
    confirmed_payload = {
        "dates": [
            {
                "date": "2026-09-20",
                "games": [
                    {
                        "gamePk": 748123,
                        "gameDate": "2026-09-20T23:05:00Z",
                        "status": {"detailedState": "Scheduled"},
                        "teams": {
                            "away": {
                                "team": {"id": 147, "name": "New York Yankees"},
                                "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole"},
                            },
                            "home": {
                                "team": {"id": 111, "name": "Boston Red Sox"},
                                "probablePitcher": {"id": 678394, "fullName": "Brayan Bello"},
                            },
                        },
                        "lineups": {
                            "awayPlayers": [
                                {"id": 101, "fullName": "Gleyber Torres", "primaryPosition": {"abbreviation": "2B"}},
                                {"id": 102, "fullName": "Juan Soto", "primaryPosition": {"abbreviation": "RF"}},
                                {"id": 103, "fullName": "Aaron Judge", "primaryPosition": {"abbreviation": "CF"}},
                                {"id": 104, "fullName": "Giancarlo Stanton", "primaryPosition": {"abbreviation": "DH"}},
                                {"id": 105, "fullName": "Jazz Chisholm", "primaryPosition": {"abbreviation": "3B"}},
                                {"id": 106, "fullName": "Anthony Rizzo", "primaryPosition": {"abbreviation": "1B"}},
                                {"id": 107, "fullName": "Anthony Volpe", "primaryPosition": {"abbreviation": "SS"}},
                                {"id": 108, "fullName": "Austin Wells", "primaryPosition": {"abbreviation": "C"}},
                                {"id": 109, "fullName": "Alex Verdugo", "primaryPosition": {"abbreviation": "LF"}},
                            ],
                            "homePlayers": [
                                {"id": 201, "fullName": "Jarren Duran", "primaryPosition": {"abbreviation": "CF"}},
                            ],
                        },
                        "venue": {"id": 3, "name": "Fenway Park"},
                        "weather": {"temp": "72", "wind": "5 mph, Calm"},
                    }
                ]
            }
        ]
    }

    def mock_get_confirmed(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = confirmed_payload
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get_confirmed)
    games, _, _ = data_fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True)
    assert games[0]["away_team"]["lineup_status"] == "CONFIRMED"
    assert len(games[0]["away_team"]["lineup"]) == 9
    assert games[0]["away_team"]["lineup"][0]["name"] == "Gleyber Torres"


# =====================================================================
# 5. Pitcher Metrics & Defensive Handling of Missing Stats
# =====================================================================

def test_pitcher_metrics_computation(data_fetcher, monkeypatch):
    """Verify FIP and xFIP in-engine computation from season pitching stats."""
    mock_pitching_payload = {
        "stats": [
            {
                "splits": [
                    {
                        "season": "2026",
                        "stat": {
                            "gamesStarted": 25,
                            "era": "3.12",
                            "inningsPitched": "150.0",
                            "strikeOuts": 175,
                            "baseOnBalls": 35,
                            "hitByPitch": 5,
                            "homeRuns": 18,
                            "airOuts": 120,
                            "groundOuts": 140,
                            "whip": "1.05",
                            "battersFaced": 600,
                        }
                    }
                ]
            }
        ]
    }

    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_pitching_payload
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get)

    metrics = data_fetcher.fetch_pitcher_metrics(543037)
    assert metrics is not None
    assert metrics["era"] == 3.12
    assert metrics["whip"] == 1.05
    # FIP formula: (13*18 + 3*(35+5) - 2*175) / 150.0 + 3.15 = (234 + 120 - 350)/150 + 3.15 = 4/150 + 3.15 = 0.0267 + 3.15 = 3.1767
    assert pytest.approx(metrics["fip"], rel=1e-2) == 3.18
    # xFIP: normalized HR = 120 * 0.115 = 13.8 -> (13*13.8 + 120 - 350)/150 + 3.15 = (179.4 - 230)/150 + 3.15 = -50.6/150 + 3.15 = 2.8127
    assert pytest.approx(metrics["xfip"], rel=1e-2) == 2.81


def test_pitcher_metrics_with_gamelog_and_inning1_splits(data_fetcher, monkeypatch):
    """Verify combined season, statSplits (i01), and gameLog payload parses rolling median IP and 1st-inning splits."""
    combined_payload = {
        "stats": [
            {
                "type": {"displayName": "season"},
                "splits": [
                    {
                        "season": "2026",
                        "stat": {
                            "gamesStarted": 10,
                            "era": "2.85",
                            "inningsPitched": "60.0",
                            "strikeOuts": 65,
                            "baseOnBalls": 15,
                            "hitByPitch": 2,
                            "homeRuns": 6,
                            "airOuts": 48,
                            "groundOuts": 55,
                            "whip": "1.08",
                            "battersFaced": 240,
                        },
                    }
                ],
            },
            {
                "type": {"displayName": "statSplits"},
                "splits": [
                    {
                        "split": {"code": "i01", "description": "First Inning"},
                        "stat": {
                            "era": "1.80",
                            "whip": "0.90",
                        },
                    }
                ],
            },
            {
                "type": {"displayName": "gameLog"},
                "splits": [
                    {"stat": {"gamesStarted": 1, "inningsPitched": "6.0"}},
                    {"stat": {"gamesStarted": 1, "inningsPitched": "5.2"}},
                    {"stat": {"gamesStarted": 1, "inningsPitched": "7.0"}},
                    {"stat": {"gamesStarted": 1, "inningsPitched": "6.1"}},
                    {"stat": {"gamesStarted": 1, "inningsPitched": "5.0"}},
                    {"stat": {"gamesStarted": 0, "inningsPitched": "2.0"}},
                ],
            },
        ]
    }

    monkeypatch.setattr(
        data_fetcher.session,
        "get",
        lambda *a, **kw: MagicMock(status_code=200, json=lambda: combined_payload, raise_for_status=lambda: None),
    )

    metrics = data_fetcher.fetch_pitcher_metrics(678394)
    assert metrics is not None
    assert metrics["era"] == 2.85
    assert metrics["whip"] == 1.08
    assert metrics["sample_ip"] == 60.0
    assert metrics["first_inning_era"] == 1.80
    assert metrics["first_inning_whip"] == 0.90
    assert metrics["median_ip"] == 6.0


def test_missing_pitcher_stats_defensive_handling(data_fetcher, monkeypatch):
    """Verify missing pitcher stats return None without raising unhandled exceptions."""
    # 1. Non-existent / empty splits
    def mock_get_empty(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"stats": [{"splits": []}]}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get_empty)
    assert data_fetcher.fetch_pitcher_metrics(9999999) is None

    # 2. Network error during stats pull
    def mock_get_error(url, *args, **kwargs):
        raise requests.RequestException("Stats API unreachable")

    monkeypatch.setattr(data_fetcher.session, "get", mock_get_error)
    assert data_fetcher.fetch_pitcher_metrics(543037) is None

    # 3. Invalid / None pitcher ID
    assert data_fetcher.fetch_pitcher_metrics(0) is None
    assert data_fetcher.fetch_pitcher_metrics(None) is None


def test_pitcher_metrics_zero_innings_pitched(data_fetcher, monkeypatch):
    """Verify pitcher with zero innings pitched uses safe fallback."""
    mock_payload = {
        "stats": [
            {
                "splits": [
                    {
                        "stat": {
                            "gamesStarted": 0,
                            "era": "0.00",
                            "inningsPitched": "0.0",
                            "strikeOuts": 0,
                            "baseOnBalls": 0,
                            "hitByPitch": 0,
                            "homeRuns": 0,
                            "whip": "0.00",
                        }
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(data_fetcher.session, "get", lambda *a, **kw: MagicMock(status_code=200, json=lambda: mock_payload, raise_for_status=lambda: None))
    metrics = data_fetcher.fetch_pitcher_metrics(123456)
    assert metrics is not None
    assert metrics["era"] == 0.0
    assert metrics["median_ip"] == 5.0


# =====================================================================
# 6. Offline Fallback & SQLite Caching Behavior
# =====================================================================

def test_offline_fallback_with_cached_data(data_fetcher, sample_game_data, monkeypatch):
    """Verify fallback to SQLite cache when network fails, displaying cached banner."""
    date_str = "2026-09-20"

    # Pre-populate cache in SQLite
    conn = data_fetcher._get_db()
    save_slate_cache(conn, date_str, [sample_game_data])

    # Simulate network failure
    def mock_network_error(*args, **kwargs):
        raise requests.ConnectionError("DNS lookup failed: statsapi.mlb.com")

    monkeypatch.setattr(data_fetcher.session, "get", mock_network_error)

    # Fetch should catch error and fall back to SQLite cache
    games, is_cached, status_msg = data_fetcher.fetch_schedule_for_date(date_str, force_refresh=True, conn=conn)

    assert is_cached is True
    assert len(games) == 1
    assert games[0]["game_pk"] == 748123
    assert "Displaying Cached Slate" in status_msg
    conn.close()


def test_offline_fallback_without_cached_data(data_fetcher, monkeypatch):
    """Verify offline behavior when neither network nor SQLite cache is available."""
    date_str = "2026-09-21"

    def mock_timeout(*args, **kwargs):
        raise requests.Timeout("Connection timed out after 8s")

    monkeypatch.setattr(data_fetcher.session, "get", mock_timeout)

    games, is_cached, status_msg = data_fetcher.fetch_schedule_for_date(date_str, force_refresh=True)

    assert is_cached is False
    assert games == []
    assert "Data unavailable / Offline" in status_msg


def test_cached_slate_returned_without_network_when_not_forced(data_fetcher, sample_game_data, monkeypatch):
    """Verify cache-first strategy: does not hit network if cache is available and force_refresh=False."""
    date_str = "2026-09-20"
    conn = data_fetcher._get_db()
    save_slate_cache(conn, date_str, [sample_game_data])

    network_called = False
    def mock_get(*args, **kwargs):
        nonlocal network_called
        network_called = True
        raise RuntimeError("Network should not be called on cache hit!")

    monkeypatch.setattr(data_fetcher.session, "get", mock_get)

    games, is_cached, status_msg = data_fetcher.fetch_schedule_for_date(date_str, force_refresh=False, conn=conn)

    assert network_called is False
    assert is_cached is True
    assert len(games) == 1
    assert "Displaying Cached Slate" in status_msg
    conn.close()


# =====================================================================
# 7. Defensive Parsing Edge Cases (Weather, Missing Starter, DB Hydration)
# =====================================================================

def test_defensive_weather_closed_dome_and_crosswind(data_fetcher):
    """Verify dome/retractable roof suppresses wind effects and parses varied wind strings."""
    # Tropicana Field dome
    w_dome = data_fetcher._parse_weather({"temp": 72, "wind": "15 mph, Out to CF"}, roof_type="Dome")
    assert w_dome["roof_status"] == "Closed"
    assert w_dome["wind_speed"] == 0
    assert w_dome["wind_dir"] == "Calm"

    # Crosswind in open stadium
    w_open = data_fetcher._parse_weather({"temp": "68", "wind": "12 mph, L to R"}, roof_type="Open")
    assert w_open["roof_status"] == "Open"
    assert w_open["temp"] == 68
    assert w_open["wind_speed"] == 12
    assert "L to R" in w_open["wind_dir"]


def test_schedule_missing_starter_defensive(data_fetcher, monkeypatch):
    """Verify schedule with missing/TBD probable starters handles gracefully."""
    tbd_payload = {
        "dates": [
            {
                "date": "2026-09-20",
                "games": [
                    {
                        "gamePk": 999111,
                        "teams": {
                            "away": {"team": {"id": 147, "name": "New York Yankees"}},
                            "home": {"team": {"id": 111, "name": "Boston Red Sox"}},
                        },
                        "venue": {"id": 3, "name": "Fenway Park"},
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(data_fetcher.session, "get", lambda *a, **kw: MagicMock(status_code=200, json=lambda: tbd_payload, raise_for_status=lambda: None))
    games, _, _ = data_fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True)
    assert len(games) == 1
    assert games[0]["away_team"]["starter"] is None
    assert games[0]["home_team"]["starter"] is None


def test_schedule_starter_hydrated_from_db(data_fetcher, monkeypatch):
    """Verify that if pitcher is stored in DB, schedule ingestion populates pitcher stats from DB."""
    conn = data_fetcher._get_db()
    save_pitcher(conn, {
        "pitcher_id": 543037,
        "name": "Gerrit Cole",
        "hand": "R",
        "era": 3.12,
        "fip": 3.25,
        "xfip": 3.30,
        "whip": 1.05,
    })

    schedule_payload = {
        "dates": [
            {
                "date": "2026-09-20",
                "games": [
                    {
                        "gamePk": 112233,
                        "teams": {
                            "away": {
                                "team": {"id": 147, "name": "New York Yankees"},
                                "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole"},
                            },
                            "home": {
                                "team": {"id": 111, "name": "Boston Red Sox"},
                            },
                        },
                        "venue": {"id": 3, "name": "Fenway Park"},
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(data_fetcher.session, "get", lambda *a, **kw: MagicMock(status_code=200, json=lambda: schedule_payload, raise_for_status=lambda: None))
    games, _, _ = data_fetcher.fetch_schedule_for_date("2026-09-20", force_refresh=True, conn=conn)

    starter = games[0]["away_team"]["starter"]
    assert starter is not None
    assert starter["id"] == 543037
    assert starter["era"] == 3.12
    assert starter["fip"] == 3.25
    conn.close()


def test_fetch_team_platoon_splits_fallback(data_fetcher):
    """Verify fetch_team_platoon_splits returns safe fallback when network fails or team invalid."""
    res_invalid = data_fetcher.fetch_team_platoon_splits(0)
    assert res_invalid["wrc_plus_vs_rhp"] == 100.0
    assert res_invalid["status"] == "Data unavailable"

    res_valid_team = data_fetcher.fetch_team_platoon_splits(147)
    assert res_valid_team["wrc_plus_vs_rhp"] == 100.0


# =====================================================================
# 8. Background Async Thread Invocation
# =====================================================================

def test_fetch_slate_async_success(data_fetcher, mock_schedule_response, monkeypatch):
    """Verify fetch_slate_async executes in daemon thread and triggers on_success callback."""
    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_schedule_response
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(data_fetcher.session, "get", mock_get)

    success_result = {}
    progress_updates = []

    def on_success(games, is_cached, status_msg):
        success_result["games"] = games
        success_result["is_cached"] = is_cached
        success_result["status_msg"] = status_msg

    def on_error(err):
        pytest.fail(f"on_error unexpectedly called: {err}")

    def on_progress(pct, msg):
        progress_updates.append((pct, msg))

    thread = data_fetcher.fetch_slate_async(
        date_str="2026-09-20",
        on_success=on_success,
        on_error=on_error,
        on_progress=on_progress,
        force_refresh=True,
    )

    assert thread.is_alive() or success_result
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "games" in success_result
    assert len(success_result["games"]) == 1
    assert success_result["is_cached"] is False
    assert len(progress_updates) > 0


def test_fetch_slate_async_error_handling(data_fetcher, monkeypatch):
    """Verify fetch_slate_async catches uncaught exceptions and routes to on_error."""
    error_result = {}

    def mock_bad_fetch(*args, **kwargs):
        raise ValueError("Simulated unexpected internal parsing failure")

    monkeypatch.setattr(data_fetcher, "fetch_schedule_for_date", mock_bad_fetch)

    def on_success(*args):
        pytest.fail("on_success should not be called on error")

    def on_error(err):
        error_result["error"] = err

    thread = data_fetcher.fetch_slate_async(
        date_str="2026-09-20",
        on_success=on_success,
        on_error=on_error,
    )

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "error" in error_result
    assert "Simulated unexpected internal parsing failure" in error_result["error"]

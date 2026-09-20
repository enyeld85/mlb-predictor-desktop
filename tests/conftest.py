import os
import tempfile
import pytest
import sqlite3

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture providing a temporary SQLite database path."""
    db_file = tmp_path / "test_mlb.db"
    return str(db_file)

@pytest.fixture
def sample_game_data():
    """Mock dictionary of an MLB game matchup (NYY @ BOS)."""
    return {
        "game_pk": 748123,
        "game_date": "2026-09-20",
        "game_time": "19:05",
        "status": "Scheduled",
        "away_team": {
            "id": 147,
            "name": "New York Yankees",
            "abbrev": "NYY",
            "record": "88-64",
            "wrc_plus_vs_rhp": 118,
            "wrc_plus_vs_lhp": 112,
            "woba_vs_rhp": 0.334,
            "woba_vs_lhp": 0.325,
            "bullpen_era": 3.42,
            "bullpen_whip": 1.18,
            "bullpen_fip": 3.55,
            "bullpen_fatigue": 0.25,
            "starter": {
                "id": 543037,
                "name": "Gerrit Cole",
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
                "sample_ip": 160.0
            },
            "lineup_status": "CONFIRMED",
            "lineup": [
                {"id": 1, "name": "Gleyber Torres", "pos": "2B", "order": 1, "woba": 0.330, "bats": "R"},
                {"id": 2, "name": "Juan Soto", "pos": "RF", "order": 2, "woba": 0.415, "bats": "L"},
                {"id": 3, "name": "Aaron Judge", "pos": "CF", "order": 3, "woba": 0.440, "bats": "R"},
                {"id": 4, "name": "Giancarlo Stanton", "pos": "DH", "order": 4, "woba": 0.345, "bats": "R"}
            ]
        },
        "home_team": {
            "id": 111,
            "name": "Boston Red Sox",
            "abbrev": "BOS",
            "record": "78-74",
            "wrc_plus_vs_rhp": 104,
            "wrc_plus_vs_lhp": 98,
            "woba_vs_rhp": 0.318,
            "woba_vs_lhp": 0.308,
            "bullpen_era": 4.10,
            "bullpen_whip": 1.28,
            "bullpen_fip": 4.05,
            "bullpen_fatigue": 0.40,
            "starter": {
                "id": 678394,
                "name": "Brayan Bello",
                "hand": "R",
                "era": 4.20,
                "fip": 4.15,
                "xfip": 4.05,
                "whip": 1.32,
                "k_pct": 0.210,
                "bb_pct": 0.080,
                "median_ip": 5.1,
                "first_inning_era": 4.50,
                "first_inning_whip": 1.40,
                "sample_ip": 145.0
            },
            "lineup_status": "CONFIRMED",
            "lineup": [
                {"id": 11, "name": "Jarren Duran", "pos": "CF", "order": 1, "woba": 0.350, "bats": "L"},
                {"id": 12, "name": "Rafael Devers", "pos": "3B", "order": 2, "woba": 0.380, "bats": "L"},
                {"id": 13, "name": "Tyler O'Neill", "pos": "LF", "order": 3, "woba": 0.355, "bats": "R"},
                {"id": 14, "name": "Triston Casas", "pos": "1B", "order": 4, "woba": 0.360, "bats": "L"}
            ]
        },
        "venue": {
            "id": 3,
            "name": "Fenway Park",
            "altitude": 20,
            "roof_type": "Open",
            "run_factor": 1.08,
            "hr_factor": 1.05
        },
        "weather": {
            "temp": 74,
            "wind_speed": 8,
            "wind_dir": "Out to CF",
            "condition": "Clear"
        }
    }

@pytest.fixture
def mock_schedule_response():
    """Mock API response matching statsapi.mlb.com/api/v1/schedule."""
    return {
        "dates": [
            {
                "date": "2026-09-20",
                "games": [
                    {
                        "gamePk": 748123,
                        "gameDate": "2026-09-20T23:05:00Z",
                        "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                        "teams": {
                            "away": {
                                "team": {"id": 147, "name": "New York Yankees"},
                                "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole"}
                            },
                            "home": {
                                "team": {"id": 111, "name": "Boston Red Sox"},
                                "probablePitcher": {"id": 678394, "fullName": "Brayan Bello"}
                            }
                        },
                        "venue": {"id": 3, "name": "Fenway Park"},
                        "weather": {"condition": "Clear", "temp": "74", "wind": "8 mph, Out to CF"}
                    }
                ]
            }
        ]
    }

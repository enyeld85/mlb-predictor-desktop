"""SQLite persistence layer and dynamic model constants for MLB Apex Predictor.

Manages relational schemas for team splits, starting pitchers, ballpark factors,
scheduled game slate caching, pick tracking, model calibration, and dynamic
season constants.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DEFAULT_MODEL_CONSTANTS: list[tuple[str, float, str]] = [
    ("dispersion_alpha", 0.12, "Negative binomial dispersion parameter"),
    ("home_field_advantage_runs", 0.18, "Home field advantage run boost"),
    ("temp_coefficient", 0.012, "Multiplicative factor per 10°F from 72°F"),
    ("wind_coefficient_out", 0.015, "Multiplicative factor per mph blowing out above 5 mph"),
    ("wind_coefficient_in", 0.012, "Multiplicative factor per mph blowing in above 5 mph"),
    ("cfip_constant", 3.15, "Constant in FIP calculation formula"),
    ("hr_fb_baseline", 0.115, "League baseline HR/FB ratio"),
    ("baseline_league_runs", 4.40, "Baseline league average runs per team per game"),
]

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS model_constants (
        param_name TEXT PRIMARY KEY,
        param_value REAL NOT NULL,
        description TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS games_cache (
        game_pk INTEGER PRIMARY KEY,
        game_date TEXT NOT NULL,
        last_updated TEXT NOT NULL,
        game_data_json TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        abbrev TEXT,
        wrc_plus_vs_rhp REAL,
        wrc_plus_vs_lhp REAL,
        woba_vs_rhp REAL,
        woba_vs_lhp REAL,
        bullpen_era REAL,
        bullpen_whip REAL,
        bullpen_fip REAL,
        bullpen_fatigue REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pitchers (
        pitcher_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        team_id INTEGER,
        hand TEXT,
        era REAL,
        fip REAL,
        xfip REAL,
        whip REAL,
        k_pct REAL,
        bb_pct REAL,
        median_ip REAL,
        first_inning_era REAL,
        first_inning_whip REAL,
        sample_ip REAL,
        last_updated TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS batters (
        batter_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        team_id INTEGER,
        hand TEXT,
        avg REAL,
        obp REAL,
        slg REAL,
        iso REAL,
        k_pct REAL,
        bb_pct REAL,
        woba_vs_rhp REAL,
        woba_vs_lhp REAL,
        hard_hit_pct REAL,
        barrel_pct REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS park_factors (
        venue_id INTEGER PRIMARY KEY,
        venue_name TEXT NOT NULL,
        altitude REAL,
        roof_type TEXT,
        run_factor REAL,
        hr_factor REAL,
        last_updated_season INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS picks_history (
        pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER,
        game_date TEXT,
        market TEXT,
        selection TEXT,
        model_prob REAL,
        uncertainty_interval REAL,
        fair_odds TEXT,
        market_odds TEXT,
        edge_pct REAL,
        ev_pct REAL,
        confidence TEXT,
        spread_delta REAL,
        result TEXT DEFAULT 'PENDING',
        closing_odds TEXT,
        units_won REAL DEFAULT 0.0
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_games_cache_date ON games_cache(game_date);",
    "CREATE INDEX IF NOT EXISTS idx_picks_history_date ON picks_history(game_date);",
    "CREATE INDEX IF NOT EXISTS idx_picks_history_game_pk ON picks_history(game_pk);",
    "CREATE INDEX IF NOT EXISTS idx_pitchers_team_id ON pitchers(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_batters_team_id ON batters(team_id);",
]


def get_db_connection(db_path: str = "mlb_analytics.db") -> sqlite3.Connection:
    """Open and return a SQLite database connection configured with Row factory."""
    if db_path != ":memory:" and os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "mlb_analytics.db") -> sqlite3.Connection:
    """Initialize SQLite database schema and seed default model constants."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)

        # Defensive migration: Ensure last_updated exists on pitchers table if table was created previously
        try:
            cursor.execute("SELECT last_updated FROM pitchers LIMIT 1")
        except sqlite3.OperationalError:
            try:
                cursor.execute("ALTER TABLE pitchers ADD COLUMN last_updated TEXT")
            except sqlite3.OperationalError:
                pass

        # Seed default constants with INSERT OR IGNORE to protect customized constants
        cursor.executemany(
            """
            INSERT OR IGNORE INTO model_constants (param_name, param_value, description)
            VALUES (?, ?, ?)
            """,
            DEFAULT_MODEL_CONSTANTS,
        )
    return conn


def get_model_constants(conn: sqlite3.Connection) -> dict[str, float]:
    """Retrieve dynamic model parameters as a dictionary."""
    cursor = conn.cursor()
    cursor.execute("SELECT param_name, param_value FROM model_constants")
    return {row[0]: float(row[1]) for row in cursor.fetchall()}


def update_model_constants(conn: sqlite3.Connection, constants: dict[str, float]) -> None:
    """Update or insert model parameters while preserving existing descriptions."""
    with conn:
        cursor = conn.cursor()
        for name, value in constants.items():
            cursor.execute(
                """
                INSERT INTO model_constants (param_name, param_value, description)
                VALUES (?, ?, '')
                ON CONFLICT(param_name) DO UPDATE SET param_value = excluded.param_value
                """,
                (name, float(value)),
            )


def save_slate_cache(conn: sqlite3.Connection, game_date: str, games: list[dict]) -> None:
    """Cache slate games as serialized JSON for a given date."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM games_cache WHERE game_date = ?", (game_date,))
        for idx, game in enumerate(games):
            pk = game.get("game_pk")
            if pk is None:
                pk = game.get("gamePk")
            if pk is None:
                pk = idx + 1
            cursor.execute(
                """
                INSERT OR REPLACE INTO games_cache (game_pk, game_date, last_updated, game_data_json)
                VALUES (?, ?, ?, ?)
                """,
                (int(pk), game_date, now_iso, json.dumps(game)),
            )


def get_cached_slate(conn: sqlite3.Connection, game_date: str) -> list[dict] | None:
    """Retrieve cached game slate for a given date, or None if not cached."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT game_data_json FROM games_cache WHERE game_date = ? ORDER BY game_pk ASC",
        (game_date,),
    )
    rows = cursor.fetchall()
    if not rows:
        return None
    return [json.loads(row[0] if isinstance(row, (tuple, list)) else row["game_data_json"]) for row in rows]


def log_pick(conn: sqlite3.Connection, pick_data: dict) -> int:
    """Insert a generated prediction or market pick into picks_history.

    Returns:
        Generated pick_id.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO picks_history (
            game_pk, game_date, market, selection, model_prob,
            uncertainty_interval, fair_odds, market_odds, edge_pct,
            ev_pct, confidence, spread_delta, result, closing_odds, units_won
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pick_data.get("game_pk"),
            pick_data.get("game_date"),
            pick_data.get("market"),
            pick_data.get("selection"),
            pick_data.get("model_prob"),
            pick_data.get("uncertainty_interval"),
            pick_data.get("fair_odds"),
            pick_data.get("market_odds"),
            pick_data.get("edge_pct"),
            pick_data.get("ev_pct"),
            pick_data.get("confidence"),
            pick_data.get("spread_delta"),
            pick_data.get("result", "PENDING"),
            pick_data.get("closing_odds", ""),
            pick_data.get("units_won", 0.0),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def update_pick_result(
    conn: sqlite3.Connection,
    pick_id: int,
    result: str,
    units_won: float,
    closing_odds: str = "",
) -> None:
    """Update settlement outcome and units won for a logged pick."""
    cursor = conn.cursor()
    if closing_odds:
        cursor.execute(
            """
            UPDATE picks_history
            SET result = ?, units_won = ?, closing_odds = ?
            WHERE pick_id = ?
            """,
            (result, float(units_won), closing_odds, pick_id),
        )
    else:
        cursor.execute(
            """
            UPDATE picks_history
            SET result = ?, units_won = ?
            WHERE pick_id = ?
            """,
            (result, float(units_won), pick_id),
        )
    conn.commit()


def get_picks_history(conn: sqlite3.Connection, limit: int = 500) -> list[dict[str, Any]]:
    """Retrieve logged picks history ordered by most recent first."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT pick_id, game_pk, game_date, market, selection,
               model_prob, uncertainty_interval, fair_odds, market_odds,
               edge_pct, ev_pct, confidence, spread_delta,
               result, closing_odds, units_won
        FROM picks_history
        ORDER BY pick_id DESC
        LIMIT ?
        """,
        (limit,),
    )
    cols = [col[0] for col in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _classify_delta_bucket(spread_delta: float) -> str:
    """Map numeric spread delta to bucket name: 0-5%, 5-10%, 10-15%, 15-20%, 20%+."""
    # Convert decimal fractions (e.g. 0.08) to percentage points (8.0)
    val = spread_delta * 100.0 if abs(spread_delta) <= 1.0 else spread_delta
    if val < 5.0:
        return "0-5%"
    if val < 10.0:
        return "5-10%"
    if val < 15.0:
        return "10-15%"
    if val < 20.0:
        return "15-20%"
    return "20%+"


def get_delta_buckets_history(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Group settled picks by spread_delta into 5 calibrated performance buckets.

    Buckets: '0-5%', '5-10%', '10-15%', '15-20%', '20%+'
    Returns:
        List of dicts with keys:
        'bucket', 'total_bets', 'wins', 'cover_pct', 'units_won', 'roi_pct'.
    """
    bucket_order = ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    buckets: dict[str, dict[str, Any]] = {
        name: {
            "bucket": name,
            "total_bets": 0,
            "wins": 0,
            "cover_pct": 0.0,
            "units_won": 0.0,
            "roi_pct": 0.0,
        }
        for name in bucket_order
    }

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT spread_delta, result, units_won
        FROM picks_history
        WHERE result IS NOT NULL
          AND UPPER(result) != 'PENDING'
          AND spread_delta IS NOT NULL
        """
    )
    rows = cursor.fetchall()

    bucket_units: dict[str, float] = {name: 0.0 for name in bucket_order}

    for row in rows:
        delta = row[0] if isinstance(row, (tuple, list)) else row["spread_delta"]
        res = str(row[1] if isinstance(row, (tuple, list)) else row["result"]).upper()
        units = float(row[2] if isinstance(row, (tuple, list)) else row["units_won"] or 0.0)

        bucket_name = _classify_delta_bucket(float(delta))
        b = buckets[bucket_name]
        b["total_bets"] += 1
        if res in ("WIN", "WON", "COVER") or units > 0:
            b["wins"] += 1
        bucket_units[bucket_name] += units

    for name in bucket_order:
        b = buckets[name]
        tot = b["total_bets"]
        wins = b["wins"]
        u_won = round(bucket_units[name], 2)
        b["units_won"] = u_won
        b["cover_pct"] = round((wins / tot) * 100.0, 2) if tot > 0 else 0.0
        b["roi_pct"] = round((u_won / tot) * 100.0, 2) if tot > 0 else 0.0

    return [buckets[name] for name in bucket_order]


def save_team(conn: sqlite3.Connection, team_data: dict) -> None:
    """Insert or replace team platoon and bullpen metrics."""
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO teams (
                team_id, name, abbrev, wrc_plus_vs_rhp, wrc_plus_vs_lhp,
                woba_vs_rhp, woba_vs_lhp, bullpen_era, bullpen_whip,
                bullpen_fip, bullpen_fatigue
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                team_data.get("team_id"),
                team_data.get("name"),
                team_data.get("abbrev"),
                team_data.get("wrc_plus_vs_rhp"),
                team_data.get("wrc_plus_vs_lhp"),
                team_data.get("woba_vs_rhp"),
                team_data.get("woba_vs_lhp"),
                team_data.get("bullpen_era"),
                team_data.get("bullpen_whip"),
                team_data.get("bullpen_fip"),
                team_data.get("bullpen_fatigue"),
            ),
        )


def get_team(conn: sqlite3.Connection, team_id: int) -> dict[str, Any] | None:
    """Retrieve team by team_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,))
    row = cursor.fetchone()
    if not row:
        return None
    cols = [col[0] for col in cursor.description]
    return dict(zip(cols, row))


def save_pitcher(conn: sqlite3.Connection, pitcher_data: dict) -> None:
    """Insert or replace pitcher statistics with UTC timestamp."""
    last_updated = pitcher_data.get("last_updated") or datetime.now(timezone.utc).isoformat()
    pid = pitcher_data.get("pitcher_id") or pitcher_data.get("id")
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO pitchers (
                pitcher_id, name, team_id, hand, era, fip, xfip, whip,
                k_pct, bb_pct, median_ip, first_inning_era, first_inning_whip, sample_ip, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                pitcher_data.get("name"),
                pitcher_data.get("team_id"),
                pitcher_data.get("hand"),
                pitcher_data.get("era"),
                pitcher_data.get("fip"),
                pitcher_data.get("xfip"),
                pitcher_data.get("whip"),
                pitcher_data.get("k_pct"),
                pitcher_data.get("bb_pct"),
                pitcher_data.get("median_ip"),
                pitcher_data.get("first_inning_era"),
                pitcher_data.get("first_inning_whip"),
                pitcher_data.get("sample_ip"),
                last_updated,
            ),
        )


def get_pitcher(
    conn: sqlite3.Connection,
    pitcher_id: int,
    max_age_hours: float | None = 12.0,
) -> dict[str, Any] | None:
    """Retrieve pitcher by pitcher_id with optional TTL freshness check."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pitchers WHERE pitcher_id = ?", (pitcher_id,))
    row = cursor.fetchone()
    if not row:
        return None
    cols = [col[0] for col in cursor.description]
    res = dict(zip(cols, row))

    # TTL freshness validation
    if max_age_hours is not None and res.get("last_updated"):
        try:
            ts_str = str(res["last_updated"]).replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if (now - ts).total_seconds() > max_age_hours * 3600.0:
                return None  # Stale, caller should refresh
        except Exception:
            pass

    return res


def save_park_factor(conn: sqlite3.Connection, park_data: dict) -> None:
    """Insert or replace venue park factor."""
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO park_factors (
                venue_id, venue_name, altitude, roof_type,
                run_factor, hr_factor, last_updated_season
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                park_data.get("venue_id"),
                park_data.get("venue_name"),
                park_data.get("altitude"),
                park_data.get("roof_type"),
                park_data.get("run_factor"),
                park_data.get("hr_factor"),
                park_data.get("last_updated_season"),
            ),
        )


def get_park_factor(conn: sqlite3.Connection, venue_id: int) -> dict[str, Any] | None:
    """Retrieve park factor by venue_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM park_factors WHERE venue_id = ?", (venue_id,))
    row = cursor.fetchone()
    if not row:
        return None
    cols = [col[0] for col in cursor.description]
    return dict(zip(cols, row))

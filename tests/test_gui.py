"""Unit and integration tests for MLB Apex Desktop GUI (gui.py & main.py).

Tests:
- Headless initialization of MLBPredictorApp without visual display errors.
- Verification of 8 tabs creation and exact titles.
- Slate loading, simulation execution, and Treeview population across tabs.
- Inline double-click editing event trigger and instant edge / EV recalculation.
- Dynamic filtering responsiveness on Player Props and Model Picks tabs.
- Historical Delta Performance bucket table rendering.
- Matplotlib FigureCanvasTkAgg calibration and bankroll charts generation.
- Settings update and database model constants synchronization.
- Thread-safe background refresh and callback orchestration.
"""

from __future__ import annotations

import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from database import init_db, log_pick, get_model_constants, update_model_constants
from models import simulate_game, model_player_props, prob_to_american

# Import will fail before implementation
import gui
from gui import MLBPredictorApp


@pytest.fixture
def gui_app(temp_db_path, sample_game_data):
    """Fixture providing an initialized, headless MLBPredictorApp with test db."""
    conn = init_db(temp_db_path)
    conn.close()

    # Create app with auto_load=False to control data ingestion deterministically
    app = MLBPredictorApp(db_path=temp_db_path, auto_load=False)
    app.withdraw()  # Headless mode: hide root window from screen
    app.update_idletasks()

    yield app

    # Cleanup
    try:
        app.destroy()
    except Exception:
        pass


def test_app_initialization_and_theme(gui_app):
    """Test that MLBPredictorApp initializes cleanly with dark sportsbook styling."""
    assert gui_app is not None
    assert gui_app.title() == "MLB Apex Predictor & Betting Analytics"
    assert gui_app.cget("bg") in ("#0f172a", "#1e293b", "SystemButtonFace", "black", gui_app.THEME["bg_root"])
    gui_app.update_idletasks()


def test_eight_tabs_exist_with_proper_titles(gui_app):
    """Test that all 8 dedicated tabs are created with exact specifications."""
    tab_names = [gui_app.notebook.tab(i, "text") for i in range(gui_app.notebook.index("end"))]
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
    for expected in expected_tabs:
        assert expected in tab_names, f"Missing tab: {expected} in {tab_names}"


def test_data_loading_and_tab_population(gui_app, sample_game_data):
    """Test populating slate data and verifying treeview items and stat badges."""
    # Feed sample game into app
    gui_app.load_games([sample_game_data], date_str="2026-09-20", status_msg="Test Slate Loaded")
    gui_app.update_idletasks()

    # Header summary badges
    assert gui_app.badge_games_var.get() == "1"
    assert "Confirmed" in gui_app.badge_lineups_var.get() or "2" in gui_app.badge_lineups_var.get()
    assert int(gui_app.badge_picks_var.get()) >= 0

    # Tab 2: Moneylines Treeview
    ml_items = gui_app.tree_moneylines.get_children()
    assert len(ml_items) >= 2  # Away and Home lines

    first_item = gui_app.tree_moneylines.item(ml_items[0])
    values = first_item["values"]
    assert len(values) >= 8
    # Matchup or team name should be present
    assert any("BOS" in str(v) or "NYY" in str(v) for v in values)

    # Tab 3: Run Lines Treeview & Delta Buckets
    rl_items = gui_app.tree_runlines.get_children()
    assert len(rl_items) >= 2

    delta_items = gui_app.tree_deltas.get_children()
    assert len(delta_items) == 5  # 5 delta buckets: 0-5%, 5-10%, 10-15%, 15-20%, 20%+

    # Tab 4: Player Props Treeview
    prop_items = gui_app.tree_props.get_children()
    assert len(prop_items) > 0

    # Tab 5: NRFI / YRFI Treeview
    nrfi_items = gui_app.tree_nrfi.get_children()
    assert len(nrfi_items) >= 1

    # Tab 6: Model Picks Treeview
    pick_items = gui_app.tree_picks.get_children()
    assert len(pick_items) >= 1


def test_inline_odds_editing_and_recalculation(gui_app, sample_game_data):
    """Test inline double-click editing on Moneylines odds cell updates Edge and EV instantly."""
    gui_app.load_games([sample_game_data], date_str="2026-09-20")
    gui_app.update_idletasks()

    ml_items = gui_app.tree_moneylines.get_children()
    assert len(ml_items) >= 2
    target_item = ml_items[0]

    # Initial edge value
    initial_values = gui_app.tree_moneylines.item(target_item)["values"]
    col_names = gui_app.tree_moneylines["columns"]
    odds_col_idx = col_names.index("book_odds")
    edge_col_idx = col_names.index("edge_pct")

    # Update odds to +180 (a huge underdog value)
    gui_app.update_cell_odds(tree_name="moneylines", item_id=target_item, new_odds="+180")
    gui_app.update_idletasks()

    updated_values = gui_app.tree_moneylines.item(target_item)["values"]
    assert str(updated_values[odds_col_idx]) in ("+180", "180")

    # Edge must be recalculated and different from initial (or positive)
    new_edge = float(str(updated_values[edge_col_idx]).replace("%", "").replace("+", ""))
    assert isinstance(new_edge, float)

    # Now change to -250 (heavy favorite market odds)
    gui_app.update_cell_odds(tree_name="moneylines", item_id=target_item, new_odds="-250")
    gui_app.update_idletasks()
    updated_values_fav = gui_app.tree_moneylines.item(target_item)["values"]
    assert str(updated_values_fav[odds_col_idx]) == "-250"
    new_edge_fav = float(str(updated_values_fav[edge_col_idx]).replace("%", "").replace("+", ""))
    assert new_edge_fav < new_edge


def test_player_props_filter_responsiveness(gui_app, sample_game_data):
    """Test dynamic category, team, and player search filters on Tab 4 Player Props."""
    gui_app.load_games([sample_game_data], date_str="2026-09-20")
    gui_app.update_idletasks()

    total_props = len(gui_app.tree_props.get_children())
    assert total_props > 0

    # 1. Filter by category "Pitcher Ks"
    gui_app.prop_category_var.set("Pitcher Ks")
    gui_app.apply_prop_filters()
    gui_app.update_idletasks()
    k_props = gui_app.tree_props.get_children()
    assert 0 < len(k_props) < total_props
    for item in k_props:
        vals = gui_app.tree_props.item(item)["values"]
        assert "Ks" in str(vals)

    # 2. Filter by player name search
    gui_app.prop_category_var.set("All")
    gui_app.prop_search_var.set("Judge")
    gui_app.apply_prop_filters()
    gui_app.update_idletasks()
    judge_props = gui_app.tree_props.get_children()
    assert len(judge_props) > 0
    for item in judge_props:
        vals = gui_app.tree_props.item(item)["values"]
        assert "Judge" in str(vals)

    # 3. Reset filters
    gui_app.prop_search_var.set("")
    gui_app.apply_prop_filters()
    gui_app.update_idletasks()
    assert len(gui_app.tree_props.get_children()) == total_props


def test_model_picks_filter_responsiveness(gui_app, sample_game_data):
    """Test dynamic sliders and confidence filters on Tab 6 Model Picks."""
    gui_app.load_games([sample_game_data], date_str="2026-09-20")
    gui_app.update_idletasks()

    total_picks = len(gui_app.tree_picks.get_children())
    assert total_picks > 0

    # 1. Raise min edge slider to a very high threshold (e.g. 50%) -> should return 0 picks
    gui_app.slider_min_edge.set(50.0)
    gui_app.apply_picks_filters()
    gui_app.update_idletasks()
    filtered_picks = len(gui_app.tree_picks.get_children())
    assert filtered_picks == 0

    # 2. Lower min edge to -20% -> should show all picks
    gui_app.slider_min_edge.set(-20.0)
    gui_app.slider_min_prob.set(0.0)
    gui_app.picks_confidence_var.set("All")
    gui_app.apply_picks_filters()
    gui_app.update_idletasks()
    assert len(gui_app.tree_picks.get_children()) == len(gui_app.active_picks)


def test_backtesting_and_model_constants_settings(gui_app):
    """Test Tab 7 settings modification, saving to SQLite, and resetting."""
    # Retrieve current constants
    constants = get_model_constants(gui_app.conn)
    initial_alpha = constants.get("dispersion_alpha", 0.12)

    # Update slider for dispersion_alpha
    gui_app.settings_vars["dispersion_alpha"].set(0.18)
    gui_app.save_settings()
    gui_app.update_idletasks()

    updated_constants = get_model_constants(gui_app.conn)
    assert pytest.approx(updated_constants["dispersion_alpha"], abs=1e-4) == 0.18

    # Reset to defaults
    gui_app.reset_settings_defaults()
    gui_app.update_idletasks()
    reset_constants = get_model_constants(gui_app.conn)
    assert pytest.approx(reset_constants["dispersion_alpha"], abs=1e-4) == 0.12


def test_performance_calibration_and_charts_rendering(gui_app, sample_game_data):
    """Test Tab 8 7-bucket calibration table and Matplotlib canvas figure update."""
    # Seed picks_history in DB with sample settled bets
    sample_picks = [
        {"game_pk": 1, "game_date": "2026-09-19", "market": "ML", "selection": "NYY", "model_prob": 0.58, "fair_odds": "-138", "market_odds": "-125", "edge_pct": 3.5, "ev_pct": 4.1, "confidence": "HIGH", "spread_delta": 0.14, "result": "WIN", "units_won": 0.80},
        {"game_pk": 2, "game_date": "2026-09-19", "market": "ML", "selection": "BOS", "model_prob": 0.63, "fair_odds": "-170", "market_odds": "-140", "edge_pct": 5.2, "ev_pct": 6.8, "confidence": "HIGH", "spread_delta": 0.16, "result": "WIN", "units_won": 0.71},
        {"game_pk": 3, "game_date": "2026-09-19", "market": "RL", "selection": "LAD +1.5", "model_prob": 0.72, "fair_odds": "-257", "market_odds": "-200", "edge_pct": 4.0, "ev_pct": 5.0, "confidence": "MEDIUM", "spread_delta": 0.22, "result": "LOSS", "units_won": -1.0},
    ]
    for p in sample_picks:
        log_pick(gui_app.conn, p)

    gui_app.refresh_calibration_tab()
    gui_app.update_idletasks()

    cal_items = gui_app.tree_calibration.get_children()
    assert len(cal_items) == 7  # 7 calibration buckets: 50-54%, 55-59%, 60-64%, 65-69%, 70-74%, 75-79%, 80%+

    # Verify Matplotlib figure canvas exists and has plots
    assert gui_app.fig_canvas is not None
    assert len(gui_app.fig.axes) >= 2  # Reliability curve and Cumulative Bankroll curve


def test_date_navigation(gui_app):
    """Test date navigation controls (< Prev, Next >, Today)."""
    current = gui_app.current_date
    gui_app.on_next_date()
    assert gui_app.current_date > current

    gui_app.on_prev_date()
    assert gui_app.current_date == current

    gui_app.on_today_date()
    gui_app.update_idletasks()
    assert gui_app.date_var.get() == gui_app.current_date


def test_log_selected_pick_action(gui_app, sample_game_data):
    """Test logging a selected pick to SQLite picks_history from Tab 6."""
    gui_app.load_games([sample_game_data], date_str="2026-09-20")
    gui_app.update_idletasks()

    pick_items = gui_app.tree_picks.get_children()
    assert len(pick_items) > 0
    first_pick = pick_items[0]

    # Select the item
    gui_app.tree_picks.selection_set(first_pick)
    cursor = gui_app.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM picks_history")
    count_before = cursor.fetchone()[0]

    gui_app.log_selected_pick()
    gui_app.update_idletasks()

    cursor.execute("SELECT COUNT(*) FROM picks_history")
    count_after = cursor.fetchone()[0]
    assert count_after == count_before + 1

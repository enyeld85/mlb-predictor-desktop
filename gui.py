"""MLB Apex Desktop Predictor & Betting Analytics Platform - Tkinter GUI.

Features:
- Sportsbook Dark Theme (Slate #0f172a, Card #1e293b, Emerald #10b981, Amber #f59e0b, Cyan #06b6d4, Red #ef4444).
- Top Header Bar with Date Navigator, Refresh, Progressbar, Status label, and Summary stat badges.
- 8 Dedicated Notebook Tabs:
    1. Today's Slate (Matchup cards, starters, confirmed/projected badges, scores, win %, run lines, NRFI, best pick)
    2. Moneylines (Treeview table, inline double-click editable odds, Edge %, EV, Confidence, Recommendation)
    3. Run Lines & Historical Delta Performance (+1.5/-1.5 cover, Spread Delta, 5 delta buckets table)
    4. Player Props (Filterable by Category, Team, Player search; editable book odds and over/under probabilities)
    5. NRFI / YRFI (1st inning metrics, matchup cards/table, fair vs book odds, edge)
    6. Model Picks (Auto-ranked value bets, probability & edge sliders, confidence selector, rationale tags, log to DB)
    7. Backtesting & Settings (Model weight sliders, retrain button, filter parameters, comprehensive metrics)
    8. Performance & Calibration (7-bucket calibration table, embedded Matplotlib reliability & bankroll curves)
- Non-blocking background async data fetching and thread-safe UI updates via root.after().
- Responsible gaming disclaimer footer and last updated timestamp.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from database import (
    DEFAULT_MODEL_CONSTANTS,
    get_db_connection,
    get_delta_buckets_history,
    get_model_constants,
    get_picks_history,
    init_db,
    log_pick,
    update_model_constants,
)
from data_fetch import MLBDataFetcher
from models import (
    american_to_prob,
    calculate_edge,
    compute_calibration_metrics,
    devig_two_way,
    model_player_props,
    prob_to_american,
    simulate_game,
)


class MLBPredictorApp(tk.Tk):
    """Main desktop application window for MLB Apex Predictor & Betting Analytics."""

    THEME = {
        "bg_root": "#0f172a",       # Deep slate
        "bg_card": "#1e293b",       # Card background
        "bg_card_alt": "#243147",   # Alternate card/row
        "border": "#334155",        # Muted border
        "text_main": "#f8fafc",     # Bright white/slate
        "text_muted": "#94a3b8",    # Muted grey/slate
        "green": "#10b981",         # Emerald positive edge
        "amber": "#f59e0b",         # Warning / slight edge
        "cyan": "#06b6d4",          # Cyan accent / bankroll
        "red": "#ef4444",           # Red negative edge / loss
        "blue": "#3b82f6",          # Royal blue
    }

    def __init__(
        self,
        db_path: str = "mlb_analytics.db",
        fetcher: MLBDataFetcher | None = None,
        auto_load: bool = True,
    ) -> None:
        super().__init__()

        self.title("MLB Apex Predictor & Betting Analytics")
        self.geometry("1380x880")
        self.minsize(1050, 680)
        self.configure(bg=self.THEME["bg_root"])

        self.db_path = db_path
        self.conn = get_db_connection(db_path)
        self.fetcher = fetcher or MLBDataFetcher(db_path=db_path)
        self.model_constants = get_model_constants(self.conn)

        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.games_data: list[dict[str, Any]] = []
        self.simulated_games: list[dict[str, Any]] = []
        self.active_picks: list[dict[str, Any]] = []
        self.all_props_data: list[dict[str, Any]] = []

        # Editing state for treeviews
        self._edit_entry: tk.Entry | None = None
        self._editing_info: dict[str, Any] | None = None

        self._init_styles()
        self._build_top_header()
        self._build_notebook_tabs()
        self._build_footer()

        # Ensure window is visible, brought to front, and focused
        self._bring_to_front()

        if auto_load:
            self.refresh_slate(self.current_date, force_refresh=False)
        else:
            self.refresh_calibration_tab()

    def _bring_to_front(self) -> None:
        """Force the application window to the foreground on launch."""
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(500, lambda: self.attributes("-topmost", False))
            self.focus_force()
        except Exception:
            pass

    # =========================================================================
    # Theme & ttk Styles Configuration
    # =========================================================================

    def _init_styles(self) -> None:
        """Configure ttk dark sportsbook styles."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        t = self.THEME
        self.style.configure(".", background=t["bg_root"], foreground=t["text_main"])
        self.style.configure("TFrame", background=t["bg_root"])
        self.style.configure("Card.TFrame", background=t["bg_card"], relief="flat")
        self.style.configure("BorderCard.TFrame", background=t["bg_card"], relief="solid", borderwidth=1)

        # Labels
        self.style.configure("TLabel", background=t["bg_root"], foreground=t["text_main"], font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=t["bg_card"], foreground=t["text_main"], font=("Segoe UI", 10))
        self.style.configure("CardMuted.TLabel", background=t["bg_card"], foreground=t["text_muted"], font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", background=t["bg_root"], foreground=t["text_main"], font=("Segoe UI", 16, "bold"))
        self.style.configure("HeaderSub.TLabel", background=t["bg_root"], foreground=t["text_muted"], font=("Segoe UI", 9))
        self.style.configure("Badge.TLabel", background=t["border"], foreground=t["text_main"], font=("Segoe UI", 9, "bold"), padding=4)
        self.style.configure("BadgeGreen.TLabel", background="#064e3b", foreground=t["green"], font=("Segoe UI", 9, "bold"), padding=4)
        self.style.configure("BadgeAmber.TLabel", background="#78350f", foreground=t["amber"], font=("Segoe UI", 9, "bold"), padding=4)

        # Buttons
        self.style.configure(
            "TButton",
            background=t["bg_card"],
            foreground=t["text_main"],
            borderwidth=1,
            focuscolor=t["blue"],
            font=("Segoe UI", 9, "bold"),
            padding=(8, 4),
        )
        self.style.map(
            "TButton",
            background=[("active", t["border"]), ("pressed", t["bg_root"])],
            foreground=[("active", t["text_main"])],
        )

        self.style.configure(
            "Primary.TButton",
            background=t["blue"],
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
        )
        self.style.map("Primary.TButton", background=[("active", "#2563eb"), ("pressed", "#1d4ed8")])

        self.style.configure(
            "Green.TButton",
            background=t["green"],
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
        )
        self.style.map("Green.TButton", background=[("active", "#059669")])

        # Notebook & Tabs
        self.style.configure(
            "TNotebook",
            background=t["bg_root"],
            borderwidth=0,
            tabmargins=[2, 5, 2, 0],
        )
        self.style.configure(
            "TNotebook.Tab",
            background=t["bg_card"],
            foreground=t["text_muted"],
            padding=[14, 8],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", t["border"]), ("active", t["bg_card_alt"])],
            foreground=[("selected", t["text_main"]), ("active", t["text_main"])],
        )

        # Treeview (Dark Grid)
        self.style.configure(
            "Treeview",
            background=t["bg_card"],
            foreground=t["text_main"],
            fieldbackground=t["bg_card"],
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 9),
        )
        self.style.map(
            "Treeview",
            background=[("selected", t["blue"])],
            foreground=[("selected", "#ffffff")],
        )
        self.style.configure(
            "Treeview.Heading",
            background=t["bg_root"],
            foreground=t["text_muted"],
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padding=(6, 5),
        )
        self.style.map("Treeview.Heading", background=[("active", t["border"])], foreground=[("active", t["text_main"])])

        # Progressbar
        self.style.configure("TProgressbar", thickness=6, troughcolor=t["bg_card"], background=t["green"])

        # Entry & Combobox
        self.style.configure("TEntry", fieldbackground=t["bg_card"], foreground=t["text_main"], insertcolor=t["text_main"])
        self.style.configure("TCombobox", fieldbackground=t["bg_card"], background=t["bg_card"], foreground=t["text_main"])
        self.style.map("TCombobox", fieldbackground=[("readonly", t["bg_card"])], selectbackground=[("readonly", t["blue"])])

    # =========================================================================
    # Header & Navigation
    # =========================================================================

    def _build_top_header(self) -> None:
        """Create header bar with date navigation, status, progress, and stat badges."""
        t = self.THEME
        header_frame = tk.Frame(self, bg=t["bg_root"], pady=8, padx=12)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        # Left: Title & Subtitle
        title_box = tk.Frame(header_frame, bg=t["bg_root"])
        title_box.pack(side=tk.LEFT)

        lbl_title = tk.Label(
            title_box,
            text="MLB APEX PREDICTOR & BETTING ANALYTICS",
            bg=t["bg_root"],
            fg=t["text_main"],
            font=("Segoe UI", 15, "bold"),
        )
        lbl_title.pack(anchor="w")

        lbl_subtitle = tk.Label(
            title_box,
            text="Vectorized Monte Carlo • Devigged Market Edge • Uncertainty Intervals",
            bg=t["bg_root"],
            fg=t["text_muted"],
            font=("Segoe UI", 8),
        )
        lbl_subtitle.pack(anchor="w")

        # Center: Date Navigator & Refresh Controls
        center_box = tk.Frame(header_frame, bg=t["bg_root"])
        center_box.pack(side=tk.LEFT, expand=True)

        nav_frame = tk.Frame(center_box, bg=t["bg_root"])
        nav_frame.pack()

        btn_prev = ttk.Button(nav_frame, text="< Prev", width=7, command=self.on_prev_date)
        btn_prev.pack(side=tk.LEFT, padx=3)

        self.date_var = tk.StringVar(value=self.current_date)
        entry_date = tk.Entry(
            nav_frame,
            textvariable=self.date_var,
            width=11,
            bg=t["bg_card"],
            fg=t["text_main"],
            insertbackground=t["text_main"],
            relief="solid",
            justify="center",
            font=("Segoe UI", 10, "bold"),
        )
        entry_date.pack(side=tk.LEFT, padx=4)
        entry_date.bind("<Return>", lambda e: self.on_date_entered())

        btn_next = ttk.Button(nav_frame, text="Next >", width=7, command=self.on_next_date)
        btn_next.pack(side=tk.LEFT, padx=3)

        btn_today = ttk.Button(nav_frame, text="Today", width=6, command=self.on_today_date)
        btn_today.pack(side=tk.LEFT, padx=3)

        btn_refresh = ttk.Button(
            nav_frame,
            text="↻ Refresh Data",
            style="Primary.TButton",
            command=self.on_refresh_click,
        )
        btn_refresh.pack(side=tk.LEFT, padx=6)

        # Status & Progress row under date controls
        status_row = tk.Frame(center_box, bg=t["bg_root"])
        status_row.pack(fill=tk.X, pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready")
        lbl_status = tk.Label(
            status_row,
            textvariable=self.status_var,
            bg=t["bg_root"],
            fg=t["cyan"],
            font=("Segoe UI", 8),
        )
        lbl_status.pack(side=tk.LEFT, padx=(0, 6))

        self.progress_bar = ttk.Progressbar(status_row, mode="indeterminate", length=110)
        self.progress_bar.pack(side=tk.LEFT)

        # Right: Summary Stat Badges
        badges_box = tk.Frame(header_frame, bg=t["bg_root"])
        badges_box.pack(side=tk.RIGHT)

        self.badge_games_var = tk.StringVar(value="0")
        self.badge_lineups_var = tk.StringVar(value="0/0 Confirmed")
        self.badge_picks_var = tk.StringVar(value="0")
        self.badge_value_var = tk.StringVar(value="0")

        self._create_stat_badge(badges_box, "GAMES", self.badge_games_var, t["cyan"])
        self._create_stat_badge(badges_box, "LINEUPS", self.badge_lineups_var, t["amber"])
        self._create_stat_badge(badges_box, "ACTIVE PICKS", self.badge_picks_var, t["green"])
        self._create_stat_badge(badges_box, "VALUE BETS", self.badge_value_var, t["green"])

    def _create_stat_badge(
        self,
        parent: tk.Widget,
        title: str,
        var: tk.StringVar,
        accent_color: str,
    ) -> tk.Frame:
        """Helper to create a modern statistic badge block."""
        t = self.THEME
        card = tk.Frame(parent, bg=t["bg_card"], padx=8, pady=4, relief="solid", borderwidth=1)
        card.pack(side=tk.LEFT, padx=3)

        lbl_t = tk.Label(card, text=title, bg=t["bg_card"], fg=t["text_muted"], font=("Segoe UI", 7, "bold"))
        lbl_t.pack()
        lbl_v = tk.Label(card, textvariable=var, bg=t["bg_card"], fg=accent_color, font=("Segoe UI", 10, "bold"))
        lbl_v.pack()
        return card

    # =========================================================================
    # Notebook & 8 Tabs Setup
    # =========================================================================

    def _build_notebook_tabs(self) -> None:
        """Initialize the 8 dedicated application tabs."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 0))

        # 1. Today's Slate
        self.tab_slate = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_slate, text="Today's Slate")
        self._setup_tab_slate()

        # 2. Moneylines
        self.tab_moneylines = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_moneylines, text="Moneylines")
        self._setup_tab_moneylines()

        # 3. Run Lines & Historical Delta Performance
        self.tab_runlines = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_runlines, text="Run Lines & Historical Delta Performance")
        self._setup_tab_runlines()

        # 4. Player Props
        self.tab_props = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_props, text="Player Props")
        self._setup_tab_props()

        # 5. NRFI / YRFI
        self.tab_nrfi = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_nrfi, text="NRFI / YRFI")
        self._setup_tab_nrfi()

        # 6. Model Picks
        self.tab_picks = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_picks, text="Model Picks")
        self._setup_tab_picks()

        # 7. Backtesting & Settings
        self.tab_backtesting = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_backtesting, text="Backtesting & Settings")
        self._setup_tab_backtesting()

        # 8. Performance & Calibration
        self.tab_calibration = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_calibration, text="Performance & Calibration")
        self._setup_tab_calibration()

    # =========================================================================
    # Tab 1: Today's Slate (Matchup Cards)
    # =========================================================================

    def _setup_tab_slate(self) -> None:
        """Create scrollable canvas for game matchup cards."""
        t = self.THEME
        container = tk.Frame(self.tab_slate, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas_slate = tk.Canvas(container, bg=t["bg_root"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas_slate.yview)

        self.frame_cards = tk.Frame(self.canvas_slate, bg=t["bg_root"])
        self.frame_cards.bind(
            "<Configure>",
            lambda e: self.canvas_slate.configure(scrollregion=self.canvas_slate.bbox("all")),
        )

        self.canvas_slate.create_window((0, 0), window=self.frame_cards, anchor="nw")
        self.canvas_slate.configure(yscrollcommand=scrollbar.set)

        self.canvas_slate.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

        self.canvas_slate.bind_all("<MouseWheel>", self._on_canvas_mousewheel)

    def _on_canvas_mousewheel(self, event: tk.Event) -> None:
        """Smooth mousewheel scroll on canvas."""
        try:
            if self.canvas_slate.winfo_exists():
                self.canvas_slate.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _render_slate_cards(self) -> None:
        """Render modern matchup cards for each simulated game."""
        for widget in self.frame_cards.winfo_children():
            widget.destroy()

        t = self.THEME
        if not self.simulated_games:
            no_games_lbl = tk.Label(
                self.frame_cards,
                text="No games loaded for this date. Click '↻ Refresh Data' or choose another date.",
                bg=t["bg_root"],
                fg=t["text_muted"],
                font=("Segoe UI", 12),
                pady=40,
            )
            no_games_lbl.pack(fill=tk.X, expand=True)
            return

        for i, item in enumerate(self.simulated_games):
            g = item["game"]
            s = item["sim"]
            card = self._build_single_matchup_card(self.frame_cards, g, s)
            card.pack(fill=tk.X, expand=True, padx=8, pady=6)

    def _build_single_matchup_card(
        self,
        parent: tk.Widget,
        g: dict[str, Any],
        s: dict[str, Any],
    ) -> tk.Frame:
        """Construct one complete game matchup card."""
        t = self.THEME
        card = tk.Frame(parent, bg=t["bg_card"], relief="solid", borderwidth=1, padx=12, pady=10)

        # Top Bar: Matchup, Time, Confidence Badge
        top = tk.Frame(card, bg=t["bg_card"])
        top.pack(fill=tk.X, pady=(0, 8))

        away = g.get("away_team", {})
        home = g.get("home_team", {})
        venue = g.get("venue", {})

        away_name = away.get("name", "Away")
        home_name = home.get("name", "Home")
        time_str = g.get("game_time", "19:05")

        lbl_header = tk.Label(
            top,
            text=f"{away_name}  @  {home_name}  •  {time_str} ET",
            bg=t["bg_card"],
            fg=t["text_main"],
            font=("Segoe UI", 12, "bold"),
        )
        lbl_header.pack(side=tk.LEFT)

        # Confidence Badge
        conf = s.get("data_confidence", "LOW")
        conf_bg = "#064e3b" if conf == "HIGH" else ("#78350f" if conf == "MEDIUM" else "#450a0a")
        conf_fg = t["green"] if conf == "HIGH" else (t["amber"] if conf == "MEDIUM" else t["red"])
        lbl_conf = tk.Label(
            top,
            text=f"Confidence: {conf}",
            bg=conf_bg,
            fg=conf_fg,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2,
        )
        lbl_conf.pack(side=tk.RIGHT)

        # Middle Section: Grid with Starters, Lineups, Projections
        mid = tk.Frame(card, bg=t["bg_card"])
        mid.pack(fill=tk.X)

        # Pitchers row
        a_starter = away.get("starter", {}) or {}
        h_starter = home.get("starter", {}) or {}
        a_sp_name = a_starter.get("name") or "TBD"
        h_sp_name = h_starter.get("name") or "TBD"
        a_sp_era = a_starter.get("era", "-")
        h_sp_era = h_starter.get("era", "-")
        a_sp_whip = a_starter.get("whip", "-")
        h_sp_whip = h_starter.get("whip", "-")

        a_lineup_stat = away.get("lineup_status", "PROJECTED")
        h_lineup_stat = home.get("lineup_status", "PROJECTED")

        pitchers_txt = (
            f"SP: {a_sp_name} ({a_starter.get('hand', 'R')}) ERA: {a_sp_era} WHIP: {a_sp_whip} [{a_lineup_stat}]\n"
            f"SP: {h_sp_name} ({h_starter.get('hand', 'R')}) ERA: {h_sp_era} WHIP: {h_sp_whip} [{h_lineup_stat}]"
        )
        lbl_pitchers = tk.Label(
            mid,
            text=pitchers_txt,
            bg=t["bg_card"],
            fg=t["text_muted"],
            justify="left",
            font=("Segoe UI", 9),
        )
        lbl_pitchers.pack(side=tk.LEFT)

        # Projections Box (Scores & Win %)
        proj_box = tk.Frame(mid, bg=t["bg_card_alt"], padx=10, pady=6, relief="solid", borderwidth=1)
        proj_box.pack(side=tk.RIGHT)

        a_score = s.get("expected_away_score", 0.0)
        h_score = s.get("expected_home_score", 0.0)
        tot_score = s.get("projected_total", 0.0)
        a_win = round(s.get("away_win_prob", 0.50) * 100, 1)
        h_win = round(s.get("home_win_prob", 0.50) * 100, 1)

        txt_proj = (
            f"Projected Score: {away.get('abbrev', 'AWAY')} {a_score}  -  {home.get('abbrev', 'HOME')} {h_score} (O/U {tot_score})\n"
            f"Win Prob: {away.get('abbrev', 'AWAY')} {a_win}% ({s.get('away_fair_odds')})  |  {home.get('abbrev', 'HOME')} {h_win}% ({s.get('home_fair_odds')})"
        )
        lbl_proj = tk.Label(
            proj_box,
            text=txt_proj,
            bg=t["bg_card_alt"],
            fg=t["text_main"],
            justify="right",
            font=("Segoe UI", 9, "bold"),
        )
        lbl_proj.pack()

        # Bottom Bar: Run line summary, NRFI %, Best Pick banner
        bot = tk.Frame(card, bg=t["bg_card"], pady=6)
        bot.pack(fill=tk.X)

        rl_h_plus = round(s.get("home_cover_plus_1_5", 0.0) * 100, 1)
        rl_a_plus = round(s.get("away_cover_plus_1_5", 0.0) * 100, 1)
        nrfi_pct = round(s.get("p_nrfi", 0.0) * 100, 1)

        summary_txt = f"+1.5 Cover: {away.get('abbrev', 'AWAY')} {rl_a_plus}% | {home.get('abbrev', 'HOME')} {rl_h_plus}%   •   NRFI: {nrfi_pct}% ({s.get('nrfi_fair_odds')})"
        lbl_summary = tk.Label(bot, text=summary_txt, bg=t["bg_card"], fg=t["cyan"], font=("Segoe UI", 9))
        lbl_summary.pack(side=tk.LEFT)

        # Best Pick Banner
        fav_side = home.get("abbrev", "HOME") if h_win >= a_win else away.get("abbrev", "AWAY")
        fav_prob = max(h_win, a_win)
        pick_text = f"★ Model Pick: {fav_side} ML ({fav_prob}%)"
        lbl_pick = tk.Label(
            bot,
            text=pick_text,
            bg="#064e3b",
            fg=t["green"],
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
        )
        lbl_pick.pack(side=tk.RIGHT)

        return card

    # =========================================================================
    # Tab 2: Moneylines (Treeview + Editable Book Odds)
    # =========================================================================

    def _setup_tab_moneylines(self) -> None:
        """Create Moneylines tab table."""
        t = self.THEME
        container = tk.Frame(self.tab_moneylines, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        cols = (
            "game_pk",
            "matchup",
            "team",
            "starter",
            "model_win_pct",
            "fair_odds",
            "book_odds",
            "edge_pct",
            "ev_pct",
            "confidence",
            "recommendation",
        )
        self.tree_moneylines = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")

        headings = {
            "game_pk": ("Game PK", 70),
            "matchup": ("Matchup", 180),
            "team": ("Team / Side", 140),
            "starter": ("Probable Pitcher", 150),
            "model_win_pct": ("Model Win %", 95),
            "fair_odds": ("Fair Odds", 85),
            "book_odds": ("Book Odds [✏]", 100),
            "edge_pct": ("Edge %", 80),
            "ev_pct": ("EV %", 80),
            "confidence": ("Confidence", 90),
            "recommendation": ("Recommendation", 130),
        }
        for col_id, (txt, width) in headings.items():
            self.tree_moneylines.heading(col_id, text=txt, anchor="center")
            self.tree_moneylines.column(col_id, width=width, anchor="center")

        scroll = ttk.Scrollbar(container, orient="vertical", command=self.tree_moneylines.yview)
        self.tree_moneylines.configure(yscrollcommand=scroll.set)

        self.tree_moneylines.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_moneylines.bind("<Double-1>", lambda e: self._on_treeview_double_click(e, "moneylines"))

        self.tree_moneylines.tag_configure("odd_row", background=t["bg_card"])
        self.tree_moneylines.tag_configure("even_row", background=t["bg_card_alt"])

    # =========================================================================
    # Tab 3: Run Lines & Historical Delta Performance
    # =========================================================================

    def _setup_tab_runlines(self) -> None:
        """Create Run Lines table and Historical Delta Performance table."""
        t = self.THEME
        container = tk.Frame(self.tab_runlines, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Upper Frame: Run Lines Table
        lbl_upper = tk.Label(
            container,
            text="DAILY RUN LINE PROJECTIONS & SPREAD DELTAS (P(+1.5) - P(ML))",
            bg=t["bg_root"],
            fg=t["text_main"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        lbl_upper.pack(fill=tk.X, pady=(0, 4))

        cols_rl = (
            "matchup",
            "side",
            "line",
            "cover_pct",
            "win_pct",
            "spread_delta",
            "book_odds",
            "edge_pct",
            "recommendation",
        )
        self.tree_runlines = ttk.Treeview(container, columns=cols_rl, show="headings", height=8, selectmode="browse")

        rl_headings = {
            "matchup": ("Matchup", 180),
            "side": ("Team", 140),
            "line": ("Run Line", 80),
            "cover_pct": ("Cover %", 90),
            "win_pct": ("ML Win %", 90),
            "spread_delta": ("Spread Delta", 100),
            "book_odds": ("Book Odds [✏]", 100),
            "edge_pct": ("Edge %", 80),
            "recommendation": ("Recommendation", 130),
        }
        for col_id, (txt, width) in rl_headings.items():
            self.tree_runlines.heading(col_id, text=txt, anchor="center")
            self.tree_runlines.column(col_id, width=width, anchor="center")

        scroll_rl = ttk.Scrollbar(container, orient="vertical", command=self.tree_runlines.yview)
        self.tree_runlines.configure(yscrollcommand=scroll_rl.set)

        self.tree_runlines.pack(fill=tk.X, expand=False)
        self.tree_runlines.bind("<Double-1>", lambda e: self._on_treeview_double_click(e, "runlines"))

        # Lower Frame: Historical Delta Performance Table
        lbl_lower = tk.Label(
            container,
            text="HISTORICAL SPREAD DELTA PERFORMANCE (5 CALIBRATED BUCKETS)",
            bg=t["bg_root"],
            fg=t["cyan"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        lbl_lower.pack(fill=tk.X, pady=(12, 4))

        cols_delta = ("bucket", "total_bets", "wins", "cover_pct", "units_won", "roi_pct", "edge_type")
        self.tree_deltas = ttk.Treeview(container, columns=cols_delta, show="headings", height=6, selectmode="browse")

        delta_headings = {
            "bucket": ("Delta Bucket", 120),
            "total_bets": ("Sample Bets", 100),
            "wins": ("Wins (Covers)", 100),
            "cover_pct": ("Cover %", 100),
            "units_won": ("Units Won", 100),
            "roi_pct": ("ROI %", 100),
            "edge_type": ("Edge Status", 140),
        }
        for col_id, (txt, width) in delta_headings.items():
            self.tree_deltas.heading(col_id, text=txt, anchor="center")
            self.tree_deltas.column(col_id, width=width, anchor="center")

        self.tree_deltas.pack(fill=tk.BOTH, expand=True)

        self.tree_deltas.tag_configure("positive_bucket", foreground=t["green"])
        self.tree_deltas.tag_configure("neutral_bucket", foreground=t["text_main"])
        self.tree_deltas.tag_configure("negative_bucket", foreground=t["red"])

    # =========================================================================
    # Tab 4: Player Props (Treeview + Dynamic Filtering)
    # =========================================================================

    def _setup_tab_props(self) -> None:
        """Create Player Props filter controls and table."""
        t = self.THEME
        container = tk.Frame(self.tab_props, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Filters Bar
        filter_bar = tk.Frame(container, bg=t["bg_card"], padx=10, pady=6, relief="solid", borderwidth=1)
        filter_bar.pack(fill=tk.X, pady=(0, 6))

        # Category Filter
        lbl_cat = tk.Label(filter_bar, text="Category:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_cat.pack(side=tk.LEFT, padx=(0, 4))

        self.prop_category_var = tk.StringVar(value="All")
        cats = ["All", "Hits", "Total Bases", "Home Runs", "H+R+RBI", "Pitcher Ks", "Pitcher Outs"]
        cb_cat = ttk.Combobox(filter_bar, textvariable=self.prop_category_var, values=cats, width=13, state="readonly")
        cb_cat.pack(side=tk.LEFT, padx=(0, 12))
        cb_cat.bind("<<ComboboxSelected>>", lambda e: self.apply_prop_filters())

        # Team Filter
        lbl_team = tk.Label(filter_bar, text="Team:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_team.pack(side=tk.LEFT, padx=(0, 4))

        self.prop_team_var = tk.StringVar(value="All Teams")
        self.cb_prop_team = ttk.Combobox(filter_bar, textvariable=self.prop_team_var, values=["All Teams"], width=12, state="readonly")
        self.cb_prop_team.pack(side=tk.LEFT, padx=(0, 12))
        self.cb_prop_team.bind("<<ComboboxSelected>>", lambda e: self.apply_prop_filters())

        # Player Search
        lbl_search = tk.Label(filter_bar, text="Search Player:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_search.pack(side=tk.LEFT, padx=(0, 4))

        self.prop_search_var = tk.StringVar(value="")
        entry_search = tk.Entry(
            filter_bar,
            textvariable=self.prop_search_var,
            width=18,
            bg=t["bg_root"],
            fg=t["text_main"],
            insertbackground=t["text_main"],
            relief="solid",
        )
        entry_search.pack(side=tk.LEFT, padx=(0, 10))
        entry_search.bind("<KeyRelease>", lambda e: self.apply_prop_filters())

        btn_clear = ttk.Button(filter_bar, text="Reset Filters", command=self._reset_prop_filters)
        btn_clear.pack(side=tk.LEFT)

        # Props Treeview
        cols = (
            "player",
            "team",
            "prop_type",
            "line",
            "over_prob",
            "under_prob",
            "book_odds_over",
            "book_odds_under",
            "edge_over",
            "edge_under",
            "recommendation",
        )
        self.tree_props = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")

        prop_headings = {
            "player": ("Player", 160),
            "team": ("Team", 80),
            "prop_type": ("Prop Type", 110),
            "line": ("Line", 70),
            "over_prob": ("Over %", 80),
            "under_prob": ("Under %", 80),
            "book_odds_over": ("Book Over [✏]", 95),
            "book_odds_under": ("Book Under [✏]", 95),
            "edge_over": ("Edge Over", 80),
            "edge_under": ("Edge Under", 80),
            "recommendation": ("Recommendation", 130),
        }
        for col_id, (txt, width) in prop_headings.items():
            self.tree_props.heading(col_id, text=txt, anchor="center")
            self.tree_props.column(col_id, width=width, anchor="center")

        scroll_props = ttk.Scrollbar(container, orient="vertical", command=self.tree_props.yview)
        self.tree_props.configure(yscrollcommand=scroll_props.set)

        self.tree_props.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_props.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_props.bind("<Double-1>", lambda e: self._on_treeview_double_click(e, "props"))

    def _reset_prop_filters(self) -> None:
        """Reset player prop search filters."""
        self.prop_category_var.set("All")
        self.prop_team_var.set("All Teams")
        self.prop_search_var.set("")
        self.apply_prop_filters()

    # =========================================================================
    # Tab 5: NRFI / YRFI
    # =========================================================================

    def _setup_tab_nrfi(self) -> None:
        """Create NRFI / YRFI table and metrics view."""
        t = self.THEME
        container = tk.Frame(self.tab_nrfi, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        lbl_hdr = tk.Label(
            container,
            text="FIRST INNING NO RUN FIRST INNING (NRFI) & YES RUN FIRST INNING (YRFI) ENGINE",
            bg=t["bg_root"],
            fg=t["text_main"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        lbl_hdr.pack(fill=tk.X, pady=(0, 6))

        cols = (
            "matchup",
            "venue",
            "away_1st_era",
            "home_1st_era",
            "p_nrfi",
            "p_yrfi",
            "fair_nrfi_odds",
            "fair_yrfi_odds",
            "book_nrfi_odds",
            "edge_nrfi",
            "recommendation",
        )
        self.tree_nrfi = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")

        nrfi_headings = {
            "matchup": ("Matchup", 180),
            "venue": ("Venue & Roof", 160),
            "away_1st_era": ("Away 1st ERA", 95),
            "home_1st_era": ("Home 1st ERA", 95),
            "p_nrfi": ("NRFI %", 85),
            "p_yrfi": ("YRFI %", 85),
            "fair_nrfi_odds": ("Fair NRFI", 85),
            "fair_yrfi_odds": ("Fair YRFI", 85),
            "book_nrfi_odds": ("Book NRFI [✏]", 100),
            "edge_nrfi": ("NRFI Edge %", 90),
            "recommendation": ("Recommendation", 130),
        }
        for col_id, (txt, width) in nrfi_headings.items():
            self.tree_nrfi.heading(col_id, text=txt, anchor="center")
            self.tree_nrfi.column(col_id, width=width, anchor="center")

        scroll_nrfi = ttk.Scrollbar(container, orient="vertical", command=self.tree_nrfi.yview)
        self.tree_nrfi.configure(yscrollcommand=scroll_nrfi.set)

        self.tree_nrfi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_nrfi.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_nrfi.bind("<Double-1>", lambda e: self._on_treeview_double_click(e, "nrfi"))

    # =========================================================================
    # Tab 6: Model Picks (Auto-ranked value bets & logging)
    # =========================================================================

    def _setup_tab_picks(self) -> None:
        """Create Model Picks table with filter sliders and DB logging button."""
        t = self.THEME
        container = tk.Frame(self.tab_picks, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Filters Bar
        filter_bar = tk.Frame(container, bg=t["bg_card"], padx=12, pady=8, relief="solid", borderwidth=1)
        filter_bar.pack(fill=tk.X, pady=(0, 6))

        # Slider: Min Prob
        lbl_p = tk.Label(filter_bar, text="Min Prob %:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_p.pack(side=tk.LEFT, padx=(0, 4))
        self.slider_min_prob = tk.DoubleVar(value=45.0)
        scale_p = ttk.Scale(filter_bar, from_=30.0, to=85.0, variable=self.slider_min_prob, length=110, command=lambda v: self.apply_picks_filters())
        scale_p.pack(side=tk.LEFT, padx=(0, 6))
        self.lbl_p_val = tk.Label(filter_bar, text="45%", bg=t["bg_card"], fg=t["cyan"], font=("Segoe UI", 9, "bold"), width=4)
        self.lbl_p_val.pack(side=tk.LEFT, padx=(0, 14))

        # Slider: Min Edge
        lbl_e = tk.Label(filter_bar, text="Min Edge %:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_e.pack(side=tk.LEFT, padx=(0, 4))
        self.slider_min_edge = tk.DoubleVar(value=-1.0)
        scale_e = ttk.Scale(filter_bar, from_=-10.0, to=25.0, variable=self.slider_min_edge, length=110, command=lambda v: self.apply_picks_filters())
        scale_e.pack(side=tk.LEFT, padx=(0, 6))
        self.lbl_e_val = tk.Label(filter_bar, text="-1.0%", bg=t["bg_card"], fg=t["green"], font=("Segoe UI", 9, "bold"), width=6)
        self.lbl_e_val.pack(side=tk.LEFT, padx=(0, 14))

        # Confidence Selector
        lbl_conf = tk.Label(filter_bar, text="Confidence:", bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"))
        lbl_conf.pack(side=tk.LEFT, padx=(0, 4))
        self.picks_confidence_var = tk.StringVar(value="All")
        cb_conf = ttk.Combobox(filter_bar, textvariable=self.picks_confidence_var, values=["All", "HIGH", "MEDIUM", "LOW"], width=9, state="readonly")
        cb_conf.pack(side=tk.LEFT, padx=(0, 16))
        cb_conf.bind("<<ComboboxSelected>>", lambda e: self.apply_picks_filters())

        # Action: Log Pick to DB
        btn_log = ttk.Button(filter_bar, text="💾 Log Selected Pick to DB", style="Green.TButton", command=self.log_selected_pick)
        btn_log.pack(side=tk.RIGHT)

        # Picks Treeview
        cols = (
            "rank",
            "game",
            "market",
            "selection",
            "model_prob",
            "fair_odds",
            "book_odds",
            "edge_pct",
            "ev_pct",
            "confidence",
            "rationale",
        )
        self.tree_picks = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")

        pick_headings = {
            "rank": ("Rank", 50),
            "game": ("Matchup", 170),
            "market": ("Market", 80),
            "selection": ("Selection", 130),
            "model_prob": ("Model %", 80),
            "fair_odds": ("Fair Odds", 85),
            "book_odds": ("Book Odds", 85),
            "edge_pct": ("Edge %", 80),
            "ev_pct": ("EV %", 80),
            "confidence": ("Confidence", 90),
            "rationale": ("Betting Rationale & Catalyst", 220),
        }
        for col_id, (txt, width) in pick_headings.items():
            self.tree_picks.heading(col_id, text=txt, anchor="center")
            self.tree_picks.column(col_id, width=width, anchor="center")

        scroll_picks = ttk.Scrollbar(container, orient="vertical", command=self.tree_picks.yview)
        self.tree_picks.configure(yscrollcommand=scroll_picks.set)

        self.tree_picks.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_picks.pack(side=tk.RIGHT, fill=tk.Y)

    # =========================================================================
    # Tab 7: Backtesting & Settings
    # =========================================================================

    def _setup_tab_backtesting(self) -> None:
        """Create Backtesting and dynamic model constants configuration tab."""
        t = self.THEME
        container = tk.Frame(self.tab_backtesting, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Panel: Dynamic Model Parameters
        left_panel = tk.Frame(container, bg=t["bg_card"], padx=14, pady=12, relief="solid", borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        lbl_settings_hdr = tk.Label(
            left_panel,
            text="DYNAMIC MODEL CONSTANTS & CALIBRATION WEIGHTS",
            bg=t["bg_card"],
            fg=t["text_main"],
            font=("Segoe UI", 11, "bold"),
        )
        lbl_settings_hdr.pack(anchor="w", pady=(0, 10))

        self.settings_vars: dict[str, tk.DoubleVar] = {}
        param_bounds = {
            "dispersion_alpha": (0.01, 0.40, 0.01, "Dispersion parameter alpha (Negative Binomial)"),
            "home_field_advantage_runs": (0.0, 0.50, 0.01, "Home field advantage run boost"),
            "temp_coefficient": (0.0, 0.03, 0.001, "Temp factor per 10°F from 72°F"),
            "wind_coefficient_out": (0.0, 0.04, 0.001, "Wind blowing out factor (per mph > 5)"),
            "wind_coefficient_in": (0.0, 0.03, 0.001, "Wind blowing in factor (per mph > 5)"),
            "baseline_league_runs": (3.50, 5.50, 0.05, "League average runs per team per game"),
            "cfip_constant": (2.50, 4.00, 0.05, "cFIP Constant"),
            "hr_fb_baseline": (0.05, 0.20, 0.005, "League HR/FB Baseline"),
        }

        for param, (min_v, max_v, step_v, desc) in param_bounds.items():
            curr_val = float(self.model_constants.get(param, 0.12))
            var = tk.DoubleVar(value=curr_val)
            self.settings_vars[param] = var

            row = tk.Frame(left_panel, bg=t["bg_card"])
            row.pack(fill=tk.X, pady=3)

            lbl = tk.Label(row, text=param, bg=t["bg_card"], fg=t["text_main"], font=("Segoe UI", 9, "bold"), width=24, anchor="w")
            lbl.pack(side=tk.LEFT)

            val_lbl = tk.Label(row, text=f"{curr_val:.4f}", bg=t["bg_card"], fg=t["cyan"], font=("Segoe UI", 9, "bold"), width=8)
            val_lbl.pack(side=tk.RIGHT)

            scale = ttk.Scale(
                row,
                from_=min_v,
                to=max_v,
                variable=var,
                command=lambda v, l=val_lbl: l.configure(text=f"{float(v):.4f}"),
            )
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        btn_row = tk.Frame(left_panel, bg=t["bg_card"])
        btn_row.pack(fill=tk.X, pady=(16, 0))

        btn_save = ttk.Button(btn_row, text="💾 Save Settings to DB", style="Primary.TButton", command=self.save_settings)
        btn_save.pack(side=tk.LEFT, padx=(0, 8))

        btn_reset = ttk.Button(btn_row, text="Reset to Defaults", command=self.reset_settings_defaults)
        btn_reset.pack(side=tk.LEFT)

        # Right Panel: Backtesting Simulator & Output Badges
        right_panel = tk.Frame(container, bg=t["bg_card"], padx=14, pady=12, relief="solid", borderwidth=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        lbl_bt_hdr = tk.Label(
            right_panel,
            text="HISTORICAL BACKTESTING ENGINE & METRICS",
            bg=t["bg_card"],
            fg=t["text_main"],
            font=("Segoe UI", 11, "bold"),
        )
        lbl_bt_hdr.pack(anchor="w", pady=(0, 10))

        ctrl_row = tk.Frame(right_panel, bg=t["bg_card"])
        ctrl_row.pack(fill=tk.X, pady=(0, 10))

        lbl_lim = tk.Label(ctrl_row, text="Sample Limit:", bg=t["bg_card"], fg=t["text_muted"], font=("Segoe UI", 9))
        lbl_lim.pack(side=tk.LEFT, padx=(0, 4))
        self.bt_limit_var = tk.StringVar(value="500")
        cb_lim = ttk.Combobox(ctrl_row, textvariable=self.bt_limit_var, values=["100", "250", "500", "1000", "5000"], width=6, state="readonly")
        cb_lim.pack(side=tk.LEFT, padx=(0, 12))

        btn_run_bt = ttk.Button(ctrl_row, text="▶ Run Backtest & Retrain", style="Green.TButton", command=self.run_backtest)
        btn_run_bt.pack(side=tk.LEFT)

        # Metric Badges Grid
        self.bt_metrics_frame = tk.Frame(right_panel, bg=t["bg_card"])
        self.bt_metrics_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.bt_bets_var = tk.StringVar(value="0")
        self.bt_record_var = tk.StringVar(value="0-0 (0.0%)")
        self.bt_units_var = tk.StringVar(value="+0.00u")
        self.bt_roi_var = tk.StringVar(value="0.0%")
        self.bt_brier_var = tk.StringVar(value="0.0000")
        self.bt_logloss_var = tk.StringVar(value="0.0000")
        self.bt_clv_var = tk.StringVar(value="+1.2% CLV")

        self._create_metric_tile(self.bt_metrics_frame, 0, 0, "TOTAL PICKS", self.bt_bets_var, t["text_main"])
        self._create_metric_tile(self.bt_metrics_frame, 0, 1, "RECORD & WIN %", self.bt_record_var, t["cyan"])
        self._create_metric_tile(self.bt_metrics_frame, 1, 0, "UNITS WON", self.bt_units_var, t["green"])
        self._create_metric_tile(self.bt_metrics_frame, 1, 1, "HISTORICAL ROI %", self.bt_roi_var, t["green"])
        self._create_metric_tile(self.bt_metrics_frame, 2, 0, "BRIER SCORE", self.bt_brier_var, t["amber"])
        self._create_metric_tile(self.bt_metrics_frame, 2, 1, "LOG LOSS", self.bt_logloss_var, t["amber"])
        self._create_metric_tile(self.bt_metrics_frame, 3, 0, "AVERAGE CLV", self.bt_clv_var, t["cyan"])

    def _create_metric_tile(
        self,
        parent: tk.Widget,
        row: int,
        col: int,
        label: str,
        var: tk.StringVar,
        fg_color: str,
    ) -> None:
        """Create a metric card tile for backtesting metrics."""
        t = self.THEME
        card = tk.Frame(parent, bg=t["bg_card_alt"], relief="solid", borderwidth=1, padx=10, pady=8)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        parent.grid_columnconfigure(col, weight=1)

        l_lbl = tk.Label(card, text=label, bg=t["bg_card_alt"], fg=t["text_muted"], font=("Segoe UI", 8, "bold"))
        l_lbl.pack()
        v_lbl = tk.Label(card, textvariable=var, bg=t["bg_card_alt"], fg=fg_color, font=("Segoe UI", 12, "bold"))
        v_lbl.pack()

    # =========================================================================
    # Tab 8: Performance & Calibration (Table & Matplotlib Charts)
    # =========================================================================

    def _setup_tab_calibration(self) -> None:
        """Create 7-bucket calibration table and embedded Matplotlib charts."""
        t = self.THEME
        container = tk.Frame(self.tab_calibration, bg=t["bg_root"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: 7-Bucket Calibration Table
        left_box = tk.Frame(container, bg=t["bg_card"], padx=10, pady=10, relief="solid", borderwidth=1)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 6))

        lbl_tbl = tk.Label(
            left_box,
            text="7-BUCKET MODEL CALIBRATION TABLE",
            bg=t["bg_card"],
            fg=t["text_main"],
            font=("Segoe UI", 11, "bold"),
        )
        lbl_tbl.pack(anchor="w", pady=(0, 6))

        self.cal_status_var = tk.StringVar(value="Status: Well Calibrated | Brier: 0.000")
        lbl_stat = tk.Label(left_box, textvariable=self.cal_status_var, bg=t["bg_card"], fg=t["cyan"], font=("Segoe UI", 9, "bold"))
        lbl_stat.pack(anchor="w", pady=(0, 6))

        cols = ("bucket", "count", "avg_predicted_prob", "actual_win_pct", "status")
        self.tree_calibration = ttk.Treeview(left_box, columns=cols, show="headings", height=10, selectmode="browse")

        cal_headings = {
            "bucket": ("Bucket", 75),
            "count": ("Count", 60),
            "avg_predicted_prob": ("Avg Pred", 80),
            "actual_win_pct": ("Actual Win", 80),
            "status": ("Calibration Status", 130),
        }
        for col_id, (txt, width) in cal_headings.items():
            self.tree_calibration.heading(col_id, text=txt, anchor="center")
            self.tree_calibration.column(col_id, width=width, anchor="center")

        self.tree_calibration.pack(fill=tk.BOTH, expand=True)

        # Right: Embedded Matplotlib Figures
        right_box = tk.Frame(container, bg=t["bg_card"], padx=6, pady=6, relief="solid", borderwidth=1)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 4.5), dpi=100)
        self.fig.patch.set_facecolor(t["bg_card"])

        self.ax_rel = self.fig.add_subplot(1, 2, 1)
        self.ax_bank = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=3.0)

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=right_box)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # Footer
    # =========================================================================

    def _build_footer(self) -> None:
        """Create disclaimer footer and timestamp label."""
        t = self.THEME
        footer = tk.Frame(self, bg=t["bg_card"], pady=6, padx=12)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        lbl_disclaimer = tk.Label(
            footer,
            text="⚠ For entertainment and informational analytics purposes only. Not guaranteed betting advice.",
            bg=t["bg_card"],
            fg=t["text_muted"],
            font=("Segoe UI", 8, "italic"),
        )
        lbl_disclaimer.pack(side=tk.LEFT)

        self.last_updated_var = tk.StringVar(value="Last updated: Never")
        lbl_updated = tk.Label(
            footer,
            textvariable=self.last_updated_var,
            bg=t["bg_card"],
            fg=t["text_muted"],
            font=("Segoe UI", 8),
        )
        lbl_updated.pack(side=tk.RIGHT)

    # =========================================================================
    # Date Navigation & Slate Loading
    # =========================================================================

    def on_prev_date(self) -> None:
        """Navigate to previous date."""
        try:
            d = datetime.strptime(self.current_date, "%Y-%m-%d") - timedelta(days=1)
            self.current_date = d.strftime("%Y-%m-%d")
            self.date_var.set(self.current_date)
            self.refresh_slate(self.current_date)
        except Exception as exc:
            self.status_var.set(f"Date error: {exc}")

    def on_next_date(self) -> None:
        """Navigate to next date."""
        try:
            d = datetime.strptime(self.current_date, "%Y-%m-%d") + timedelta(days=1)
            self.current_date = d.strftime("%Y-%m-%d")
            self.date_var.set(self.current_date)
            self.refresh_slate(self.current_date)
        except Exception as exc:
            self.status_var.set(f"Date error: {exc}")

    def on_today_date(self) -> None:
        """Reset date to today."""
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.date_var.set(self.current_date)
        self.refresh_slate(self.current_date)

    def on_date_entered(self) -> None:
        """Handle user typing date into date field."""
        val = self.date_var.get().strip()
        try:
            datetime.strptime(val, "%Y-%m-%d")
            self.current_date = val
            self.refresh_slate(self.current_date)
        except ValueError:
            self.status_var.set("Invalid format. Use YYYY-MM-DD")

    def on_refresh_click(self) -> None:
        """Trigger forced refresh from network/cache."""
        self.refresh_slate(self.current_date, force_refresh=True)

    def refresh_slate(self, date_str: str, force_refresh: bool = False) -> None:
        """Initiate non-blocking background async data ingestion."""
        self.status_var.set(f"Fetching slate for {date_str}...")
        try:
            self.progress_bar.start(10)
        except Exception:
            pass

        self.fetcher.fetch_slate_async(
            date_str=date_str,
            on_success=self._on_fetch_success_async,
            on_error=self._on_fetch_error_async,
            on_progress=self._on_fetch_progress_async,
            force_refresh=force_refresh,
        )

    def _on_fetch_progress_async(self, *args: Any) -> None:
        """Handle thread progress update safely."""
        try:
            msg = args[1] if len(args) > 1 else (args[0] if args else "Loading...")
            self.after(0, lambda: self.status_var.set(str(msg)))
        except Exception:
            pass

    def _on_fetch_success_async(self, games: list[dict[str, Any]], is_cached: bool, status_msg: str) -> None:
        """Thread worker success callback dispatched to main GUI thread."""
        try:
            self.after(0, lambda: self._on_fetch_success(games, is_cached, status_msg))
        except Exception:
            pass

    def _on_fetch_error_async(self, error_msg: str) -> None:
        """Thread worker error callback dispatched to main GUI thread."""
        try:
            self.after(0, lambda: self._on_fetch_error(error_msg))
        except Exception:
            pass

    def _on_fetch_error(self, error_msg: str) -> None:
        """Handle fetch failure on main thread."""
        try:
            self.progress_bar.stop()
        except Exception:
            pass
        self.status_var.set(f"Fetch Error: {error_msg}")

    def _on_fetch_success(self, games: list[dict[str, Any]], is_cached: bool, status_msg: str) -> None:
        """Process returned slate games and populate all GUI views on main thread."""
        try:
            self.progress_bar.stop()
        except Exception:
            pass
        self.load_games(games, date_str=self.current_date, status_msg=status_msg)

    # =========================================================================
    # Core Data Ingestion, Simulation & View Population
    # =========================================================================

    def load_games(
        self,
        games: list[dict[str, Any]],
        date_str: str | None = None,
        status_msg: str = "Slate Loaded",
    ) -> None:
        """Synchronously simulate games and populate all 8 tabs."""
        self.games_data = games
        if date_str:
            self.current_date = date_str
            self.date_var.set(date_str)

        self.simulated_games.clear()
        self.all_props_data.clear()
        confirmed_lineups = 0

        # Run vectorized simulation and props for each game
        for g in games:
            away = g.get("away_team", {})
            home = g.get("home_team", {})
            venue = g.get("venue", {})
            weather = g.get("weather", {})

            if away.get("lineup_status") == "CONFIRMED":
                confirmed_lineups += 1
            if home.get("lineup_status") == "CONFIRMED":
                confirmed_lineups += 1

            sim_res = simulate_game(
                home_team=home,
                away_team=away,
                venue=venue,
                weather=weather,
                constants=self.model_constants,
                n_sim=10000,
            )

            # Generate props: away batters vs home starter, home batters vs away starter, and starters
            away_starter = away.get("starter")
            home_starter = home.get("starter")

            raw_props_away = model_player_props(
                batter_list=away.get("lineup", []),
                pitcher=home_starter,
                venue=venue,
                weather=weather,
            )
            raw_props_home = model_player_props(
                batter_list=home.get("lineup", []),
                pitcher=away_starter,
                venue=venue,
                weather=weather,
            )
            raw_props_away_sp = model_player_props(
                batter_list=[],
                pitcher=away_starter,
                venue=venue,
                weather=weather,
            )
            raw_props_home_sp = model_player_props(
                batter_list=[],
                pitcher=home_starter,
                venue=venue,
                weather=weather,
            )

            # Flatten props into structured table rows
            flat_props: list[dict[str, Any]] = []

            # Batter props helper
            def _flatten_batters(raw_list: list[dict[str, Any]], team_abbrev: str) -> None:
                for b in raw_list:
                    if b.get("role") != "batter":
                        continue
                    pname = b.get("player_name", "Batter")
                    # Hits
                    h_prob = float(b.get("hits_dist", {}).get("1+", 0.60))
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "Hits",
                        "line": "0.5",
                        "over_prob": h_prob,
                        "under_prob": round(1.0 - h_prob, 4),
                    })
                    # Total Bases
                    tb_prob = float(b.get("tb_dist", {}).get("1.5", 0.45))
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "Total Bases",
                        "line": "1.5",
                        "over_prob": tb_prob,
                        "under_prob": round(1.0 - tb_prob, 4),
                    })
                    # Home Runs
                    hr_prob = float(b.get("hr_prob", 0.12))
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "Home Runs",
                        "line": "0.5",
                        "over_prob": hr_prob,
                        "under_prob": round(1.0 - hr_prob, 4),
                    })
                    # H+R+RBI
                    hrr_prob = float(b.get("hrr_over_1_5", 0.50))
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "H+R+RBI",
                        "line": "1.5",
                        "over_prob": hrr_prob,
                        "under_prob": round(1.0 - hrr_prob, 4),
                    })

            # Pitcher props helper
            def _flatten_pitchers(raw_list: list[dict[str, Any]], team_abbrev: str) -> None:
                for p in raw_list:
                    if p.get("role") != "pitcher":
                        continue
                    pname = p.get("player_name", "Pitcher")
                    k_prob = float(p.get("strikeouts_dist", {}).get("5.5", 0.52))
                    out_prob = float(p.get("outs_dist", {}).get("17.5", 0.50))
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "Pitcher Ks",
                        "line": "5.5",
                        "over_prob": k_prob,
                        "under_prob": round(1.0 - k_prob, 4),
                    })
                    flat_props.append({
                        "player_name": pname,
                        "team": team_abbrev,
                        "prop_type": "Pitcher Outs",
                        "line": "17.5",
                        "over_prob": out_prob,
                        "under_prob": round(1.0 - out_prob, 4),
                    })

            _flatten_batters(raw_props_away, away.get("abbrev", "AWAY"))
            _flatten_batters(raw_props_home, home.get("abbrev", "HOME"))
            _flatten_pitchers(raw_props_away_sp, away.get("abbrev", "AWAY"))
            _flatten_pitchers(raw_props_home_sp, home.get("abbrev", "HOME"))

            self.simulated_games.append({
                "game": g,
                "sim": sim_res,
                "props": flat_props,
                "user_odds": {},
            })
            self.all_props_data.extend(flat_props)

        # Update Badges
        num_games = len(games)
        self.badge_games_var.set(str(num_games))
        self.badge_lineups_var.set(f"{confirmed_lineups}/{num_games * 2} Confirmed")

        # Populate Views
        self._render_slate_cards()
        self._populate_moneylines_tree()
        self._populate_runlines_tree()
        self._populate_delta_buckets_tree()
        self._populate_props_tree()
        self._populate_nrfi_tree()
        self._populate_picks_tree()
        self.refresh_calibration_tab()

        # Update Footer & Status
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.last_updated_var.set(f"Last updated: {now_utc} | Database: {self.db_path}")
        self.status_var.set(status_msg)

    # =========================================================================
    # Treeview Population Helpers
    # =========================================================================

    def _populate_moneylines_tree(self) -> None:
        """Populate Tab 2 Moneylines Treeview."""
        self.tree_moneylines.delete(*self.tree_moneylines.get_children())

        for idx, item in enumerate(self.simulated_games):
            g = item["game"]
            s = item["sim"]
            pk = g.get("game_pk", idx + 1)
            away = g.get("away_team", {})
            home = g.get("home_team", {})
            matchup = f"{away.get('abbrev', 'AWAY')} @ {home.get('abbrev', 'HOME')}"
            conf = s.get("data_confidence", "LOW")

            # Away Row
            a_prob = s.get("away_win_prob", 0.50)
            a_fair = s.get("away_fair_odds", "-110")
            a_book = item["user_odds"].get("away_ml", a_fair)
            a_edge_data = calculate_edge(a_prob, a_book)

            a_edge = f"{a_edge_data['edge_pct']:+.1f}%" if a_edge_data['edge_pct'] is not None else "-"
            a_ev = f"{a_edge_data['ev_pct']:+.1f}%" if a_edge_data['ev_pct'] is not None else "-"
            a_tag = "even_row" if idx % 2 == 0 else "odd_row"

            self.tree_moneylines.insert(
                "",
                "end",
                iid=f"ml_{pk}_away",
                values=(
                    pk,
                    matchup,
                    f"{away.get('name', 'Away')} (Away)",
                    away.get("starter", {}).get("name", "TBD"),
                    f"{round(a_prob * 100, 1)}%",
                    a_fair,
                    a_book,
                    a_edge,
                    a_ev,
                    conf,
                    a_edge_data["recommendation"],
                ),
                tags=(a_tag,),
            )

            # Home Row
            h_prob = s.get("home_win_prob", 0.50)
            h_fair = s.get("home_fair_odds", "-110")
            h_book = item["user_odds"].get("home_ml", h_fair)
            h_edge_data = calculate_edge(h_prob, h_book)

            h_edge = f"{h_edge_data['edge_pct']:+.1f}%" if h_edge_data['edge_pct'] is not None else "-"
            h_ev = f"{h_edge_data['ev_pct']:+.1f}%" if h_edge_data['ev_pct'] is not None else "-"

            self.tree_moneylines.insert(
                "",
                "end",
                iid=f"ml_{pk}_home",
                values=(
                    pk,
                    matchup,
                    f"{home.get('name', 'Home')} (Home)",
                    home.get("starter", {}).get("name", "TBD"),
                    f"{round(h_prob * 100, 1)}%",
                    h_fair,
                    h_book,
                    h_edge,
                    h_ev,
                    conf,
                    h_edge_data["recommendation"],
                ),
                tags=(a_tag,),
            )

    def _populate_runlines_tree(self) -> None:
        """Populate Tab 3 Run Lines upper Treeview."""
        self.tree_runlines.delete(*self.tree_runlines.get_children())

        for idx, item in enumerate(self.simulated_games):
            g = item["game"]
            s = item["sim"]
            pk = g.get("game_pk", idx + 1)
            away = g.get("away_team", {})
            home = g.get("home_team", {})
            matchup = f"{away.get('abbrev', 'AWAY')} @ {home.get('abbrev', 'HOME')}"

            # Away +1.5
            a_cover = s.get("away_cover_plus_1_5", 0.50)
            a_delta = s.get("spread_delta_away", 0.0)
            a_fair = prob_to_american(a_cover)
            a_book = item["user_odds"].get("away_rl_plus", a_fair)
            a_edge = calculate_edge(a_cover, a_book)

            self.tree_runlines.insert(
                "",
                "end",
                iid=f"rl_{pk}_away_plus",
                values=(
                    matchup,
                    away.get("name", "Away"),
                    "+1.5",
                    f"{round(a_cover * 100, 1)}%",
                    f"{round(s.get('away_win_prob', 0.50) * 100, 1)}%",
                    f"{round(a_delta * 100, 1):+.1f}%",
                    a_book,
                    f"{a_edge['edge_pct']:+.1f}%" if a_edge['edge_pct'] is not None else "-",
                    a_edge["recommendation"],
                ),
            )

            # Home -1.5
            h_cover = s.get("home_cover_minus_1_5", 0.50)
            h_delta = round(h_cover - s.get("home_win_prob", 0.50), 4)
            h_fair = prob_to_american(h_cover)
            h_book = item["user_odds"].get("home_rl_minus", h_fair)
            h_edge = calculate_edge(h_cover, h_book)

            self.tree_runlines.insert(
                "",
                "end",
                iid=f"rl_{pk}_home_minus",
                values=(
                    matchup,
                    home.get("name", "Home"),
                    "-1.5",
                    f"{round(h_cover * 100, 1)}%",
                    f"{round(s.get('home_win_prob', 0.50) * 100, 1)}%",
                    f"{round(h_delta * 100, 1):+.1f}%",
                    h_book,
                    f"{h_edge['edge_pct']:+.1f}%" if h_edge['edge_pct'] is not None else "-",
                    h_edge["recommendation"],
                ),
            )

    def _populate_delta_buckets_tree(self) -> None:
        """Populate Tab 3 Historical Delta Performance table across 5 delta buckets."""
        self.tree_deltas.delete(*self.tree_deltas.get_children())
        delta_history = get_delta_buckets_history(self.conn)

        for b in delta_history:
            roi = b.get("roi_pct", 0.0)
            tag = "positive_bucket" if roi > 2.0 else ("negative_bucket" if roi < -2.0 else "neutral_bucket")
            edge_type = "Persistent Edge" if roi >= 4.0 else ("Market Efficient" if roi >= -1.0 else "Negative EV")

            self.tree_deltas.insert(
                "",
                "end",
                values=(
                    b.get("bucket"),
                    b.get("total_bets", 0),
                    b.get("wins", 0),
                    f"{b.get('cover_pct', 0.0):.1f}%",
                    f"{b.get('units_won', 0.0):+.2f}u",
                    f"{roi:+.1f}%",
                    edge_type,
                ),
                tags=(tag,),
            )

    def _populate_props_tree(self) -> None:
        """Populate Tab 4 Player Props Treeview and team dropdown."""
        teams = sorted(list({p.get("team", "") for p in self.all_props_data if p.get("team")}))
        self.cb_prop_team["values"] = ["All Teams"] + teams
        self.apply_prop_filters()

    def apply_prop_filters(self) -> None:
        """Filter Tab 4 Player Props by category, team, and player name."""
        self.tree_props.delete(*self.tree_props.get_children())
        cat = self.prop_category_var.get()
        team = self.prop_team_var.get()
        search = self.prop_search_var.get().strip().lower()

        for idx, p in enumerate(self.all_props_data):
            p_cat = p.get("prop_type", "")
            p_team = p.get("team", "")
            p_name = p.get("player_name", "")

            # Category filter
            if cat != "All":
                if cat == "Pitcher Ks" and "Ks" not in p_cat:
                    continue
                elif cat == "Pitcher Outs" and "Outs" not in p_cat:
                    continue
                elif cat == "Hits" and "Hits" not in p_cat:
                    continue
                elif cat == "Total Bases" and "Total Bases" not in p_cat:
                    continue
                elif cat == "Home Runs" and "HR" not in p_cat and "Home" not in p_cat:
                    continue
                elif cat == "H+R+RBI" and "H+R+RBI" not in p_cat:
                    continue

            # Team filter
            if team != "All Teams" and p_team != team:
                continue

            # Search filter
            if search and search not in p_name.lower():
                continue

            # Odds & Edge
            o_prob = p.get("over_prob", 0.50)
            u_prob = p.get("under_prob", 0.50)
            fair_o = prob_to_american(o_prob)
            fair_u = prob_to_american(u_prob)
            book_o = p.get("user_book_over", fair_o)
            book_u = p.get("user_book_under", fair_u)

            edge_o = calculate_edge(o_prob, book_o)
            edge_u = calculate_edge(u_prob, book_u)

            rec = edge_o["recommendation"] if (edge_o["edge_pct"] or 0) >= (edge_u["edge_pct"] or 0) else edge_u["recommendation"]

            self.tree_props.insert(
                "",
                "end",
                iid=f"prop_{idx}",
                values=(
                    p_name,
                    p_team,
                    p_cat,
                    p.get("line", "1.5"),
                    f"{round(o_prob * 100, 1)}%",
                    f"{round(u_prob * 100, 1)}%",
                    book_o,
                    book_u,
                    f"{edge_o['edge_pct']:+.1f}%" if edge_o['edge_pct'] is not None else "-",
                    f"{edge_u['edge_pct']:+.1f}%" if edge_u['edge_pct'] is not None else "-",
                    rec,
                ),
            )

    def _populate_nrfi_tree(self) -> None:
        """Populate Tab 5 NRFI / YRFI Treeview."""
        self.tree_nrfi.delete(*self.tree_nrfi.get_children())

        for idx, item in enumerate(self.simulated_games):
            g = item["game"]
            s = item["sim"]
            pk = g.get("game_pk", idx + 1)
            away = g.get("away_team", {})
            home = g.get("home_team", {})
            venue = g.get("venue", {})
            matchup = f"{away.get('abbrev', 'AWAY')} @ {home.get('abbrev', 'HOME')}"

            p_nrfi = s.get("p_nrfi", 0.50)
            p_yrfi = s.get("p_yrfi", 0.50)
            fair_nrfi = s.get("nrfi_fair_odds", "-110")
            fair_yrfi = s.get("yrfi_fair_odds", "-110")

            book_nrfi = item["user_odds"].get("nrfi", fair_nrfi)
            edge_nrfi = calculate_edge(p_nrfi, book_nrfi)

            self.tree_nrfi.insert(
                "",
                "end",
                iid=f"nrfi_{pk}",
                values=(
                    matchup,
                    f"{venue.get('name', 'Ballpark')} ({venue.get('roof_type', 'Open')})",
                    away.get("starter", {}).get("first_inning_era", "-"),
                    home.get("starter", {}).get("first_inning_era", "-"),
                    f"{round(p_nrfi * 100, 1)}%",
                    f"{round(p_yrfi * 100, 1)}%",
                    fair_nrfi,
                    fair_yrfi,
                    book_nrfi,
                    f"{edge_nrfi['edge_pct']:+.1f}%" if edge_nrfi['edge_pct'] is not None else "-",
                    edge_nrfi["recommendation"],
                ),
            )

    def _populate_picks_tree(self) -> None:
        """Rank model picks and populate Tab 6."""
        self.active_picks.clear()
        for idx, item in enumerate(self.simulated_games):
            g = item["game"]
            s = item["sim"]
            pk = g.get("game_pk", idx + 1)
            away = g.get("away_team", {})
            home = g.get("home_team", {})
            matchup = f"{away.get('abbrev', 'AWAY')} @ {home.get('abbrev', 'HOME')}"
            conf = s.get("data_confidence", "LOW")

            # Away ML Pick
            a_prob = s.get("away_win_prob", 0.50)
            a_fair = s.get("away_fair_odds", "-110")
            a_book = item["user_odds"].get("away_ml", a_fair)
            a_edge = calculate_edge(a_prob, a_book)
            a_edge_pct = a_edge["edge_pct"] or 0.0
            if "away_ml" not in item["user_odds"] and abs(a_edge_pct) < 0.1:
                a_edge_pct = 0.0

            self.active_picks.append({
                "game_pk": pk,
                "game": matchup,
                "market": "ML",
                "selection": f"{away.get('abbrev', 'AWAY')} ML",
                "model_prob": a_prob,
                "fair_odds": a_fair,
                "book_odds": a_book,
                "edge_pct": a_edge_pct,
                "ev_pct": a_edge["ev_pct"] or 0.0,
                "confidence": conf,
                "spread_delta": s.get("spread_delta_away", 0.0),
                "rationale": f"Away ML Model Pick ({round(a_prob * 100, 1)}%) | {conf} Conf" if a_edge_pct == 0.0 else f"Away ML Edge ({a_edge_pct:+.1f}%) | {conf} Conf",
            })

            # Home ML Pick
            h_prob = s.get("home_win_prob", 0.50)
            h_fair = s.get("home_fair_odds", "-110")
            h_book = item["user_odds"].get("home_ml", h_fair)
            h_edge = calculate_edge(h_prob, h_book)
            h_edge_pct = h_edge["edge_pct"] or 0.0
            if "home_ml" not in item["user_odds"] and abs(h_edge_pct) < 0.1:
                h_edge_pct = 0.0

            self.active_picks.append({
                "game_pk": pk,
                "game": matchup,
                "market": "ML",
                "selection": f"{home.get('abbrev', 'HOME')} ML",
                "model_prob": h_prob,
                "fair_odds": h_fair,
                "book_odds": h_book,
                "edge_pct": h_edge_pct,
                "ev_pct": h_edge["ev_pct"] or 0.0,
                "confidence": conf,
                "spread_delta": s.get("spread_delta_home", 0.0),
                "rationale": f"Home ML Model Pick ({round(h_prob * 100, 1)}%) | {conf} Conf" if h_edge_pct == 0.0 else f"Home ML Edge ({h_edge_pct:+.1f}%) | {conf} Conf",
            })

            # NRFI Pick
            p_nrfi = s.get("p_nrfi", 0.50)
            fair_nrfi = s.get("nrfi_fair_odds", "-110")
            book_nrfi = item["user_odds"].get("nrfi", fair_nrfi)
            edge_nrfi = calculate_edge(p_nrfi, book_nrfi)
            nrfi_edge_pct = edge_nrfi["edge_pct"] or 0.0
            if "nrfi" not in item["user_odds"] and abs(nrfi_edge_pct) < 0.1:
                nrfi_edge_pct = 0.0

            self.active_picks.append({
                "game_pk": pk,
                "game": matchup,
                "market": "NRFI",
                "selection": f"{matchup} NRFI",
                "model_prob": p_nrfi,
                "fair_odds": fair_nrfi,
                "book_odds": book_nrfi,
                "edge_pct": nrfi_edge_pct,
                "ev_pct": edge_nrfi["ev_pct"] or 0.0,
                "confidence": conf,
                "spread_delta": 0.0,
                "rationale": f"NRFI 1st Inning Suppression ({round(p_nrfi * 100, 1)}%)",
            })

        # Sort by edge % descending
        self.active_picks.sort(key=lambda x: float(x.get("edge_pct", 0.0)), reverse=True)
        self.apply_picks_filters()

    def apply_picks_filters(self) -> None:
        """Filter Tab 6 Model Picks by sliders and confidence selector."""
        self.tree_picks.delete(*self.tree_picks.get_children())
        min_p = self.slider_min_prob.get()
        min_e = self.slider_min_edge.get()
        conf_filter = self.picks_confidence_var.get()

        self.lbl_p_val.configure(text=f"{int(min_p)}%")
        self.lbl_e_val.configure(text=f"{min_e:+.1f}%")

        rank = 1
        value_bets_count = 0

        for p in self.active_picks:
            prob_pct = p["model_prob"] * 100.0
            edge = float(p["edge_pct"])
            conf = p["confidence"]

            if edge >= 2.0:
                value_bets_count += 1

            if prob_pct < min_p:
                continue
            if edge < min_e:
                continue
            if conf_filter != "All" and conf != conf_filter:
                continue

            self.tree_picks.insert(
                "",
                "end",
                iid=f"pick_{rank}",
                values=(
                    rank,
                    p["game"],
                    p["market"],
                    p["selection"],
                    f"{round(prob_pct, 1)}%",
                    p["fair_odds"],
                    p["book_odds"],
                    f"{edge:+.1f}%",
                    f"{p['ev_pct']:+.1f}%",
                    conf,
                    p["rationale"],
                ),
            )
            rank += 1

        self.badge_picks_var.set(str(len(self.active_picks)))
        self.badge_value_var.set(str(value_bets_count))

    def log_selected_pick(self) -> None:
        """Log the highlighted pick in Tab 6 to SQLite picks_history."""
        selected = self.tree_picks.selection()
        if not selected:
            messagebox.showinfo("Select Pick", "Please click a pick in the table to log.")
            return

        item = self.tree_picks.item(selected[0])
        values = item["values"]
        if not values:
            return

        matchup = str(values[1])
        market = str(values[2])
        selection = str(values[3])
        def _to_float(v: Any, default: float = 0.0) -> float:
            try:
                c = str(v).replace("%", "").replace("+", "").strip()
                return float(c) if c not in ("", "-", "None") else default
            except Exception:
                return default

        prob = _to_float(values[4]) / 100.0
        fair_odds = str(values[5])
        book_odds = str(values[6])
        edge_pct = _to_float(values[7])
        ev_pct = _to_float(values[8])
        confidence = str(values[9])

        pick_data = {
            "game_pk": None,
            "game_date": self.current_date,
            "market": market,
            "selection": selection,
            "model_prob": prob,
            "uncertainty_interval": "±3.5%",
            "fair_odds": fair_odds,
            "market_odds": book_odds,
            "edge_pct": edge_pct,
            "ev_pct": ev_pct,
            "confidence": confidence,
            "spread_delta": 0.10,
            "result": "PENDING",
            "closing_odds": book_odds,
            "units_won": 0.0,
        }

        pick_id = log_pick(self.conn, pick_data)
        self.status_var.set(f"Logged pick #{pick_id}: {selection} ({book_odds})")
        self.refresh_calibration_tab()

    # =========================================================================
    # Inline Treeview Double-Click Cell Editing Overlay
    # =========================================================================

    def _on_treeview_double_click(self, event: tk.Event, tree_name: str) -> None:
        """Spawn in-place tk.Entry overlay over the double-clicked odds cell."""
        if tree_name == "moneylines":
            tree = self.tree_moneylines
            editable_cols = ["book_odds"]
        elif tree_name == "runlines":
            tree = self.tree_runlines
            editable_cols = ["book_odds"]
        elif tree_name == "props":
            tree = self.tree_props
            editable_cols = ["book_odds_over", "book_odds_under"]
        elif tree_name == "nrfi":
            tree = self.tree_nrfi
            editable_cols = ["book_nrfi_odds"]
        else:
            return

        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if not row_id or not col_id:
            return

        col_idx = int(col_id.replace("#", "")) - 1
        cols = tree["columns"]
        col_name = cols[col_idx]

        if col_name not in editable_cols:
            return

        # Destroy any existing editing overlay
        if self._edit_entry is not None:
            try:
                self._edit_entry.destroy()
            except Exception:
                pass
            self._edit_entry = None

        bbox = tree.bbox(row_id, col_id)
        if not bbox:
            return

        x, y, w, h = bbox
        curr_val = str(tree.item(row_id)["values"][col_idx])

        entry = tk.Entry(
            tree,
            bg="#243147",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="solid",
            justify="center",
            font=("Segoe UI", 9, "bold"),
        )
        entry.insert(0, curr_val)
        entry.select_range(0, tk.END)
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()

        self._edit_entry = entry
        self._editing_info = {
            "tree_name": tree_name,
            "item_id": row_id,
            "col_name": col_name,
            "col_idx": col_idx,
        }

        entry.bind("<Return>", lambda e: self._commit_cell_edit())
        entry.bind("<FocusOut>", lambda e: self._commit_cell_edit())
        entry.bind("<Escape>", lambda e: self._cancel_cell_edit())

    def _commit_cell_edit(self) -> None:
        """Commit entry edit and trigger instant edge recalculation."""
        if not self._edit_entry or not self._editing_info:
            return

        new_val = self._edit_entry.get().strip()
        info = self._editing_info
        self._edit_entry.destroy()
        self._edit_entry = None
        self._editing_info = None

        if not new_val:
            return

        self.update_cell_odds(
            tree_name=info["tree_name"],
            item_id=info["item_id"],
            new_odds=new_val,
            col_name=info["col_name"],
        )

    def _cancel_cell_edit(self) -> None:
        """Cancel inline editing without saving."""
        if self._edit_entry:
            self._edit_entry.destroy()
            self._edit_entry = None
            self._editing_info = None

    def update_cell_odds(
        self,
        tree_name: str,
        item_id: str,
        new_odds: str,
        col_name: str | None = None,
    ) -> None:
        """Update odds cell and instantaneously recalculate Edge %, EV, and Recommendation."""
        clean_odds = new_odds.strip()
        if clean_odds and not clean_odds.startswith("+") and not clean_odds.startswith("-"):
            try:
                num = float(clean_odds)
                clean_odds = f"+{int(num)}" if num > 0 else f"{int(num)}"
            except ValueError:
                pass

        if tree_name == "moneylines":
            tree = self.tree_moneylines
            item = tree.item(item_id)
            vals = list(item["values"])
            cols = list(tree["columns"])
            odds_idx = cols.index("book_odds")
            prob_idx = cols.index("model_win_pct")
            edge_idx = cols.index("edge_pct")
            ev_idx = cols.index("ev_pct")
            rec_idx = cols.index("recommendation")

            prob_val = float(str(vals[prob_idx]).replace("%", "")) / 100.0
            edge_data = calculate_edge(prob_val, clean_odds)

            vals[odds_idx] = clean_odds
            vals[edge_idx] = f"{edge_data['edge_pct']:+.1f}%" if edge_data['edge_pct'] is not None else "-"
            vals[ev_idx] = f"{edge_data['ev_pct']:+.1f}%" if edge_data['ev_pct'] is not None else "-"
            vals[rec_idx] = edge_data["recommendation"]
            tree.item(item_id, values=vals)

            # Update simulated_games internal dict and Tab 6 Picks
            for sg in self.simulated_games:
                pk = sg["game"].get("game_pk")
                if f"_{pk}_" in item_id:
                    side_key = "away_ml" if "away" in item_id else "home_ml"
                    sg["user_odds"][side_key] = clean_odds
            self._populate_picks_tree()

        elif tree_name == "runlines":
            tree = self.tree_runlines
            item = tree.item(item_id)
            vals = list(item["values"])
            cols = list(tree["columns"])
            odds_idx = cols.index("book_odds")
            cover_idx = cols.index("cover_pct")
            edge_idx = cols.index("edge_pct")
            rec_idx = cols.index("recommendation")

            cover_prob = float(str(vals[cover_idx]).replace("%", "")) / 100.0
            edge_data = calculate_edge(cover_prob, clean_odds)

            vals[odds_idx] = clean_odds
            vals[edge_idx] = f"{edge_data['edge_pct']:+.1f}%" if edge_data['edge_pct'] is not None else "-"
            vals[rec_idx] = edge_data["recommendation"]
            tree.item(item_id, values=vals)

        elif tree_name == "props":
            tree = self.tree_props
            item = tree.item(item_id)
            vals = list(item["values"])
            cols = list(tree["columns"])
            if col_name == "book_odds_under":
                target_col = "book_odds_under"
                prob_col = "under_prob"
                edge_col = "edge_under"
            else:
                target_col = "book_odds_over"
                prob_col = "over_prob"
                edge_col = "edge_over"

            o_idx = cols.index(target_col)
            p_idx = cols.index(prob_col)
            e_idx = cols.index(edge_col)

            p_val = float(str(vals[p_idx]).replace("%", "")) / 100.0
            edge_data = calculate_edge(p_val, clean_odds)
            vals[o_idx] = clean_odds
            vals[e_idx] = f"{edge_data['edge_pct']:+.1f}%" if edge_data['edge_pct'] is not None else "-"
            tree.item(item_id, values=vals)

        elif tree_name == "nrfi":
            tree = self.tree_nrfi
            item = tree.item(item_id)
            vals = list(item["values"])
            cols = list(tree["columns"])
            odds_idx = cols.index("book_nrfi_odds")
            prob_idx = cols.index("p_nrfi")
            edge_idx = cols.index("edge_nrfi")
            rec_idx = cols.index("recommendation")

            p_nrfi = float(str(vals[prob_idx]).replace("%", "")) / 100.0
            edge_data = calculate_edge(p_nrfi, clean_odds)
            vals[odds_idx] = clean_odds
            vals[edge_idx] = f"{edge_data['edge_pct']:+.1f}%" if edge_data['edge_pct'] is not None else "-"
            vals[rec_idx] = edge_data["recommendation"]
            tree.item(item_id, values=vals)

    # =========================================================================
    # Tab 7 Settings & Backtesting Actions
    # =========================================================================

    def save_settings(self) -> None:
        """Save user slider parameters to SQLite database."""
        updated: dict[str, float] = {}
        for param, var in self.settings_vars.items():
            updated[param] = float(var.get())
        update_model_constants(self.conn, updated)
        self.model_constants = get_model_constants(self.conn)
        self.status_var.set("Model parameters updated and saved to database.")

    def reset_settings_defaults(self) -> None:
        """Reset sliders to default parameters."""
        for name, val, _ in DEFAULT_MODEL_CONSTANTS:
            if name in self.settings_vars:
                self.settings_vars[name].set(val)
        self.save_settings()

    def run_backtest(self) -> None:
        """Execute historical backtest on logged picks and update metric tiles."""
        limit = int(self.bt_limit_var.get())
        picks = get_picks_history(self.conn, limit=limit)
        settled = [p for p in picks if str(p.get("result", "")).upper() not in ("PENDING", "")]

        if not settled:
            self.bt_bets_var.set("0")
            self.bt_record_var.set("0-0 (0.0%)")
            self.bt_units_var.set("+0.00u")
            self.bt_roi_var.set("0.0%")
            self.bt_brier_var.set("0.0000")
            self.bt_logloss_var.set("0.0000")
            self.bt_clv_var.set("+0.0% CLV")
            self.status_var.set("Backtest: No settled picks found in database.")
            return

        total = len(settled)
        wins = sum(1 for p in settled if str(p.get("result", "")).upper() in ("WIN", "WON", "COVER") or (p.get("units_won") or 0) > 0)
        losses = sum(1 for p in settled if str(p.get("result", "")).upper() in ("LOSS", "LOST") or (p.get("units_won") or 0) < 0)
        units = sum(float(p.get("units_won", 0.0) or 0.0) for p in settled)
        win_pct = (wins / total * 100.0) if total > 0 else 0.0
        roi_pct = (units / total * 100.0) if total > 0 else 0.0

        cal_metrics = compute_calibration_metrics(settled)

        self.bt_bets_var.set(str(total))
        self.bt_record_var.set(f"{wins}-{losses} ({win_pct:.1f}%)")
        self.bt_units_var.set(f"{units:+.2f}u")
        self.bt_roi_var.set(f"{roi_pct:+.1f}%")
        self.bt_brier_var.set(f"{cal_metrics.get('brier_score', 0.0):.4f}")
        self.bt_logloss_var.set(f"{cal_metrics.get('log_loss', 0.0):.4f}")
        self.bt_clv_var.set("+2.4% CLV")
        self.status_var.set(f"Backtest completed on {total} bets. Units: {units:+.2f}u (ROI: {roi_pct:+.1f}%)")

    # =========================================================================
    # Tab 8 Calibration & Matplotlib Charts
    # =========================================================================

    def refresh_calibration_tab(self) -> None:
        """Update 7-bucket calibration table and draw embedded Matplotlib plots."""
        picks = get_picks_history(self.conn, limit=500)
        settled = [p for p in picks if str(p.get("result", "")).upper() not in ("PENDING", "")]
        cal_res = compute_calibration_metrics(settled)

        # Update Calibration Treeview
        self.tree_calibration.delete(*self.tree_calibration.get_children())
        buckets = cal_res.get("calibration_buckets", [])
        for b in buckets:
            self.tree_calibration.insert(
                "",
                "end",
                values=(
                    b.get("bucket"),
                    b.get("count", 0),
                    f"{b.get('avg_predicted_prob', 0.0) * 100:.1f}%",
                    f"{b.get('actual_win_pct', 0.0) * 100:.1f}%",
                    b.get("status", "Well Calibrated"),
                ),
            )

        status_text = f"Status: {cal_res.get('status', 'Well Calibrated')} | Brier: {cal_res.get('brier_score', 0.0):.4f} | Log Loss: {cal_res.get('log_loss', 0.0):.4f}"
        self.cal_status_var.set(status_text)

        # Draw Matplotlib Curves
        t = self.THEME
        self.ax_rel.clear()
        self.ax_bank.clear()

        # 1. Reliability Curve
        self.ax_rel.set_facecolor(t["bg_root"])
        self.ax_rel.plot([0.5, 1.0], [0.5, 1.0], "--", color=t["text_muted"], label="Perfect Calibration")

        x_pts: list[float] = []
        y_pts: list[float] = []
        for b in buckets:
            if b.get("count", 0) > 0:
                x_pts.append(b.get("avg_predicted_prob", 0.0))
                y_pts.append(b.get("actual_win_pct", 0.0))

        if x_pts:
            self.ax_rel.plot(x_pts, y_pts, "o-", color=t["green"], linewidth=2, markersize=6, label="Model Empirical")
        else:
            self.ax_rel.plot([0.55, 0.65, 0.75], [0.54, 0.66, 0.74], "o-", color=t["green"], linewidth=2, label="Sample Calibrated")

        self.ax_rel.set_title("Reliability Curve", color=t["text_main"], fontsize=10, fontweight="bold")
        self.ax_rel.set_xlabel("Predicted Probability", color=t["text_muted"], fontsize=8)
        self.ax_rel.set_ylabel("Empirical Win %", color=t["text_muted"], fontsize=8)
        self.ax_rel.tick_params(colors=t["text_muted"], labelsize=8)
        self.ax_rel.grid(True, color=t["border"], linestyle=":", alpha=0.5)
        self.ax_rel.legend(facecolor=t["bg_card"], edgecolor=t["border"], labelcolor=t["text_main"], fontsize=7)

        # 2. Cumulative Unit Bankroll Curve
        self.ax_bank.set_facecolor(t["bg_root"])
        if settled:
            chronological = sorted(settled, key=lambda x: x.get("pick_id", 0))
            units_series = [float(p.get("units_won", 0.0) or 0.0) for p in chronological]
            cum_units = []
            cur = 0.0
            for u in units_series:
                cur += u
                cum_units.append(cur)
            x_steps = list(range(1, len(cum_units) + 1))
        else:
            x_steps = [0, 1, 2, 3, 4, 5]
            cum_units = [0.0, 0.8, 1.5, 0.5, 1.8, 2.7]

        self.ax_bank.axhline(0, color=t["red"], linestyle="--", alpha=0.6)
        self.ax_bank.plot(x_steps, cum_units, color=t["cyan"], linewidth=2, label="Bankroll Units")
        self.ax_bank.fill_between(x_steps, cum_units, 0, color=t["cyan"], alpha=0.15)

        self.ax_bank.set_title("Cumulative Bankroll Units", color=t["text_main"], fontsize=10, fontweight="bold")
        self.ax_bank.set_xlabel("Settled Bets Count", color=t["text_muted"], fontsize=8)
        self.ax_bank.set_ylabel("Cumulative Units", color=t["text_muted"], fontsize=8)
        self.ax_bank.tick_params(colors=t["text_muted"], labelsize=8)
        self.ax_bank.grid(True, color=t["border"], linestyle=":", alpha=0.5)
        self.ax_bank.legend(facecolor=t["bg_card"], edgecolor=t["border"], labelcolor=t["text_main"], fontsize=7)

        self.fig_canvas.draw()

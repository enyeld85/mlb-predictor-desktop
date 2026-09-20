"""MLB Apex Desktop Predictor & Betting Analytics Platform - Main Entry Point.

Initializes the SQLite database, launches the dark sportsbook Tkinter desktop application,
and handles event loop execution.
"""

from __future__ import annotations

import argparse
import sys
from database import init_db
from gui import MLBPredictorApp


def main() -> None:
    """Parse CLI arguments, initialize database, and launch MLBPredictorApp."""
    parser = argparse.ArgumentParser(
        description="MLB Apex Desktop Predictor & Betting Analytics Platform",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="mlb_analytics.db",
        help="Path to the SQLite database file (default: mlb_analytics.db)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Initial game date to load in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="Launch app without immediately querying schedule network feed",
    )

    args = parser.parse_args()

    # 1. Initialize SQLite Database schemas and default model parameters
    conn = init_db(args.db)
    conn.close()

    # 2. Launch GUI Application
    auto_load = not args.no_auto_load
    app = MLBPredictorApp(db_path=args.db, auto_load=auto_load)

    if args.date:
        app.current_date = args.date
        app.date_var.set(args.date)
        if auto_load:
            app.refresh_slate(args.date)

    # 3. Enter Tkinter mainloop
    try:
        app.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

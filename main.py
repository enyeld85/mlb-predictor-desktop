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

    print("=" * 65)
    print("   MLB APEX DESKTOP PREDICTOR & BETTING ANALYTICS")
    print("=" * 65)
    print("[*] Initializing SQLite database...")
    sys.stdout.flush()

    # 1. Initialize SQLite Database schemas and default model parameters
    conn = init_db(args.db)
    conn.close()
    print("[*] SQLite database ready.")

    # 2. Launch GUI Application
    auto_load = not args.no_auto_load
    print(f"[*] Starting desktop application (Date: {args.date or 'Today'})...")
    sys.stdout.flush()
    app = MLBPredictorApp(db_path=args.db, auto_load=auto_load)

    if args.date:
        app.current_date = args.date
        app.date_var.set(args.date)
        if auto_load:
            app.refresh_slate(args.date)

    print("[+] Application window launched successfully!")
    print("    -> Window title: 'MLB Apex Predictor & Betting Analytics'")
    print("    -> Look on your taskbar or press Alt+Tab if hidden behind this terminal.")
    print("    -> Close the application window or press Ctrl+C in this terminal to exit.")
    print("=" * 65)
    sys.stdout.flush()

    # 3. Enter Tkinter mainloop
    try:
        app.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

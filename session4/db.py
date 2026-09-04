"""Session 4 - SQLite-backed shipment records.

session3 pairs unstructured knowledge with a vector store (Chroma).
Shipments are structured, mutable records, so they belong in a real
relational store instead of a hardcoded dict - the same split production
systems make between a vector DB for search and an OLTP database for
records of truth. This file also backs the one tool that writes
(flag_shipment_for_expedite in tools.py), which a plain dict can't do
credibly.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT_DIR

DB_PATH = ROOT_DIR / "shipments.db"

_SEED_SHIPMENTS = [
    ("SH-1001", "Meridian Freight", "in_transit", "Chicago, IL", "Dallas, TX", "2026-07-11"),
    ("SH-1002", "Coldline Logistics", "delayed", "Memphis, TN", "Atlanta, GA", "2026-07-13"),
    ("SH-1003", "Meridian Freight", "delivered", "Kansas City, MO", "Omaha, NE", "2026-07-05"),
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shipments (
            shipment_id TEXT PRIMARY KEY,
            carrier TEXT NOT NULL,
            status TEXT NOT NULL,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            eta TEXT NOT NULL,
            expedite_requested INTEGER NOT NULL DEFAULT 0,
            expedite_reason TEXT
        )
        """
    )
    if conn.execute("SELECT COUNT(*) FROM shipments").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO shipments (shipment_id, carrier, status, origin, destination, eta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _SEED_SHIPMENTS,
        )
        conn.commit()
    return conn


def reset_db() -> None:
    """Drop and reseed the table. Used by the eval harness for a clean slate."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS shipments")
    conn.commit()
    conn.close()
    get_connection().close()


if __name__ == "__main__":
    conn = get_connection()
    rows = conn.execute("SELECT * FROM shipments").fetchall()
    for row in rows:
        print(dict(row))
    conn.close()

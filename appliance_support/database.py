"""SQLite persistence: tickets, warranty records and agent telemetry."""
from __future__ import annotations

import datetime as dt
import random
import sqlite3
import uuid

import pandas as pd

import config


def db() -> sqlite3.Connection:
    return sqlite3.connect(config.DB_PATH, check_same_thread=False)


def init_db() -> None:
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets(
                id TEXT PRIMARY KEY, created_at TEXT, customer TEXT, product TEXT, category TEXT,
                priority TEXT, sentiment TEXT, status TEXT, resolved_by TEXT, csat INTEGER,
                resolution_minutes REAL, summary TEXT, has_image INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS agent_runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent TEXT, latency_s REAL);
            CREATE TABLE IF NOT EXISTS warranty(
                serial TEXT PRIMARY KEY, product TEXT, purchase_date TEXT,
                warranty_months INTEGER, customer TEXT);
            """
        )
        if c.execute("SELECT COUNT(*) FROM warranty").fetchone()[0] == 0:
            today = dt.date.today()
            rows = [
                ("WM-1001", "Washing Machine", str(today - dt.timedelta(days=200)), 24, "Arun K"),
                ("RF-2002", "Refrigerator", str(today - dt.timedelta(days=900)), 24, "Meena S"),
                ("AC-3003", "Air Conditioner", str(today - dt.timedelta(days=400)), 36, "Karthik R"),
                ("DW-4004", "Dishwasher", str(today - dt.timedelta(days=60)), 12, "Priya N"),
            ]
            c.executemany("INSERT INTO warranty VALUES (?,?,?,?,?)", rows)


def seed_demo_data(n: int = 160) -> None:
    """Populate the dashboard with synthetic history on first run."""
    with db() as c:
        if c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] > 0:
            return
        rnd = random.Random(7)
        now = dt.datetime.now()
        for _ in range(n):
            created = now - dt.timedelta(days=rnd.random() * 30, hours=rnd.random() * 10)
            ai_resolved = rnd.random() < 0.68
            status = ("Resolved" if ai_resolved or rnd.random() < 0.5
                      else rnd.choice(["Escalated", "In Progress"]))
            c.execute(
                "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"TKT-{uuid.uuid4().hex[:8].upper()}", created.isoformat(timespec="seconds"),
                 rnd.choice(["Arun", "Meena", "Karthik", "Priya", "Suresh", "Divya"]),
                 rnd.choice(config.PRODUCTS), rnd.choice(config.CATEGORIES),
                 rnd.choices(config.PRIORITIES, [4, 5, 3, 1])[0],
                 rnd.choices(config.SENTIMENTS, [2, 5, 3])[0], status,
                 "AI Agents" if ai_resolved else "Human Technician",
                 rnd.choice([3, 4, 4, 5, 5, 2, 1]) if status == "Resolved" else None,
                 round(rnd.uniform(1, 12), 1) if ai_resolved else round(rnd.uniform(60, 600), 1),
                 "Seeded demo ticket", int(rnd.random() < 0.3)),
            )
            for ag in ["Triage", "Knowledge", "Resolution", "QA"]:
                c.execute("INSERT INTO agent_runs(ts,agent,latency_s) VALUES (?,?,?)",
                          (created.isoformat(timespec="seconds"), ag, round(rnd.uniform(0.6, 5.5), 2)))


def log_run(agent: str, latency: float) -> None:
    with db() as c:
        c.execute("INSERT INTO agent_runs(ts,agent,latency_s) VALUES (?,?,?)",
                  (dt.datetime.now().isoformat(timespec="seconds"), agent, round(latency, 2)))


def save_ticket(customer, product, category, priority, sentiment, status, resolved_by,
                summary, resolution_minutes, has_image) -> str:
    tid = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    with db() as c:
        c.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (tid, dt.datetime.now().isoformat(timespec="seconds"), customer or "Guest",
                   product, category, priority, sentiment, status, resolved_by, None,
                   resolution_minutes, summary, int(bool(has_image))))
    return tid


def set_csat(ticket_id: str, score: int) -> None:
    with db() as c:
        c.execute("UPDATE tickets SET csat=? WHERE id=?", (score, ticket_id))


def lookup_warranty(serial: str):
    with db() as c:
        return c.execute(
            "SELECT product, purchase_date, warranty_months, customer FROM warranty WHERE serial=?",
            (serial.strip().upper(),)).fetchone()


def lookup_ticket(ticket_id: str):
    with db() as c:
        return c.execute(
            "SELECT id, status, product, category, created_at FROM tickets WHERE id=?",
            (ticket_id.strip().upper(),)).fetchone()


def load_tickets() -> pd.DataFrame:
    with db() as c:
        return pd.read_sql_query("SELECT * FROM tickets", c, parse_dates=["created_at"])


def load_runs() -> pd.DataFrame:
    with db() as c:
        return pd.read_sql_query("SELECT * FROM agent_runs", c, parse_dates=["ts"])

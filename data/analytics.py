import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

import streamlit as st


@st.cache_resource
def get_conn(base_path: str):
    """Return (conn, kind) where kind is 'sqlite' or 'pg'."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
        try:
            import psycopg2

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    visitor_id TEXT NOT NULL,
                    page TEXT NOT NULL,
                    first_ts TEXT NOT NULL,
                    last_ts TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    user_agent TEXT,
                    ip TEXT,
                    PRIMARY KEY (visitor_id, page)
                )
                """
            )
            conn.commit()
            return conn, "pg"
        except Exception as e:
            logging.warning(f"Postgres unavailable, falling back to SQLite: {e}")
    # SQLite fallback
    path = os.path.join(base_path, "analytics.db")
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS visits (
            visitor_id TEXT NOT NULL,
            page TEXT NOT NULL,
            first_ts TEXT NOT NULL,
            last_ts TEXT NOT NULL,
            count INTEGER NOT NULL,
            user_agent TEXT,
            ip TEXT,
            PRIMARY KEY (visitor_id, page)
        )
        """
    )
    try:
        cur.execute("PRAGMA table_info(visits)")
        cols = {r[1] for r in cur.fetchall()}
        if "user_agent" not in cols:
            cur.execute("ALTER TABLE visits ADD COLUMN user_agent TEXT")
        if "ip" not in cols:
            cur.execute("ALTER TABLE visits ADD COLUMN ip TEXT")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    return conn, "sqlite"


def record_visit(conn, kind: str, visitor_id: str, page: str, ua: str, ip: str):
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.cursor()
        if kind == "pg":
            cur.execute(
                """
                INSERT INTO visits (visitor_id, page, first_ts, last_ts, count, user_agent, ip)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (visitor_id, page)
                DO UPDATE SET last_ts=EXCLUDED.last_ts, count=visits.count+1, user_agent=EXCLUDED.user_agent, ip=EXCLUDED.ip
                """,
                (visitor_id, page, now, now, 1, ua, ip),
            )
        else:
            cur.execute(
                "SELECT count FROM visits WHERE visitor_id=? AND page=?",
                (visitor_id, page),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO visits (visitor_id, page, first_ts, last_ts, count, user_agent, ip) VALUES (?,?,?,?,?,?,?)",
                    (visitor_id, page, now, now, 1, ua, ip),
                )
            else:
                cur.execute(
                    "UPDATE visits SET last_ts=?, count=count+1, user_agent=?, ip=? WHERE visitor_id=? AND page=?",
                    (now, ua, ip, visitor_id, page),
                )
        conn.commit()
    except Exception as e:
        logging.debug(f"record_visit failed: {e}")


def query_summary(conn, kind: str, where: str = "", params: List = None) -> Dict:
    params = params or []
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(DISTINCT visitor_id) FROM visits {where}", params)
    unique_visitors = cur.fetchone()[0] or 0
    cur.execute(f"SELECT COALESCE(SUM(count),0) FROM visits {where}", params)
    total_visits = cur.fetchone()[0] or 0
    return {"unique_visitors": unique_visitors, "total_visits": total_visits}

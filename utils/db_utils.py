from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ---------- basic connections ----------
def get_connection(db_path: str) -> sqlite3.Connection:
    p = Path(db_path).expanduser().resolve()
    return sqlite3.connect(str(p))


# ---------- schema helpers ----------
def get_schema_and_columns(db_path: str, table_name: str) -> Tuple[str, List[str]]:
    """
    Returns (CREATE TABLE DDL string, [column names]) for `table_name` in the sqlite DB.
    """
    with get_connection(db_path) as conn:
        # Try to read the original CREATE statement if available
        q = "SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
        row = conn.execute(q, (table_name,)).fetchone()
        if row and row[0]:
            ddl = row[0].strip()
        else:
            # Fallback: build a CREATE from PRAGMA info
            info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            if not info:
                raise ValueError(f"Table '{table_name}' not found in DB '{db_path}'.")
            cols = [f'    {r[1]} {r[2] or ""}'.rstrip() for r in info]  # (cid,name,type,notnull,dflt,pk)
            ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n)"

        # Columns list
        pragma = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = [r[1] for r in pragma]

    return ddl, columns


# ---------- execution helpers ----------
def run_sql(db_path: str, sql: str) -> pd.DataFrame:
    """
    Executes SQL. If it returns rows, return a DataFrame; otherwise returns an empty DF.
    """
    with get_connection(db_path) as conn:
        sql_strip = sql.strip().lower()
        if sql_strip.startswith("select") or sql_strip.startswith("with"):
            return pd.read_sql_query(sql, conn)
        else:
            conn.execute(sql)
            conn.commit()
            # no rows to return
            return pd.DataFrame()
